"""Visual utils for checking fit of learned model.

Compares the categorical softmax baseline against the flow world model on
next-state prediction for JaxMARL CoinGame, split by transition regime:

* Deterministic transitions (player moves): did the model put mass on the one
  correct next cell? Shown as per-cell exact-prediction accuracy.
* Stochastic respawns (a coin is collected and reappears ~uniformly): is the
  predicted next-cell distribution close to uniform? Shown as the aggregate
  next-cell distribution against the uniform target.

A single aggregate per-entity marginal would smear deterministic moves across
the grid and let both models look "right" while being wrong on every
individual transition, so the regime split is what makes the plot diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from baselines.softmax_model import (
    GRID_SIZE,
    NUM_CELLS,
    SoftmaxBaselinePredictions,
    collected_coin_masks,
    deterministic_transition_mask,
    softmax_np,
)

_RED_COIN_ENTITY = 2
_BLUE_COIN_ENTITY = 3
_OBSERVED_AGENT_FRAME = 0


@dataclass(frozen=True)
class NextStateComparison:
    """Regime-split next-state comparison arrays for softmax vs flow.

    All grid fields are ``[GRID, GRID]``. Deterministic fields hold per-cell
    exact-prediction accuracy (binned by the true next cell); respawn fields
    hold the aggregate next-cell probability over coin respawn events.
    """

    det_softmax_accuracy: np.ndarray
    det_flow_accuracy: np.ndarray
    respawn_uniform: np.ndarray
    respawn_empirical: np.ndarray
    respawn_softmax: np.ndarray
    respawn_flow: np.ndarray
    det_exact_softmax: float
    det_exact_flow: float
    det_transition_count: int
    respawn_event_count: int
    respawn_tv_softmax: float
    respawn_tv_flow: float
    respawn_tv_empirical: float
    respawn_kl_softmax: float
    respawn_kl_flow: float
    respawn_kl_empirical: float
    num_flow_samples: int

    def to_metrics(self) -> dict[str, Any]:
        """JSON-friendly scalar summary for run logging."""
        return {
            "deterministic_transition_count": self.det_transition_count,
            "deterministic_full_state_exact_accuracy": {
                "softmax": _nan_to_none(self.det_exact_softmax),
                "flow": _nan_to_none(self.det_exact_flow),
            },
            "respawn_event_count": self.respawn_event_count,
            "respawn_total_variation_to_uniform": {
                "softmax": _nan_to_none(self.respawn_tv_softmax),
                "flow": _nan_to_none(self.respawn_tv_flow),
                "empirical": _nan_to_none(self.respawn_tv_empirical),
            },
            "respawn_kl_uniform_to_model": {
                "softmax": _nan_to_none(self.respawn_kl_softmax),
                "flow": _nan_to_none(self.respawn_kl_flow),
                "empirical": _nan_to_none(self.respawn_kl_empirical),
            },
            "num_flow_samples": self.num_flow_samples,
        }


def build_next_state_comparison(
    validation_data: Any,
    softmax_predictions: SoftmaxBaselinePredictions,
    flow_position_samples: np.ndarray,
) -> NextStateComparison:
    """Aggregate softmax and flow next-state predictions into regime-split grids.

    ``flow_position_samples`` are decoded flow next-position ids shaped ``[K, N, 2, 4]``
    (``K`` samples per validation transition). A trailing-only ``[N, 2, 4]``
    array is treated as a single sample.
    """
    positions = np.asarray(validation_data.positions, dtype=np.int32)
    actions = np.asarray(validation_data.actions, dtype=np.int32)
    dones = np.asarray(validation_data.dones, dtype=np.float32)
    true_next = np.asarray(validation_data.next_positions, dtype=np.int32)
    softmax_logits = np.asarray(
        softmax_predictions.next_position_logits, dtype=np.float64
    )
    softmax_pred = softmax_predictions.next_positions

    flow_position_samples = np.asarray(flow_position_samples, dtype=np.int32)
    if flow_position_samples.ndim == 3:
        flow_position_samples = flow_position_samples[None, ...]
    if flow_position_samples.ndim != 4 or flow_position_samples.shape[1:] != true_next.shape:
        raise ValueError(
            "flow_position_samples must be shaped [K, N, 2, 4] matching the "
            f"validation set; got {flow_position_samples.shape} vs {true_next.shape}"
        )
    num_flow_samples = int(flow_position_samples.shape[0])

    det_mask = np.asarray(
        deterministic_transition_mask(validation_data),
        dtype=bool,
    )
    det_softmax_acc, det_exact_softmax = _deterministic_accuracy(
        true_next, softmax_pred[None, ...], det_mask
    )
    det_flow_acc, det_exact_flow = _deterministic_accuracy(
        true_next, flow_position_samples, det_mask
    )

    respawn = _respawn_distributions(
        positions, actions, dones, true_next, softmax_logits, flow_position_samples
    )
    uniform = np.full((NUM_CELLS,), 1.0 / NUM_CELLS, dtype=np.float64)

    return NextStateComparison(
        det_softmax_accuracy=det_softmax_acc,
        det_flow_accuracy=det_flow_acc,
        respawn_uniform=_as_grid(uniform),
        respawn_empirical=_as_grid(respawn["empirical"]),
        respawn_softmax=_as_grid(respawn["softmax"]),
        respawn_flow=_as_grid(respawn["flow"]),
        det_exact_softmax=det_exact_softmax,
        det_exact_flow=det_exact_flow,
        det_transition_count=int(det_mask.sum()),
        respawn_event_count=respawn["count"],
        respawn_tv_softmax=_total_variation(respawn["softmax"], uniform),
        respawn_tv_flow=_total_variation(respawn["flow"], uniform),
        respawn_tv_empirical=_total_variation(respawn["empirical"], uniform),
        respawn_kl_softmax=_kl_uniform_to(respawn["softmax"], uniform),
        respawn_kl_flow=_kl_uniform_to(respawn["flow"], uniform),
        respawn_kl_empirical=_kl_uniform_to(respawn["empirical"], uniform),
        num_flow_samples=num_flow_samples,
    )


def plot_next_state_comparison(
    comparison: NextStateComparison,
    output: Path | None = None,
) -> None:
    """Render the regime-split softmax-vs-flow next-state comparison figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_panels = [
        (comparison.det_softmax_accuracy, "det: softmax accuracy"),
        (comparison.det_flow_accuracy, "det: flow accuracy"),
        (comparison.respawn_uniform, "respawn: uniform target"),
        (comparison.respawn_empirical, "respawn: empirical heldout"),
        (comparison.respawn_softmax, "respawn: softmax"),
        (comparison.respawn_flow, "respawn: flow"),
    ]
    bottom_panels = [
        (_deviation_from(comparison.det_softmax_accuracy, 1.0), "det: softmax miss"),
        (_deviation_from(comparison.det_flow_accuracy, 1.0), "det: flow miss"),
        (_deviation_from(comparison.respawn_uniform, None), "|uniform - unif|"),
        (_deviation_from(comparison.respawn_empirical, None), "|empirical - unif|"),
        (_deviation_from(comparison.respawn_softmax, None), "|softmax - unif|"),
        (_deviation_from(comparison.respawn_flow, None), "|flow - unif|"),
    ]

    fig, axes = plt.subplots(
        2,
        7,
        figsize=(18, 6.4),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 0.08]},
    )
    top_mappable = _draw_panel_row(
        plt, axes[0, :6], top_panels, vmin=0.0, vmax=1.0, cmap="viridis"
    )
    bottom_vmax = _row_vmax(bottom_panels)
    bottom_mappable = _draw_panel_row(
        plt, axes[1, :6], bottom_panels, vmin=0.0, vmax=bottom_vmax, cmap="magma"
    )

    fig.colorbar(top_mappable, cax=axes[0, 6], label="probability / accuracy")
    fig.colorbar(bottom_mappable, cax=axes[1, 6], label="abs. deviation from ideal")

    fig.suptitle(_suptitle(comparison), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _finish_figure(fig, output)


def _deterministic_accuracy(
    true_next: np.ndarray,
    predicted_samples: np.ndarray,
    det_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Per-cell exact accuracy (binned by true next cell) and full-state exact.

    ``predicted_samples`` is ``[K, N, 2, 4]`` of predicted next-cell ids.
    """
    if not bool(det_mask.any()):
        return _as_grid(np.full((NUM_CELLS,), np.nan, dtype=np.float64)), float("nan")

    det_true = true_next[det_mask]
    det_pred = predicted_samples[:, det_mask]
    true_broadcast = np.broadcast_to(det_true[None, ...], det_pred.shape)
    correct = (det_pred == true_broadcast).reshape(-1).astype(np.float64)
    true_flat = true_broadcast.reshape(-1)
    counts = np.bincount(true_flat, minlength=NUM_CELLS).astype(np.float64)
    hits = np.bincount(true_flat, weights=correct, minlength=NUM_CELLS)
    accuracy = np.divide(hits, counts, out=np.full(NUM_CELLS, np.nan), where=counts > 0)

    full_exact = np.all(det_pred == det_true[None, ...], axis=(2, 3))
    return _as_grid(accuracy), float(full_exact.mean())


def _respawn_distributions(
    positions: np.ndarray,
    actions: np.ndarray,
    dones: np.ndarray,
    true_next: np.ndarray,
    softmax_logits: np.ndarray,
    flow_position_samples: np.ndarray,
) -> dict[str, Any]:
    """Aggregate next-cell distributions over coin respawn events.

    Mirrors ``stochastic_respawn_metrics``: the respawned coin is read from the
    observed agent's frame (entity 2 = red coin, entity 3 = blue coin), masked
    by collection events on non-terminal transitions.
    """
    nonterminal = ~np.any(dones > 0.0, axis=1)
    red_collected, blue_collected = collected_coin_masks(positions, actions)
    red_collected = np.asarray(red_collected, dtype=bool)
    blue_collected = np.asarray(blue_collected, dtype=bool)
    selections = (
        (red_collected & nonterminal, _RED_COIN_ENTITY),
        (blue_collected & nonterminal, _BLUE_COIN_ENTITY),
    )

    softmax_probs = softmax_np(softmax_logits, axis=-1)
    uniform = np.full((NUM_CELLS,), 1.0 / NUM_CELLS, dtype=np.float64)

    softmax_rows: list[np.ndarray] = []
    flow_cells: list[np.ndarray] = []
    empirical_cells: list[np.ndarray] = []
    for mask, entity in selections:
        if not bool(mask.any()):
            continue
        softmax_rows.append(softmax_probs[mask, _OBSERVED_AGENT_FRAME, entity, :])
        flow_cells.append(
            flow_position_samples[:, mask, _OBSERVED_AGENT_FRAME, entity].reshape(-1)
        )
        empirical_cells.append(true_next[mask, _OBSERVED_AGENT_FRAME, entity])

    if not softmax_rows:
        return {
            "count": 0,
            "softmax": uniform.copy(),
            "flow": uniform.copy(),
            "empirical": uniform.copy(),
        }

    softmax_dist = np.concatenate(softmax_rows, axis=0).mean(axis=0)
    flow_dist = _normalized_histogram(np.concatenate(flow_cells, axis=0))
    empirical_dist = _normalized_histogram(np.concatenate(empirical_cells, axis=0))
    return {
        "count": int(sum(row.shape[0] for row in softmax_rows)),
        "softmax": softmax_dist,
        "flow": flow_dist,
        "empirical": empirical_dist,
    }


def _normalized_histogram(cells: np.ndarray) -> np.ndarray:
    counts = np.bincount(cells, minlength=NUM_CELLS).astype(np.float64)
    return counts / max(float(counts.sum()), 1.0)


def _total_variation(distribution: np.ndarray, uniform: np.ndarray) -> float:
    return float(0.5 * np.abs(distribution - uniform).sum())


def _kl_uniform_to(distribution: np.ndarray, uniform: np.ndarray) -> float:
    clipped = np.clip(distribution, 1e-12, 1.0)
    return float(np.sum(uniform * (np.log(uniform) - np.log(clipped))))


def _as_grid(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(
        (GRID_SIZE, GRID_SIZE)
    )


def _deviation_from(grid: np.ndarray, ideal: float | None) -> np.ndarray:
    if ideal is None:
        ideal = 1.0 / NUM_CELLS
    return np.abs(grid - ideal)


def _row_vmax(panels: list[tuple[np.ndarray, str]]) -> float:
    finite = [np.nanmax(grid) for grid, _ in panels if np.isfinite(grid).any()]
    candidate = max(finite) if finite else 0.0
    return float(candidate) if candidate > 0.0 else 1.0


def _draw_panel_row(plt, axes, panels, *, vmin: float, vmax: float, cmap: str):
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("lightgrey")
    mappable = None
    for ax, (grid, title) in zip(axes, panels, strict=True):
        mappable = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap=colormap, origin="upper")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        _annotate_cells(ax, grid, vmin=vmin, vmax=vmax)
    if mappable is None:
        raise ValueError("expected at least one panel to draw")
    return mappable


def _annotate_cells(ax, grid: np.ndarray, *, vmin: float, vmax: float) -> None:
    midpoint = 0.5 * (vmin + vmax)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            value = grid[row, col]
            if not np.isfinite(value):
                continue
            color = "white" if value < midpoint else "black"
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )


def _suptitle(comparison: NextStateComparison) -> str:
    det = (
        f"deterministic exact ({comparison.det_transition_count} transitions): "
        f"sm={_fmt(comparison.det_exact_softmax)} "
        f"flow={_fmt(comparison.det_exact_flow)}"
    )
    respawn = (
        f"respawn KL(unif||.) ({comparison.respawn_event_count} events): "
        f"sm={_fmt(comparison.respawn_kl_softmax)} "
        f"flow={_fmt(comparison.respawn_kl_flow)} "
        f"(empirical floor={_fmt(comparison.respawn_kl_empirical)})"
    )
    return (
        f"Next-state prediction: softmax vs flow ({comparison.num_flow_samples} "
        f"flow samples)\n{det}  |  {respawn}"
    )


def _fmt(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.3f}"


def _nan_to_none(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def _finish_figure(fig, output: Path | None) -> None:
    """Save or show a figure and always close it."""
    import matplotlib.pyplot as plt

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
