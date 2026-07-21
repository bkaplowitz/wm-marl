"""Plot decoded imagined rollouts against real trajectories for JEPA runs.

Reads ``decoder_rollout.npz`` artifacts written by ``train_dmc_jepa`` when
``--decoder-train-steps`` is set, and renders two figures next to each npz:

- ``decoder_rollout_frames.png``: per-trajectory filmstrips (observation dim x
  time heatmaps) for the real trajectory, the decoder's reconstruction of the
  encoder latents, the decoded imagined open-loop rollout, and the absolute
  real-vs-imagined error. The dashed line marks where imagination starts.
- ``decoder_rollout_traces.png``: per-dimension line traces of the same three
  signals, one column per trajectory.

The reconstruction row isolates decoder error from dynamics error: if the
reconstruction tracks the real frames but the imagined row drifts, the gap is
the world model's, not the decoder's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REAL_COLOR = "#52514e"
RECON_COLOR = "#2a78d6"
IMAGINED_COLOR = "#e34948"
FRAMES_FILENAME = "decoder_rollout_frames.png"
TRACES_FILENAME = "decoder_rollout_traces.png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help=(
            "A train_dmc_jepa run or experiment directory (repeatable); "
            "decoder_rollout.npz files are found recursively."
        ),
    )
    parser.add_argument(
        "--max-trace-dims",
        type=int,
        default=12,
        help="Cap on observation dimensions in the traces figure (by variance).",
    )
    return parser.parse_args(argv)


def _full_strips(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Concatenate context and horizon segments into full time strips."""
    real = np.concatenate(
        [payload["context_observations"], payload["real_observations"]],
        axis=1,
    )
    reconstructed = np.concatenate(
        [payload["decoded_context"], payload["reconstructed_observations"]],
        axis=1,
    )
    imagined = np.concatenate(
        [payload["decoded_context"], payload["imagined_observations"]],
        axis=1,
    )
    return {"real": real, "reconstructed": reconstructed, "imagined": imagined}


def _mean_cosine(payload: dict[str, np.ndarray], index: int) -> float:
    validity = payload["validity"][index]
    total = float(validity.sum())
    if total == 0.0:
        return float("nan")
    return float((payload["open_loop_cosine"][index] * validity).sum() / total)


def _mark_invalid_steps(ax, validity: np.ndarray, context_window: int) -> None:
    for step, valid in enumerate(np.asarray(validity)):
        if valid == 0.0:
            ax.axvspan(
                context_window + step - 0.5,
                context_window + step + 0.5,
                color="white",
                alpha=0.55,
                lw=0,
            )


def save_rollout_frames_plot(
    payload: dict[str, np.ndarray],
    out_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    out_path = Path(out_path)
    strips = _full_strips(payload)
    error = np.abs(strips["real"] - strips["imagined"])
    num_traj, strip_length, _ = strips["real"].shape
    context_window = payload["context_observations"].shape[1]

    signed_values = np.concatenate([strip.ravel() for strip in strips.values()])
    signed_max = float(np.percentile(np.abs(signed_values), 99.5)) or 1.0
    error_max = float(np.percentile(error, 99.5)) or 1.0
    signed_norm = TwoSlopeNorm(vmin=-signed_max, vcenter=0.0, vmax=signed_max)
    error_norm = Normalize(vmin=0.0, vmax=error_max)

    row_specs = [
        ("real", strips["real"], "RdBu_r", signed_norm),
        ("decoder(E(o))", strips["reconstructed"], "RdBu_r", signed_norm),
        ("imagined", strips["imagined"], "RdBu_r", signed_norm),
        ("|real - imagined|", error, "Blues", error_norm),
    ]
    fig, axes = plt.subplots(
        len(row_specs),
        num_traj,
        figsize=(3.4 * num_traj + 1.6, 8.0),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for column in range(num_traj):
        for row, (label, values, cmap, norm) in enumerate(row_specs):
            ax = axes[row][column]
            ax.imshow(
                values[column].T,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
            )
            ax.axvline(context_window - 0.5, color="black", ls="--", lw=1.0)
            _mark_invalid_steps(ax, payload["validity"][column], context_window)
            if column == 0:
                ax.set_ylabel(f"{label}\nobs dim")
            if row == 0:
                ax.set_title(
                    f"trajectory {column} · cos {_mean_cosine(payload, column):.3f}",
                    fontsize=10,
                )
            if row == len(row_specs) - 1:
                ax.set_xlabel("step in window")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=signed_norm, cmap="RdBu_r"),
        ax=[axes[row][-1] for row in range(3)],
        extend="both",
        label="observation value",
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=error_norm, cmap="Blues"),
        ax=[axes[3][-1]],
        extend="max",
        label="abs error",
    )
    fig.suptitle(
        title
        or (
            "Real vs imagined decoded rollout "
            f"(context={context_window}, horizon={strip_length - context_window})"
        )
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_rollout_traces_plot(
    payload: dict[str, np.ndarray],
    out_path: str | Path,
    *,
    title: str | None = None,
    max_dims: int = 12,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    strips = _full_strips(payload)
    num_traj, strip_length, observation_dim = strips["real"].shape
    context_window = payload["context_observations"].shape[1]

    variances = strips["real"].reshape((-1, observation_dim)).var(axis=0)
    shown_dims = np.sort(np.argsort(variances)[::-1][:max_dims])
    dim_note = ""
    if len(shown_dims) < observation_dim:
        dim_note = f" (top {len(shown_dims)} of {observation_dim} dims by variance)"

    fig, axes = plt.subplots(
        len(shown_dims),
        num_traj,
        figsize=(3.2 * num_traj, 1.15 * len(shown_dims) + 1.6),
        squeeze=False,
        sharex=True,
    )
    steps = np.arange(strip_length)
    for column in range(num_traj):
        for row, dim in enumerate(shown_dims):
            ax = axes[row][column]
            ax.plot(
                steps,
                strips["real"][column, :, dim],
                color=REAL_COLOR,
                lw=1.8,
                label="real",
            )
            ax.plot(
                steps,
                strips["reconstructed"][column, :, dim],
                color=RECON_COLOR,
                lw=1.4,
                ls=":",
                label="decoder(E(o))",
            )
            ax.plot(
                steps,
                strips["imagined"][column, :, dim],
                color=IMAGINED_COLOR,
                lw=1.8,
                label="imagined",
            )
            ax.axvline(context_window - 0.5, color="black", ls="--", lw=0.8)
            ax.grid(True, alpha=0.25)
            if column == 0:
                ax.set_ylabel(f"dim {dim}", fontsize=9)
            if row == 0:
                ax.set_title(
                    f"trajectory {column} · cos {_mean_cosine(payload, column):.3f}",
                    fontsize=10,
                )
            if row == len(shown_dims) - 1:
                ax.set_xlabel("step in window")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncols=3,
        bbox_to_anchor=(0.5, 0.955),
        frameon=False,
    )
    fig.suptitle(
        (
            title
            or (
                "Real vs imagined decoded traces "
                f"(context={context_window}, horizon={strip_length - context_window})"
            )
        )
        + dim_note,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _load_payload(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    npz_paths: list[Path] = []
    for run_dir in args.run_dir:
        if not run_dir.exists():
            print(f"missing run dir: {run_dir}", file=sys.stderr)
            raise SystemExit(2)
        npz_paths.extend(sorted(run_dir.rglob("decoder_rollout.npz")))
    if not npz_paths:
        print(
            "no decoder_rollout.npz found; run train_dmc_jepa with "
            "--decoder-train-steps first",
            file=sys.stderr,
        )
        raise SystemExit(2)
    for npz_path in npz_paths:
        payload = _load_payload(npz_path)
        frames = save_rollout_frames_plot(
            payload,
            npz_path.parent / FRAMES_FILENAME,
        )
        traces = save_rollout_traces_plot(
            payload,
            npz_path.parent / TRACES_FILENAME,
            max_dims=args.max_trace_dims,
        )
        print(frames)
        print(traces)


if __name__ == "__main__":
    main()
