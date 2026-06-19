"""Compare next-state predictors on CoinGame: discrete + linear flow vs a dumb baseline.

Fits three next-state predictors on the same held-out-fair env data and compares
them on training convergence, single-step held-out accuracy, on 25-step
autoregressive rollout:

  * ``discrete`` -- conditional discrete flow matching (CTMC tau-leaping, per-factor
    cross-entropy).
  * ``linear``   -- conditional continuous (OT/linear) flow matching (MSE).
  * ``baseline`` -- a simple one-shot categorical MLP classifier with cross-entropy loss.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from jaxmarl.environments.coin_game.coin_game import MOVES

from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter
from world_marl.scripts.verify_fitted_env import (
    _collect_combined_batch,
    _create_initial_policy_state,
    _write_loss_csv,
)
from world_marl.world_model import (
    VectorTransitionBatch,
    VectorWorldModelConfig,
    _cond_dim,
    _num_factors,
    _pack_cond_vars,
    _pack_discrete_tokens,
    _unpack_discrete_onehot,
    create_world_model_state,
    predict_next,
)
from world_marl.world_model_training import fit_world_model_steps

MOVES_NP = np.asarray(MOVES)

# Factor layout after _pack_discrete_tokens (d = num_agents*channels = 8):
#   f = agent*4 + channel; agent-0 grid is the absolute frame
#   [red_player, blue_player, red_coin, blue_coin]; agent-1 grid is colour-swapped.
PLAYER_FACTORS = (0, 1, 4, 5)
COIN_FACTORS = (2, 3, 6, 7)
# Which action column drives each player factor (red action = col 0, blue = col 1).
PLAYER_ACTION_COL = {0: 0, 1: 1, 4: 1, 5: 0}
# Agent-0 absolute-frame channels used for state-distribution / rollout rendering.
AGENT0_CHANNEL_LABELS = ("red_player", "blue_player", "red_coin", "blue_coin")


# --------------------------------------------------------------------------------------
# Dumb categorical baseline: one-shot MLP classifier trained with cross-entropy.
# --------------------------------------------------------------------------------------
class CategoricalBaseline(nn.Module):
    """One-shot per-factor classifier: ``cond_vars -> (d*V)`` logits.

    Mirrors :class:`MLPVectorField`'s trunk (Dense + SiLU) for capacity parity with
    the flow models, but takes only the conditioning (no ``x``/``t``) and emits
    ``d*V`` logits reshaped to ``(B, d, V)`` -- a plain classifier, not a vector field.
    """

    hidden_dims: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, cond_vars: jax.Array) -> jax.Array:
        layers = [
            layer for dim in self.hidden_dims for layer in (nn.Dense(dim), nn.silu)
        ]
        return nn.Sequential(layers + [nn.Dense(self.output_dim)])(cond_vars)


def create_baseline_state(
    key: jax.Array,
    config: VectorWorldModelConfig,
    hidden_dims: tuple[int, ...],
    learning_rate: float,
) -> TrainState:
    output_dim = _num_factors(config) * config.num_categories
    model = CategoricalBaseline(hidden_dims=hidden_dims, output_dim=output_dim)
    params = model.init(key, jnp.zeros((1, _cond_dim(config))))["params"]
    return TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(learning_rate)
    )


def _baseline_loss(params, apply_fn, batch: VectorTransitionBatch, config) -> jax.Array:
    """Cross-entropy of the clean next-state tokens, summed over factors then meaned.

    Identical reduction to ``conditioned_discrete_flow_matching_loss`` so the curves
    are co-plottable on one nats axis (both start at ``d*ln(V)``).
    """
    cond_vars = _pack_cond_vars(batch.states, batch.actions, config)
    z = _pack_discrete_tokens(batch.next_states, config)  # (B, d)
    logits = apply_fn({"params": params}, cond_vars).reshape(
        z.shape[0], z.shape[1], config.num_categories
    )
    token_ce = optax.softmax_cross_entropy_with_integer_labels(logits, z)
    return jnp.mean(jnp.sum(token_ce, axis=-1))


@partial(jax.jit, static_argnames=("config", "steps"))
def _fit_baseline_updates(state, batch, config, *, steps):
    """Fused full-batch CE fitting: one ``lax.scan`` step per gradient update."""

    def update(carry, _):
        loss, grads = jax.value_and_grad(_baseline_loss)(
            carry.params, carry.apply_fn, batch, config
        )
        return carry.apply_gradients(grads=grads), loss

    state, loss_history = jax.lax.scan(update, state, xs=None, length=steps)
    return state, loss_history


def fit_categorical_baseline_steps(state, batch, config, *, steps):
    """Scan-based baseline fitter mirroring ``fit_world_model_steps`` (no rng needed)."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    state, loss_history = _fit_baseline_updates(state, batch, config, steps=steps)
    return state, loss_history


def predict_next_baseline(state, key, states, actions, config, inference):
    """Baseline next-state: sample (or argmax) per-factor tokens, return one-hot grids."""
    cond_vars = _pack_cond_vars(states, actions, config)
    logits = state.apply_fn({"params": state.params}, cond_vars).reshape(
        states.shape[0], _num_factors(config), config.num_categories
    )
    if inference == "sample":
        tokens = jax.random.categorical(key, logits, axis=-1)
    else:
        tokens = jnp.argmax(logits, axis=-1)
    return _unpack_discrete_onehot(tokens, config)


def chunk_sizes(total: int, chunk: int) -> list[int]:
    full, rem = divmod(total, chunk)
    return [chunk] * full + ([rem] if rem else [])


def fit_fm_chunked(model_state, rng, batch, config, total_steps, chunk, label):
    """Fit a flow model in jitted-scan chunks, returning the full per-step history."""
    history, done, t0 = [], 0, time.time()
    for n in chunk_sizes(total_steps, chunk):
        model_state, rng, _, hist = fit_world_model_steps(
            model_state, rng, batch, config, steps=n
        )
        hist = np.asarray(jax.block_until_ready(hist))
        history.append(hist)
        done += n
        rate = done / max(time.time() - t0, 1e-9)
        print(
            f"[{label}] step {done}/{total_steps} loss={hist[-1]:.4f} ({rate:.1f} it/s)",
            flush=True,
        )
    return model_state, rng, np.concatenate(history)


def fit_baseline_chunked(state, batch, config, total_steps, chunk, label):
    history, done, t0 = [], 0, time.time()
    for n in chunk_sizes(total_steps, chunk):
        state, hist = fit_categorical_baseline_steps(state, batch, config, steps=n)
        hist = np.asarray(jax.block_until_ready(hist))
        history.append(hist)
        done += n
        rate = done / max(time.time() - t0, 1e-9)
        print(
            f"[{label}] step {done}/{total_steps} loss={hist[-1]:.4f} ({rate:.1f} it/s)",
            flush=True,
        )
    return state, np.concatenate(history)


# --------------------------------------------------------------------------------------
# Rollout fidelity (ported from the memory-validated throwaway, generalized over
# an arbitrary predict_fn so discrete / linear / baseline share one code path).
# --------------------------------------------------------------------------------------
def pack(states, decode_config) -> np.ndarray:
    return np.asarray(_pack_discrete_tokens(jnp.asarray(states), decode_config))


def _move_player_tokens(prev_tokens: np.ndarray, actions: np.ndarray) -> dict:
    """Deterministic next player token per player factor: (pos + MOVES[a]) % 3."""
    out = {}
    for f, col in PLAYER_ACTION_COL.items():
        row, colp = prev_tokens[:, f] // 3, prev_tokens[:, f] % 3
        mv = MOVES_NP[actions[:, col]]
        nr, nc = (row + mv[:, 0]) % 3, (colp + mv[:, 1]) % 3
        out[f] = nr * 3 + nc
    return out


def coin_reset_mask(
    states: np.ndarray | jnp.ndarray, env_actions: np.ndarray
) -> np.ndarray:
    """Per-factor ``(B, 8)`` bool: which coin factors respawn this step.

    Mirrors the collision predicate in ``coin_game_reward_done``: a coin resets iff
    either player's *new* position lands on it. Player factors are always False.
    """
    states = jnp.asarray(states)
    num_envs = states.shape[0]
    grid = states[:, 0].reshape((num_envs, 3, 3, 4))

    def pos(ch: int) -> jnp.ndarray:
        flat = jnp.argmax(grid[..., ch].reshape((num_envs, 9)), axis=-1)
        return jnp.stack([flat // 3, flat % 3], axis=-1)

    red_p, blue_p, red_c, blue_c = pos(0), pos(1), pos(2), pos(3)
    a = jnp.asarray(env_actions, dtype=jnp.int32)
    new_red = (red_p + MOVES[a[:, 0]]) % 3
    new_blue = (blue_p + MOVES[a[:, 1]]) % 3
    red_event = jnp.all(new_red == red_c, -1) | jnp.all(new_blue == red_c, -1)
    blue_event = jnp.all(new_red == blue_c, -1) | jnp.all(new_blue == blue_c, -1)
    f = jnp.zeros_like(red_event)
    mask = jnp.stack([f, f, red_event, blue_event, f, f, blue_event, red_event], axis=1)
    return np.asarray(mask)


def collect_real_trajectory(adapter, args, decode_config):
    """Fixed-action real rollout. Returns real tokens, states, action seq, reset mask."""
    num_agents = adapter.num_agents
    rng = np.random.default_rng(args.seed + 100)
    s0 = adapter.reset()  # (B, A, 36)
    real_states = [np.asarray(s0)]
    actions_seq, step_reset = [], []
    for _ in range(args.horizon):
        a = rng.integers(
            0, adapter.action_dim, size=(args.num_envs, num_agents)
        ).astype(np.int32)
        actions_seq.append(a)
        step_reset.append(coin_reset_mask(real_states[-1], a))
        step = adapter.step(a)
        real_states.append(np.asarray(step.observations))
    real_tokens = np.stack([pack(s, decode_config) for s in real_states])
    return real_tokens, real_states, np.stack(actions_seq), np.stack(step_reset)


def validate_real(real_tokens, actions_seq, step_reset):
    """Layout + episode-boundary guardrails. Raises on violation; returns cum mask."""
    horizon = step_reset.shape[0]
    cum = np.zeros_like(real_tokens, dtype=bool)
    coin_sel = np.zeros(real_tokens.shape[-1], dtype=bool)
    coin_sel[list(COIN_FACTORS)] = True
    for h in range(1, horizon + 1):
        cum[h] = cum[h - 1] | step_reset[h - 1]
        not_reset_coin = (~cum[h]) & coin_sel[None, :]
        if not np.all(real_tokens[h][not_reset_coin] == real_tokens[0][not_reset_coin]):
            raise AssertionError(
                f"non-reset coin changed at step {h} -- reset mask or layout is wrong"
            )
    teleports = 0
    for h in range(1, horizon + 1):
        expected = _move_player_tokens(real_tokens[h - 1], actions_seq[h - 1])
        for f, exp in expected.items():
            teleports += int(np.sum(exp != real_tokens[h][:, f]))
    if teleports:
        raise AssertionError(
            f"{teleports} player teleports -- episode boundary fired within horizon; "
            "reduce horizon below max_cycles"
        )
    return cum


def imagined_trajectory(predict_fn, s0, actions_seq, decode_config, key):
    """Autoregressive rollout feeding each predicted state back as the next input."""
    imagined = jnp.asarray(s0)
    tokens = [pack(imagined, decode_config)]
    for h in range(actions_seq.shape[0]):
        key, k = jax.random.split(key)
        imagined = predict_fn(imagined, jnp.asarray(actions_seq[h]), k)
        tokens.append(pack(imagined, decode_config))
    return np.stack(tokens)  # (H+1, B, 8)


def compute_metrics(real, img, cum, step_reset):
    """Per-step match curves + aggregate reset-cell distribution."""
    horizon = step_reset.shape[0]
    p, c = list(PLAYER_FACTORS), list(COIN_FACTORS)
    out = {
        "step": list(range(1, horizon + 1)),
        "player_imagined": [],
        "player_persistence": [],
        "coin_nonreset_imagined": [],
        "coin_nonreset_persistence": [],
        "coin_nonreset_count": [],
        "predictable_all_imagined": [],
    }
    for h in range(1, horizon + 1):
        out["player_imagined"].append(float(np.mean(img[h][:, p] == real[h][:, p])))
        out["player_persistence"].append(float(np.mean(real[h][:, p] == real[0][:, p])))
        coin_ok = ~cum[h][:, c]
        n = int(coin_ok.sum())
        out["coin_nonreset_count"].append(n)
        ci = np.array([])
        if n:
            ci = (img[h][:, c] == real[h][:, c])[coin_ok]
            cp = (real[h][:, c] == real[0][:, c])[coin_ok]
            out["coin_nonreset_imagined"].append(float(np.mean(ci)))
            out["coin_nonreset_persistence"].append(float(np.mean(cp)))
        else:
            out["coin_nonreset_imagined"].append(float("nan"))
            out["coin_nonreset_persistence"].append(float("nan"))
        pred_ok = np.concatenate(
            [(img[h][:, p] == real[h][:, p]).reshape(-1), ci if n else np.array([])]
        )
        out["predictable_all_imagined"].append(float(np.mean(pred_ok)))

    img_vals, real_vals = [], []
    for t in range(horizon):
        m = step_reset[t][:, c]
        if m.any():
            img_vals.append(img[t + 1][:, c][m])
            real_vals.append(real[t + 1][:, c][m])
    img_vals = np.concatenate(img_vals) if img_vals else np.array([], dtype=int)
    real_vals = np.concatenate(real_vals) if real_vals else np.array([], dtype=int)

    def hist(v):
        hh = np.bincount(v, minlength=9)[:9].astype(float)
        return (hh / hh.sum()).tolist() if hh.sum() else [0.0] * 9

    def entropy(v):
        hh = np.bincount(v, minlength=9)[:9].astype(float)
        hh = hh / hh.sum() if hh.sum() else hh
        nz = hh[hh > 0]
        return float(-(nz * np.log(nz)).sum())

    out["reset_distribution"] = {
        "num_events": int(img_vals.size),
        "argmax_match_rate": (
            float(np.mean(img_vals == real_vals)) if img_vals.size else float("nan")
        ),
        "imagined_cell_hist": hist(img_vals),
        "real_cell_hist": hist(real_vals),
        "imagined_entropy": entropy(img_vals),
        "real_entropy": entropy(real_vals),
        "uniform_entropy": float(np.log(9)),
    }
    return out


def token_accuracy_breakdown(predict_fn, batch, decode_config, key):
    """Single-step held-out per-factor accuracy, split player/coin, vs copy-through."""
    pred_next = predict_fn(jnp.asarray(batch.states), jnp.asarray(batch.actions), key)
    pred = pack(pred_next, decode_config)
    true = pack(batch.next_states, decode_config)
    cur = pack(batch.states, decode_config)
    p, c = list(PLAYER_FACTORS), list(COIN_FACTORS)

    def acc(a, b, cols=None):
        return (
            float(np.mean(a == b))
            if cols is None
            else float(np.mean(a[:, cols] == b[:, cols]))
        )

    return {
        "overall": acc(pred, true),
        "player": acc(pred, true, p),
        "coin": acc(pred, true, c),
        "copy_through_overall": acc(cur, true),
        "copy_through_player": acc(cur, true, p),
        "copy_through_coin": acc(cur, true, c),
    }


def _downsample(y, max_points=4000):
    y = np.asarray(y)
    if len(y) <= max_points:
        return np.arange(1, len(y) + 1), y
    k = int(np.ceil(len(y) / max_points))
    idx = np.arange(0, len(y), k)
    return idx + 1, y[idx]


def plot_loss_curves(out_dir: Path, loss_histories: dict, uniform_ce: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ce_names = [n for n in loss_histories if n in ("discrete", "baseline")]
    mse_names = [n for n in loss_histories if n not in ("discrete", "baseline")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ce_ax, mse_ax = axes

    for name in ce_names:
        x, y = _downsample(loss_histories[name])
        ce_ax.plot(x, y, label=name)
    ce_ax.axhline(
        uniform_ce, color="k", ls=":", alpha=0.6, label=f"uniform={uniform_ce:.2f}"
    )
    ce_ax.set_xlabel("fit step")
    ce_ax.set_ylabel("cross-entropy (nats, sum over factors)")
    ce_ax.set_title(
        "Convergence DIAGNOSTIC (not a ranking):\nCE units, discrete vs baseline"
    )
    ce_ax.set_yscale("log")
    ce_ax.legend()
    ce_ax.grid(True, alpha=0.25)
    if not ce_names:
        ce_ax.text(0.5, 0.5, "n/a", ha="center", va="center")

    for name in mse_names:
        x, y = _downsample(loss_histories[name])
        mse_ax.plot(x, y, label=name)
    mse_ax.set_xlabel("fit step")
    mse_ax.set_ylabel("flow-matching MSE")
    mse_ax.set_title("Continuous flow loss (separate scale)")
    mse_ax.legend()
    mse_ax.grid(True, alpha=0.25)
    if not mse_names:
        mse_ax.text(0.5, 0.5, "n/a", ha="center", va="center")

    fig.text(
        0.5,
        0.005,
        "Discrete FM CE is averaged over corruption levels t~U(0,1), NOT a clean-conditioned NLL; "
        "use held-out accuracy + rollout fidelity to rank models.",
        ha="center",
        fontsize=8,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_dir / "loss_curves.png")
    plt.close(fig)


def plot_curves(out_dir: Path, results: dict, horizon: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = list(range(1, horizon + 1))
    any_m = next(iter(results.values()))

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, m in results.items():
        ax.plot(steps, m["player_imagined"], marker="o", ms=3, label=f"{name} imagined")
    ax.plot(
        steps, any_m["player_persistence"], "k--", label="persistence (copy-through)"
    )
    ax.axhline(1 / 9, color="red", alpha=0.3, label="1/9 chance")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("player-factor match")
    ax.set_title("Player trajectory fidelity (deterministic; should stay ~1.0)")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "player_drift.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, m in results.items():
        ax.plot(
            steps,
            m["coin_nonreset_imagined"],
            marker="o",
            ms=3,
            label=f"{name} imagined",
        )
    ax.plot(steps, any_m["coin_nonreset_persistence"], "k--", label="persistence")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("non-reset coin match")
    ax.set_title("Non-reset coin fidelity (copy-through is the bar)")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "coin_nonreset.png")
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(results), figsize=(5 * len(results), 4), squeeze=False
    )
    cells = np.arange(9)
    for ax, (name, m) in zip(axes[0], results.items()):
        rd = m["reset_distribution"]
        ax.bar(cells - 0.2, rd["real_cell_hist"], width=0.4, label="real (uniform)")
        ax.bar(cells + 0.2, rd["imagined_cell_hist"], width=0.4, label=f"{name} sample")
        ax.axhline(1 / 9, color="k", ls=":", alpha=0.6)
        ax.set_title(
            f"{name} reset cells (n={rd['num_events']})\n"
            f"H_img={rd['imagined_entropy']:.2f} vs H_unif={rd['uniform_entropy']:.2f}"
        )
        ax.set_xlabel("grid cell")
        ax.set_ylabel("frequency")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reset_distribution.png")
    plt.close(fig)


def plot_occupancy(
    out_dir: Path, real_tokens, imagined_tokens: dict, horizon: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = {"real": real_tokens, **imagined_tokens}

    def occ(tokens, factor):
        vals = tokens[1 : horizon + 1, :, factor].ravel()
        h = np.bincount(vals, minlength=9)[:9].astype(float)
        h = h / h.sum() if h.sum() else h
        return h.reshape(3, 3)

    for factor, label in enumerate(AGENT0_CHANNEL_LABELS):
        fig, axes = plt.subplots(
            1, len(sources), figsize=(3 * len(sources), 3.4), squeeze=False
        )
        for ax, (name, tokens) in zip(axes[0], sources.items()):
            im = ax.imshow(occ(tokens, factor), cmap="viridis", vmin=0)
            ax.set_title(name)
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"{label} cell occupancy over rollout (real vs predicted)")
        fig.tight_layout()
        fig.savefig(out_dir / f"occupancy_{label}.png")
        plt.close(fig)


def _board_from_tokens(row: np.ndarray) -> np.ndarray:
    """Agent-0 frame -> 3x3 int board: 0 empty, 1 red_coin, 2 blue_coin, 3 red_player, 4 blue_player."""
    board = np.zeros((3, 3), dtype=int)
    for token, code in ((row[2], 1), (row[3], 2), (row[0], 3), (row[1], 4)):
        board[token // 3, token % 3] = code  # players drawn last (overwrite coins)
    return board


def plot_example_rollout(
    out_dir: Path, real_tokens, imagined_tokens: dict, horizon: int, env: int = 0
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    sources = {"real": real_tokens, **imagined_tokens}
    rows = list(sources)
    cols = sorted(
        set(np.linspace(0, horizon, min(horizon + 1, 8)).astype(int).tolist())
    )
    cmap = mcolors.ListedColormap(["white", "#ffb3b3", "#b3b3ff", "#cc0000", "#0000cc"])

    fig, axes = plt.subplots(
        len(rows), len(cols), figsize=(1.5 * len(cols), 1.6 * len(rows)), squeeze=False
    )
    for r, name in enumerate(rows):
        tokens = sources[name]
        for cidx, step in enumerate(cols):
            ax = axes[r][cidx]
            ax.imshow(_board_from_tokens(tokens[step, env]), cmap=cmap, vmin=0, vmax=4)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={step}", fontsize=9)
            if cidx == 0:
                ax.set_ylabel(name, fontsize=9)
    legend = [
        mpatches.Patch(color="#cc0000", label="red player"),
        mpatches.Patch(color="#0000cc", label="blue player"),
        mpatches.Patch(color="#ffb3b3", label="red coin"),
        mpatches.Patch(color="#b3b3ff", label="blue coin"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(f"Example rollout (env {env}): real vs predicted")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out_dir / "example_rollout.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=("ippo", "mappo"), default="ippo")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-cycles", type=int, default=1000)
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--fit-steps", type=int, default=40000)
    p.add_argument("--chunk-steps", type=int, default=2000)
    p.add_argument("--train-random-rollouts", type=int, default=64)
    p.add_argument("--train-initial-rollouts", type=int, default=64)
    p.add_argument("--heldout-random-rollouts", type=int, default=16)
    p.add_argument("--heldout-initial-rollouts", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--integration-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--num-categories", type=int, default=9)
    p.add_argument("--flow-types", nargs="+", default=["discrete", "linear"])
    p.add_argument(
        "--baseline-inference", choices=("sample", "argmax"), default="sample"
    )
    p.add_argument("--out-dir", default="runs/compare_world_models")
    args = p.parse_args()
    for name in ("num_envs", "horizon", "fit_steps", "chunk_steps", "hidden_dim"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"jax devices: {jax.devices()}", flush=True)

    adapter = JaxMARLCoinGameVectorAdapter(
        num_envs=args.num_envs, max_cycles=args.max_cycles, seed=args.seed
    )
    try:
        state_dim = adapter.observation_shape[0]
        hidden_dims = (args.hidden_dim, args.hidden_dim)
        decode_config = VectorWorldModelConfig(
            state_dim=state_dim,
            num_agents=adapter.num_agents,
            action_dim=adapter.action_dim,
            hidden_dims=hidden_dims,
            learning_rate=args.learning_rate,
            integration_steps=args.integration_steps,
            num_categories=args.num_categories,
        )
        uniform_ce = float(_num_factors(decode_config) * np.log(args.num_categories))

        rng = jax.random.PRNGKey(args.seed)
        rng, policy_key = jax.random.split(rng)
        policy_state = _create_initial_policy_state(args, adapter, policy_key)

        observations = adapter.reset()
        train_batch, observations, rng = _collect_combined_batch(
            args, adapter, policy_state, observations, rng,
            random_seed=args.seed + 1,
            random_rollouts=args.train_random_rollouts,
            initial_rollouts=args.train_initial_rollouts,
        )
        observations = adapter.reset()
        heldout_batch, _, rng = _collect_combined_batch(
            args, adapter, policy_state, observations, rng,
            random_seed=args.seed + 2,
            random_rollouts=args.heldout_random_rollouts,
            initial_rollouts=args.heldout_initial_rollouts,
        )
        print(
            f"train transitions: {int(train_batch.states.shape[0])} | "
            f"heldout: {int(heldout_batch.states.shape[0])}",
            flush=True,
        )

        real_tokens, real_states, actions_seq, step_reset = collect_real_trajectory(
            adapter, args, decode_config
        )
        cum = validate_real(real_tokens, actions_seq, step_reset)
        total_resets = int(step_reset[:, :, list(COIN_FACTORS)].sum())
        print(
            f"guardrails OK | horizon={args.horizon} | coin reset events={total_resets} "
            f"| 0 player teleports",
            flush=True,
        )

        # Shared init seed + shared eval keys -> identical comparison across predictors.
        model_key = jax.random.PRNGKey(args.seed + 7)
        fit_rng = jax.random.PRNGKey(args.seed + 11)
        roll_key = jax.random.PRNGKey(args.seed + 200)
        heldout_key = jax.random.PRNGKey(args.seed + 300)

        loss_histories, results, imagined_tokens, heldout_acc = {}, {}, {}, {}

        for flow in args.flow_types:
            config = dataclasses.replace(
                decode_config,
                flow_type=flow,
                num_categories=(args.num_categories if flow == "discrete" else 0),
            )
            model_state = create_world_model_state(model_key, config)
            model_state, _, history = fit_fm_chunked(
                model_state, fit_rng, train_batch, config,
                args.fit_steps, args.chunk_steps, flow,
            )
            loss_histories[flow] = history

            def predict_fn(s, a, k, st=model_state, cfg=config):
                return predict_next(st, k, s, a, cfg)

            img = imagined_trajectory(
                predict_fn, real_states[0], actions_seq, decode_config, roll_key
            )
            results[flow] = compute_metrics(real_tokens, img, cum, step_reset)
            imagined_tokens[flow] = img
            heldout_acc[flow] = token_accuracy_breakdown(
                predict_fn, heldout_batch, decode_config, heldout_key
            )
            _report(flow, results[flow], heldout_acc[flow], history)

        # --- dumb categorical baseline ---
        baseline_state = create_baseline_state(
            model_key, decode_config, hidden_dims, args.learning_rate
        )
        baseline_state, history = fit_baseline_chunked(
            baseline_state, train_batch, decode_config,
            args.fit_steps, args.chunk_steps, "baseline",
        )
        loss_histories["baseline"] = history

        def predict_fn_baseline(s, a, k):
            return predict_next_baseline(
                baseline_state, k, s, a, decode_config, args.baseline_inference
            )

        img = imagined_trajectory(
            predict_fn_baseline, real_states[0], actions_seq, decode_config, roll_key
        )
        results["baseline"] = compute_metrics(real_tokens, img, cum, step_reset)
        imagined_tokens["baseline"] = img
        heldout_acc["baseline"] = token_accuracy_breakdown(
            predict_fn_baseline, heldout_batch, decode_config, heldout_key
        )
        _report("baseline", results["baseline"], heldout_acc["baseline"], history)

        # --- plots + artifacts ---
        plot_loss_curves(out_dir, loss_histories, uniform_ce)
        plot_curves(out_dir, results, args.horizon)
        plot_occupancy(out_dir, real_tokens, imagined_tokens, args.horizon)
        plot_example_rollout(out_dir, real_tokens, imagined_tokens, args.horizon)
        for name, history in loss_histories.items():
            _write_loss_csv(out_dir / f"loss_{name}.csv", history.tolist())

        summary = {
            "args": vars(args),
            "decode_config": dataclasses.asdict(decode_config),
            "uniform_cross_entropy": uniform_ce,
            "train_transition_count": int(train_batch.states.shape[0]),
            "heldout_transition_count": int(heldout_batch.states.shape[0]),
            "total_coin_reset_events": total_resets,
            "loss_summary": {
                name: {
                    "first": float(h[0]),
                    "last": float(h[-1]),
                    "min": float(np.min(h)),
                    "mean_last_50": float(np.mean(h[-min(50, len(h)) :])),
                }
                for name, h in loss_histories.items()
            },
            "heldout_accuracy": heldout_acc,
            "rollout_results": results,
        }
        (out_dir / "compare_world_models.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {out_dir / 'compare_world_models.json'} + plots", flush=True)
    finally:
        adapter.close()


def _report(name, metrics, acc, history):
    rd = metrics["reset_distribution"]
    print(
        f"[{name}] fit loss first={history[0]:.4f} last={history[-1]:.4f} "
        f"min={np.min(history):.4f}",
        flush=True,
    )
    print(
        f"[{name}] heldout acc overall={acc['overall']:.3f} player={acc['player']:.3f} "
        f"coin={acc['coin']:.3f} (copy-through overall={acc['copy_through_overall']:.3f}) | "
        f"rollout player step1={metrics['player_imagined'][0]:.3f} "
        f"stepH={metrics['player_imagined'][-1]:.3f} | reset H_img="
        f"{rd['imagined_entropy']:.2f}/{rd['uniform_entropy']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
