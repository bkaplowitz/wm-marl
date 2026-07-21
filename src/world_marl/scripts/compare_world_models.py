"""Compare next-state predictors on CoinGame: discrete + linear flow vs a dumb baseline.

Fits three next-state predictors on identical env data and scores them on training
convergence plus two evaluation regimes. Every number reported is one (factor kind ×
regime) cell, so the metric names below are organized around those two axes.

FACTOR KINDS -- what the true next value looks like, given (state, action):
  * DETERMINISTIC -- players move ``(pos + MOVES[a]) % 3`` and uncollected coins stay put,
    so there is exactly one correct next cell. The score is ACCURACY (fraction of cells
    matching the single truth): a perfect model reaches 1.0, and COPY_THROUGH (freeze the
    current state and predict no change) is the baseline to beat.
  * STOCHASTIC -- a collected coin RESPAWNS uniformly over the 9 grid cells, so there is no
    single truth and per-cell accuracy is meaningless (ceiling 1/9). We instead compare the
    predicted respawn-cell DISTRIBUTION (histogram + entropy) against the real draws and the
    uniform reference.

REGIMES -- how far ahead we predict:
  * SINGLE_STEP (``single_step_accuracy``) -- one step ahead on a held-out batch.
  * ROLLOUT (``rollout_tracking``) -- an H-step autoregressive rollout that feeds each
    prediction back as the next input: per-step player / uncollected-coin accuracy plus the
    aggregate respawn distribution.

Predictors compared:
  * ``discrete`` -- conditional discrete flow matching (CTMC tau-leaping, per-factor
    cross-entropy).
  * ``linear``   -- conditional continuous (OT/linear) flow matching (MSE).
  * ``baseline`` -- a dumb one-shot categorical MLP classifier with cross-entropy loss.
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
    LLaDA2WorldModelConfig,
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


# Dumb categorical baseline: one-shot MLP classifier trained with cross-entropy.
class CategoricalBaseline(nn.Module):
    """Per-factor classifier ``cond_vars -> (d*V)`` logits; mirrors MLPVectorField's trunk for capacity parity (no ``x``/``t``)."""

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
    """CE of next-state tokens, summed over factors then meaned -- same reduction as
    ``conditioned_discrete_flow_matching_loss`` so both curves start at ``d*ln(V)``."""
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


def predict_next_baseline(
    state: TrainState,
    key: jax.Array,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    config: VectorWorldModelConfig,
    inference: str,
) -> jnp.ndarray:
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


def fit_chunked(carry, step_fn, total_steps, chunk, label):
    """Drive any scan-based fitter in jitted chunks; ``step_fn(carry, n) -> (carry, hist)``."""
    history, done, t0 = [], 0, time.time()
    for n in chunk_sizes(total_steps, chunk):
        carry, hist = step_fn(carry, n)
        hist = np.asarray(jax.block_until_ready(hist))
        history.append(hist)
        done += n
        rate = done / max(time.time() - t0, 1e-9)
        print(
            f"[{label}] step {done}/{total_steps} loss={hist[-1]:.4f} ({rate:.1f} it/s)",
            flush=True,
        )
    return carry, np.concatenate(history)


# Token packing + rollout helpers, generalized over an arbitrary predict_fn so the
# discrete / linear / baseline predictors share one code path.
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


def coin_respawn_mask(
    states: np.ndarray | jnp.ndarray, env_actions: np.ndarray
) -> np.ndarray:
    """Per-factor ``(B, 8)`` bool: which coin factors respawn this step.

    Mirrors the collision predicate in ``coin_game_reward_done``: a coin respawns iff
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


def collect_real_rollout(adapter, args, decode_config):
    """Fixed-action real rollout. Returns real tokens, states, action seq, per-step respawn mask."""
    num_agents = adapter.num_agents
    rng = np.random.default_rng(args.seed + 100)
    s0 = adapter.reset()  # (B, A, 36)
    real_states = [np.asarray(s0)]
    actions_seq, step_respawn = [], []
    for _ in range(args.horizon):
        a = rng.integers(
            0, adapter.action_dim, size=(args.num_envs, num_agents)
        ).astype(np.int32)
        actions_seq.append(a)
        step_respawn.append(coin_respawn_mask(real_states[-1], a))
        step = adapter.step(a)
        real_states.append(np.asarray(step.observations))
    real_tokens = np.stack([pack(s, decode_config) for s in real_states])
    return real_tokens, real_states, np.stack(actions_seq), np.stack(step_respawn)


def validate_real_rollout(real_tokens, actions_seq, step_respawn):
    """Layout + episode-boundary guardrails. Raises on violation; returns the cumulative
    respawn mask (True once a coin factor has respawned by step h)."""
    horizon = step_respawn.shape[0]
    respawned = np.zeros_like(real_tokens, dtype=bool)
    coin_sel = np.zeros(real_tokens.shape[-1], dtype=bool)
    coin_sel[list(COIN_FACTORS)] = True
    for h in range(1, horizon + 1):
        respawned[h] = respawned[h - 1] | step_respawn[h - 1]
        uncollected_coin = (~respawned[h]) & coin_sel[None, :]
        if not np.all(
            real_tokens[h][uncollected_coin] == real_tokens[0][uncollected_coin]
        ):
            raise AssertionError(
                f"uncollected coin changed at step {h} -- respawn mask or layout is wrong"
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
    return respawned


def predicted_rollout(predict_fn, s0, actions_seq, decode_config, key):
    """Autoregressive rollout feeding each predicted state back as the next input. (H+1, B, 8)."""
    predicted = jnp.asarray(s0)
    tokens = [pack(predicted, decode_config)]
    for h in range(actions_seq.shape[0]):
        key, k = jax.random.split(key)
        predicted = predict_fn(predicted, jnp.asarray(actions_seq[h]), k)
        tokens.append(pack(predicted, decode_config))
    return np.stack(tokens)  # (H+1, B, 8)


def _norm_hist(v) -> np.ndarray:
    """9-cell occupancy distribution of token values (all-zeros if empty)."""
    hh = np.bincount(v, minlength=9)[:9].astype(float)
    return hh / hh.sum() if hh.sum() else hh


def _entropy(p) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def rollout_tracking_metrics(real, predicted, respawned, step_respawn):
    """Score an autoregressive rollout against the real one.

    Deterministic factors (players, uncollected coins) get per-step ACCURACY curves plus
    their copy-through baseline; the just-respawned (stochastic) coins instead feed the
    aggregate respawn-cell DISTRIBUTION. ``real``/``predicted`` are token arrays ``(H+1, B,
    8)``; ``respawned`` is the cumulative respawn mask, ``step_respawn`` the per-step one.
    """
    horizon = step_respawn.shape[0]
    p, c = list(PLAYER_FACTORS), list(COIN_FACTORS)
    out = {
        "step": list(range(1, horizon + 1)),
        "player_accuracy": [],
        "player_copy_through": [],
        "uncollected_coin_accuracy": [],
        "uncollected_coin_copy_through": [],
        "uncollected_coin_count": [],
        "deterministic_accuracy": [],
    }
    for h in range(1, horizon + 1):
        out["player_accuracy"].append(
            float(np.mean(predicted[h][:, p] == real[h][:, p]))
        )
        out["player_copy_through"].append(
            float(np.mean(real[h][:, p] == real[0][:, p]))
        )
        uncollected = ~respawned[h][:, c]
        n = int(uncollected.sum())
        out["uncollected_coin_count"].append(n)
        coin_hit = np.array([])
        if n:
            coin_hit = (predicted[h][:, c] == real[h][:, c])[uncollected]
            coin_copy = (real[h][:, c] == real[0][:, c])[uncollected]
            out["uncollected_coin_accuracy"].append(float(np.mean(coin_hit)))
            out["uncollected_coin_copy_through"].append(float(np.mean(coin_copy)))
        else:
            out["uncollected_coin_accuracy"].append(float("nan"))
            out["uncollected_coin_copy_through"].append(float("nan"))
        deterministic_hit = np.concatenate(
            [
                (predicted[h][:, p] == real[h][:, p]).reshape(-1),
                coin_hit if n else np.array([]),
            ]
        )
        out["deterministic_accuracy"].append(float(np.mean(deterministic_hit)))

    pred_vals, real_vals = [], []
    for t in range(horizon):
        m = step_respawn[t][:, c]
        if m.any():
            pred_vals.append(predicted[t + 1][:, c][m])
            real_vals.append(real[t + 1][:, c][m])
    pred_vals = np.concatenate(pred_vals) if pred_vals else np.array([], dtype=int)
    real_vals = np.concatenate(real_vals) if real_vals else np.array([], dtype=int)

    pred_p, real_p = _norm_hist(pred_vals), _norm_hist(real_vals)
    out["respawn_distribution"] = {
        "num_respawn_events": int(pred_vals.size),
        "predicted_cell_hist": pred_p.tolist(),
        "real_cell_hist": real_p.tolist(),
        "predicted_entropy": _entropy(pred_p),
        "real_entropy": _entropy(real_p),
        "uniform_entropy": float(np.log(9)),
    }
    return out


def single_step_accuracy(predict_fn, batch, decode_config, key):
    """One-step-ahead per-factor accuracy on a held-out batch, split player/coin, against
    the copy-through (freeze current state) baseline."""
    pred_next = predict_fn(jnp.asarray(batch.states), jnp.asarray(batch.actions), key)
    predicted = pack(pred_next, decode_config)
    real = pack(batch.next_states, decode_config)
    current = pack(batch.states, decode_config)
    p, c = list(PLAYER_FACTORS), list(COIN_FACTORS)

    def acc(a, b, cols=None):
        return (
            float(np.mean(a == b))
            if cols is None
            else float(np.mean(a[:, cols] == b[:, cols]))
        )

    return {
        "overall": acc(predicted, real),
        "player": acc(predicted, real, p),
        "coin": acc(predicted, real, c),
        "copy_through_overall": acc(current, real),
        "copy_through_player": acc(current, real, p),
        "copy_through_coin": acc(current, real, c),
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

    ce_names = [
        n for n in loss_histories if n in ("discrete", "transformer", "baseline")
    ]
    mse_names = [
        n for n in loss_histories if n not in ("discrete", "transformer", "baseline")
    ]

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
        "use single-step accuracy + rollout tracking to rank models.",
        ha="center",
        fontsize=8,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_dir / "loss_curves.png")
    plt.close(fig)


def _plot_rollout_accuracy(ax, steps, rollout_metrics, key, copy_key, *, chance_line):
    """One per-step accuracy panel: each predictor's curve plus the shared copy-through bar."""
    for name, m in rollout_metrics.items():
        ax.plot(steps, m[key], marker="o", ms=3, label=f"{name} predicted")
    any_m = next(iter(rollout_metrics.values()))
    ax.plot(steps, any_m[copy_key], "k--", label="copy-through")
    if chance_line:
        ax.axhline(1 / 9, color="red", alpha=0.3, label="1/9 chance")
    ax.set_xlabel("rollout step")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.25)


def plot_rollout_tracking(out_dir: Path, rollout_metrics: dict, horizon: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = list(range(1, horizon + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_rollout_accuracy(
        ax,
        steps,
        rollout_metrics,
        "player_accuracy",
        "player_copy_through",
        chance_line=True,
    )
    ax.set_ylabel("player-factor cell accuracy")
    ax.set_title("Rollout player accuracy (deterministic target; should stay ~1.0)")
    fig.tight_layout()
    fig.savefig(out_dir / "rollout_player_accuracy.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_rollout_accuracy(
        ax,
        steps,
        rollout_metrics,
        "uncollected_coin_accuracy",
        "uncollected_coin_copy_through",
        chance_line=False,
    )
    ax.set_ylabel("uncollected-coin cell accuracy")
    ax.set_title("Rollout uncollected-coin accuracy (copy-through is the bar)")
    fig.tight_layout()
    fig.savefig(out_dir / "rollout_uncollected_coin_accuracy.png")
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(rollout_metrics), figsize=(5 * len(rollout_metrics), 4), squeeze=False
    )
    cells = np.arange(9)
    for ax, (name, m) in zip(axes[0], rollout_metrics.items()):
        rd = m["respawn_distribution"]
        ax.bar(cells - 0.2, rd["real_cell_hist"], width=0.4, label="real (uniform)")
        ax.bar(
            cells + 0.2, rd["predicted_cell_hist"], width=0.4, label=f"{name} sample"
        )
        ax.axhline(1 / 9, color="k", ls=":", alpha=0.6)
        ax.set_title(
            f"{name} respawn cells (n={rd['num_respawn_events']})\n"
            f"H_pred={rd['predicted_entropy']:.2f} vs H_unif={rd['uniform_entropy']:.2f}"
        )
        ax.set_xlabel("grid cell")
        ax.set_ylabel("frequency")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "respawn_distribution.png")
    plt.close(fig)


def plot_occupancy(
    out_dir: Path, real_tokens, predicted_tokens: dict, horizon: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = {"real": real_tokens, **predicted_tokens}

    def occ(tokens, factor):
        vals = tokens[1 : horizon + 1, :, factor].ravel()
        return _norm_hist(vals).reshape(3, 3)

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
    out_dir: Path, real_tokens, predicted_tokens: dict, horizon: int, env: int = 0
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    sources = {"real": real_tokens, **predicted_tokens}
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


def count_params(params, *, num_experts: int = 1, expert_top_k: int = 1) -> dict:
    """Total leaf params + a FLOPs-comparable 'active' count.

    MoE experts live as separate named submodules (``expert_{e}``); only
    ``expert_top_k`` of ``num_experts`` run per token, so ``active`` discounts the
    un-selected experts (router/attention/embeds are always active). Arms with no
    ``expert_`` leaves get ``active == total``.
    """
    total = expert = 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        size = int(leaf.size)
        total += size
        if "expert_" in jax.tree_util.keystr(path):
            expert += size
    if num_experts > 0:
        active = (total - expert) + int(round(expert * expert_top_k / num_experts))
    else:
        active = total
    return {"total": total, "active": active}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=("ippo", "mappo"), default="ippo")
    p.add_argument(
        "--num-envs", "--rollout-envs", dest="num_envs", type=int, default=64
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-cycles", type=int, default=1000)
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--fit-steps", type=int, default=40000)
    p.add_argument("--chunk-steps", type=int, default=2000)
    p.add_argument("--train-random-rollouts", type=int, default=64)
    p.add_argument("--train-initial-rollouts", type=int, default=64)
    p.add_argument("--heldout-random-rollouts", type=int, default=16)
    p.add_argument("--heldout-initial-rollouts", type=int, default=16)
    p.add_argument("--heldout-seeds", type=int)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--integration-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--num-categories", type=int, default=9)
    # Transformer-arm capacity (shared by 'transformer' and 'llada2'); kept separate
    # from --hidden-dim (the MLP arms). The printed param audit makes matching the MLP
    # arm's count explicit -- set --transformer-dim from it. ffn width is 4x model_dim
    # (so the defaults reproduce the prior model_dim=64, ffn=256 transformer).
    p.add_argument("--transformer-dim", type=int, default=64)
    p.add_argument("--transformer-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    # LLaDA2.0 block-diffusion arm knobs (orthogonal; one parameter each).
    p.add_argument("--block-size", type=int, default=4)
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--expert-top-k", type=int, default=2)
    p.add_argument("--alpha-min", type=float, default=0.15)
    p.add_argument("--alpha-max", type=float, default=0.95)
    p.add_argument("--mask-schedule", default="linear")
    p.add_argument("--confidence-threshold", type=float, default=0.9)
    p.add_argument("--steps-per-block", type=int, default=4)
    p.add_argument("--cap-lambda", type=float, default=0.1)
    p.add_argument(
        "--complementary-masking", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--wsd", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--flow-types",
        nargs="+",
        default=["discrete", "linear"],
        help=(
            "predictor arms: 'discrete'/'transformer' are token denoisers (MLP vs "
            "transformer), 'llada2' is the faithful LLaDA2.0 block-diffusion arm, "
            "'gaussian'/'linear' are continuous flows. Pass 'discrete transformer "
            "llada2 linear' for the full token-architecture ablation."
        ),
    )
    p.add_argument(
        "--baseline-inference", choices=("sample", "argmax"), default="sample"
    )
    p.add_argument("--out-dir", default="runs/compare_world_models")
    args = p.parse_args()
    if args.heldout_seeds is not None:
        args.heldout_random_rollouts = args.heldout_seeds
        args.heldout_initial_rollouts = args.heldout_seeds
    positive = (
        "num_envs",
        "horizon",
        "fit_steps",
        "chunk_steps",
        "hidden_dim",
        "transformer_dim",
        "transformer_layers",
        "num_heads",
        "block_size",
        "num_experts",
        "expert_top_k",
        "steps_per_block",
    )
    for name in positive:
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.heldout_seeds is not None and args.heldout_seeds < 1:
        p.error("--heldout-seeds must be >= 1")
    if args.transformer_dim % args.num_heads != 0:
        p.error("--transformer-dim must be divisible by --num-heads")
    if args.expert_top_k > args.num_experts:
        p.error("--expert-top-k must be <= --num-experts")
    if not 0.0 <= args.alpha_min < args.alpha_max <= 1.0:
        p.error("require 0 <= --alpha-min < --alpha-max <= 1")
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
        # ffn width = 4x model_dim per layer; single source of truth for transformer
        # depth (len) and per-layer/MoE-expert width (the LLaDA2 backbone reads [0]).
        ffn_hidden_dims = (4 * args.transformer_dim,) * args.transformer_layers
        decode_config = LLaDA2WorldModelConfig(
            state_dim=state_dim,
            num_agents=adapter.num_agents,
            action_dim=adapter.action_dim,
            hidden_dims=hidden_dims,
            learning_rate=args.learning_rate,
            integration_steps=args.integration_steps,
            num_categories=args.num_categories,
            model_dim=args.transformer_dim,
            num_heads=args.num_heads,
            ffn_hidden_dims=ffn_hidden_dims,
            block_size=args.block_size,
            num_experts=args.num_experts,
            expert_top_k=args.expert_top_k,
            mask_schedule=args.mask_schedule,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            cap_lambda=args.cap_lambda,
            complementary_masking=args.complementary_masking,
            confidence_threshold=args.confidence_threshold,
            steps_per_block=args.steps_per_block,
            wsd_enabled=args.wsd,
        )
        uniform_ce = float(_num_factors(decode_config) * np.log(args.num_categories))

        rng = jax.random.PRNGKey(args.seed)
        rng, policy_key = jax.random.split(rng)
        policy_state = _create_initial_policy_state(args, adapter, policy_key)

        observations = adapter.reset()
        train_batch, observations, rng = _collect_combined_batch(
            args,
            adapter,
            policy_state,
            observations,
            rng,
            random_seed=args.seed + 1,
            random_rollouts=args.train_random_rollouts,
            initial_rollouts=args.train_initial_rollouts,
        )
        observations = adapter.reset()
        heldout_batch, _, rng = _collect_combined_batch(
            args,
            adapter,
            policy_state,
            observations,
            rng,
            random_seed=args.seed + 2,
            random_rollouts=args.heldout_random_rollouts,
            initial_rollouts=args.heldout_initial_rollouts,
        )
        print(
            f"train transitions: {int(train_batch.states.shape[0])} | "
            f"heldout: {int(heldout_batch.states.shape[0])}",
            flush=True,
        )

        real_tokens, real_states, actions_seq, step_respawn = collect_real_rollout(
            adapter, args, decode_config
        )
        respawned = validate_real_rollout(real_tokens, actions_seq, step_respawn)
        total_respawn_events = int(step_respawn[:, :, list(COIN_FACTORS)].sum())
        print(
            f"guardrails OK | horizon={args.horizon} | coin respawn events={total_respawn_events} "
            f"| 0 player teleports",
            flush=True,
        )

        # Shared init seed + shared eval keys -> identical comparison across predictors.
        model_key = jax.random.PRNGKey(args.seed + 7)
        fit_rng = jax.random.PRNGKey(args.seed + 11)
        roll_key = jax.random.PRNGKey(args.seed + 200)
        heldout_key = jax.random.PRNGKey(args.seed + 300)

        loss_histories, rollout_metrics, predicted_tokens, single_step_acc = (
            {},
            {},
            {},
            {},
        )
        param_counts: dict[str, dict] = {}

        def evaluate_predictor(name, predict_fn, history):
            loss_histories[name] = history
            predicted = predicted_rollout(
                predict_fn, real_states[0], actions_seq, decode_config, roll_key
            )
            rollout_metrics[name] = rollout_tracking_metrics(
                real_tokens, predicted, respawned, step_respawn
            )
            predicted_tokens[name] = predicted
            single_step_acc[name] = single_step_accuracy(
                predict_fn, heldout_batch, decode_config, heldout_key
            )
            _report(name, rollout_metrics[name], single_step_acc[name], history)

        for flow in args.flow_types:
            is_token = flow in ("discrete", "transformer", "llada2")
            if flow == "llada2":
                flow_type = "llada2"
            elif flow in ("discrete", "transformer"):
                flow_type = "discrete"
            else:
                flow_type = flow
            config = dataclasses.replace(
                decode_config,
                flow_type=flow_type,
                num_categories=(args.num_categories if is_token else 0),
                discrete_arch=("transformer" if flow == "transformer" else "mlp"),
            )
            model_state = create_world_model_state(model_key, config)
            param_counts[flow] = count_params(
                model_state.params,
                num_experts=config.num_experts,
                expert_top_k=config.expert_top_k,
            )
            print(
                f"[{flow}] params total={param_counts[flow]['total']} "
                f"active={param_counts[flow]['active']}",
                flush=True,
            )

            def fit_step(carry, n, cfg=config):
                ms, r = carry
                ms, r, _, hist = fit_world_model_steps(ms, r, train_batch, cfg, steps=n)
                return (ms, r), hist

            (model_state, _), history = fit_chunked(
                (model_state, fit_rng), fit_step, args.fit_steps, args.chunk_steps, flow
            )

            def predict_fn(s, a, k, st=model_state, cfg=config):
                return predict_next(st, k, s, a, cfg)

            evaluate_predictor(flow, predict_fn, history)

        baseline_state = create_baseline_state(
            model_key, decode_config, hidden_dims, args.learning_rate
        )
        param_counts["baseline"] = count_params(baseline_state.params)
        print(
            f"[baseline] params total={param_counts['baseline']['total']}", flush=True
        )

        def baseline_step(state, n):
            return fit_categorical_baseline_steps(
                state, train_batch, decode_config, steps=n
            )

        baseline_state, history = fit_chunked(
            baseline_state, baseline_step, args.fit_steps, args.chunk_steps, "baseline"
        )

        def predict_fn_baseline(s, a, k):
            return predict_next_baseline(
                baseline_state, k, s, a, decode_config, args.baseline_inference
            )

        evaluate_predictor("baseline", predict_fn_baseline, history)

        # --- plots + artifacts ---
        plot_loss_curves(out_dir, loss_histories, uniform_ce)
        plot_rollout_tracking(out_dir, rollout_metrics, args.horizon)
        plot_occupancy(out_dir, real_tokens, predicted_tokens, args.horizon)
        plot_example_rollout(out_dir, real_tokens, predicted_tokens, args.horizon)
        for name, history in loss_histories.items():
            _write_loss_csv(out_dir / f"loss_{name}.csv", history.tolist())

        summary = {
            "args": vars(args),
            "decode_config": dataclasses.asdict(decode_config),
            "uniform_cross_entropy": uniform_ce,
            "train_transition_count": int(train_batch.states.shape[0]),
            "heldout_transition_count": int(heldout_batch.states.shape[0]),
            "total_respawn_events": total_respawn_events,
            "loss_summary": {
                name: {
                    "first": float(h[0]),
                    "last": float(h[-1]),
                    "min": float(np.min(h)),
                    "mean_last_50": float(np.mean(h[-min(50, len(h)) :])),
                }
                for name, h in loss_histories.items()
            },
            "single_step_accuracy": single_step_acc,
            "rollout_tracking": rollout_metrics,
            "param_counts": param_counts,
        }
        (out_dir / "compare_world_models.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {out_dir / 'compare_world_models.json'} + plots", flush=True)
    finally:
        adapter.close()


def _report(name, metrics, acc, history):
    rd = metrics["respawn_distribution"]
    print(
        f"[{name}] fit loss first={history[0]:.4f} last={history[-1]:.4f} "
        f"min={np.min(history):.4f}",
        flush=True,
    )
    print(
        f"[{name}] 1-step heldout: overall={acc['overall']:.3f} player={acc['player']:.3f} "
        f"coin={acc['coin']:.3f} (copy-through={acc['copy_through_overall']:.3f}) | "
        f"rollout player accuracy step1={metrics['player_accuracy'][0]:.3f} "
        f"stepH={metrics['player_accuracy'][-1]:.3f} | respawn H_pred="
        f"{rd['predicted_entropy']:.2f}/{rd['uniform_entropy']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
