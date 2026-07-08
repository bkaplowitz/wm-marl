"""Train PPO inside a generative world model on a single-agent environment.

Model-based counterpart of ``train_dmc_jepa`` for the generative arms: fit a
conditional next-observation model (``discrete-transformer`` CTMC flow,
``continuous-transformer`` flow, or ``llada2`` block diffusion) plus a learned
reward/continue head on real replay transitions, then train the policy entirely
on imagined rollouts. Quantile tokenizers are fit once on the initial random
replay and stay frozen (refitting would silently re-map every token id the
model has already learned); later out-of-range observations clip into the edge
bins.

``--tokenizer genie`` replaces the frozen quantile bins with a Genie-style
transformer VQ-VAE (token arms only): the codebook's ids become the world
model's targets, the policy/head consume flattened codebook embeddings, and
the tokenizer co-trains on the growing raw replay each online iteration with
the world model retrained on the re-encoded ids to absorb code drift.

``--collect-steps``/``--online-collect-steps`` are per-env steps (total
transitions = steps x ``--num-envs``), the same units as ``train_dmc_jepa``.
On gymnax/brax, collection and evaluation run on-device through the adapters'
``scan_rollout``; DMC falls back to the host step loop.

The extra ``model-free`` arm skips the world model entirely and spends the
identical real-step budget on PPO over real rollouts — the baseline the
model-based arms must beat on sample efficiency.

Runs ``--num-runs`` seeded runs and applies an improvement gate in
``summarize()``. Exit code 1 means the gate failed, not that the program
crashed. Artifacts mirror ``train_dmc_jepa``: ``config.json`` at the experiment
root, ``outcome.json`` per run, ``summary.json`` at the root, so a policy-level
comparison harness can glob both layouts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.genwm import (
    GENWM_ARMS,
    CodebookTokenizer,
    GenieTokenizer,
    GenWMConfig,
    PPOConfig,
    QuantileTokenizer,
    create_genie_state,
    create_genwm_state,
    create_head_state,
    create_policy_state,
    fit_quantile_tokenizer,
    genie_train_step,
    genwm_train_step,
    head_train_step,
    imagined_rollout,
    make_genie_encode,
    ppo_update,
)
from world_marl.genwm.imagination import ImaginedBatch
from world_marl.jepa.training import load_frozen_encoder
from world_marl.world_model_training import _replay_scan_episode_bookkeeping

MODEL_FREE_ARM = "model-free"
ARM_CHOICES = (*GENWM_ARMS, MODEL_FREE_ARM)

# Defaults for the swept hyperparameters; flags parse as None and resolve to
# the arm's tuned entry, falling back to the base value. llada2 values are the
# best 3-seed-mean Optuna trial on gymnax CartPole (740.6 vs 354.7 for the
# base defaults; runs/runpod/optuna-single-genwm/20260708T013406Z/).
BASE_DEFAULTS: dict[str, int | float] = {
    "imag_horizon": 15,
    "model_dim": 128,
    "num_layers": 2,
    "wm_learning_rate": 1e-3,
    "obs_bins": 32,
    "block_size": 4,
    "steps_per_block": 4,
    "ppo_learning_rate": 3e-4,
    "ent_coef": 0.01,
}
ARM_TUNED_DEFAULTS: dict[str, dict[str, int | float]] = {
    "llada2": {
        "imag_horizon": 15,
        "model_dim": 256,
        "num_layers": 4,
        "wm_learning_rate": 0.00017171860465888458,
        "obs_bins": 64,
        "block_size": 1,
        "steps_per_block": 4,
        "ppo_learning_rate": 0.0009478968677765566,
        "ent_coef": 0.007824023983213257,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        required=True,
        help="dmc:<domain>/<task>, brax:<env>, or gymnax:<env_id>",
    )
    parser.add_argument("--arm", required=True, choices=ARM_CHOICES)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/genwm"))
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--max-cycles", type=int, default=500)
    parser.add_argument(
        "--brax-backend",
        default=None,
        help="Optional Brax physics backend to pass through to brax.envs.create.",
    )
    # Real-env and update budgets (defaults match the JEPA mainline preset).
    parser.add_argument(
        "--collect-steps",
        type=int,
        default=8192,
        help="Per-env real steps for the initial random collection "
        "(total transitions = steps x --num-envs, matching train_dmc_jepa).",
    )
    parser.add_argument("--train-steps", type=int, default=12000)
    parser.add_argument("--policy-train-steps", type=int, default=3000)
    parser.add_argument("--online-iterations", type=int, default=6)
    parser.add_argument(
        "--online-collect-steps",
        type=int,
        default=4096,
        help="Per-env real steps collected per online iteration.",
    )
    parser.add_argument("--online-train-steps", type=int, default=3000)
    parser.add_argument("--online-policy-train-steps", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--policy-batch-size", type=int, default=512)
    parser.add_argument("--imag-horizon", type=int, default=None)
    # World-model capacity (base defaults match the JEPA mainline preset;
    # None-defaulted flags resolve per arm via ARM_TUNED_DEFAULTS).
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--wm-learning-rate", type=float, default=None)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--integration-steps", type=int, default=8)
    parser.add_argument("--obs-bins", type=int, default=None)
    parser.add_argument("--action-bins", type=int, default=8)
    parser.add_argument(
        "--latent-encoder",
        default=None,
        help="Path to a jepa checkpoint dir; tokenize and model its frozen "
        "latents instead of raw observations.",
    )
    parser.add_argument(
        "--tokenizer",
        choices=("quantile", "genie"),
        default="quantile",
        help="Observation tokenizer for the token arms: frozen quantile bins, "
        "or a Genie-style transformer VQ-VAE whose learned code ids are the "
        "world model's targets (codebook size = --obs-bins).",
    )
    parser.add_argument("--genie-code-dim", type=int, default=16)
    parser.add_argument("--genie-model-dim", type=int, default=64)
    parser.add_argument("--genie-heads", type=int, default=4)
    parser.add_argument("--genie-layers", type=int, default=2)
    parser.add_argument("--genie-learning-rate", type=float, default=3e-4)
    parser.add_argument("--genie-train-steps", type=int, default=2000)
    parser.add_argument("--genie-online-train-steps", type=int, default=500)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--steps-per-block", type=int, default=None)
    # PPO.
    parser.add_argument(
        "--mf-rollout-steps",
        type=int,
        default=128,
        help="model-free arm: on-policy segment length per PPO update "
        "(segment x num-envs must divide evenly into --ppo-num-minibatches).",
    )
    parser.add_argument("--ppo-learning-rate", type=float, default=None)
    parser.add_argument("--ppo-update-epochs", type=int, default=4)
    parser.add_argument("--ppo-num-minibatches", type=int, default=4)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    # Evaluation and gate.
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Exit 0 even when the improvement gate fails.",
    )
    parser.add_argument("--quiet", action="store_true")
    # Weights & Biases (disabled unless --wandb-project is set).
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    args = parser.parse_args(argv)
    tuned = ARM_TUNED_DEFAULTS.get(args.arm, {})
    for key, base in BASE_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, tuned.get(key, base))
    if not args.env.startswith(("dmc:", "brax:", "gymnax:")):
        parser.error(
            "--env must be formatted as dmc:<domain>/<task>, brax:<env>, "
            "or gymnax:<env_id>"
        )
    if args.arm == MODEL_FREE_ARM and args.env.startswith("dmc:"):
        parser.error("the model-free arm needs a scan_rollout adapter (gymnax/brax)")
    return args


def _make_adapter(args: argparse.Namespace, *, seed: int):
    if args.env.startswith("dmc:"):
        from world_marl.envs.dmc_adapter import DMCVectorAdapter, dmc_env_name

        return DMCVectorAdapter(
            dmc_env_name(args.env),
            num_envs=args.num_envs,
            max_cycles=args.max_cycles,
            seed=seed,
        )
    if args.env.startswith("brax:"):
        from world_marl.envs.brax_adapter import BraxVectorAdapter, brax_env_name

        return BraxVectorAdapter(
            brax_env_name(args.env),
            num_envs=args.num_envs,
            max_cycles=args.max_cycles,
            seed=seed,
            backend=args.brax_backend,
        )
    from world_marl.envs.gymnax_adapter import GymnaxVectorAdapter, gymnax_env_name

    return GymnaxVectorAdapter(
        gymnax_env_name(args.env),
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        seed=seed,
    )


@jax.jit
def _sample_policy_actions(policy_state, key, observations):
    policy, _ = policy_state.apply_fn({"params": policy_state.params}, observations)
    return policy.sample(seed=key)


@jax.jit
def _mode_policy_actions(policy_state, observations):
    policy, _ = policy_state.apply_fn({"params": policy_state.params}, observations)
    return policy.mode()


def _flat_observations(observations: np.ndarray, num_envs: int) -> np.ndarray:
    return np.asarray(observations, dtype=np.float32)[:, 0].reshape((num_envs, -1))


def _shape_for_step(actions: np.ndarray, action_mode: str) -> np.ndarray:
    if action_mode == "discrete":
        return actions.reshape((-1, 1))
    return actions[:, None, :]


def _policy_actions(
    policy_state,
    flat_obs: np.ndarray,
    key: jax.Array | None,
    *,
    action_mode: str,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
) -> np.ndarray:
    if key is None:
        actions = np.asarray(_mode_policy_actions(policy_state, jnp.asarray(flat_obs)))
    else:
        actions = np.asarray(
            _sample_policy_actions(policy_state, key, jnp.asarray(flat_obs))
        )
    if action_mode == "discrete":
        return actions.astype(np.int32)
    return np.clip(actions.astype(np.float32), action_low, action_high)


def _make_scan_action_fns(
    adapter,
    action_mode: str,
    *,
    encode_fn: Any | None = None,
    encoder_in_state: bool = False,
) -> dict[str, Any]:
    """Action callbacks in ``scan_rollout``'s 3-arg shape.

    ``scan_rollout`` caches compiled programs by ``id(fn)``, so one set is
    created per run and reused across every collection/eval call. When
    ``encode_fn`` is given the policy consumes latents, not raw observations;
    with ``encoder_in_state`` the train_state is a ``(policy_state,
    encoder_params)`` tuple so co-trained encoder params ride through the scan
    as traced values instead of baked-in constants.
    """
    action_dim = adapter.action_dim
    if action_mode == "discrete":

        def random_fn(_state, key, obs_flat):
            rows = obs_flat.shape[0]
            actions = jax.random.randint(key, (rows,), 0, action_dim)
            zeros = jnp.zeros((rows,), dtype=jnp.float32)
            return actions.astype(jnp.int32), zeros, zeros, zeros

    else:
        action_low = jnp.asarray(adapter.action_low, dtype=jnp.float32)
        action_high = jnp.asarray(adapter.action_high, dtype=jnp.float32)

        def random_fn(_state, key, obs_flat):
            rows = obs_flat.shape[0]
            actions = jax.random.uniform(
                key, (rows, action_dim), minval=action_low, maxval=action_high
            )
            zeros = jnp.zeros((rows,), dtype=jnp.float32)
            return actions, zeros, zeros, zeros

    def unpack(train_state):
        if encoder_in_state:
            return train_state
        return train_state, None

    def policy_inputs(encoder_params, obs_flat):
        if encode_fn is None:
            return obs_flat
        if encoder_in_state:
            return encode_fn(encoder_params, obs_flat)
        return encode_fn(obs_flat)

    def sample_fn(train_state, key, obs_flat):
        policy_state, encoder_params = unpack(train_state)
        policy, values = policy_state.apply_fn(
            {"params": policy_state.params}, policy_inputs(encoder_params, obs_flat)
        )
        actions = policy.sample(seed=key)
        return actions, policy.log_prob(actions), values, policy.entropy()

    def mode_fn(train_state, _key, obs_flat):
        policy_state, encoder_params = unpack(train_state)
        policy, values = policy_state.apply_fn(
            {"params": policy_state.params}, policy_inputs(encoder_params, obs_flat)
        )
        actions = policy.mode()
        return actions, policy.log_prob(actions), values, policy.entropy()

    return {"random": random_fn, "sample": sample_fn, "mode": mode_fn}


def _scan_collect_transitions(
    adapter,
    action_fn,
    policy_state,
    *,
    steps_per_env: int,
    key: jax.Array,
) -> dict[str, np.ndarray]:
    """On-device twin of ``_collect_transitions`` via ``scan_rollout``.

    Next observations are the observation sequence shifted by one plus the
    final carry, which reproduces the host loop's post-reset next-obs at
    terminal steps.
    """
    observations = adapter.reset()
    ys, last_obs_flat = adapter.scan_rollout(
        action_fn,
        policy_state,
        steps_per_env,
        policy_key=key,
        observations=observations,
    )
    obs_seq, action_seq, _lp, _values, _ent, reward_seq, done_seq = ys
    _replay_scan_episode_bookkeeping(adapter, ys, steps_per_env)
    obs = np.asarray(obs_seq, dtype=np.float32)
    next_obs = np.concatenate(
        [obs[1:], np.asarray(last_obs_flat, dtype=np.float32)[None]], axis=0
    )
    total = obs.shape[0] * obs.shape[1]
    actions = np.asarray(action_seq)
    return {
        "observations": obs.reshape((total, -1)),
        "actions": actions.reshape((total, *actions.shape[2:])),
        "rewards": np.asarray(reward_seq, dtype=np.float32).reshape((total,)),
        "dones": np.asarray(done_seq, dtype=np.float32).reshape((total,)),
        "next_observations": next_obs.reshape((total, -1)),
    }


def _scan_eval_return(
    adapter,
    action_fn,
    policy_state,
    *,
    episodes: int,
    key: jax.Array,
) -> float:
    """Scan-based twin of ``_evaluate_policy`` with the same total step bound."""
    returns: list[float] = []
    observations = adapter.reset()
    block = max(1, adapter.max_cycles)
    total_step_budget = max(1, episodes * adapter.max_cycles * 4 // adapter.num_envs)
    max_blocks = max(1, -(-total_step_budget // block))
    for _ in range(max_blocks):
        key, block_key = jax.random.split(key)
        ys, last_obs_flat = adapter.scan_rollout(
            action_fn,
            policy_state,
            block,
            policy_key=block_key,
            observations=observations,
        )
        completed_returns, _ = _replay_scan_episode_bookkeeping(adapter, ys, block)
        returns.extend(float(item[0]) for item in completed_returns)
        observations = np.asarray(last_obs_flat, dtype=np.float32)
        if len(returns) >= episodes:
            break
    if not returns:
        return float("nan")
    return float(np.mean(returns[:episodes]))


def _resolve_latent_encoder(
    latent_encoder: str | None, adapter, obs_dim: int, *, arm: str
) -> tuple[Any | None, int]:
    """Load the frozen jepa encoder and swap the model's observation space to latents."""
    if latent_encoder is None:
        return None, obs_dim
    if arm == MODEL_FREE_ARM:
        raise ValueError(
            "--latent-encoder requires a world-model arm, not model-free; "
            "_train_policy_real feeds raw scan observations to PPO"
        )
    if not hasattr(adapter, "scan_rollout"):
        raise ValueError(
            "--latent-encoder requires an adapter with scan_rollout; the loop "
            "collection path feeds raw observations to the policy"
        )
    return load_frozen_encoder(latent_encoder)


def _resolve_genie(args: argparse.Namespace, adapter) -> GenieTokenizer | None:
    """Build the Genie VQ-VAE tokenizer module (None for the quantile default)."""
    if args.tokenizer != "genie":
        return None
    if args.arm not in ("discrete-transformer", "llada2"):
        raise ValueError(
            "--tokenizer genie requires a token arm (discrete-transformer or "
            f"llada2), got {args.arm!r}"
        )
    if args.latent_encoder is not None:
        raise ValueError(
            "--tokenizer genie and --latent-encoder are mutually exclusive; "
            "both replace the observation representation"
        )
    if not hasattr(adapter, "scan_rollout"):
        raise ValueError(
            "--tokenizer genie requires an adapter with scan_rollout; the loop "
            "collection path feeds raw observations to the policy"
        )
    return GenieTokenizer(
        obs_dim=int(np.prod(adapter.observation_shape)),
        codebook_size=args.obs_bins,
        code_dim=args.genie_code_dim,
        model_dim=args.genie_model_dim,
        num_heads=args.genie_heads,
        num_layers=args.genie_layers,
        mlp_ratio=args.mlp_ratio,
    )


def _train_genie(
    genie_state,
    data: dict[str, np.ndarray],
    *,
    steps: int,
    batch_size: int,
    rng: np.random.Generator,
    log_every: int,
    quiet: bool,
    label: str,
):
    """Fit the VQ-VAE by reconstruction on raw replay observations."""
    samples = np.concatenate([data["observations"], data["next_observations"]])
    metrics: dict[str, Any] = {}
    for step_index in range(steps):
        index = rng.integers(0, samples.shape[0], size=batch_size)
        genie_state, step_metrics = genie_train_step(
            genie_state, jnp.asarray(samples[index])
        )
        if step_index % log_every == 0 or step_index == steps - 1:
            metrics = {name: float(value) for name, value in step_metrics.items()}
            if not quiet:
                print(
                    f"[{label}] step {step_index + 1}/{steps} "
                    f"recon={metrics['genie_recon_loss']:.4f} "
                    f"vq={metrics['genie_codebook_loss']:.4f}",
                    flush=True,
                )
    return genie_state, metrics


def _encode_replay(encode_fn, replay: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map replay observations through the frozen encoder; other keys pass through."""
    encoded = dict(replay)
    for key in ("observations", "next_observations"):
        encoded[key] = np.asarray(
            encode_fn(jnp.asarray(replay[key], dtype=jnp.float32)), dtype=np.float32
        )
    return encoded


def _collect_replay(
    adapter,
    *,
    steps_per_env: int,
    rng: np.random.Generator,
    collect_key: jax.Array,
    action_fns: dict[str, Any] | None,
    policy_state,
    action_mode: str,
) -> dict[str, np.ndarray]:
    if action_fns is not None:
        fn = action_fns["random"] if policy_state is None else action_fns["sample"]
        return _scan_collect_transitions(
            adapter, fn, policy_state, steps_per_env=steps_per_env, key=collect_key
        )
    return _collect_transitions(
        adapter,
        num_steps=steps_per_env,
        rng=rng,
        policy_state=policy_state,
        policy_key=collect_key if policy_state is not None else None,
        action_mode=action_mode,
    )


def _eval_return(
    adapter,
    *,
    episodes: int,
    rng: np.random.Generator,
    eval_key: jax.Array,
    action_fns: dict[str, Any] | None,
    policy_state,
    action_mode: str,
) -> float:
    if action_fns is not None:
        fn = action_fns["random"] if policy_state is None else action_fns["mode"]
        return _scan_eval_return(
            adapter, fn, policy_state, episodes=episodes, key=eval_key
        )
    return _evaluate_policy(
        adapter,
        policy_state,
        episodes=episodes,
        action_mode=action_mode,
        rng=rng,
    )


def _collect_transitions(
    adapter,
    *,
    num_steps: int,
    rng: np.random.Generator,
    policy_state=None,
    policy_key: jax.Array | None = None,
    action_mode: str,
) -> dict[str, np.ndarray]:
    """Collect ``num_steps`` per-env steps on the host (random when state is None)."""
    num_envs = adapter.num_envs
    iterations = max(1, num_steps)
    observations = adapter.reset()
    records: dict[str, list[np.ndarray]] = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "next_observations": [],
    }
    for _ in range(iterations):
        flat = _flat_observations(observations, num_envs)
        if policy_state is None:
            step_actions = adapter.sample_actions(rng)
            actions = step_actions[:, 0]
        else:
            assert policy_key is not None
            policy_key, action_key = jax.random.split(policy_key)
            actions = _policy_actions(
                policy_state,
                flat,
                action_key,
                action_mode=action_mode,
                action_low=adapter.action_low,
                action_high=adapter.action_high,
            )
            step_actions = _shape_for_step(actions, action_mode)
        step = adapter.step(step_actions)
        records["observations"].append(flat)
        records["actions"].append(np.asarray(actions))
        records["rewards"].append(np.asarray(step.rewards[:, 0], dtype=np.float32))
        records["dones"].append(np.asarray(step.dones[:, 0], dtype=np.float32))
        records["next_observations"].append(
            _flat_observations(step.observations, num_envs)
        )
        observations = step.observations
    return {name: np.concatenate(chunks) for name, chunks in records.items()}


def _evaluate_policy(
    adapter,
    policy_state,
    *,
    episodes: int,
    action_mode: str,
    rng: np.random.Generator,
) -> float:
    """Mean return over completed episodes (random policy when state is None)."""
    returns: list[float] = []
    observations = adapter.reset()
    step_limit = max(1, episodes * adapter.max_cycles * 4 // adapter.num_envs)
    for _ in range(step_limit):
        if policy_state is None:
            step_actions = adapter.sample_actions(rng)
        else:
            flat = _flat_observations(observations, adapter.num_envs)
            actions = _policy_actions(
                policy_state,
                flat,
                None,
                action_mode=action_mode,
                action_low=adapter.action_low,
                action_high=adapter.action_high,
            )
            step_actions = _shape_for_step(actions, action_mode)
        step = adapter.step(step_actions)
        returns.extend(float(total[0]) for total in step.completed_returns)
        observations = step.observations
        if len(returns) >= episodes:
            break
    if not returns:
        return float("nan")
    return float(np.mean(returns[:episodes]))


def _action_feature_array(actions: np.ndarray, config: GenWMConfig) -> np.ndarray:
    if config.action_mode == "discrete":
        return np.eye(config.action_dim, dtype=np.float32)[actions.astype(np.int64)]
    return actions.astype(np.float32)


def _fit_models(
    wm_state,
    head_state,
    key: jax.Array,
    data: dict[str, np.ndarray],
    obs_tokenizer: QuantileTokenizer,
    action_tokenizer: QuantileTokenizer | None,
    config: GenWMConfig,
    *,
    steps: int,
    batch_size: int,
    rng: np.random.Generator,
    log_every: int,
    quiet: bool,
    label: str,
):
    """Interleave world-model and reward/continue-head updates on replay data.

    The world model only sees non-terminal transitions (the stored next
    observation at a terminal step is post-reset); the head trains on every
    transition since its targets (r, done) belong to (s, a).
    """
    non_terminal = np.flatnonzero(data["dones"] < 0.5)
    if non_terminal.size == 0:
        raise RuntimeError("replay contains only terminal transitions")
    features = _action_feature_array(data["actions"], config)
    continues = 1.0 - data["dones"]
    total = data["observations"].shape[0]
    wm_loss = float("nan")
    head_metrics: dict[str, Any] = {}
    for step_index in range(steps):
        wm_index = non_terminal[rng.integers(0, non_terminal.size, size=batch_size)]
        key, step_key = jax.random.split(key)
        wm_state, loss = genwm_train_step(
            wm_state,
            step_key,
            jnp.asarray(data["observations"][wm_index]),
            jnp.asarray(data["actions"][wm_index]),
            jnp.asarray(data["next_observations"][wm_index]),
            obs_tokenizer,
            action_tokenizer,
            config,
        )
        head_index = rng.integers(0, total, size=batch_size)
        head_state, metrics = head_train_step(
            head_state,
            jnp.asarray(data["observations"][head_index]),
            jnp.asarray(features[head_index]),
            jnp.asarray(data["rewards"][head_index]),
            jnp.asarray(continues[head_index]),
        )
        if step_index % log_every == 0 or step_index == steps - 1:
            wm_loss = float(loss)
            head_metrics = {name: float(value) for name, value in metrics.items()}
            if not quiet:
                print(
                    f"[{label}] step {step_index + 1}/{steps} "
                    f"wm_loss={wm_loss:.4f} "
                    f"head_loss={head_metrics['head_total_loss']:.4f}",
                    flush=True,
                )
    return wm_state, head_state, wm_loss, head_metrics


def _train_policy(
    policy_state,
    wm_state,
    head_state,
    obs_tokenizer: QuantileTokenizer,
    action_tokenizer: QuantileTokenizer | None,
    start_pool: np.ndarray,
    key: jax.Array,
    config: GenWMConfig,
    ppo_config: PPOConfig,
    *,
    steps: int,
    batch_size: int,
    horizon: int,
    rng: np.random.Generator,
    log_every: int,
    quiet: bool,
    label: str,
):
    metrics: dict[str, Any] = {}
    for step_index in range(steps):
        starts = jnp.asarray(
            start_pool[rng.integers(0, start_pool.shape[0], size=batch_size)]
        )
        key, rollout_key, update_key = jax.random.split(key, 3)
        batch, last_values = imagined_rollout(
            policy_state,
            wm_state,
            head_state,
            obs_tokenizer,
            action_tokenizer,
            starts,
            rollout_key,
            horizon=horizon,
            config=config,
            ppo_config=ppo_config,
        )
        policy_state, step_metrics = ppo_update(
            policy_state, batch, last_values, update_key, ppo_config
        )
        if step_index % log_every == 0 or step_index == steps - 1:
            metrics = {name: float(value) for name, value in step_metrics.items()}
            if not quiet:
                print(
                    f"[{label}] step {step_index + 1}/{steps} "
                    f"ppo_loss={metrics['total_loss']:.4f} "
                    f"entropy={metrics['entropy']:.4f}",
                    flush=True,
                )
    return policy_state, metrics


@jax.jit
def _policy_last_values(policy_state, observations):
    _, values = policy_state.apply_fn({"params": policy_state.params}, observations)
    return values


def _train_policy_real(
    policy_state,
    adapter,
    sample_fn,
    key: jax.Array,
    ppo_config: PPOConfig,
    *,
    steps_per_env: int,
    rollout_steps: int,
    log_every: int,
    quiet: bool,
    label: str,
    wandb_run=None,
):
    """PPO on real on-policy rollouts (the model-free arm's phase trainer).

    Spends the same per-env step budget a world-model arm would use for replay
    collection in this phase, updating on each ``rollout_steps`` segment.
    """
    num_updates = max(1, steps_per_env // rollout_steps)
    observations = adapter.reset()
    metrics: dict[str, Any] = {}
    for update_index in range(num_updates):
        key, rollout_key, update_key = jax.random.split(key, 3)
        ys, last_obs_flat = adapter.scan_rollout(
            sample_fn,
            policy_state,
            rollout_steps,
            policy_key=rollout_key,
            observations=observations,
        )
        obs_seq, action_seq, log_prob_seq, value_seq, _ent, reward_seq, done_seq = ys
        _replay_scan_episode_bookkeeping(adapter, ys, rollout_steps)
        batch = ImaginedBatch(
            observations=obs_seq,
            actions=action_seq,
            log_probs=log_prob_seq,
            rewards=reward_seq,
            dones=done_seq.astype(jnp.float32),
            values=value_seq,
        )
        last_values = _policy_last_values(policy_state, jnp.asarray(last_obs_flat))
        policy_state, step_metrics = ppo_update(
            policy_state, batch, last_values, update_key, ppo_config
        )
        observations = np.asarray(last_obs_flat, dtype=np.float32)
        if update_index % log_every == 0 or update_index == num_updates - 1:
            metrics = {name: float(value) for name, value in step_metrics.items()}
            if wandb_run is not None:
                wandb_run.log({f"ppo/{name}": value for name, value in metrics.items()})
            if not quiet:
                print(
                    f"[{label}] update {update_index + 1}/{num_updates} "
                    f"ppo_loss={metrics['total_loss']:.4f} "
                    f"entropy={metrics['entropy']:.4f}",
                    flush=True,
                )
    return policy_state, metrics


def _init_wandb(args: argparse.Namespace, *, run_index: int):
    """Create a W&B run when --wandb-project is set (returns None otherwise)."""
    if not args.wandb_project:
        return None
    import wandb

    env_slug = args.env.replace(":", "_").replace("/", "_")
    config = {
        name: (str(value) if isinstance(value, Path) else value)
        for name, value in vars(args).items()
    }
    config["run_index"] = run_index
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group or f"{env_slug}-{args.arm}",
        name=f"{env_slug}-{args.arm}-run{run_index:02d}",
        config=config,
        reinit=True,
    )
    run.define_metric("eval/real_env_steps")
    run.define_metric("eval/return", step_metric="eval/real_env_steps")
    return run


def run_one(args: argparse.Namespace, *, run_dir: Path, run_index: int) -> dict:
    started = time.time()
    seed = args.seed + 10_000 * run_index
    rng = np.random.default_rng(seed)
    adapter = _make_adapter(args, seed=seed)
    num_envs = int(adapter.num_envs)
    wandb_run = _init_wandb(args, run_index=run_index)
    eval_points: list[dict[str, Any]] = []

    def record_eval(tag: str, steps_per_env: int, value: float) -> None:
        real_steps = steps_per_env * num_envs
        eval_points.append({"tag": tag, "real_env_steps": real_steps, "return": value})
        if wandb_run is not None:
            wandb_run.log({"eval/return": value, "eval/real_env_steps": real_steps})

    try:
        action_mode = "discrete" if args.env.startswith("gymnax:") else "continuous"
        obs_dim = int(np.prod(adapter.observation_shape))
        model_based = args.arm != MODEL_FREE_ARM
        encode_fn, obs_dim = _resolve_latent_encoder(
            args.latent_encoder, adapter, obs_dim, arm=args.arm
        )
        genie_module = _resolve_genie(args, adapter)
        genie_encode = (
            make_genie_encode(genie_module) if genie_module is not None else None
        )
        action_fns = (
            _make_scan_action_fns(
                adapter,
                action_mode,
                encode_fn=genie_encode if genie_module is not None else encode_fn,
                encoder_in_state=genie_module is not None,
            )
            if hasattr(adapter, "scan_rollout")
            else None
        )
        config = GenWMConfig(
            # model-free uses the config only for policy creation, and the arm
            # field is validated against GENWM_ARMS, so give it a placeholder.
            arm=args.arm if model_based else GENWM_ARMS[0],
            obs_dim=obs_dim,
            action_dim=adapter.action_dim,
            action_mode=action_mode,
            obs_bins=args.obs_bins,
            action_bins=args.action_bins,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            mlp_ratio=args.mlp_ratio,
            learning_rate=args.wm_learning_rate,
            integration_steps=args.integration_steps,
            block_size=args.block_size,
            steps_per_block=args.steps_per_block,
            code_dim=args.genie_code_dim if genie_module is not None else 1,
        )
        ppo_config = PPOConfig(
            learning_rate=args.ppo_learning_rate,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            update_epochs=args.ppo_update_epochs,
            num_minibatches=args.ppo_num_minibatches,
        )
        key = jax.random.PRNGKey(seed)
        key, wm_key, head_key, policy_key = jax.random.split(key, 4)
        wm_state = head_state = None
        if model_based:
            wm_state = create_genwm_state(wm_key, config)
            head_state = create_head_state(
                head_key, config, learning_rate=args.head_learning_rate
            )
        policy_state = create_policy_state(policy_key, config, ppo_config)
        genie_state = None
        genie_metrics: dict[str, Any] = {}
        if genie_module is not None:
            key, genie_key = jax.random.split(key)
            genie_state = create_genie_state(
                genie_key, genie_module, learning_rate=args.genie_learning_rate
            )

        def policy_carry(state):
            """Pair the policy with the current encoder params for the scan fns."""
            if genie_state is None or state is None:
                return state
            return (state, genie_state.params)

        def genie_replay(replay):
            """Re-encode raw replay observations with the current codebook."""
            params = genie_state.params
            return _encode_replay(lambda obs: genie_encode(params, obs), replay)

        key, random_eval_key, initial_eval_key = jax.random.split(key, 3)
        random_return = _eval_return(
            adapter,
            episodes=args.eval_episodes,
            rng=rng,
            eval_key=random_eval_key,
            action_fns=action_fns,
            policy_state=None,
            action_mode=action_mode,
        )
        initial_return = _eval_return(
            adapter,
            episodes=args.eval_episodes,
            rng=rng,
            eval_key=initial_eval_key,
            action_fns=action_fns,
            policy_state=policy_carry(policy_state),
            action_mode=action_mode,
        )
        record_eval("initial", 0, initial_return)
        if not args.quiet:
            print(
                f"[run {run_index}] random={random_return:.2f} "
                f"initial={initial_return:.2f}"
            )

        wm_loss: float | None = None
        head_metrics: dict[str, Any] = {}
        data = None
        obs_tokenizer = action_tokenizer = None
        if model_based:
            if not args.quiet:
                print(
                    f"[run {run_index}] collecting {args.collect_steps} "
                    "random steps per env"
                )
            key, collect_key = jax.random.split(key)
            data = _collect_replay(
                adapter,
                steps_per_env=args.collect_steps,
                rng=rng,
                collect_key=collect_key,
                action_fns=action_fns,
                policy_state=None,
                action_mode=action_mode,
            )
            if encode_fn is not None:
                data = _encode_replay(encode_fn, data)
            raw_data = data
            if genie_module is not None:
                genie_state, genie_metrics = _train_genie(
                    genie_state,
                    raw_data,
                    steps=args.genie_train_steps,
                    batch_size=args.batch_size,
                    rng=rng,
                    log_every=max(1, args.genie_train_steps // 10),
                    quiet=args.quiet,
                    label=f"run {run_index} genie",
                )
                obs_tokenizer = CodebookTokenizer(
                    codebook=genie_state.params["codebook"]
                )
                data = genie_replay(raw_data)
            else:
                obs_tokenizer = fit_quantile_tokenizer(
                    np.concatenate([data["observations"], data["next_observations"]]),
                    args.obs_bins,
                )
            if action_mode == "continuous":
                action_tokenizer = fit_quantile_tokenizer(
                    data["actions"], args.action_bins
                )

            key, fit_key = jax.random.split(key)
            wm_state, head_state, wm_loss, head_metrics = _fit_models(
                wm_state,
                head_state,
                fit_key,
                data,
                obs_tokenizer,
                action_tokenizer,
                config,
                steps=args.train_steps,
                batch_size=args.batch_size,
                rng=rng,
                log_every=max(1, args.train_steps // 10),
                quiet=args.quiet,
                label=f"run {run_index} fit",
            )
            key, policy_fit_key = jax.random.split(key)
            policy_state, ppo_metrics = _train_policy(
                policy_state,
                wm_state,
                head_state,
                obs_tokenizer,
                action_tokenizer,
                data["observations"],
                policy_fit_key,
                config,
                ppo_config,
                steps=args.policy_train_steps,
                batch_size=args.policy_batch_size,
                horizon=args.imag_horizon,
                rng=rng,
                log_every=max(1, args.policy_train_steps // 10),
                quiet=args.quiet,
                label=f"run {run_index} policy",
            )
        else:
            assert action_fns is not None
            key, offline_key = jax.random.split(key)
            policy_state, ppo_metrics = _train_policy_real(
                policy_state,
                adapter,
                action_fns["sample"],
                offline_key,
                ppo_config,
                steps_per_env=args.collect_steps,
                rollout_steps=args.mf_rollout_steps,
                log_every=max(1, args.collect_steps // args.mf_rollout_steps // 10),
                quiet=args.quiet,
                label=f"run {run_index} model-free",
                wandb_run=wandb_run,
            )

        key, offline_eval_key = jax.random.split(key)
        offline_return = _eval_return(
            adapter,
            episodes=args.eval_episodes,
            rng=rng,
            eval_key=offline_eval_key,
            action_fns=action_fns,
            policy_state=policy_carry(policy_state),
            action_mode=action_mode,
        )
        record_eval("offline", args.collect_steps, offline_return)
        if not args.quiet:
            print(f"[run {run_index}] offline return={offline_return:.2f}")

        iteration_returns: list[float] = []
        for iteration in range(args.online_iterations):
            key, collect_key = jax.random.split(key)
            if model_based:
                fresh = _collect_replay(
                    adapter,
                    steps_per_env=args.online_collect_steps,
                    rng=rng,
                    collect_key=collect_key,
                    action_fns=action_fns,
                    policy_state=policy_carry(policy_state),
                    action_mode=action_mode,
                )
                if encode_fn is not None:
                    fresh = _encode_replay(encode_fn, fresh)
                if genie_module is not None:
                    raw_data = {
                        name: np.concatenate([raw_data[name], fresh[name]])
                        for name in raw_data
                    }
                    genie_state, genie_metrics = _train_genie(
                        genie_state,
                        raw_data,
                        steps=args.genie_online_train_steps,
                        batch_size=args.batch_size,
                        rng=rng,
                        log_every=max(1, args.genie_online_train_steps // 5),
                        quiet=args.quiet,
                        label=f"run {run_index} online {iteration} genie",
                    )
                    obs_tokenizer = CodebookTokenizer(
                        codebook=genie_state.params["codebook"]
                    )
                    data = genie_replay(raw_data)
                else:
                    data = {
                        name: np.concatenate([data[name], fresh[name]]) for name in data
                    }
                key, fit_key, policy_fit_key = jax.random.split(key, 3)
                wm_state, head_state, wm_loss, head_metrics = _fit_models(
                    wm_state,
                    head_state,
                    fit_key,
                    data,
                    obs_tokenizer,
                    action_tokenizer,
                    config,
                    steps=args.online_train_steps,
                    batch_size=args.batch_size,
                    rng=rng,
                    log_every=max(1, args.online_train_steps // 5),
                    quiet=args.quiet,
                    label=f"run {run_index} online {iteration} fit",
                )
                policy_state, ppo_metrics = _train_policy(
                    policy_state,
                    wm_state,
                    head_state,
                    obs_tokenizer,
                    action_tokenizer,
                    data["observations"],
                    policy_fit_key,
                    config,
                    ppo_config,
                    steps=args.online_policy_train_steps,
                    batch_size=args.policy_batch_size,
                    horizon=args.imag_horizon,
                    rng=rng,
                    log_every=max(1, args.online_policy_train_steps // 5),
                    quiet=args.quiet,
                    label=f"run {run_index} online {iteration} policy",
                )
            else:
                policy_state, ppo_metrics = _train_policy_real(
                    policy_state,
                    adapter,
                    action_fns["sample"],
                    collect_key,
                    ppo_config,
                    steps_per_env=args.online_collect_steps,
                    rollout_steps=args.mf_rollout_steps,
                    log_every=max(
                        1, args.online_collect_steps // args.mf_rollout_steps // 5
                    ),
                    quiet=args.quiet,
                    label=f"run {run_index} model-free online {iteration}",
                    wandb_run=wandb_run,
                )
            key, iter_eval_key = jax.random.split(key)
            iteration_return = _eval_return(
                adapter,
                episodes=args.eval_episodes,
                rng=rng,
                eval_key=iter_eval_key,
                action_fns=action_fns,
                policy_state=policy_carry(policy_state),
                action_mode=action_mode,
            )
            iteration_returns.append(iteration_return)
            record_eval(
                f"online_{iteration}",
                args.collect_steps + (iteration + 1) * args.online_collect_steps,
                iteration_return,
            )
            if not args.quiet:
                print(
                    f"[run {run_index}] online iteration {iteration}: "
                    f"return={iteration_return:.2f}"
                )

        total_steps_per_env = args.collect_steps + (
            args.online_iterations * args.online_collect_steps
        )
        key, final_eval_key = jax.random.split(key)
        trained_return = _eval_return(
            adapter,
            episodes=args.eval_episodes,
            rng=rng,
            eval_key=final_eval_key,
            action_fns=action_fns,
            policy_state=policy_carry(policy_state),
            action_mode=action_mode,
        )
        record_eval("final", total_steps_per_env, trained_return)
        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    "policy_random_mean": random_return,
                    "policy_initial_mean": initial_return,
                    "policy_trained_mean": trained_return,
                }
            )
    finally:
        adapter.close()
        if wandb_run is not None:
            wandb_run.finish()

    real_env_steps = total_steps_per_env * num_envs
    outcome = {
        "run_index": run_index,
        "seed": seed,
        "arm": args.arm,
        "env": args.env,
        "action_mode": action_mode,
        "obs_dim": obs_dim,
        "action_dim": int(adapter.action_dim),
        "policy_random_mean": random_return,
        "policy_initial_mean": initial_return,
        "policy_offline_mean": offline_return,
        "policy_trained_mean": trained_return,
        "policy_iteration_returns": iteration_returns,
        "eval_points": eval_points,
        "tokenizer": args.tokenizer,
        "world_model_final_loss": wm_loss,
        "genie_final_metrics": genie_metrics,
        "head_final_metrics": head_metrics,
        "ppo_final_metrics": ppo_metrics,
        "real_env_steps": real_env_steps,
        "real_env_steps_per_env": total_steps_per_env,
        "replay_transitions": (
            int(data["observations"].shape[0]) if data is not None else real_env_steps
        ),
        "runtime_seconds": time.time() - started,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "outcome.json").write_text(json.dumps(outcome, indent=2))
    return outcome


def summarize(outcomes: list[dict], *, min_improvement: float) -> dict:
    trained = np.array([o["policy_trained_mean"] for o in outcomes])
    random_ = np.array([o["policy_random_mean"] for o in outcomes])
    initial = np.array([o["policy_initial_mean"] for o in outcomes])
    baseline = max(float(random_.mean()), float(initial.mean()))
    improvement = float(trained.mean()) - baseline
    passed = bool(
        np.all(np.isfinite(trained))
        and np.isfinite(improvement)
        and improvement >= min_improvement
    )
    return {
        "num_runs": len(outcomes),
        "policy_random_mean": float(random_.mean()),
        "policy_initial_mean": float(initial.mean()),
        "policy_trained_mean": float(trained.mean()),
        "improvement_over_baseline": improvement,
        "min_improvement": min_improvement,
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    env_slug = args.env.replace(":", "_").replace("/", "_")
    experiment_dir = args.out_dir / f"genwm_{env_slug}_{args.arm}_{stamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        name: (str(value) if isinstance(value, Path) else value)
        for name, value in vars(args).items()
    }
    (experiment_dir / "config.json").write_text(json.dumps(config_payload, indent=2))
    print(f"experiment dir: {experiment_dir}", flush=True)

    outcomes = []
    for run_index in range(args.num_runs):
        outcomes.append(
            run_one(
                args,
                run_dir=experiment_dir / f"run_{run_index:02d}",
                run_index=run_index,
            )
        )

    summary = summarize(outcomes, min_improvement=args.min_improvement)
    (experiment_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["passed"] and not args.allow_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
