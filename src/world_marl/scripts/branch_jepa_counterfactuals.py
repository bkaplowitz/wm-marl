"""Short-horizon counterfactual DMC branch test for JEPA checkpoints.

This diagnostic restores the same live DMC simulator state and executes several
candidate action sequences. It then asks whether the JEPA world model ranks
those within-state counterfactual futures in the same order as the real
environment.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from world_marl.checkpointing import load_metadata, load_params
from world_marl.envs.dmc_adapter import DMCVectorAdapter, dmc_env_name
from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.training import create_jepa_train_state, select_continuous_actions
from world_marl.scripts.cem_dmc_jepa import score_action_sequences
from world_marl.scripts.eval_jepa_wm import parameter_counts, to_jsonable
from world_marl.scripts.rank_jepa_futures import (
    binary_auc,
    count_values,
    pearson,
    safe_mean,
    spearman,
)


CSV_FIELDS = (
    "context_id",
    "candidate",
    "candidate_group",
    "horizon",
    "trajectory_type",
    "real_reward_sum",
    "real_reward_mean",
    "pred_reward_sum",
    "reward_sum_error",
    "reward_sum_abs_error",
    "latent_cosine_mean",
    "latent_cosine_last",
    "uncertainty_sum",
    "uncertainty_mean",
    "continue_product",
    "action_abs_mean",
    "action_saturation",
    "done_observed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Branch DMC from identical simulator states and compare JEPA "
            "predicted ranking against real counterfactual returns."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--env", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--contexts", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument(
        "--context-policy",
        choices=("random", "actor"),
        default="actor",
        help="Policy used to reach branch states.",
    )
    parser.add_argument(
        "--logged-policy",
        choices=("random", "actor"),
        default="actor",
        help="Policy used for the logged future candidate.",
    )
    parser.add_argument(
        "--candidates",
        default="logged,logged_noise,actor,zero,random,cem",
        help=(
            "Comma-separated candidates: logged, logged_noise, actor, zero, "
            "random, cem, random_pool, actor_noise_pool, cem_pool, cem_actor_pool."
        ),
    )
    parser.add_argument("--noise-scale", type=float, default=0.10)
    parser.add_argument("--random-pool-size", type=int, default=16)
    parser.add_argument("--actor-noise-pool-size", type=int, default=16)
    parser.add_argument("--cem-pool-size", type=int, default=16)
    parser.add_argument("--cem-actor-pool-size", type=int, default=16)
    parser.add_argument("--cem-candidates", type=int, default=64)
    parser.add_argument("--cem-iterations", type=int, default=2)
    parser.add_argument("--cem-elite-fraction", type=float, default=0.125)
    parser.add_argument("--cem-init-std", type=float, default=1.0)
    parser.add_argument("--cem-min-std", type=float, default=0.05)
    parser.add_argument("--success-mean-threshold", type=float, default=0.9)
    parser.add_argument("--soft-failure-mean-threshold", type=float, default=0.7)
    parser.add_argument("--hard-failure-mean-threshold", type=float, default=0.1)
    parser.add_argument(
        "--escape-mean-threshold",
        type=float,
        default=0.7,
        help="Per-step reward threshold used to mark an oracle branch as an escape.",
    )
    parser.add_argument("--action-saturation-threshold", type=float, default=0.95)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.contexts < 1:
        parser.error("--contexts must be >= 1")
    if args.horizon < 1:
        parser.error("--horizon must be >= 1")
    if args.cem_candidates < 2:
        parser.error("--cem-candidates must be >= 2")
    for name in (
        "random_pool_size",
        "actor_noise_pool_size",
        "cem_pool_size",
        "cem_actor_pool_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if not (0.0 < args.cem_elite_fraction <= 1.0):
        parser.error("--cem-elite-fraction must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.checkpoint)
    env = args.env or metadata.get("env")
    if not isinstance(env, str) or not env.startswith("dmc:"):
        raise ValueError("--env is required unless checkpoint metadata contains a DMC env")
    seed = int(metadata.get("seed", 0) if args.seed is None else args.seed)
    config = JepaConfig(**metadata["jepa_config"])
    candidates = parse_candidates(args.candidates)

    state = create_jepa_train_state(jax.random.PRNGKey(seed + 17), config)
    state = state.replace(
        params=load_params(args.checkpoint / "checkpoint.msgpack", state.params)
    )

    adapter = DMCVectorAdapter(
        dmc_env_name(env),
        num_envs=1,
        max_cycles=args.max_cycles,
        seed=seed + 800_000,
        num_workers=1,
    )
    rng = np.random.default_rng(seed + 900_000)
    key = jax.random.PRNGKey(seed + 1_000_000)
    rows: list[dict[str, Any]] = []
    try:
        observations = adapter.reset()
        progress = tqdm(
            range(args.contexts),
            desc="branch contexts",
            unit="context",
            disable=args.quiet,
        )
        for context_id in progress:
            key, context_key, candidate_key = jax.random.split(key, 3)
            observations, obs_context, action_context = collect_context(
                adapter,
                observations,
                state,
                config,
                rng,
                context_key,
                policy=args.context_policy,
            )
            snapshot = capture_snapshot(adapter, observations)
            action_sequences = build_candidate_sequences(
                args,
                state,
                config,
                adapter,
                snapshot,
                obs_context,
                action_context,
                rng,
                candidate_key,
                candidates=candidates,
            )
            for candidate_name, action_sequence in action_sequences.items():
                real = rollout_fixed_actions(adapter, snapshot, action_sequence)
                scores = score_real_future(
                    state,
                    config,
                    obs_context[None],
                    action_context[None],
                    action_sequence[None],
                    real["observations"][None],
                )
                rows.append(
                    row_from_branch(
                        scores,
                        real,
                        action_sequence,
                        context_id=context_id,
                        candidate=candidate_name,
                        horizon=args.horizon,
                        success_mean_threshold=args.success_mean_threshold,
                        soft_failure_mean_threshold=args.soft_failure_mean_threshold,
                        hard_failure_mean_threshold=args.hard_failure_mean_threshold,
                        action_saturation_threshold=args.action_saturation_threshold,
                    )
                )
            # Continue data collection from the logged branch so contexts are not
            # identical but remain on the selected collection distribution.
            restore_snapshot(adapter, snapshot)
            logged = action_sequences["logged"]
            for action in logged:
                step = adapter.step(action[None, None, :])
                observations = step.observations
    finally:
        adapter.close()

    csv_out = args.csv_out or args.out.with_suffix(".csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_out, rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "metadata": {
            "env": env,
            "seed": seed,
            "algorithm": metadata.get("algorithm"),
            "control": metadata.get("control"),
            "jepa_config": dataclasses.asdict(config),
        },
        "parameter_counts": parameter_counts(state.params),
        "eval": {
            "contexts": args.contexts,
            "horizon": args.horizon,
            "context_policy": args.context_policy,
            "logged_policy": args.logged_policy,
            "candidates": candidates,
        },
        "csv": str(csv_out),
        "summary": summarize_rows(
            rows,
            escape_mean_threshold=args.escape_mean_threshold,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


def collect_context(
    adapter: DMCVectorAdapter,
    observations: np.ndarray,
    state,
    config: JepaConfig,
    rng: np.random.Generator,
    key: jax.Array,
    *,
    policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_items: list[np.ndarray] = [np.asarray(observations[0, 0], dtype=np.float32)]
    actions: list[np.ndarray] = []
    action_key = key
    for _ in range(config.context_window - 1):
        action_key, step_key = jax.random.split(action_key)
        action = choose_action(
            adapter,
            observations,
            state,
            config,
            rng,
            step_key,
            policy=policy,
        )
        step = adapter.step(action[None, None, :])
        actions.append(action)
        observations = step.observations
        obs_items.append(np.asarray(observations[0, 0], dtype=np.float32))
    action_context = np.zeros((config.context_window, config.action_dim), dtype=np.float32)
    if actions:
        action_context[: len(actions)] = np.asarray(actions, dtype=np.float32)
    return observations, np.asarray(obs_items, dtype=np.float32), action_context


def build_candidate_sequences(
    args: argparse.Namespace,
    state,
    config: JepaConfig,
    adapter: DMCVectorAdapter,
    snapshot: dict[str, Any],
    obs_context: np.ndarray,
    action_context: np.ndarray,
    rng: np.random.Generator,
    key: jax.Array,
    *,
    candidates: list[str],
) -> dict[str, np.ndarray]:
    sequences: dict[str, np.ndarray] = {}
    keys = dict(zip(candidates, jax.random.split(key, max(1, len(candidates))), strict=False))
    actor_sequence = None
    if "logged" in candidates or "logged_noise" in candidates:
        logged = rollout_policy_actions(
            adapter,
            snapshot,
            state,
            config,
            rng,
            keys.get("logged", key),
            policy=args.logged_policy,
            horizon=args.horizon,
        )
        if "logged" in candidates:
            sequences["logged"] = logged
        if "logged_noise" in candidates:
            scale = args.noise_scale * (adapter.action_high - adapter.action_low)
            noise = rng.normal(0.0, scale, size=logged.shape).astype(np.float32)
            sequences["logged_noise"] = np.clip(
                logged + noise,
                adapter.action_low,
                adapter.action_high,
            ).astype(np.float32)
    if "actor" in candidates:
        actor_sequence = rollout_policy_actions(
            adapter,
            snapshot,
            state,
            config,
            rng,
            keys.get("actor", key),
            policy="actor",
            horizon=args.horizon,
        )
        sequences["actor"] = actor_sequence
    if "actor_noise_pool" in candidates and actor_sequence is None:
        actor_sequence = rollout_policy_actions(
            adapter,
            snapshot,
            state,
            config,
            rng,
            keys.get("actor_noise_pool", key),
            policy="actor",
            horizon=args.horizon,
        )
    if "zero" in candidates:
        sequences["zero"] = np.zeros((args.horizon, config.action_dim), dtype=np.float32)
    if "random" in candidates:
        sequences["random"] = rng.uniform(
            adapter.action_low,
            adapter.action_high,
            size=(args.horizon, config.action_dim),
        ).astype(np.float32)
    if "random_pool" in candidates:
        for index in range(args.random_pool_size):
            sequences[f"random_{index:03d}"] = rng.uniform(
                adapter.action_low,
                adapter.action_high,
                size=(args.horizon, config.action_dim),
            ).astype(np.float32)
    if "actor_noise_pool" in candidates:
        assert actor_sequence is not None
        scale = args.noise_scale * (adapter.action_high - adapter.action_low)
        for index in range(args.actor_noise_pool_size):
            noise = rng.normal(0.0, scale, size=actor_sequence.shape).astype(np.float32)
            sequences[f"actor_noise_{index:03d}"] = np.clip(
                actor_sequence + noise,
                adapter.action_low,
                adapter.action_high,
            ).astype(np.float32)
    if "cem" in candidates:
        sequences["cem"] = np.asarray(
            cem_action_sequence(
                state,
                config,
                jnp.asarray(obs_context[None], dtype=jnp.float32),
                jnp.asarray(action_context[None], dtype=jnp.float32),
                jnp.asarray(adapter.action_low, dtype=jnp.float32),
                jnp.asarray(adapter.action_high, dtype=jnp.float32),
                keys.get("cem", key),
                horizon=args.horizon,
                candidates=args.cem_candidates,
                elite_count=max(1, int(round(args.cem_candidates * args.cem_elite_fraction))),
                iterations=args.cem_iterations,
                init_std=args.cem_init_std,
                min_std=args.cem_min_std,
            )[0]
        )
    if "cem_pool" in candidates:
        pool = np.asarray(
            cem_action_sequence_pool(
                state,
                config,
                jnp.asarray(obs_context[None], dtype=jnp.float32),
                jnp.asarray(action_context[None], dtype=jnp.float32),
                jnp.asarray(adapter.action_low, dtype=jnp.float32),
                jnp.asarray(adapter.action_high, dtype=jnp.float32),
                jnp.zeros((1, args.horizon, config.action_dim), dtype=jnp.float32),
                keys.get("cem_pool", key),
                horizon=args.horizon,
                candidates=args.cem_candidates,
                elite_count=max(1, int(round(args.cem_candidates * args.cem_elite_fraction))),
                iterations=args.cem_iterations,
                pool_size=min(args.cem_pool_size, args.cem_candidates),
                init_std=args.cem_init_std,
                min_std=args.cem_min_std,
            )[0]
        )
        for index, sequence in enumerate(pool):
            sequences[f"cem_zero_{index:03d}"] = np.asarray(sequence, dtype=np.float32)
    if "cem_actor_pool" in candidates:
        if actor_sequence is None:
            actor_sequence = rollout_policy_actions(
                adapter,
                snapshot,
                state,
                config,
                rng,
                keys.get("cem_actor_pool", key),
                policy="actor",
                horizon=args.horizon,
            )
        initial_mean = normalize_actions(
            jnp.asarray(actor_sequence[None], dtype=jnp.float32),
            jnp.asarray(adapter.action_low, dtype=jnp.float32),
            jnp.asarray(adapter.action_high, dtype=jnp.float32),
        )
        pool = np.asarray(
            cem_action_sequence_pool(
                state,
                config,
                jnp.asarray(obs_context[None], dtype=jnp.float32),
                jnp.asarray(action_context[None], dtype=jnp.float32),
                jnp.asarray(adapter.action_low, dtype=jnp.float32),
                jnp.asarray(adapter.action_high, dtype=jnp.float32),
                initial_mean,
                keys.get("cem_actor_pool", key),
                horizon=args.horizon,
                candidates=args.cem_candidates,
                elite_count=max(1, int(round(args.cem_candidates * args.cem_elite_fraction))),
                iterations=args.cem_iterations,
                pool_size=min(args.cem_actor_pool_size, args.cem_candidates),
                init_std=args.cem_init_std,
                min_std=args.cem_min_std,
            )[0]
        )
        for index, sequence in enumerate(pool):
            sequences[f"cem_actor_{index:03d}"] = np.asarray(sequence, dtype=np.float32)
    return sequences


@jax.jit
def _encode_context(state, obs_context: jax.Array) -> jax.Array:
    return state.apply_fn(
        {"params": state.params},
        obs_context,
        method=JepaWorldModel.encode,
    )


@jax.jit
def _score_sequence(state, obs_context, action_context, future_actions, future_obs):
    config = None
    del config
    return obs_context, action_context, future_actions, future_obs


def score_real_future(
    state,
    config: JepaConfig,
    obs_context: jax.Array,
    action_context: jax.Array,
    future_actions: jax.Array,
    future_obs: jax.Array,
) -> dict[str, np.ndarray]:
    return to_jsonable(
        _score_real_future_jit(
            state,
            obs_context,
            action_context,
            future_actions,
            future_obs,
            config,
            future_actions.shape[1],
        )
    )


@partial(jax.jit, static_argnames=("config", "horizon"))
def _score_real_future_jit(
    state,
    obs_context: jax.Array,
    action_context: jax.Array,
    future_actions: jax.Array,
    future_obs: jax.Array,
    config: JepaConfig,
    horizon: int,
) -> dict[str, jax.Array]:
    context = state.apply_fn(
        {"params": state.params},
        obs_context,
        method=JepaWorldModel.encode,
    )
    target_latents = normalize(
        state.apply_fn(
            {"params": state.params},
            future_obs,
            method=JepaWorldModel.encode,
        )
    )

    def step_fn(carry, action_t):
        latent_context, current_action_context = carry
        model_action_context = current_action_context.at[:, -1].set(action_t)
        z_ensemble, reward_ensemble, continue_logit_ensemble = state.apply_fn(
            {"params": state.params},
            latent_context,
            model_action_context,
            method=JepaWorldModel.predict_next_ensemble_from_history,
        )
        z_next = jnp.mean(z_ensemble, axis=0)
        reward = jnp.mean(reward_ensemble, axis=0)
        continue_prob = jnp.mean(jax.nn.sigmoid(continue_logit_ensemble), axis=0)
        normalized_ensemble = normalize(z_ensemble)
        mean_direction = jnp.mean(normalized_ensemble, axis=0)
        uncertainty = 1.0 - jnp.sum(jnp.square(mean_direction), axis=-1)
        latent_context = jnp.concatenate(
            [latent_context[:, 1:], z_next[:, None, :]],
            axis=1,
        )
        current_action_context = jnp.concatenate(
            [model_action_context[:, 1:], action_t[:, None, :]],
            axis=1,
        )
        return (latent_context, current_action_context), {
            "pred_latent": z_next,
            "pred_reward": reward,
            "pred_continue": continue_prob,
            "uncertainty": uncertainty,
        }

    _, rollout = jax.lax.scan(
        step_fn,
        (context, action_context),
        jnp.swapaxes(future_actions[:, :horizon], 0, 1),
    )
    pred_latents = jnp.swapaxes(rollout["pred_latent"], 0, 1)
    pred_rewards = jnp.swapaxes(rollout["pred_reward"], 0, 1)
    pred_continues = jnp.swapaxes(rollout["pred_continue"], 0, 1)
    uncertainty = jnp.swapaxes(rollout["uncertainty"], 0, 1)
    latent_cosine = jnp.sum(normalize(pred_latents) * target_latents[:, :horizon], axis=-1)
    return {
        "pred_rewards": pred_rewards,
        "pred_continues": pred_continues,
        "uncertainty": uncertainty,
        "latent_cosine": latent_cosine,
    }


@partial(
    jax.jit,
    static_argnames=("config", "horizon", "candidates", "elite_count", "iterations"),
)
def cem_action_sequence(
    state,
    config: JepaConfig,
    obs_context: jax.Array,
    action_context: jax.Array,
    action_low: jax.Array,
    action_high: jax.Array,
    key: jax.Array,
    *,
    horizon: int,
    candidates: int,
    elite_count: int,
    iterations: int,
    init_std: float,
    min_std: float,
) -> jax.Array:
    latent_context = _encode_context(state, obs_context)
    mean = jnp.zeros((1, horizon, config.action_dim), dtype=jnp.float32)
    std = jnp.full_like(mean, init_std)

    def iteration(carry, _):
        mean, std, rng = carry
        rng, sample_key = jax.random.split(rng)
        samples = jnp.clip(
            mean[:, None]
            + std[:, None]
            * jax.random.normal(
                sample_key,
                (1, candidates, horizon, config.action_dim),
                dtype=mean.dtype,
            ),
            -1.0,
            1.0,
        )
        scores, _ = score_action_sequences(
            state,
            latent_context,
            action_context,
            samples,
            action_low,
            action_high,
            config,
            uncertainty_penalty=0.0,
            uncertainty_latent_weight=1.0,
            uncertainty_reward_weight=1.0,
            uncertainty_continue_weight=1.0,
            bootstrap_value=False,
        )
        elite_indices = jnp.argsort(scores, axis=1)[:, -elite_count:]
        elite = jnp.take_along_axis(samples, elite_indices[:, :, None, None], axis=1)
        return (
            jnp.mean(elite, axis=1),
            jnp.maximum(jnp.std(elite, axis=1), min_std),
            rng,
        ), None

    (mean, _, _), _ = jax.lax.scan(iteration, (mean, std, key), None, length=iterations)
    return scale_actions(mean, action_low, action_high)


@partial(
    jax.jit,
    static_argnames=(
        "config",
        "horizon",
        "candidates",
        "elite_count",
        "iterations",
        "pool_size",
    ),
)
def cem_action_sequence_pool(
    state,
    config: JepaConfig,
    obs_context: jax.Array,
    action_context: jax.Array,
    action_low: jax.Array,
    action_high: jax.Array,
    initial_mean: jax.Array,
    key: jax.Array,
    *,
    horizon: int,
    candidates: int,
    elite_count: int,
    iterations: int,
    pool_size: int,
    init_std: float,
    min_std: float,
) -> jax.Array:
    latent_context = _encode_context(state, obs_context)
    mean = jnp.asarray(initial_mean, dtype=jnp.float32)
    std = jnp.full_like(mean, init_std)

    def iteration(carry, _):
        mean, std, rng = carry
        rng, sample_key = jax.random.split(rng)
        samples = jnp.clip(
            mean[:, None]
            + std[:, None]
            * jax.random.normal(
                sample_key,
                (1, candidates, horizon, config.action_dim),
                dtype=mean.dtype,
            ),
            -1.0,
            1.0,
        )
        scores, _ = score_action_sequences(
            state,
            latent_context,
            action_context,
            samples,
            action_low,
            action_high,
            config,
            uncertainty_penalty=0.0,
            uncertainty_latent_weight=1.0,
            uncertainty_reward_weight=1.0,
            uncertainty_continue_weight=1.0,
            bootstrap_value=False,
        )
        elite_indices = jnp.argsort(scores, axis=1)[:, -elite_count:]
        elite = jnp.take_along_axis(samples, elite_indices[:, :, None, None], axis=1)
        return (
            jnp.mean(elite, axis=1),
            jnp.maximum(jnp.std(elite, axis=1), min_std),
            rng,
        ), None

    (mean, std, rng), _ = jax.lax.scan(iteration, (mean, std, key), None, length=iterations)
    rng, sample_key = jax.random.split(rng)
    samples = jnp.clip(
        mean[:, None]
        + std[:, None]
        * jax.random.normal(
            sample_key,
            (1, candidates, horizon, config.action_dim),
            dtype=mean.dtype,
        ),
        -1.0,
        1.0,
    )
    scores, _ = score_action_sequences(
        state,
        latent_context,
        action_context,
        samples,
        action_low,
        action_high,
        config,
        uncertainty_penalty=0.0,
        uncertainty_latent_weight=1.0,
        uncertainty_reward_weight=1.0,
        uncertainty_continue_weight=1.0,
        bootstrap_value=False,
    )
    top_indices = jnp.argsort(scores, axis=1)[:, -pool_size:][:, ::-1]
    top_samples = jnp.take_along_axis(samples, top_indices[:, :, None, None], axis=1)
    return scale_actions(top_samples, action_low, action_high)


def choose_action(
    adapter: DMCVectorAdapter,
    observations: np.ndarray,
    state,
    config: JepaConfig,
    rng: np.random.Generator,
    key: jax.Array,
    *,
    policy: str,
) -> np.ndarray:
    if policy == "random":
        return adapter.sample_actions(rng)[0, 0]
    action = select_continuous_actions(
        state,
        jnp.asarray(observations[:, 0], dtype=jnp.float32),
        config,
        jnp.asarray(adapter.action_low, dtype=jnp.float32),
        jnp.asarray(adapter.action_high, dtype=jnp.float32),
        key=key,
        stochastic=False,
    )
    return np.asarray(action[0], dtype=np.float32)


def rollout_policy_actions(
    adapter: DMCVectorAdapter,
    snapshot: dict[str, Any],
    state,
    config: JepaConfig,
    rng: np.random.Generator,
    key: jax.Array,
    *,
    policy: str,
    horizon: int,
) -> np.ndarray:
    restore_snapshot(adapter, snapshot)
    observations = snapshot["observation"].copy()
    actions = []
    action_key = key
    for _ in range(horizon):
        action_key, step_key = jax.random.split(action_key)
        action = choose_action(
            adapter,
            observations,
            state,
            config,
            rng,
            step_key,
            policy=policy,
        )
        actions.append(action)
        observations = adapter.step(action[None, None, :]).observations
    return np.asarray(actions, dtype=np.float32)


def rollout_fixed_actions(
    adapter: DMCVectorAdapter,
    snapshot: dict[str, Any],
    actions: np.ndarray,
) -> dict[str, np.ndarray]:
    restore_snapshot(adapter, snapshot)
    observations = []
    rewards = []
    dones = []
    for action in actions:
        step = adapter.step(action[None, None, :])
        observations.append(np.asarray(step.observations[0, 0], dtype=np.float32))
        rewards.append(float(step.rewards[0, 0]))
        dones.append(float(step.dones[0, 0]))
        if dones[-1] > 0.5:
            # Continue recording reset observations/rewards if auto-reset fires,
            # but downstream validity masks stop at the first done.
            pass
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
    }


def capture_snapshot(adapter: DMCVectorAdapter, observations: np.ndarray) -> dict[str, Any]:
    env = adapter._envs[0]
    return {
        "physics_state": np.asarray(env.physics.get_state(), dtype=np.float64).copy(),
        "episode_returns": adapter._episode_returns.copy(),
        "episode_lengths": adapter._episode_lengths.copy(),
        "observation": np.asarray(observations, dtype=np.float32).copy(),
    }


def restore_snapshot(adapter: DMCVectorAdapter, snapshot: dict[str, Any]) -> None:
    env = adapter._envs[0]
    env.physics.set_state(snapshot["physics_state"])
    env.physics.forward()
    adapter._episode_returns[:] = snapshot["episode_returns"]
    adapter._episode_lengths[:] = snapshot["episode_lengths"]


def row_from_branch(
    scores: dict[str, Any],
    real: dict[str, np.ndarray],
    actions: np.ndarray,
    *,
    context_id: int,
    candidate: str,
    horizon: int,
    success_mean_threshold: float,
    soft_failure_mean_threshold: float,
    hard_failure_mean_threshold: float,
    action_saturation_threshold: float,
) -> dict[str, Any]:
    rewards = np.asarray(real["rewards"], dtype=np.float64)
    dones = np.asarray(real["dones"], dtype=np.float64)
    validity = transition_validity_np(dones)
    valid_steps = float(np.sum(validity))
    denom = max(valid_steps, 1.0)
    real_sum = float(np.sum(rewards * validity))
    pred_rewards = np.asarray(scores["pred_rewards"], dtype=np.float64)[0]
    pred_continues = np.asarray(scores["pred_continues"], dtype=np.float64)[0]
    uncertainty = np.asarray(scores["uncertainty"], dtype=np.float64)[0]
    latent_cosine = np.asarray(scores["latent_cosine"], dtype=np.float64)[0]
    pred_sum = float(np.sum(pred_rewards[:horizon] * validity))
    action_abs = np.abs(actions[:horizon])
    real_mean = real_sum / denom
    return {
        "context_id": int(context_id),
        "candidate": candidate,
        "candidate_group": candidate_group(candidate),
        "horizon": int(horizon),
        "trajectory_type": classify_trajectory(
            real_mean,
            success_mean_threshold=success_mean_threshold,
            soft_failure_mean_threshold=soft_failure_mean_threshold,
            hard_failure_mean_threshold=hard_failure_mean_threshold,
        ),
        "real_reward_sum": real_sum,
        "real_reward_mean": real_mean,
        "pred_reward_sum": pred_sum,
        "reward_sum_error": pred_sum - real_sum,
        "reward_sum_abs_error": abs(pred_sum - real_sum),
        "latent_cosine_mean": float(np.sum(latent_cosine * validity) / denom),
        "latent_cosine_last": float(latent_cosine[min(horizon - 1, latent_cosine.size - 1)]),
        "uncertainty_sum": float(np.sum(uncertainty * validity)),
        "uncertainty_mean": float(np.sum(uncertainty * validity) / denom),
        "continue_product": float(np.prod(np.clip(pred_continues, 0.0, 1.0))),
        "action_abs_mean": float(np.mean(action_abs)),
        "action_saturation": float(np.mean(action_abs >= action_saturation_threshold)),
        "done_observed": bool(np.any(dones > 0.5)),
    }


def transition_validity_np(dones: np.ndarray) -> np.ndarray:
    starts = np.ones((1,), dtype=np.float64)
    if dones.size <= 1:
        return starts[: dones.size]
    return np.cumprod(np.concatenate([starts, 1.0 - dones[:-1]]))


def classify_trajectory(
    reward_mean: float,
    *,
    success_mean_threshold: float,
    soft_failure_mean_threshold: float,
    hard_failure_mean_threshold: float,
) -> str:
    if reward_mean >= success_mean_threshold:
        return "success"
    if reward_mean <= hard_failure_mean_threshold:
        return "hard_failure"
    if reward_mean <= soft_failure_mean_threshold:
        return "soft_failure"
    return "middle"


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    escape_mean_threshold: float = 0.7,
) -> dict[str, Any]:
    real = np.asarray([row["real_reward_sum"] for row in rows], dtype=np.float64)
    pred = np.asarray([row["pred_reward_sum"] for row in rows], dtype=np.float64)
    abs_error = np.asarray([row["reward_sum_abs_error"] for row in rows], dtype=np.float64)
    success = np.asarray([row["trajectory_type"] == "success" for row in rows], dtype=bool)
    hard_failure = np.asarray(
        [row["trajectory_type"] == "hard_failure" for row in rows],
        dtype=bool,
    )
    by_candidate = summarize_groups(rows, key="candidate")
    by_candidate_group = summarize_groups(rows, key="candidate_group")
    return {
        "overall": {
            "count": int(len(rows)),
            "real_reward_sum_mean": safe_mean(real),
            "pred_reward_sum_mean": safe_mean(pred),
            "reward_sum_abs_error_mean": safe_mean(abs_error),
            "spearman_pred_real": spearman(pred, real),
            "pearson_pred_real": pearson(pred, real),
            "success_rate": safe_mean(success.astype(np.float64)),
            "hard_failure_rate": safe_mean(hard_failure.astype(np.float64)),
            "success_auc_pred_reward": binary_auc(pred, success),
            "failure_auc_negative_pred_reward": binary_auc(-pred, hard_failure),
        },
        "by_candidate": by_candidate,
        "by_candidate_group": by_candidate_group,
        "within_context": within_context_metrics(
            rows,
            escape_mean_threshold=escape_mean_threshold,
        ),
    }


def summarize_groups(rows: list[dict[str, Any]], *, key: str) -> dict[str, Any]:
    grouped = {}
    for value in sorted({str(row[key]) for row in rows}):
        subset = [row for row in rows if row[key] == value]
        c_real = np.asarray([row["real_reward_sum"] for row in subset], dtype=np.float64)
        c_pred = np.asarray([row["pred_reward_sum"] for row in subset], dtype=np.float64)
        c_abs = np.asarray(
            [row["reward_sum_abs_error"] for row in subset],
            dtype=np.float64,
        )
        c_success = np.asarray(
            [row["trajectory_type"] == "success" for row in subset],
            dtype=bool,
        )
        c_fail = np.asarray(
            [row["trajectory_type"] == "hard_failure" for row in subset],
            dtype=bool,
        )
        grouped[value] = {
            "count": int(len(subset)),
            "real_reward_sum_mean": safe_mean(c_real),
            "pred_reward_sum_mean": safe_mean(c_pred),
            "reward_sum_abs_error_mean": safe_mean(c_abs),
            "spearman_pred_real": spearman(c_pred, c_real),
            "pearson_pred_real": pearson(c_pred, c_real),
            "success_rate": safe_mean(c_success.astype(np.float64)),
            "hard_failure_rate": safe_mean(c_fail.astype(np.float64)),
            "success_auc_pred_reward": binary_auc(c_pred, c_success),
            "failure_auc_negative_pred_reward": binary_auc(-c_pred, c_fail),
            "trajectory_type_counts": count_values(
                [str(row["trajectory_type"]) for row in subset]
            ),
        }
    return grouped


def within_context_metrics(
    rows: list[dict[str, Any]],
    *,
    escape_mean_threshold: float | None = None,
) -> dict[str, Any]:
    if escape_mean_threshold is None:
        escape_mean_threshold = 0.7
    by_context: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_context.setdefault(int(row["context_id"]), []).append(row)
    spearmans = []
    top_pred_real = []
    top_real_pred = []
    cem_real_advantage = []
    oracle_regrets = []
    top_pred_real_ranks = []
    top_pred_is_real_best = []
    escape_available = []
    top_pred_escape = []
    real_best_rewards = []
    top_pred_groups = []
    real_best_groups = []
    for context_rows in by_context.values():
        if len(context_rows) < 2:
            continue
        real = np.asarray([row["real_reward_sum"] for row in context_rows], dtype=np.float64)
        pred = np.asarray([row["pred_reward_sum"] for row in context_rows], dtype=np.float64)
        current_spearman = spearman(pred, real)
        if current_spearman is not None:
            spearmans.append(current_spearman)
        top_pred = context_rows[int(np.argmax(pred))]
        top_real = context_rows[int(np.argmax(real))]
        top_pred_real.append(top_pred["real_reward_sum"])
        top_real_pred.append(top_real["pred_reward_sum"])
        best_real = float(top_real["real_reward_sum"])
        selected_real = float(top_pred["real_reward_sum"])
        oracle_regrets.append(best_real - selected_real)
        real_best_rewards.append(best_real)
        rank = 1 + int(np.sum(real > selected_real))
        top_pred_real_ranks.append(rank)
        top_pred_is_real_best.append(rank == 1)
        threshold_sum = float(escape_mean_threshold * top_pred["horizon"])
        escape_available.append(best_real >= threshold_sum)
        top_pred_escape.append(selected_real >= threshold_sum)
        top_pred_groups.append(str(top_pred["candidate_group"]))
        real_best_groups.append(str(top_real["candidate_group"]))
        cem = next((row for row in context_rows if row["candidate"] == "cem"), None)
        logged = next((row for row in context_rows if row["candidate"] == "logged"), None)
        if cem is not None and logged is not None:
            cem_real_advantage.append(cem["real_reward_sum"] - logged["real_reward_sum"])
    return {
        "mean_spearman_pred_real": safe_mean(np.asarray(spearmans, dtype=np.float64)),
        "median_spearman_pred_real": safe_median(np.asarray(spearmans, dtype=np.float64)),
        "top_pred_real_reward_mean": safe_mean(np.asarray(top_pred_real, dtype=np.float64)),
        "top_real_pred_reward_mean": safe_mean(np.asarray(top_real_pred, dtype=np.float64)),
        "real_best_reward_mean": safe_mean(np.asarray(real_best_rewards, dtype=np.float64)),
        "oracle_regret_mean": safe_mean(np.asarray(oracle_regrets, dtype=np.float64)),
        "oracle_regret_p50": safe_median(np.asarray(oracle_regrets, dtype=np.float64)),
        "oracle_regret_p90": safe_quantile(np.asarray(oracle_regrets, dtype=np.float64), 0.90),
        "top_pred_real_rank_mean": safe_mean(
            np.asarray(top_pred_real_ranks, dtype=np.float64)
        ),
        "top_pred_is_real_best_rate": safe_mean(
            np.asarray(top_pred_is_real_best, dtype=np.float64)
        ),
        "escape_mean_threshold": float(escape_mean_threshold),
        "escape_available_rate": safe_mean(np.asarray(escape_available, dtype=np.float64)),
        "top_pred_escape_rate": safe_mean(np.asarray(top_pred_escape, dtype=np.float64)),
        "top_pred_group_counts": count_values(top_pred_groups),
        "real_best_group_counts": count_values(real_best_groups),
        "cem_minus_logged_real_reward_mean": safe_mean(
            np.asarray(cem_real_advantage, dtype=np.float64)
        ),
        "contexts": int(len(by_context)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})


def parse_candidates(value: str) -> list[str]:
    allowed = {
        "logged",
        "logged_noise",
        "actor",
        "zero",
        "random",
        "cem",
        "random_pool",
        "actor_noise_pool",
        "cem_pool",
        "cem_actor_pool",
    }
    items = [item.strip() for item in value.split(",") if item.strip()]
    bad = [item for item in items if item not in allowed]
    if bad:
        raise ValueError(f"unknown candidates: {bad}")
    if "logged" not in items:
        items.insert(0, "logged")
    return items


def scale_actions(normalized_actions: jax.Array, low: jax.Array, high: jax.Array) -> jax.Array:
    return low + 0.5 * (normalized_actions + 1.0) * (high - low)


def normalize_actions(actions: jax.Array, low: jax.Array, high: jax.Array) -> jax.Array:
    return jnp.clip(2.0 * (actions - low) / (high - low + 1e-6) - 1.0, -1.0, 1.0)


def normalize(x: jax.Array) -> jax.Array:
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


def candidate_group(candidate: str) -> str:
    for prefix in ("actor_noise", "cem_actor", "cem_zero", "random"):
        if candidate.startswith(prefix + "_"):
            return prefix
    return candidate


def safe_median(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.median(values))


def safe_quantile(values: np.ndarray, q: float) -> float | None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.quantile(values, q))


if __name__ == "__main__":
    main()
