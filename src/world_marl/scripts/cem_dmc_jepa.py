"""CEM planner diagnostic for DMC JEPA checkpoints."""

from __future__ import annotations

import argparse
import dataclasses
import json
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from world_marl.checkpointing import load_metadata, load_params
from world_marl.envs.dmc_adapter import DMCVectorAdapter, dmc_env_name
from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.training import (
    JepaTrainState,
    append_action_context,
    create_jepa_train_state,
    ensemble_transition_uncertainty,
    replace_last_action_context,
    scale_normalized_actions,
    select_continuous_actions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether a saved JEPA world model supports model-predictive "
            "CEM control on DMC vector observations."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint directory containing checkpoint.msgpack.",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Environment, e.g. dmc:reacher/easy. Defaults to checkpoint metadata.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--env-workers", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--elite-fraction", type=float, default=0.125)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--init-std", type=float, default=0.8)
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument(
        "--plan-every",
        type=int,
        default=2,
        help="Replan every N environment steps and repeat the first action between plans.",
    )
    parser.add_argument("--uncertainty-penalty", type=float, default=0.1)
    parser.add_argument("--uncertainty-latent-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-reward-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-continue-weight", type=float, default=1.0)
    parser.add_argument(
        "--bootstrap-value",
        action="store_true",
        help="Add the JEPA critic value at the terminal imagined state to the CEM score.",
    )
    parser.add_argument(
        "--actor-baseline",
        action="store_true",
        help="Also evaluate the checkpoint actor with the same episode budget.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    if args.num_envs < 1:
        parser.error("--num-envs must be >= 1")
    if args.horizon < 1:
        parser.error("--horizon must be >= 1")
    if args.candidates < 2:
        parser.error("--candidates must be >= 2")
    if not (0.0 < args.elite_fraction <= 1.0):
        parser.error("--elite-fraction must be in (0, 1]")
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.plan_every < 1:
        parser.error("--plan-every must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint
    metadata = load_metadata(checkpoint)
    env = args.env or metadata.get("env")
    if not isinstance(env, str) or not env.startswith("dmc:"):
        raise ValueError("--env is required unless checkpoint metadata contains a DMC env")
    seed = int(metadata.get("seed", 0) if args.seed is None else args.seed)

    config = JepaConfig(**metadata["jepa_config"])
    state = create_jepa_train_state(jax.random.PRNGKey(seed + 17), config)
    state = state.replace(
        params=load_params(checkpoint / "checkpoint.msgpack", state.params)
    )

    adapter = DMCVectorAdapter(
        dmc_env_name(env),
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        seed=seed + 4_000_000,
        num_workers=min(args.env_workers, args.num_envs),
    )
    try:
        cem_eval = evaluate_cem(
            args,
            state,
            config,
            adapter,
            seed=seed + 5_000_000,
        )
    finally:
        adapter.close()

    actor_eval = None
    if args.actor_baseline:
        actor_adapter = DMCVectorAdapter(
            dmc_env_name(env),
            num_envs=args.num_envs,
            max_cycles=args.max_cycles,
            seed=seed + 4_000_000,
            num_workers=min(args.env_workers, args.num_envs),
        )
        try:
            actor_eval = evaluate_actor(
                args,
                state,
                config,
                actor_adapter,
                seed=seed + 6_000_000,
            )
        finally:
            actor_adapter.close()

    result = {
        "checkpoint": str(checkpoint),
        "metadata": {
            "env": env,
            "seed": seed,
            "algorithm": metadata.get("algorithm"),
            "control": metadata.get("control"),
            "jepa_config": dataclasses.asdict(config),
        },
        "cem": cem_eval,
        "actor": actor_eval,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_jsonable(result), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(result), indent=2, sort_keys=True))


def evaluate_cem(
    args: argparse.Namespace,
    state: JepaTrainState,
    config: JepaConfig,
    adapter: DMCVectorAdapter,
    *,
    seed: int,
) -> dict[str, Any]:
    observations = adapter.reset()
    obs_history = np.repeat(
        observations[:, :1, :],
        repeats=config.context_window,
        axis=1,
    )
    action_history = np.zeros(
        (adapter.num_envs, config.context_window, config.action_dim),
        dtype=np.float32,
    )
    last_actions = np.zeros((adapter.num_envs, config.action_dim), dtype=np.float32)
    replan_countdown = np.zeros((adapter.num_envs,), dtype=np.int32)

    returns: list[float] = []
    lengths: list[int] = []
    plan_records: list[dict[str, float]] = []
    step_calls = 0
    key = jax.random.PRNGKey(seed)
    action_low = jnp.asarray(adapter.action_low, dtype=jnp.float32)
    action_high = jnp.asarray(adapter.action_high, dtype=jnp.float32)
    elite_count = max(1, int(round(args.candidates * args.elite_fraction)))

    with tqdm(
        total=args.episodes,
        desc="cem eval",
        unit="episode",
        disable=args.quiet,
    ) as progress:
        while len(returns) < args.episodes:
            before = len(returns)
            needs_plan = bool(np.any(replan_countdown <= 0))
            if needs_plan:
                key, plan_key = jax.random.split(key)
                planned_actions, diagnostics = cem_plan_actions(
                    state,
                    jnp.asarray(obs_history, dtype=jnp.float32),
                    jnp.asarray(action_history, dtype=jnp.float32),
                    action_low,
                    action_high,
                    plan_key,
                    config,
                    horizon=args.horizon,
                    candidates=args.candidates,
                    elite_count=elite_count,
                    iterations=args.iterations,
                    init_std=args.init_std,
                    min_std=args.min_std,
                    uncertainty_penalty=args.uncertainty_penalty,
                    uncertainty_latent_weight=args.uncertainty_latent_weight,
                    uncertainty_reward_weight=args.uncertainty_reward_weight,
                    uncertainty_continue_weight=args.uncertainty_continue_weight,
                    bootstrap_value=args.bootstrap_value,
                )
                planned_np = np.asarray(planned_actions)
                mask = replan_countdown <= 0
                last_actions[mask] = planned_np[mask]
                replan_countdown[mask] = args.plan_every
                diag_np = jax.tree_util.tree_map(np.asarray, diagnostics)
                plan_records.extend(
                    _diagnostic_records(diag_np, mask=mask, limit_remaining=512)
                )

            actions = last_actions
            step = adapter.step(actions[:, None, :])
            step_calls += 1
            returns.extend(float(item[0]) for item in step.completed_returns)
            lengths.extend(int(item) for item in step.completed_lengths)

            model_action_context = np.asarray(
                replace_last_action_context(
                    jnp.asarray(action_history, dtype=jnp.float32),
                    jnp.asarray(actions, dtype=jnp.float32),
                    config,
                )
            ).copy()
            action_history = np.asarray(
                append_action_context(
                    jnp.asarray(model_action_context, dtype=jnp.float32),
                    jnp.zeros_like(jnp.asarray(actions, dtype=jnp.float32)),
                    config,
                )
            ).copy()
            obs_history = np.concatenate(
                [obs_history[:, 1:], step.observations[:, :1, :]],
                axis=1,
            )
            replan_countdown -= 1

            done_mask = np.asarray(step.dones[:, 0] > 0.0)
            if np.any(done_mask):
                reset_obs = step.observations[done_mask]
                obs_history[done_mask] = np.repeat(
                    reset_obs[:, :1, :],
                    repeats=config.context_window,
                    axis=1,
                )
                action_history[done_mask] = 0.0
                last_actions[done_mask] = 0.0
                replan_countdown[done_mask] = 0

            _update_progress(progress, before, len(returns), args.episodes)

    returns = returns[: args.episodes]
    lengths = lengths[: args.episodes]
    return {
        "episodes": len(returns),
        "num_envs": adapter.num_envs,
        "env_steps": step_calls * adapter.num_envs,
        "completed_episode_steps": int(sum(lengths)),
        "plan_every": args.plan_every,
        "horizon": args.horizon,
        "candidates": args.candidates,
        "elite_count": elite_count,
        "iterations": args.iterations,
        "uncertainty_penalty": args.uncertainty_penalty,
        "bootstrap_value": bool(args.bootstrap_value),
        "returns": returns,
        "lengths": lengths,
        **return_metrics(returns, lengths),
        "plan_diagnostics": summarize_plan_records(plan_records),
    }


def evaluate_actor(
    args: argparse.Namespace,
    state: JepaTrainState,
    config: JepaConfig,
    adapter: DMCVectorAdapter,
    *,
    seed: int,
) -> dict[str, Any]:
    observations = adapter.reset()
    returns: list[float] = []
    lengths: list[int] = []
    step_calls = 0
    key = jax.random.PRNGKey(seed)
    action_low = jnp.asarray(adapter.action_low, dtype=jnp.float32)
    action_high = jnp.asarray(adapter.action_high, dtype=jnp.float32)

    with tqdm(
        total=args.episodes,
        desc="actor eval",
        unit="episode",
        disable=args.quiet,
    ) as progress:
        while len(returns) < args.episodes:
            before = len(returns)
            key, action_key = jax.random.split(key)
            actions = np.asarray(
                select_continuous_actions(
                    state,
                    jnp.asarray(observations[:, 0], dtype=jnp.float32),
                    config,
                    action_low,
                    action_high,
                    key=action_key,
                    stochastic=False,
                )
            )
            step = adapter.step(actions[:, None, :])
            step_calls += 1
            returns.extend(float(item[0]) for item in step.completed_returns)
            lengths.extend(int(item) for item in step.completed_lengths)
            observations = step.observations
            _update_progress(progress, before, len(returns), args.episodes)

    returns = returns[: args.episodes]
    lengths = lengths[: args.episodes]
    return {
        "episodes": len(returns),
        "num_envs": adapter.num_envs,
        "env_steps": step_calls * adapter.num_envs,
        "completed_episode_steps": int(sum(lengths)),
        "returns": returns,
        "lengths": lengths,
        **return_metrics(returns, lengths),
    }


@jax.jit
def _actor_values(state: JepaTrainState, latents: jnp.ndarray) -> jnp.ndarray:
    _, values = state.apply_fn(
        {"params": state.params},
        latents,
        method=JepaWorldModel.actor_value_from_latent,
    )
    return values


def _diagnostic_records(
    diagnostics: dict[str, np.ndarray],
    *,
    mask: np.ndarray,
    limit_remaining: int,
) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    selected = np.flatnonzero(mask)
    for env_index in selected[:limit_remaining]:
        records.append(
            {
                key: float(np.asarray(value)[env_index])
                for key, value in diagnostics.items()
            }
        )
    return records


@jax.jit
def _encode_context(
    state: JepaTrainState,
    observations: jnp.ndarray,
) -> jnp.ndarray:
    return state.apply_fn(
        {"params": state.params},
        observations,
        method=JepaWorldModel.encode,
    )


def cem_plan_actions(
    state: JepaTrainState,
    obs_history: jnp.ndarray,
    action_history: jnp.ndarray,
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    key: jax.Array,
    config: JepaConfig,
    *,
    horizon: int,
    candidates: int,
    elite_count: int,
    iterations: int,
    init_std: float,
    min_std: float,
    uncertainty_penalty: float,
    uncertainty_latent_weight: float,
    uncertainty_reward_weight: float,
    uncertainty_continue_weight: float,
    bootstrap_value: bool,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    plan_fn = _cem_plan_actions_jit
    return plan_fn(
        state,
        obs_history,
        action_history,
        action_low,
        action_high,
        key,
        config,
        horizon,
        candidates,
        elite_count,
        iterations,
        init_std,
        min_std,
        uncertainty_penalty,
        uncertainty_latent_weight,
        uncertainty_reward_weight,
        uncertainty_continue_weight,
        bootstrap_value,
    )


@partial(
    jax.jit,
    static_argnames=(
        "config",
        "horizon",
        "candidates",
        "elite_count",
        "iterations",
        "bootstrap_value",
    )
)
def _cem_plan_actions_jit(
    state: JepaTrainState,
    obs_history: jnp.ndarray,
    action_history: jnp.ndarray,
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    key: jax.Array,
    config: JepaConfig,
    horizon: int,
    candidates: int,
    elite_count: int,
    iterations: int,
    init_std: float,
    min_std: float,
    uncertainty_penalty: float,
    uncertainty_latent_weight: float,
    uncertainty_reward_weight: float,
    uncertainty_continue_weight: float,
    bootstrap_value: bool,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    batch_size = obs_history.shape[0]
    action_dim = config.action_dim
    latent_context = _encode_context(state, obs_history)
    mean = jnp.zeros((batch_size, horizon, action_dim), dtype=jnp.float32)
    std = jnp.full_like(mean, init_std)

    def iteration(carry, _):
        mean, std, rng = carry
        rng, sample_key = jax.random.split(rng)
        noise = jax.random.normal(
            sample_key,
            (batch_size, candidates, horizon, action_dim),
            dtype=mean.dtype,
        )
        samples = jnp.clip(mean[:, None] + std[:, None] * noise, -1.0, 1.0)
        scores, _ = score_action_sequences(
            state,
            latent_context,
            action_history,
            samples,
            action_low,
            action_high,
            config,
            uncertainty_penalty=uncertainty_penalty,
            uncertainty_latent_weight=uncertainty_latent_weight,
            uncertainty_reward_weight=uncertainty_reward_weight,
            uncertainty_continue_weight=uncertainty_continue_weight,
            bootstrap_value=bootstrap_value,
        )
        elite_indices = jnp.argsort(scores, axis=1)[:, -elite_count:]
        elite = jnp.take_along_axis(
            samples,
            elite_indices[:, :, None, None],
            axis=1,
        )
        next_mean = jnp.mean(elite, axis=1)
        next_std = jnp.maximum(jnp.std(elite, axis=1), min_std)
        return (next_mean, next_std, rng), None

    (mean, std, _), _ = jax.lax.scan(
        iteration,
        (mean, std, key),
        xs=None,
        length=iterations,
    )
    final_scores, final_diag = score_action_sequences(
        state,
        latent_context,
        action_history,
        mean[:, None],
        action_low,
        action_high,
        config,
        uncertainty_penalty=uncertainty_penalty,
        uncertainty_latent_weight=uncertainty_latent_weight,
        uncertainty_reward_weight=uncertainty_reward_weight,
        uncertainty_continue_weight=uncertainty_continue_weight,
        bootstrap_value=bootstrap_value,
    )
    normalized_action = mean[:, 0]
    actions = scale_normalized_actions(normalized_action, action_low, action_high)
    diagnostics = {
        "predicted_score": final_scores[:, 0],
        "predicted_reward_return": final_diag["reward_return"][:, 0],
        "predicted_uncertainty_sum": final_diag["uncertainty_sum"][:, 0],
        "predicted_continue_product": final_diag["continue_product"][:, 0],
        "planned_action_abs_mean": jnp.mean(jnp.abs(normalized_action), axis=-1),
        "planned_action_saturation": jnp.mean(
            (jnp.abs(normalized_action) >= 0.95).astype(jnp.float32),
            axis=-1,
        ),
        "cem_final_std_mean": jnp.mean(std, axis=(1, 2)),
    }
    return actions, diagnostics


def score_action_sequences(
    state: JepaTrainState,
    latent_context: jnp.ndarray,
    action_history: jnp.ndarray,
    normalized_sequences: jnp.ndarray,
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    config: JepaConfig,
    *,
    uncertainty_penalty: float,
    uncertainty_latent_weight: float,
    uncertainty_reward_weight: float,
    uncertainty_continue_weight: float,
    bootstrap_value: bool,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    batch_size, candidates, horizon, action_dim = normalized_sequences.shape
    flat_batch = batch_size * candidates
    context = jnp.repeat(latent_context, candidates, axis=0)
    action_context = jnp.repeat(action_history, candidates, axis=0)
    flat_sequences = normalized_sequences.reshape((flat_batch, horizon, action_dim))
    gamma = jnp.asarray(config.gamma, dtype=context.dtype)
    discounts = jnp.ones((flat_batch,), dtype=context.dtype)
    score = jnp.zeros((flat_batch,), dtype=context.dtype)
    reward_return = jnp.zeros_like(score)
    uncertainty_sum = jnp.zeros_like(score)
    continue_product = jnp.ones_like(score)

    def step(carry, normalized_action):
        (
            context,
            action_context,
            discounts,
            score,
            reward_return,
            uncertainty_sum,
            continue_product,
        ) = carry
        action = scale_normalized_actions(normalized_action, action_low, action_high)
        model_action_context = replace_last_action_context(
            action_context,
            action,
            config,
        )
        z_ensemble, reward_ensemble, continue_logit_ensemble = state.apply_fn(
            {"params": state.params},
            context,
            model_action_context,
            method=JepaWorldModel.predict_next_ensemble_from_history,
        )
        next_z = jnp.mean(z_ensemble, axis=0)
        raw_reward = jnp.mean(reward_ensemble, axis=0)
        reward = (
            jnp.clip(
                raw_reward,
                config.imagined_reward_min,
                config.imagined_reward_max,
            )
            if config.clip_imagined_rewards
            else raw_reward
        )
        continues = jnp.mean(jax.nn.sigmoid(continue_logit_ensemble), axis=0)
        uncertainty = ensemble_transition_uncertainty(
            z_ensemble,
            reward_ensemble,
            continue_logit_ensemble,
            latent_weight=uncertainty_latent_weight,
            reward_weight=uncertainty_reward_weight,
            continue_weight=uncertainty_continue_weight,
        )
        step_score = reward - uncertainty_penalty * uncertainty
        score = score + discounts * step_score
        reward_return = reward_return + discounts * reward
        uncertainty_sum = uncertainty_sum + discounts * uncertainty
        discounts = discounts * gamma * continues
        continue_product = continue_product * continues
        next_context = jnp.concatenate([context[:, 1:], next_z[:, None, :]], axis=1)
        next_action_context = append_action_context(
            model_action_context,
            jnp.zeros_like(action),
            config,
        )
        return (
            next_context,
            next_action_context,
            discounts,
            score,
            reward_return,
            uncertainty_sum,
            continue_product,
        ), None

    normalized_by_time = jnp.swapaxes(flat_sequences, 0, 1)
    (
        final_context,
        _,
        discounts,
        score,
        reward_return,
        uncertainty_sum,
        continue_product,
    ), _ = jax.lax.scan(
        step,
        (
            context,
            action_context,
            discounts,
            score,
            reward_return,
            uncertainty_sum,
            continue_product,
        ),
        normalized_by_time,
    )
    if bootstrap_value:
        score = score + discounts * _actor_values(state, final_context[:, -1])
    return score.reshape((batch_size, candidates)), {
        "reward_return": reward_return.reshape((batch_size, candidates)),
        "uncertainty_sum": uncertainty_sum.reshape((batch_size, candidates)),
        "continue_product": continue_product.reshape((batch_size, candidates)),
    }


def return_metrics(returns: list[float], lengths: list[int]) -> dict[str, Any]:
    values = np.asarray(returns, dtype=np.float64)
    if values.size == 0:
        return {}
    sorted_values = np.sort(values)
    tail_count = max(1, int(np.ceil(0.10 * values.size)))
    failures = values < 100.0
    successes = values >= 900.0
    nonfailures = values[~failures]
    return {
        "mean_return": float(np.mean(values)),
        "std_return": float(np.std(values)),
        "mean_length": float(np.mean(lengths)) if lengths else None,
        "failure_count": int(np.sum(failures)),
        "failure_rate": float(np.mean(failures)),
        "success_count": int(np.sum(successes)),
        "success_rate": float(np.mean(successes)),
        "return_min": float(np.min(values)),
        "return_max": float(np.max(values)),
        "return_p05": float(np.quantile(values, 0.05)),
        "return_p10": float(np.quantile(values, 0.10)),
        "return_p25": float(np.quantile(values, 0.25)),
        "return_p50": float(np.quantile(values, 0.50)),
        "return_p75": float(np.quantile(values, 0.75)),
        "return_p90": float(np.quantile(values, 0.90)),
        "return_cvar10": float(np.mean(sorted_values[:tail_count])),
        "nonfailure_mean_return": (
            float(np.mean(nonfailures)) if nonfailures.size else None
        ),
    }


def summarize_plan_records(records: list[dict[str, float]]) -> dict[str, Any]:
    if not records:
        return {}
    keys = sorted(records[0])
    summary: dict[str, Any] = {"records": len(records)}
    for key in keys:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
        summary[f"{key}_p10"] = float(np.quantile(values, 0.10))
        summary[f"{key}_p50"] = float(np.quantile(values, 0.50))
        summary[f"{key}_p90"] = float(np.quantile(values, 0.90))
    return summary


def _update_progress(progress: tqdm, before: int, after: int, target: int) -> None:
    progress.update(max(0, min(after, target) - min(before, target)))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
