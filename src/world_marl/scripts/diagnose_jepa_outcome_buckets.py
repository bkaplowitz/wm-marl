"""Diagnose JEPA prediction quality by deterministic-policy outcome bucket."""

from __future__ import annotations

import argparse
import csv
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
from world_marl.scripts.eval_jepa_wm import (
    _jepa_config_from_metadata,
    parameter_counts,
    to_jsonable,
)


DETAIL_FIELDS = (
    "episode",
    "episode_return",
    "outcome_bucket",
    "context",
    "context_step",
    "horizon",
    "candidate",
    "candidate_group",
    "real_reward_sum",
    "pred_reward_sum",
    "reward_sum_error",
    "reward_step_mae",
    "latent_cosine_mean",
    "latent_cosine_last",
    "action_abs_mean",
    "action_saturation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare JEPA rollout and counterfactual ranking quality between "
            "failed, intermediate, and solved policy episodes."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--details-out", type=Path, default=None)
    parser.add_argument("--env", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--contexts-per-episode", type=int, default=4)
    parser.add_argument("--horizons", default="4,8")
    parser.add_argument("--actor-noise-candidates", type=int, default=15)
    parser.add_argument("--random-candidates", type=int, default=16)
    parser.add_argument("--actor-noise-scale", type=float, default=0.20)
    parser.add_argument("--failure-threshold", type=float, default=100.0)
    parser.add_argument("--success-threshold", type=float, default=900.0)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--action-saturation-threshold", type=float, default=0.95)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    if args.contexts_per_episode < 1:
        parser.error("--contexts-per-episode must be >= 1")
    if args.actor_noise_candidates < 0 or args.random_candidates < 0:
        parser.error("candidate counts must be >= 0")
    if args.failure_threshold >= args.success_threshold:
        parser.error("--failure-threshold must be below --success-threshold")
    args.horizons = parse_horizons(args.horizons)
    return args


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.checkpoint)
    env = args.env or metadata.get("env")
    if not isinstance(env, str) or not env.startswith("dmc:"):
        raise ValueError("--env is required unless checkpoint metadata contains a DMC env")
    seed = int(metadata.get("seed", 0) if args.seed is None else args.seed)
    config, ignored_config_keys = _jepa_config_from_metadata(metadata)
    state = create_jepa_train_state(jax.random.PRNGKey(seed + 17), config)
    state = state.replace(
        params=load_params(args.checkpoint / "checkpoint.msgpack", state.params)
    )

    adapter = DMCVectorAdapter(
        dmc_env_name(env),
        num_envs=1,
        max_cycles=args.max_cycles,
        seed=seed + 8_000_000,
        num_workers=1,
    )
    try:
        rows, context_rows, episode_rows = collect_diagnostics(
            args,
            state,
            config,
            adapter,
            seed=seed + 9_000_000,
        )
    finally:
        adapter.close()

    details_out = args.details_out or args.out.with_suffix(".csv")
    write_details(details_out, rows)
    summary = summarize(rows, context_rows, episode_rows)
    result = {
        "checkpoint": str(args.checkpoint),
        "metadata": {
            "env": env,
            "seed": seed,
            "algorithm": metadata.get("algorithm"),
            "control": metadata.get("control"),
            "ignored_legacy_jepa_config_keys": ignored_config_keys,
        },
        "parameter_counts": parameter_counts(state.params),
        "eval": {
            "episodes": args.episodes,
            "contexts_per_episode": args.contexts_per_episode,
            "horizons": args.horizons,
            "actor_noise_candidates": args.actor_noise_candidates,
            "random_candidates": args.random_candidates,
            "actor_noise_scale": args.actor_noise_scale,
            "failure_threshold": args.failure_threshold,
            "success_threshold": args.success_threshold,
            "policy": "deterministic_latest_checkpoint_actor",
        },
        "details_csv": str(details_out),
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_jsonable(result), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(result), indent=2, sort_keys=True))


def collect_diagnostics(
    args: argparse.Namespace,
    state,
    config: JepaConfig,
    adapter: DMCVectorAdapter,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    action_key = jax.random.PRNGKey(seed + 1)
    observations = adapter.reset()
    context_steps = evenly_spaced_context_steps(
        max_cycles=args.max_cycles,
        context_window=config.context_window,
        max_horizon=max(args.horizons),
        count=args.contexts_per_episode,
    )
    rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    progress = tqdm(
        total=args.episodes,
        desc="outcome-bucket diagnostics",
        unit="episode",
        disable=args.quiet,
    )
    try:
        for episode in range(args.episodes):
            episode_observations = [np.asarray(observations[0, 0], dtype=np.float32)]
            episode_actions: list[np.ndarray] = []
            saved_contexts: list[dict[str, Any]] = []
            completed_return = None
            while completed_return is None:
                step_index = len(episode_actions)
                if step_index in context_steps:
                    saved_contexts.append(
                        {
                            "context": len(saved_contexts),
                            "context_step": step_index,
                            "snapshot": capture_snapshot(adapter, observations),
                            "observations": np.asarray(
                                episode_observations[-config.context_window :],
                                dtype=np.float32,
                            ),
                            "actions": action_context(
                                episode_actions,
                                config.context_window,
                                config.action_dim,
                            ),
                        }
                    )
                action_key, step_key = jax.random.split(action_key)
                action = actor_action(
                    state,
                    config,
                    adapter,
                    observations,
                    step_key,
                )
                step = adapter.step(action[None, None, :])
                episode_actions.append(action)
                if step.completed_returns:
                    completed_return = float(step.completed_returns[0][0])
                    continuation = capture_snapshot(adapter, step.observations)
                else:
                    observations = step.observations
                    episode_observations.append(
                        np.asarray(observations[0, 0], dtype=np.float32)
                    )

            bucket = outcome_bucket(
                completed_return,
                failure_threshold=args.failure_threshold,
                success_threshold=args.success_threshold,
            )
            episode_rows.append(
                {
                    "episode": episode,
                    "episode_return": completed_return,
                    "outcome_bucket": bucket,
                    "contexts": len(saved_contexts),
                }
            )
            for saved in saved_contexts:
                action_key, branch_key = jax.random.split(action_key)
                branch_rows, branch_summary = diagnose_context(
                    args,
                    state,
                    config,
                    adapter,
                    saved,
                    rng,
                    branch_key,
                    episode=episode,
                    episode_return=completed_return,
                    bucket=bucket,
                )
                rows.extend(branch_rows)
                context_rows.extend(branch_summary)
            restore_snapshot(adapter, continuation)
            observations = np.asarray(continuation["observation"], dtype=np.float32)
            progress.update(1)
    finally:
        progress.close()
    return rows, context_rows, episode_rows


def diagnose_context(
    args: argparse.Namespace,
    state,
    config: JepaConfig,
    adapter: DMCVectorAdapter,
    saved: dict[str, Any],
    rng: np.random.Generator,
    key: jax.Array,
    *,
    episode: int,
    episode_return: float,
    bucket: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    max_horizon = max(args.horizons)
    actor_sequence = rollout_actor_sequence(
        state,
        config,
        adapter,
        saved["snapshot"],
        key,
        horizon=max_horizon,
    )
    candidates: list[tuple[str, str, np.ndarray]] = [
        ("actor", "actor", actor_sequence),
        ("zero", "zero", np.zeros_like(actor_sequence)),
    ]
    for index in range(args.actor_noise_candidates):
        noisy = np.clip(
            actor_sequence
            + rng.normal(0.0, args.actor_noise_scale, actor_sequence.shape),
            adapter.action_low,
            adapter.action_high,
        ).astype(np.float32)
        candidates.append((f"actor_noise_{index:03d}", "actor_noise", noisy))
    for index in range(args.random_candidates):
        random_actions = rng.uniform(
            adapter.action_low,
            adapter.action_high,
            size=actor_sequence.shape,
        ).astype(np.float32)
        candidates.append((f"random_{index:03d}", "random", random_actions))

    real_futures = [
        rollout_fixed_actions(adapter, saved["snapshot"], actions)
        for _, _, actions in candidates
    ]
    scores = score_futures(
        state,
        config,
        np.repeat(saved["observations"][None], len(candidates), axis=0),
        np.repeat(saved["actions"][None], len(candidates), axis=0),
        np.asarray([item[2] for item in candidates], dtype=np.float32),
        np.asarray([item["observations"] for item in real_futures], dtype=np.float32),
    )
    rows = []
    context_summaries = []
    for horizon in args.horizons:
        horizon_rows = []
        for index, (name, group, actions) in enumerate(candidates):
            real = real_futures[index]
            reward_valid = transition_validity(real["dones"][:horizon])
            latent_valid = reward_valid * (1.0 - real["dones"][:horizon])
            reward_denom = max(float(np.sum(reward_valid)), 1.0)
            latent_denom = max(float(np.sum(latent_valid)), 1.0)
            real_rewards = real["rewards"][:horizon]
            pred_rewards = scores["pred_rewards"][index, :horizon]
            latent_cosine = scores["latent_cosine"][index, :horizon]
            real_sum = float(np.sum(real_rewards * reward_valid))
            pred_sum = float(np.sum(pred_rewards * reward_valid))
            action_abs = np.abs(actions[:horizon])
            row = {
                "episode": episode,
                "episode_return": episode_return,
                "outcome_bucket": bucket,
                "context": saved["context"],
                "context_step": saved["context_step"],
                "horizon": horizon,
                "candidate": name,
                "candidate_group": group,
                "real_reward_sum": real_sum,
                "pred_reward_sum": pred_sum,
                "reward_sum_error": pred_sum - real_sum,
                "reward_step_mae": float(
                    np.sum(np.abs(pred_rewards - real_rewards) * reward_valid)
                    / reward_denom
                ),
                "latent_cosine_mean": float(
                    np.sum(latent_cosine * latent_valid) / latent_denom
                ),
                "latent_cosine_last": float(latent_cosine[horizon - 1]),
                "action_abs_mean": float(np.mean(action_abs)),
                "action_saturation": float(
                    np.mean(action_abs >= args.action_saturation_threshold)
                ),
            }
            rows.append(row)
            horizon_rows.append(row)
        context_summaries.append(summarize_context(horizon_rows))
    return rows, context_summaries


def capture_snapshot(
    adapter: DMCVectorAdapter,
    observations: np.ndarray,
) -> dict[str, Any]:
    env = adapter._envs[0]
    task_rng = env._task._random
    algorithm, keys, position, has_gauss, cached_gaussian = task_rng.get_state()
    return {
        "physics_state": np.asarray(env.physics.get_state(), dtype=np.float64).copy(),
        "physics_time": float(env.physics.data.time),
        "task_rng_state": (
            str(algorithm),
            np.asarray(keys, dtype=np.uint32).copy(),
            int(position),
            int(has_gauss),
            float(cached_gaussian),
        ),
        "step_count": int(env._step_count),
        "reset_next_step": bool(env._reset_next_step),
        "episode_returns": adapter._episode_returns.copy(),
        "episode_lengths": adapter._episode_lengths.copy(),
        "observation": np.asarray(observations, dtype=np.float32).copy(),
    }


def restore_snapshot(adapter: DMCVectorAdapter, snapshot: dict[str, Any]) -> None:
    env = adapter._envs[0]
    env.physics.set_state(snapshot["physics_state"])
    env.physics.data.time = snapshot["physics_time"]
    env.physics.forward()
    env._task._random.set_state(snapshot["task_rng_state"])
    env._step_count = snapshot["step_count"]
    env._reset_next_step = snapshot["reset_next_step"]
    adapter._episode_returns[:] = snapshot["episode_returns"]
    adapter._episode_lengths[:] = snapshot["episode_lengths"]


def actor_action(state, config, adapter, observations, key) -> np.ndarray:
    return np.asarray(
        select_continuous_actions(
            state,
            jnp.asarray(observations[:, 0], dtype=jnp.float32),
            config,
            jnp.asarray(adapter.action_low, dtype=jnp.float32),
            jnp.asarray(adapter.action_high, dtype=jnp.float32),
            key=key,
            stochastic=False,
        )
    )[0]


def rollout_actor_sequence(
    state,
    config,
    adapter,
    snapshot,
    key,
    *,
    horizon: int,
) -> np.ndarray:
    restore_snapshot(adapter, snapshot)
    observations = np.asarray(snapshot["observation"], dtype=np.float32)
    actions = []
    for _ in range(horizon):
        key, step_key = jax.random.split(key)
        action = actor_action(state, config, adapter, observations, step_key)
        actions.append(action)
        observations = adapter.step(action[None, None, :]).observations
    return np.asarray(actions, dtype=np.float32)


def rollout_fixed_actions(adapter, snapshot, actions) -> dict[str, np.ndarray]:
    restore_snapshot(adapter, snapshot)
    observations = []
    rewards = []
    dones = []
    for action in actions:
        step = adapter.step(action[None, None, :])
        observations.append(np.asarray(step.observations[0, 0], dtype=np.float32))
        rewards.append(float(step.rewards[0, 0]))
        dones.append(float(step.dones[0, 0]))
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
    }


def score_futures(
    state,
    config: JepaConfig,
    observation_context: np.ndarray,
    action_history: np.ndarray,
    future_actions: np.ndarray,
    future_observations: np.ndarray,
) -> dict[str, np.ndarray]:
    result = _score_futures_jit(
        state,
        jnp.asarray(observation_context),
        jnp.asarray(action_history),
        jnp.asarray(future_actions),
        jnp.asarray(future_observations),
        config,
        future_actions.shape[1],
    )
    return {key: np.asarray(value) for key, value in result.items()}


@partial(jax.jit, static_argnames=("config", "horizon"))
def _score_futures_jit(
    state,
    observation_context: jax.Array,
    action_history: jax.Array,
    future_actions: jax.Array,
    future_observations: jax.Array,
    config: JepaConfig,
    horizon: int,
) -> dict[str, jax.Array]:
    latent_context = state.apply_fn(
        {"params": state.params},
        observation_context,
        method=JepaWorldModel.encode,
    )
    target_latents = normalize(
        state.apply_fn(
            {"params": state.params},
            future_observations,
            method=JepaWorldModel.encode,
        )
    )

    def step_fn(carry, action_t):
        current_latents, current_actions = carry
        model_actions = current_actions.at[:, -1].set(action_t)
        z_ensemble, reward_ensemble, _ = state.apply_fn(
            {"params": state.params},
            current_latents,
            model_actions,
            method=JepaWorldModel.predict_next_ensemble_from_history,
        )
        z_next = jnp.mean(z_ensemble, axis=0)
        reward = jnp.mean(reward_ensemble, axis=0)
        next_latents = jnp.concatenate(
            [current_latents[:, 1:], z_next[:, None]],
            axis=1,
        )
        next_actions = jnp.concatenate(
            [model_actions[:, 1:], action_t[:, None]],
            axis=1,
        )
        return (next_latents, next_actions), (z_next, reward)

    _, (pred_latents, pred_rewards) = jax.lax.scan(
        step_fn,
        (latent_context, action_history),
        jnp.swapaxes(future_actions[:, :horizon], 0, 1),
    )
    pred_latents = jnp.swapaxes(pred_latents, 0, 1)
    pred_rewards = jnp.swapaxes(pred_rewards, 0, 1)
    latent_cosine = jnp.sum(
        normalize(pred_latents) * target_latents[:, :horizon],
        axis=-1,
    )
    return {
        "pred_rewards": pred_rewards,
        "latent_cosine": latent_cosine,
    }


def action_context(
    episode_actions: list[np.ndarray],
    context_window: int,
    action_dim: int,
) -> np.ndarray:
    result = np.zeros((context_window, action_dim), dtype=np.float32)
    previous = episode_actions[-(context_window - 1) :]
    if previous:
        result[: len(previous)] = np.asarray(previous, dtype=np.float32)
    return result


def evenly_spaced_context_steps(
    *,
    max_cycles: int,
    context_window: int,
    max_horizon: int,
    count: int,
) -> set[int]:
    low = context_window - 1
    high = max(low, max_cycles - max_horizon - 1)
    points = np.linspace(low, high, count + 2, dtype=np.int64)[1:-1]
    return {int(point) for point in points}


def transition_validity(dones: np.ndarray) -> np.ndarray:
    if not dones.size:
        return np.zeros((0,), dtype=np.float64)
    return np.cumprod(
        np.concatenate([np.ones((1,), dtype=np.float64), 1.0 - dones[:-1]])
    )


def outcome_bucket(
    episode_return: float,
    *,
    failure_threshold: float,
    success_threshold: float,
) -> str:
    if episode_return < failure_threshold:
        return "failure"
    if episode_return >= success_threshold:
        return "success"
    return "intermediate"


def summarize_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pred = np.asarray([row["pred_reward_sum"] for row in rows], dtype=np.float64)
    real = np.asarray([row["real_reward_sum"] for row in rows], dtype=np.float64)
    selected = int(np.argmax(pred))
    best = float(np.max(real))
    selected_real = float(real[selected])
    return {
        "episode": rows[0]["episode"],
        "episode_return": rows[0]["episode_return"],
        "outcome_bucket": rows[0]["outcome_bucket"],
        "context": rows[0]["context"],
        "context_step": rows[0]["context_step"],
        "horizon": rows[0]["horizon"],
        "spearman_pred_real": spearman(pred, real),
        "top1_regret": best - selected_real,
        "top1_real_rank": 1 + int(np.sum(real > selected_real)),
        "top1_is_real_best": bool(np.isclose(selected_real, best)),
        "actor_real_reward": float(
            next(row["real_reward_sum"] for row in rows if row["candidate"] == "actor")
        ),
        "oracle_real_reward": best,
    }


def summarize(
    rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    buckets = ("failure", "intermediate", "success")
    by_bucket: dict[str, Any] = {}
    for bucket in buckets:
        bucket_episodes = [row for row in episode_rows if row["outcome_bucket"] == bucket]
        bucket_rows = [row for row in rows if row["outcome_bucket"] == bucket]
        bucket_contexts = [
            row for row in context_rows if row["outcome_bucket"] == bucket
        ]
        by_horizon = {}
        for horizon in sorted({int(row["horizon"]) for row in rows}):
            h_rows = [row for row in bucket_rows if row["horizon"] == horizon]
            h_contexts = [
                row for row in bucket_contexts if row["horizon"] == horizon
            ]
            by_horizon[str(horizon)] = summarize_bucket_horizon(h_rows, h_contexts)
        by_bucket[bucket] = {
            "episodes": len(bucket_episodes),
            "episode_fraction": (
                len(bucket_episodes) / len(episode_rows) if episode_rows else None
            ),
            "episode_return_mean": safe_mean(
                [row["episode_return"] for row in bucket_episodes]
            ),
            "by_horizon": by_horizon,
        }
    return {
        "episodes": len(episode_rows),
        "episode_return_mean": safe_mean(
            [row["episode_return"] for row in episode_rows]
        ),
        "episode_return_std": safe_std(
            [row["episode_return"] for row in episode_rows]
        ),
        "outcome_counts": {
            bucket: sum(row["outcome_bucket"] == bucket for row in episode_rows)
            for bucket in buckets
        },
        "by_outcome_bucket": by_bucket,
        "intermediate_minus_success": bucket_contrast(by_bucket),
    }


def summarize_bucket_horizon(
    rows: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "contexts": len(contexts),
        "candidates": len(rows),
        "reward_step_mae_mean": safe_mean([row["reward_step_mae"] for row in rows]),
        "reward_sum_abs_error_mean": safe_mean(
            [abs(row["reward_sum_error"]) for row in rows]
        ),
        "reward_sum_bias_mean": safe_mean([row["reward_sum_error"] for row in rows]),
        "latent_cosine_mean": safe_mean([row["latent_cosine_mean"] for row in rows]),
        "latent_cosine_last_mean": safe_mean(
            [row["latent_cosine_last"] for row in rows]
        ),
        "within_context_spearman_mean": safe_mean(
            [row["spearman_pred_real"] for row in contexts]
        ),
        "top1_regret_mean": safe_mean([row["top1_regret"] for row in contexts]),
        "top1_real_rank_mean": safe_mean(
            [row["top1_real_rank"] for row in contexts]
        ),
        "top1_real_best_fraction": safe_mean(
            [float(row["top1_is_real_best"]) for row in contexts]
        ),
        "actor_to_oracle_gap_mean": safe_mean(
            [row["oracle_real_reward"] - row["actor_real_reward"] for row in contexts]
        ),
    }


def bucket_contrast(by_bucket: dict[str, Any]) -> dict[str, Any]:
    result = {}
    intermediate = by_bucket["intermediate"]["by_horizon"]
    success = by_bucket["success"]["by_horizon"]
    for horizon in sorted(set(intermediate) | set(success), key=int):
        left = intermediate.get(horizon, {})
        right = success.get(horizon, {})
        result[horizon] = {
            metric: numeric_difference(left.get(metric), right.get(metric))
            for metric in (
                "reward_step_mae_mean",
                "reward_sum_abs_error_mean",
                "latent_cosine_mean",
                "latent_cosine_last_mean",
                "within_context_spearman_mean",
                "top1_regret_mean",
                "top1_real_best_fraction",
                "actor_to_oracle_gap_mean",
            )
        }
    return result


def numeric_difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def normalize(values: jax.Array) -> jax.Array:
    return values / (jnp.linalg.norm(values, axis=-1, keepdims=True) + 1e-6)


def safe_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def safe_std(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.std(finite)) if finite else None


def parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not horizons or horizons[0] < 1:
        raise ValueError("--horizons must contain positive integers")
    return horizons


def write_details(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
