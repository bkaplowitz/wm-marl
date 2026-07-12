"""Rank logged futures with a saved JEPA world model.

This diagnostic asks a narrower question than policy evaluation:

    Does the world model assign better scores to real futures that actually
    produce higher returns?

It samples contiguous replay windows, rolls the model forward from the real
context under logged/shuffled/zero/random action futures, and writes both a
window-level CSV and aggregate ranking metrics.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from world_marl.checkpointing import load_metadata, load_params
from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.replay import ReplayBatch, SequenceReplayBuffer
from world_marl.jepa.training import create_jepa_train_state
from world_marl.scripts.eval_jepa_wm import (
    collect_dmc_replay,
    parameter_counts,
    replay_stats,
    to_jsonable,
)


CSV_FIELDS = (
    "window_id",
    "control_mode",
    "horizon",
    "trajectory_type",
    "valid_steps",
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether a JEPA checkpoint ranks real action-conditioned "
            "futures by their actual reward."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Summary JSON path.")
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Window-level CSV path. Defaults to --out with .csv suffix.",
    )
    parser.add_argument("--replay", type=Path, default=None)
    parser.add_argument("--env", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--env-workers", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--collect-steps", type=int, default=1024)
    parser.add_argument(
        "--collect-policy",
        choices=("random", "actor"),
        default="actor",
        help="Policy used when collecting fresh replay.",
    )
    parser.add_argument("--save-collected-replay", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument(
        "--horizons",
        default="1,2,4,8,16,32,64",
        help="Comma-separated cumulative reward horizons.",
    )
    parser.add_argument(
        "--controls",
        default="logged,shuffled,zero,random",
        help="Comma-separated action futures: logged, shuffled, zero, random.",
    )
    parser.add_argument("--success-mean-threshold", type=float, default=0.9)
    parser.add_argument("--soft-failure-mean-threshold", type=float, default=0.7)
    parser.add_argument("--hard-failure-mean-threshold", type=float, default=0.1)
    parser.add_argument(
        "--action-saturation-threshold",
        type=float,
        default=0.95,
        help="Absolute action value threshold for saturation metric.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.batches < 1:
        parser.error("--batches must be >= 1")
    if args.collect_steps < 2:
        parser.error("--collect-steps must be >= 2")
    return args


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.checkpoint)
    env = args.env or metadata.get("env")
    if not isinstance(env, str) or not env.startswith("dmc:"):
        raise ValueError("--env is required unless checkpoint metadata contains a DMC env")
    seed = int(metadata.get("seed", 0) if args.seed is None else args.seed)
    config = JepaConfig(**metadata["jepa_config"])
    horizons = parse_int_list(args.horizons)
    controls = parse_control_list(args.controls)
    max_horizon = max(horizons)

    state = create_jepa_train_state(jax.random.PRNGKey(seed + 17), config)
    state = state.replace(
        params=load_params(args.checkpoint / "checkpoint.msgpack", state.params)
    )

    if args.replay is None:
        replay, replay_source = collect_dmc_replay(args, state, config, env, seed=seed)
        if args.save_collected_replay is not None:
            args.save_collected_replay.parent.mkdir(parents=True, exist_ok=True)
            replay.save_npz(args.save_collected_replay)
    else:
        replay = SequenceReplayBuffer.load_npz(args.replay)
        replay_source = {"mode": "loaded_npz", "path": str(args.replay)}

    if not replay.can_sample(
        chunk_length=config.context_window,
        max_horizon=max_horizon,
    ):
        raise ValueError(
            "replay is too short for requested horizons: "
            f"size={replay.size}, need={config.context_window + max_horizon}"
        )

    rng = np.random.default_rng(seed + 600_000)
    key = jax.random.PRNGKey(seed + 700_000)
    rows: list[dict[str, Any]] = []
    progress = tqdm(
        range(args.batches),
        desc="rank futures",
        unit="batch",
        disable=args.quiet,
    )
    window_offset = 0
    for batch_index in progress:
        batch = replay.sample(
            rng,
            batch_size=args.batch_size,
            chunk_length=config.context_window,
            max_horizon=max_horizon,
        )
        key, future_key = jax.random.split(key)
        futures = make_action_futures(
            batch,
            config,
            controls=controls,
            key=future_key,
            horizon=max_horizon,
        )
        for control_mode, future_actions in futures.items():
            scores = score_future_actions(
                state,
                batch,
                future_actions,
                config,
                horizon=max_horizon,
            )
            rows.extend(
                rows_from_scores(
                    scores,
                    future_actions=np.asarray(future_actions),
                    horizons=horizons,
                    control_mode=control_mode,
                    window_offset=window_offset,
                    success_mean_threshold=args.success_mean_threshold,
                    soft_failure_mean_threshold=args.soft_failure_mean_threshold,
                    hard_failure_mean_threshold=args.hard_failure_mean_threshold,
                    action_saturation_threshold=args.action_saturation_threshold,
                )
            )
        window_offset += args.batch_size

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
        "replay": {
            **replay_source,
            "size_per_env": replay.size,
            "num_envs": replay.num_envs,
            "env_steps": replay.size * replay.num_envs,
            "observation_shape": replay.observation_shape,
            "action_shape": replay.action_shape,
            "action_dtype": str(replay.action_dtype),
            "stats": replay_stats(replay),
        },
        "eval": {
            "batch_size": args.batch_size,
            "batches": args.batches,
            "windows": args.batch_size * args.batches,
            "horizons": horizons,
            "controls": controls,
            "success_mean_threshold": args.success_mean_threshold,
            "soft_failure_mean_threshold": args.soft_failure_mean_threshold,
            "hard_failure_mean_threshold": args.hard_failure_mean_threshold,
        },
        "csv": str(csv_out),
        "summary": summarize_rows(rows, controls=controls, horizons=horizons),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


def make_action_futures(
    batch: ReplayBatch,
    config: JepaConfig,
    *,
    controls: list[str],
    key: jax.Array,
    horizon: int,
) -> dict[str, jax.Array]:
    context_window = config.context_window
    logged = batch.actions[:, context_window - 1 : context_window - 1 + horizon]
    futures: dict[str, jax.Array] = {}
    keys = dict(zip(controls, jax.random.split(key, max(len(controls), 1)), strict=False))
    for control in controls:
        if control == "logged":
            futures[control] = logged
        elif control == "zero":
            futures[control] = jnp.zeros_like(logged)
        elif control == "shuffled":
            futures[control] = jax.random.permutation(keys[control], logged, axis=0)
        elif control == "random":
            futures[control] = jax.random.uniform(
                keys[control],
                logged.shape,
                minval=-1.0,
                maxval=1.0,
                dtype=logged.dtype,
            )
        else:
            raise ValueError(f"unknown control: {control}")
    return futures


@partial(jax.jit, static_argnames=("config", "horizon"))
def score_future_actions(
    state,
    batch: ReplayBatch,
    future_actions: jax.Array,
    config: JepaConfig,
    *,
    horizon: int,
) -> dict[str, jax.Array]:
    context_window = config.context_window
    all_latents = state.apply_fn(
        {"params": state.params},
        batch.observations[:, : context_window + horizon],
        method=JepaWorldModel.encode,
    )
    target_latents = normalize(all_latents[:, context_window : context_window + horizon])
    context = all_latents[:, :context_window]
    action_context = batch.actions[:, :context_window]
    real_rewards = batch.rewards[:, context_window - 1 : context_window - 1 + horizon]
    dones = batch.dones[:, context_window - 1 : context_window - 1 + horizon]
    validity = transition_validity(dones)

    def step_fn(carry, action_t):
        latent_context, current_action_context = carry
        current_action_context = current_action_context.at[:, -1].set(action_t)
        z_ensemble, reward_ensemble, continue_logit_ensemble = state.apply_fn(
            {"params": state.params},
            latent_context,
            current_action_context,
            method=JepaWorldModel.predict_next_ensemble_from_history,
        )
        normalized_ensemble = normalize(z_ensemble)
        mean_direction = jnp.mean(normalized_ensemble, axis=0)
        latent_disagreement = 1.0 - jnp.sum(jnp.square(mean_direction), axis=-1)
        z_next = jnp.mean(z_ensemble, axis=0)
        reward = jnp.mean(reward_ensemble, axis=0)
        continue_prob = jnp.mean(jax.nn.sigmoid(continue_logit_ensemble), axis=0)
        latent_context = jnp.concatenate(
            [latent_context[:, 1:], z_next[:, None, :]],
            axis=1,
        )
        current_action_context = jnp.concatenate(
            [current_action_context[:, 1:], action_t[:, None, :]],
            axis=1,
        )
        outputs = {
            "pred_latent": z_next,
            "pred_reward": reward,
            "pred_continue": continue_prob,
            "uncertainty": latent_disagreement,
            "reward_std": jnp.std(reward_ensemble, axis=0),
            "continue_std": jnp.std(jax.nn.sigmoid(continue_logit_ensemble), axis=0),
        }
        return (latent_context, current_action_context), outputs

    _, rollout = jax.lax.scan(
        step_fn,
        (context, action_context),
        jnp.swapaxes(future_actions, 0, 1),
    )
    pred_latents = jnp.swapaxes(rollout["pred_latent"], 0, 1)
    pred_rewards = jnp.swapaxes(rollout["pred_reward"], 0, 1)
    pred_continues = jnp.swapaxes(rollout["pred_continue"], 0, 1)
    uncertainty = jnp.swapaxes(rollout["uncertainty"], 0, 1)
    reward_std = jnp.swapaxes(rollout["reward_std"], 0, 1)
    continue_std = jnp.swapaxes(rollout["continue_std"], 0, 1)
    latent_cosine = jnp.sum(normalize(pred_latents) * target_latents, axis=-1)
    return {
        "real_rewards": real_rewards,
        "pred_rewards": pred_rewards,
        "pred_continues": pred_continues,
        "validity": validity,
        "latent_cosine": latent_cosine,
        "uncertainty": uncertainty,
        "reward_std": reward_std,
        "continue_std": continue_std,
    }


def transition_validity(dones: jax.Array) -> jax.Array:
    if dones.shape[1] < 1:
        return jnp.ones_like(dones)
    starts = jnp.ones_like(dones[:, :1])
    previous_not_done = 1.0 - dones[:, :-1]
    return jnp.cumprod(jnp.concatenate([starts, previous_not_done], axis=1), axis=1)


def rows_from_scores(
    scores: dict[str, jax.Array],
    *,
    future_actions: np.ndarray,
    horizons: list[int],
    control_mode: str,
    window_offset: int,
    success_mean_threshold: float,
    soft_failure_mean_threshold: float,
    hard_failure_mean_threshold: float,
    action_saturation_threshold: float,
) -> list[dict[str, Any]]:
    real_rewards = np.asarray(scores["real_rewards"], dtype=np.float64)
    pred_rewards = np.asarray(scores["pred_rewards"], dtype=np.float64)
    pred_continues = np.asarray(scores["pred_continues"], dtype=np.float64)
    validity = np.asarray(scores["validity"], dtype=np.float64)
    latent_cosine = np.asarray(scores["latent_cosine"], dtype=np.float64)
    uncertainty = np.asarray(scores["uncertainty"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for index in range(real_rewards.shape[0]):
        window_id = window_offset + index
        for horizon in horizons:
            mask = validity[index, :horizon]
            valid_steps = float(np.sum(mask))
            denom = max(valid_steps, 1.0)
            real_sum = float(np.sum(real_rewards[index, :horizon] * mask))
            pred_sum = float(np.sum(pred_rewards[index, :horizon] * mask))
            real_mean = real_sum / denom
            pred_continue = np.clip(pred_continues[index, :horizon], 0.0, 1.0)
            action_window = future_actions[index, :horizon]
            trajectory_type = classify_trajectory(
                real_mean,
                success_mean_threshold=success_mean_threshold,
                soft_failure_mean_threshold=soft_failure_mean_threshold,
                hard_failure_mean_threshold=hard_failure_mean_threshold,
            )
            rows.append(
                {
                    "window_id": window_id,
                    "control_mode": control_mode,
                    "horizon": int(horizon),
                    "trajectory_type": trajectory_type,
                    "valid_steps": valid_steps,
                    "real_reward_sum": real_sum,
                    "real_reward_mean": real_mean,
                    "pred_reward_sum": pred_sum,
                    "reward_sum_error": pred_sum - real_sum,
                    "reward_sum_abs_error": abs(pred_sum - real_sum),
                    "latent_cosine_mean": float(
                        np.sum(latent_cosine[index, :horizon] * mask) / denom
                    ),
                    "latent_cosine_last": float(
                        latent_cosine[index, min(horizon - 1, latent_cosine.shape[1] - 1)]
                    ),
                    "uncertainty_sum": float(np.sum(uncertainty[index, :horizon] * mask)),
                    "uncertainty_mean": float(
                        np.sum(uncertainty[index, :horizon] * mask) / denom
                    ),
                    "continue_product": float(np.prod(pred_continue)),
                    "action_abs_mean": float(np.mean(np.abs(action_window))),
                    "action_saturation": float(
                        np.mean(np.abs(action_window) >= action_saturation_threshold)
                    ),
                }
            )
    return rows


def classify_trajectory(
    real_reward_mean: float,
    *,
    success_mean_threshold: float,
    soft_failure_mean_threshold: float,
    hard_failure_mean_threshold: float,
) -> str:
    if real_reward_mean >= success_mean_threshold:
        return "success"
    if real_reward_mean <= hard_failure_mean_threshold:
        return "hard_failure"
    if real_reward_mean <= soft_failure_mean_threshold:
        return "soft_failure"
    return "middle"


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    controls: list[str],
    horizons: list[int],
) -> dict[str, Any]:
    by_control: dict[str, dict[str, Any]] = {}
    for control in controls:
        control_rows = [row for row in rows if row["control_mode"] == control]
        by_control[control] = summarize_group(control_rows, horizons=horizons)
    return {
        "by_control": by_control,
        "logged_vs_controls": logged_vs_controls(rows, controls=controls, horizons=horizons),
    }


def summarize_group(rows: list[dict[str, Any]], *, horizons: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in horizons:
        subset = [row for row in rows if int(row["horizon"]) == horizon]
        real = np.asarray([row["real_reward_sum"] for row in subset], dtype=np.float64)
        pred = np.asarray([row["pred_reward_sum"] for row in subset], dtype=np.float64)
        abs_error = np.asarray(
            [row["reward_sum_abs_error"] for row in subset],
            dtype=np.float64,
        )
        uncertainty = np.asarray(
            [row["uncertainty_sum"] for row in subset],
            dtype=np.float64,
        )
        success = np.asarray(
            [row["trajectory_type"] == "success" for row in subset],
            dtype=bool,
        )
        failure = np.asarray(
            [row["trajectory_type"] == "hard_failure" for row in subset],
            dtype=bool,
        )
        top_mask = top_fraction_mask(pred, fraction=0.10)
        result[str(horizon)] = {
            "count": int(len(subset)),
            "real_reward_sum_mean": safe_mean(real),
            "pred_reward_sum_mean": safe_mean(pred),
            "reward_sum_abs_error_mean": safe_mean(abs_error),
            "spearman_pred_real": spearman(pred, real),
            "pearson_pred_real": pearson(pred, real),
            "spearman_uncertainty_abs_error": spearman(uncertainty, abs_error),
            "success_rate": safe_mean(success.astype(np.float64)),
            "hard_failure_rate": safe_mean(failure.astype(np.float64)),
            "success_auc_pred_reward": binary_auc(pred, success),
            "failure_auc_negative_pred_reward": binary_auc(-pred, failure),
            "top10_success_precision": safe_mean(success[top_mask].astype(np.float64))
            if np.any(top_mask)
            else None,
            "top10_hard_failure_false_positive_rate": safe_mean(
                failure[top_mask].astype(np.float64)
            )
            if np.any(top_mask)
            else None,
            "trajectory_type_counts": count_values(
                [str(row["trajectory_type"]) for row in subset]
            ),
        }
    return result


def logged_vs_controls(
    rows: list[dict[str, Any]],
    *,
    controls: list[str],
    horizons: list[int],
) -> dict[str, Any]:
    if "logged" not in controls:
        return {}
    indexed = {
        (row["window_id"], row["horizon"], row["control_mode"]): row for row in rows
    }
    result: dict[str, Any] = {}
    for horizon in horizons:
        window_ids = sorted(
            {
                row["window_id"]
                for row in rows
                if row["horizon"] == horizon and row["control_mode"] == "logged"
            }
        )
        horizon_result: dict[str, Any] = {}
        logged_lowest_abs_error = []
        logged_highest_pred = []
        for window_id in window_ids:
            available = [
                indexed[(window_id, horizon, control)]
                for control in controls
                if (window_id, horizon, control) in indexed
            ]
            logged = indexed.get((window_id, horizon, "logged"))
            if logged is None or not available:
                continue
            logged_lowest_abs_error.append(
                logged["reward_sum_abs_error"]
                <= min(row["reward_sum_abs_error"] for row in available) + 1e-9
            )
            logged_highest_pred.append(
                logged["pred_reward_sum"]
                >= max(row["pred_reward_sum"] for row in available) - 1e-9
            )
        horizon_result["p_logged_lowest_reward_abs_error"] = safe_mean(
            np.asarray(logged_lowest_abs_error, dtype=np.float64)
        )
        horizon_result["p_logged_highest_pred_reward"] = safe_mean(
            np.asarray(logged_highest_pred, dtype=np.float64)
        )
        for control in controls:
            if control == "logged":
                continue
            deltas = []
            latent_deltas = []
            pred_deltas = []
            for window_id in window_ids:
                logged = indexed.get((window_id, horizon, "logged"))
                other = indexed.get((window_id, horizon, control))
                if logged is None or other is None:
                    continue
                deltas.append(
                    other["reward_sum_abs_error"] - logged["reward_sum_abs_error"]
                )
                latent_deltas.append(
                    logged["latent_cosine_mean"] - other["latent_cosine_mean"]
                )
                pred_deltas.append(other["pred_reward_sum"] - logged["pred_reward_sum"])
            horizon_result[f"{control}_minus_logged_abs_error_mean"] = safe_mean(
                np.asarray(deltas, dtype=np.float64)
            )
            horizon_result[f"logged_minus_{control}_latent_cosine_mean"] = safe_mean(
                np.asarray(latent_deltas, dtype=np.float64)
            )
            horizon_result[f"{control}_minus_logged_pred_reward_mean"] = safe_mean(
                np.asarray(pred_deltas, dtype=np.float64)
            )
        result[str(horizon)] = horizon_result
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})


def parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items or min(items) < 1:
        raise ValueError("horizon values must be >= 1")
    return sorted(set(items))


def parse_control_list(value: str) -> list[str]:
    allowed = {"logged", "shuffled", "zero", "random"}
    items = [item.strip() for item in value.split(",") if item.strip()]
    bad = [item for item in items if item not in allowed]
    if bad:
        raise ValueError(f"unknown controls: {bad}")
    return items


def count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def top_fraction_mask(values: np.ndarray, *, fraction: float) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(values, dtype=bool)
    count = max(1, int(math.ceil(values.size * fraction)))
    order = np.argsort(values)
    mask = np.zeros_like(values, dtype=bool)
    mask[order[-count:]] = True
    return mask


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = finite_values(x)
    y = finite_values(y)
    if x.size != y.size or x.size < 2:
        return None
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size != y.size or x.size < 2:
        return None
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return None
    return pearson(rankdata(x[mask]), rankdata(y[mask]))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    mask = np.isfinite(scores)
    scores = scores[mask]
    labels = labels[mask]
    positives = int(np.sum(labels))
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(scores)
    rank_sum_pos = float(np.sum(ranks[labels]))
    auc = (rank_sum_pos - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )
    return float(auc)


def safe_mean(values: np.ndarray) -> float | None:
    values = finite_values(values)
    if values.size == 0:
        return None
    return float(np.mean(values))


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def normalize(x: jax.Array) -> jax.Array:
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


if __name__ == "__main__":
    main()
