"""World-model-only diagnostics for JEPA checkpoints.

This script deliberately avoids actor/critic return as the primary metric. It
asks whether a saved JEPA checkpoint is a reliable action-conditioned simulator
on replay data or freshly collected DMC trajectories.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from world_marl.checkpointing import load_metadata, load_params
from world_marl.envs.dmc_adapter import DMCVectorAdapter, dmc_env_name
from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.replay import ReplayBatch, SequenceReplayBuffer
from world_marl.jepa.training import (
    create_jepa_train_state,
    evaluate_open_loop,
    evaluate_world_model_loss,
    prediction_validity,
    select_continuous_actions,
    transition_start_validity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a JEPA world model as a dynamics/reward/continue model, "
            "separate from actor-critic returns."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Optional SequenceReplayBuffer NPZ. If omitted, collect fresh DMC replay.",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="DMC env, e.g. dmc:reacher/easy. Defaults to checkpoint metadata.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--env-workers", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument(
        "--collect-steps",
        type=int,
        default=1024,
        help="Vector steps to collect when --replay is omitted.",
    )
    parser.add_argument(
        "--collect-policy",
        choices=("random", "actor"),
        default="random",
        help="Policy used for freshly collected replay.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=None,
        help="Sequence length for supervised WM metrics. Defaults to metadata or 64.",
    )
    parser.add_argument(
        "--open-loop-horizons",
        default="1,2,4,8,16,32",
        help="Comma-separated horizons for autoregressive latent rollout metrics.",
    )
    parser.add_argument(
        "--save-collected-replay",
        type=Path,
        default=None,
        help="Optional NPZ path for freshly collected replay.",
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
        raise ValueError(
            "--env is required unless checkpoint metadata contains a DMC env"
        )
    seed = int(metadata.get("seed", 0) if args.seed is None else args.seed)
    config, ignored_config_keys = _jepa_config_from_metadata(metadata)
    chunk_length = int(
        args.chunk_length
        or metadata.get("chunk_length")
        or metadata.get("model_chunk_length")
        or 64
    )
    horizons = parse_int_list(args.open_loop_horizons)

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
        replay_source = {
            "mode": "loaded_npz",
            "path": str(args.replay),
        }

    rng = np.random.default_rng(seed + 100_000)
    key = jax.random.PRNGKey(seed + 200_000)
    metrics = evaluate_model(
        state,
        config,
        replay,
        rng,
        key,
        batch_size=args.batch_size,
        batches=args.batches,
        chunk_length=chunk_length,
        horizons=horizons,
    )

    result = {
        "checkpoint": str(args.checkpoint),
        "metadata": {
            "env": env,
            "seed": seed,
            "algorithm": metadata.get("algorithm"),
            "jepa_config": dataclasses.asdict(config),
            "ignored_legacy_jepa_config_keys": ignored_config_keys,
        },
        "config_summary": {
            "latent_dim": config.latent_dim,
            "model_dim": config.model_dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "context_window": config.context_window,
            "model_horizon": config.max_horizon,
            "dynamics": "deterministic_residual",
            "reward_prediction": "symlog_twohot",
            "target_gradient": "stopgrad",
            "regularizer": "sigreg",
            "regularizer_weight": config.regularizer_weight,
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
            "chunk_length": chunk_length,
            "open_loop_horizons": horizons,
        },
        "metrics": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_jsonable(result), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(result), indent=2, sort_keys=True))


def _jepa_config_from_metadata(
    metadata: dict[str, Any],
) -> tuple[JepaConfig, list[str]]:
    payload = metadata.get("jepa_config")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint metadata is missing jepa_config")
    field_names = {field.name for field in dataclasses.fields(JepaConfig)}
    ignored = sorted(set(payload) - field_names)
    return JepaConfig(
        **{key: value for key, value in payload.items() if key in field_names}
    ), ignored


def collect_dmc_replay(
    args: argparse.Namespace,
    state,
    config: JepaConfig,
    env: str,
    *,
    seed: int,
) -> tuple[SequenceReplayBuffer, dict[str, Any]]:
    adapter = DMCVectorAdapter(
        dmc_env_name(env),
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        seed=seed + 300_000,
        num_workers=min(args.env_workers, args.num_envs),
    )
    try:
        replay = SequenceReplayBuffer(
            capacity=max(2, args.collect_steps),
            num_envs=adapter.num_envs,
            observation_shape=(config.observation_dim,),
            action_shape=(config.action_dim,),
            action_dtype=np.float32,
        )
        observations = adapter.reset()
        rng = np.random.default_rng(seed + 400_000)
        action_key = jax.random.PRNGKey(seed + 500_000)
        action_low = jnp.asarray(adapter.action_low, dtype=jnp.float32)
        action_high = jnp.asarray(adapter.action_high, dtype=jnp.float32)
        for _ in tqdm(
            range(args.collect_steps),
            desc=f"collect {args.collect_policy} replay",
            unit="step",
            disable=args.quiet,
        ):
            if args.collect_policy == "random":
                actions = adapter.sample_actions(rng)[:, 0]
            else:
                action_key, step_key = jax.random.split(action_key)
                actions = np.asarray(
                    select_continuous_actions(
                        state,
                        jnp.asarray(observations[:, 0], dtype=jnp.float32),
                        config,
                        action_low,
                        action_high,
                        key=step_key,
                        stochastic=False,
                    )
                )
            step = adapter.step(actions[:, None, :])
            replay.add_step(
                observations=observations[:, 0],
                actions=actions,
                rewards=step.rewards[:, 0],
                dones=step.dones[:, 0],
            )
            observations = step.observations
        return replay, {
            "mode": "fresh_dmc",
            "env": env,
            "collect_policy": args.collect_policy,
            "collect_steps": args.collect_steps,
        }
    finally:
        adapter.close()


def evaluate_model(
    state,
    config: JepaConfig,
    replay: SequenceReplayBuffer,
    rng: np.random.Generator,
    key: jax.Array,
    *,
    batch_size: int,
    batches: int,
    chunk_length: int,
    horizons: list[int],
) -> dict[str, Any]:
    max_eval_horizon = max([config.max_horizon, *horizons])
    if not replay.can_sample(chunk_length=chunk_length, max_horizon=config.max_horizon):
        raise ValueError(
            "replay is too short for supervised metrics: "
            f"size={replay.size}, need={chunk_length + config.max_horizon}"
        )
    if not replay.can_sample(
        chunk_length=config.context_window,
        max_horizon=max_eval_horizon,
    ):
        raise ValueError(
            "replay is too short for requested open-loop horizons: "
            f"size={replay.size}, need={config.context_window + max_eval_horizon}"
        )

    wm_records: list[dict[str, Any]] = []
    horizon_records: list[dict[str, Any]] = []
    action_records: list[dict[str, Any]] = []
    for batch_index in range(batches):
        batch_key = jax.random.fold_in(key, batch_index)
        wm_batch = replay.sample(
            rng,
            batch_size=batch_size,
            chunk_length=chunk_length,
            max_horizon=config.max_horizon,
        )
        wm_metrics = evaluate_world_model_loss(
            state,
            batch_key,
            wm_batch,
            config,
            chunk_length=chunk_length,
        )
        wm_records.append(to_jsonable(wm_metrics))
        horizon_records.append(
            to_jsonable(
                per_horizon_metrics(
                    state,
                    config,
                    wm_batch,
                    chunk_length=chunk_length,
                )
            )
        )
        action_records.append(
            to_jsonable(action_sensitivity(state, config, wm_batch))
        )

    open_loop: dict[str, Any] = {}
    for horizon in horizons:
        records = []
        for batch_index in range(batches):
            open_batch = replay.sample(
                rng,
                batch_size=batch_size,
                chunk_length=config.context_window,
                max_horizon=horizon,
            )
            records.append(
                to_jsonable(
                    evaluate_open_loop(
                        state,
                        open_batch,
                        config,
                        horizon=horizon,
                    )
                )
            )
        open_loop[str(horizon)] = aggregate_records(records)

    return {
        "world_model_loss": aggregate_records(wm_records),
        "per_horizon": aggregate_nested_records(horizon_records),
        "open_loop": open_loop,
        "action_sensitivity": aggregate_records(action_records),
    }


@jax.jit
def _noop_jit_barrier(x):
    return x


def per_horizon_metrics(
    state,
    config: JepaConfig,
    batch: ReplayBatch,
    *,
    chunk_length: int,
) -> dict[str, jax.Array]:
    outputs = state.apply_fn(
        {"params": state.params},
        batch.observations,
        batch.actions,
        chunk_length=chunk_length,
        dones=batch.dones,
        method=JepaWorldModel.sequence_outputs,
    )
    pred = normalize(outputs["predicted_latents"])
    target = normalize(outputs["target_latents"])
    ensemble_axis = pred.ndim == target.ndim + 1
    if ensemble_axis:
        pred_for_latent = jnp.mean(pred, axis=-2)
        reward_values = jnp.mean(outputs["reward_values"], axis=-1)
        continue_probs = jnp.mean(jax.nn.sigmoid(outputs["continue_logits"]), axis=-1)
        latent_disagreement = 1.0 - jnp.sum(
            jnp.square(jnp.mean(pred, axis=-2)),
            axis=-1,
        )
        reward_std = jnp.std(outputs["reward_values"], axis=-1)
        continue_std = jnp.std(jax.nn.sigmoid(outputs["continue_logits"]), axis=-1)
    else:
        pred_for_latent = pred
        reward_values = outputs["reward_values"]
        continue_probs = jax.nn.sigmoid(outputs["continue_logits"])
        latent_disagreement = jnp.zeros(pred_for_latent.shape[:-1])
        reward_std = jnp.zeros_like(reward_values)
        continue_std = jnp.zeros_like(continue_probs)

    latent_validity = prediction_validity(batch.dones, chunk_length, config.max_horizon)
    transition_validity = transition_start_validity(
        batch.dones,
        chunk_length,
        config.max_horizon,
    )
    reward_targets = jnp.stack(
        [
            batch.rewards[:, offset : offset + chunk_length]
            for offset in range(config.max_horizon)
        ],
        axis=2,
    )
    done_targets = jnp.stack(
        [
            batch.dones[:, offset : offset + chunk_length]
            for offset in range(config.max_horizon)
        ],
        axis=2,
    )
    continue_targets = 1.0 - done_targets
    cosine = jnp.sum(pred_for_latent * target, axis=-1)
    reward_error = reward_values - reward_targets
    continue_pred = continue_probs >= 0.5
    continue_true = continue_targets >= 0.5

    metrics = {}
    for horizon_index in range(config.max_horizon):
        label = f"h{horizon_index + 1}"
        latent_mask = latent_validity[:, :, horizon_index]
        transition_mask = transition_validity[:, :, horizon_index]
        metrics[f"{label}/latent_cosine"] = masked_mean(
            cosine[:, :, horizon_index],
            latent_mask,
        )
        metrics[f"{label}/latent_loss"] = masked_mean(
            1.0 - cosine[:, :, horizon_index],
            latent_mask,
        )
        metrics[f"{label}/reward_mae"] = masked_mean(
            jnp.abs(reward_error[:, :, horizon_index]),
            transition_mask,
        )
        metrics[f"{label}/reward_rmse"] = jnp.sqrt(
            masked_mean(
                jnp.square(reward_error[:, :, horizon_index]),
                transition_mask,
            )
        )
        metrics[f"{label}/reward_target_mean"] = masked_mean(
            reward_targets[:, :, horizon_index],
            transition_mask,
        )
        metrics[f"{label}/reward_pred_mean"] = masked_mean(
            reward_values[:, :, horizon_index],
            transition_mask,
        )
        metrics[f"{label}/continue_accuracy"] = masked_mean(
            (
                continue_pred[:, :, horizon_index] == continue_true[:, :, horizon_index]
            ).astype(jnp.float32),
            transition_mask,
        )
        metrics[f"{label}/ensemble_latent_disagreement"] = masked_mean(
            latent_disagreement[:, :, horizon_index],
            transition_mask,
        )
        metrics[f"{label}/ensemble_reward_std"] = masked_mean(
            reward_std[:, :, horizon_index],
            transition_mask,
        )
        metrics[f"{label}/ensemble_continue_std"] = masked_mean(
            continue_std[:, :, horizon_index],
            transition_mask,
        )
    return metrics


def action_sensitivity(
    state,
    config: JepaConfig,
    batch: ReplayBatch,
) -> dict[str, jax.Array]:
    obs = batch.observations[:, 0]
    latents = state.apply_fn(
        {"params": state.params}, obs, method=JepaWorldModel.encode
    )
    latent_context = latents[:, None, :]
    if config.action_mode != "continuous":
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return {
            "latent_low_high_l2": zero,
            "reward_low_high_abs": zero,
            "continue_low_high_abs": zero,
        }
    low = -jnp.ones((latents.shape[0], 1, config.action_dim), dtype=jnp.float32)
    high = jnp.ones((latents.shape[0], 1, config.action_dim), dtype=jnp.float32)
    z_low, r_low, c_low = state.apply_fn(
        {"params": state.params},
        latent_context,
        low,
        method=JepaWorldModel.predict_next_from_history,
    )
    z_high, r_high, c_high = state.apply_fn(
        {"params": state.params},
        latent_context,
        high,
        method=JepaWorldModel.predict_next_from_history,
    )
    return {
        "latent_low_high_l2": jnp.mean(jnp.linalg.norm(z_high - z_low, axis=-1)),
        "reward_low_high_abs": jnp.mean(jnp.abs(r_high - r_low)),
        "continue_low_high_abs": jnp.mean(
            jnp.abs(jax.nn.sigmoid(c_high) - jax.nn.sigmoid(c_low))
        ),
    }


def normalize(x: jax.Array) -> jax.Array:
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


def masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    values = jnp.asarray(values, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    return jnp.sum(values * mask) / (jnp.sum(mask) + 1e-6)


def replay_stats(replay: SequenceReplayBuffer) -> dict[str, Any]:
    observations, actions, rewards, dones = replay._ordered_arrays()
    return {
        "observation_mean_abs": float(np.mean(np.abs(observations))),
        "observation_std": float(np.std(observations)),
        "action_mean_abs": float(np.mean(np.abs(actions))),
        "action_std": float(np.std(actions)),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "done_fraction": float(np.mean(dones)),
    }


def parameter_counts(params) -> dict[str, Any]:
    counts = {
        group: int(sum(x.size for x in jax.tree_util.tree_leaves(subtree)))
        for group, subtree in params.items()
    }
    actor = counts.get("actor_head", 0)
    critic = counts.get("value_head", 0)
    total = int(sum(counts.values()))
    return {
        "total": total,
        "world_model": int(total - actor - critic),
        "actor": int(actor),
        "critic": int(critic),
        "by_group": counts,
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for record in records for key in record})
    result = {}
    for key in keys:
        values = [record[key] for record in records if is_number(record.get(key))]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        result[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return result


def aggregate_nested_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return aggregate_records(records)


def parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items or min(items) < 1:
        raise ValueError("integer list values must be >= 1")
    return sorted(set(items))
def is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(
        float(value)
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, jax.Array):
        return to_jsonable(np.asarray(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


if __name__ == "__main__":
    main()
