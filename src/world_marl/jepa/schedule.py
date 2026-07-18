"""Pure scheduling and replay-sampling helpers for canonical JEPA training."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.jepa.models import JepaConfig
from world_marl.jepa.replay import ReplayBatch, SequenceReplayBuffer


def scheduled_recent_world_model_fraction(
    args: argparse.Namespace,
    *,
    train_env_steps: int,
) -> float:
    until = args.online_recent_world_model_until_env_steps
    if until is not None and train_env_steps >= until:
        return 0.0
    return float(args.online_recent_world_model_fraction)


def scheduled_policy_reset_start_fraction(
    args: argparse.Namespace,
    *,
    train_env_steps: int,
) -> float:
    if train_env_steps < args.policy_reset_start_fraction_start_env_steps:
        return 0.0
    return float(args.policy_reset_start_fraction)


def recent_batch_size(
    batch_size: int,
    *,
    recent_replay: SequenceReplayBuffer | None,
    recent_fraction: float,
) -> int:
    if recent_replay is None or recent_fraction <= 0.0:
        return 0
    return max(0, min(batch_size, int(round(batch_size * recent_fraction))))


def effective_recent_fraction(
    requested_fraction: float,
    *,
    full_replay_size: int,
    recent_replay_size: int,
    max_oversample: float,
) -> float:
    """Cap recent replay pressure by per-transition oversampling."""

    if requested_fraction <= 0.0 or recent_replay_size <= 0:
        return 0.0
    if max_oversample <= 0.0:
        return float(requested_fraction)
    if max_oversample <= 1.0:
        return 0.0

    extra_weight = max_oversample - 1.0
    capped_fraction = (
        extra_weight
        * float(recent_replay_size)
        / (float(full_replay_size) + extra_weight * float(recent_replay_size))
    )
    return float(min(requested_fraction, capped_fraction))


def recent_oversample_ratio(
    recent_fraction: float,
    *,
    full_replay_size: int,
    recent_replay_size: int,
) -> float | None:
    """Return the recent-to-old per-transition sampling probability ratio."""

    if recent_replay_size <= 0 or full_replay_size <= recent_replay_size:
        return 1.0
    if recent_fraction <= 0.0:
        return 1.0
    if recent_fraction >= 1.0:
        return None
    return float(
        1.0
        + recent_fraction
        * float(full_replay_size)
        / ((1.0 - recent_fraction) * float(recent_replay_size))
    )


def sample_replay_batch(
    replay: SequenceReplayBuffer,
    rng: np.random.Generator,
    *,
    recent_replay: SequenceReplayBuffer | None,
    recent_fraction: float,
    batch_size: int,
    chunk_length: int,
    max_horizon: int,
) -> ReplayBatch:
    recent_size = recent_batch_size(
        batch_size,
        recent_replay=recent_replay,
        recent_fraction=recent_fraction,
    )
    if recent_size == 0:
        return replay.sample(
            rng,
            batch_size=batch_size,
            chunk_length=chunk_length,
            max_horizon=max_horizon,
        )
    assert recent_replay is not None
    full_size = batch_size - recent_size
    batches = []
    if full_size:
        batches.append(
            replay.sample(
                rng,
                batch_size=full_size,
                chunk_length=chunk_length,
                max_horizon=max_horizon,
            )
        )
    batches.append(
        recent_replay.sample(
            rng,
            batch_size=recent_size,
            chunk_length=chunk_length,
            max_horizon=max_horizon,
        )
    )
    if len(batches) == 1:
        return batches[0]
    return ReplayBatch(
        observations=jnp.concatenate([batch.observations for batch in batches], axis=0),
        actions=jnp.concatenate([batch.actions for batch in batches], axis=0),
        rewards=jnp.concatenate([batch.rewards for batch in batches], axis=0),
        dones=jnp.concatenate([batch.dones for batch in batches], axis=0),
    )


def sample_policy_starts_with_reset_mix(
    replay: SequenceReplayBuffer,
    rng: np.random.Generator,
    *,
    config: JepaConfig,
    batch_size: int,
    reset_start_indices: tuple[np.ndarray, np.ndarray] | None = None,
    reset_start_fraction: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    reset_size = int(round(batch_size * reset_start_fraction))
    if reset_size > 0 and reset_start_indices is None:
        raise ValueError("reset_start_indices are required for reset-start sampling")
    if reset_size == 0:
        return sample_policy_starts(
            replay,
            rng,
            config=config,
            batch_size=batch_size,
        )
    full_size = batch_size - reset_size
    chunks = []
    if full_size:
        chunks.append(
            sample_policy_starts(
                replay,
                rng,
                config=config,
                batch_size=full_size,
            )
        )
    assert reset_start_indices is not None
    candidate_starts, candidate_envs = reset_start_indices
    selected = rng.integers(0, candidate_starts.size, size=(reset_size,))
    reset_batch = replay.sample_from_indices(
        candidate_starts[selected],
        candidate_envs[selected],
        chunk_length=config.context_window,
        max_horizon=1,
    )
    chunks.append(
        (
            reset_batch.observations[:, : config.context_window],
            reset_batch.actions[:, : config.context_window],
        )
    )
    return (
        jnp.concatenate([chunk[0] for chunk in chunks], axis=0),
        jnp.concatenate([chunk[1] for chunk in chunks], axis=0),
    )


def sample_policy_starts(
    replay: SequenceReplayBuffer,
    rng: np.random.Generator,
    *,
    config: JepaConfig,
    batch_size: int,
) -> tuple[jax.Array, jax.Array]:
    observation_chunks = []
    action_chunks = []
    collected = 0
    attempts = 0
    sample_size = max(64, 2 * batch_size)
    while collected < batch_size and attempts < 64:
        attempts += 1
        batch = replay.sample(
            rng,
            batch_size=sample_size,
            chunk_length=config.context_window,
            max_horizon=1,
        )
        done_context = np.asarray(batch.dones[:, : config.context_window])
        valid_indices = np.flatnonzero(np.sum(done_context, axis=1) == 0.0)
        if valid_indices.size == 0:
            continue
        valid_indices = valid_indices[: batch_size - collected]
        observation_chunks.append(
            batch.observations[valid_indices, : config.context_window]
        )
        action_chunks.append(batch.actions[valid_indices, : config.context_window])
        collected += int(valid_indices.size)
    if collected < batch_size:
        raise ValueError(
            "could not sample enough policy starts without episode boundaries; "
            f"collected {collected}/{batch_size} after {attempts} attempts"
        )
    return (
        jnp.concatenate(observation_chunks, axis=0)[:batch_size],
        jnp.concatenate(action_chunks, axis=0)[:batch_size],
    )


def scheduled_value_clip(
    args: argparse.Namespace,
    *,
    train_env_steps: int,
) -> float:
    final_clip = args.value_clip_final
    start = args.value_clip_schedule_start_env_steps
    end = args.value_clip_schedule_end_env_steps
    if final_clip is None or start is None or end is None:
        return float(args.value_clip)
    if train_env_steps <= start:
        return float(args.value_clip)
    if train_env_steps >= end:
        return float(final_clip)
    progress = (train_env_steps - start) / (end - start)
    return float(args.value_clip + progress * (final_clip - args.value_clip))


def scheduled_online_actor_update_interval(
    args: argparse.Namespace,
    *,
    train_env_steps: int,
) -> int:
    if train_env_steps < args.online_policy_actor_update_interval_start_env_steps:
        return 1
    return int(args.online_policy_actor_update_interval)


def scheduled_online_encoder_freeze(
    args: argparse.Namespace,
    *,
    train_env_steps: int,
) -> bool:
    start = args.online_freeze_encoder_after_env_steps
    return start is not None and train_env_steps >= start
