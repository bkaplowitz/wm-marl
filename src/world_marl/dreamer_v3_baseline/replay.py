from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from world_marl.dreamer_v3_baseline.config import DreamerV3Config
from world_marl.dreamer_v3_baseline.rssm import RSSMState
from world_marl.world_model_foundation.replay import JaxSequenceBatch


class JaxDreamerReplay(NamedTuple):
    sequence: JaxSequenceBatch
    deterministic: jax.Array
    stochastic: jax.Array
    logits: jax.Array
    valid_start_logits: jax.Array
    online_starts: jax.Array
    online_count: jax.Array
    online_cursor: jax.Array
    size: jax.Array
    write_index: jax.Array
    total_inserted: jax.Array


class DreamerReplaySample(NamedTuple):
    batch: JaxSequenceBatch
    initial_state: RSSMState
    previous_actions: jax.Array
    context_starts: jax.Array
    time_indices: jax.Array
    sequence_indices: jax.Array
    online_items: jax.Array


def initialize_empty_dreamer_replay(
    config: DreamerV3Config,
    *,
    num_sequences: int,
    capacity_time: int,
    sequence_length: int,
) -> JaxDreamerReplay:
    if num_sequences <= 0:
        raise ValueError("num_sequences must be positive")
    if capacity_time <= sequence_length:
        raise ValueError("Dreamer replay capacity must exceed sequence_length")
    if config.action_mode == "discrete":
        actions = jnp.zeros((capacity_time, num_sequences), dtype=jnp.int32)
    else:
        actions = jnp.zeros(
            (capacity_time, num_sequences, config.action_dim),
            dtype=jnp.float32,
        )
    observations = jnp.zeros(
        (capacity_time, num_sequences, *config.observation_shape),
        dtype=jnp.float32,
    )
    scalars = jnp.zeros((capacity_time, num_sequences), dtype=jnp.float32)
    masks = jnp.zeros((capacity_time, num_sequences), dtype=bool)
    sequence = JaxSequenceBatch(
        observations=observations,
        actions=actions,
        rewards=scalars,
        continues=scalars,
        is_first=masks,
        is_terminal=masks,
        is_last=masks,
    )
    deterministic = jnp.zeros(
        (capacity_time, num_sequences, config.rssm.deterministic_size),
        dtype=jnp.float32,
    )
    stochastic = jnp.zeros(
        (
            capacity_time,
            num_sequences,
            config.rssm.stochastic_size,
            config.rssm.discrete_classes,
        ),
        dtype=jnp.float32,
    )
    logits = jnp.zeros_like(stochastic)
    max_starts = capacity_time - sequence_length
    context_length = sequence_length + 1
    max_online_windows = (capacity_time + context_length - 1) // context_length
    sentinel = jnp.asarray(capacity_time * num_sequences, dtype=jnp.int32)
    return JaxDreamerReplay(
        sequence=sequence,
        deterministic=deterministic,
        stochastic=stochastic,
        logits=logits,
        valid_start_logits=jnp.full(
            (max_starts * num_sequences,),
            -jnp.inf,
            dtype=jnp.float32,
        ),
        online_starts=jnp.full(
            (max_online_windows * num_sequences,),
            sentinel,
            dtype=jnp.int32,
        ),
        online_count=jnp.zeros((), dtype=jnp.int32),
        online_cursor=jnp.zeros((), dtype=jnp.int32),
        size=jnp.zeros((), dtype=jnp.int32),
        write_index=jnp.zeros((), dtype=jnp.int32),
        total_inserted=jnp.zeros((), dtype=jnp.int32),
    )


def _valid_context_starts(
    sequence: JaxSequenceBatch,
    *,
    sequence_length: int,
) -> jax.Array:
    time_steps, num_sequences = sequence.observations.shape[:2]
    num_starts = time_steps - sequence_length
    if num_starts <= 0:
        raise ValueError("Dreamer replay requires sequence_length + 1 context records")
    return jnp.ones(
        (num_starts, num_sequences),
        dtype=bool,
    )


def initialize_dreamer_replay(
    sequence: JaxSequenceBatch,
    world_model_params: Any,
    config: DreamerV3Config,
    key: jax.Array,
    *,
    sequence_length: int,
    capacity_time: int | None = None,
) -> JaxDreamerReplay:
    from world_marl.dreamer_v3_baseline.training import observe_dreamer_sequence

    action_dtype = jnp.int32 if config.action_mode == "discrete" else jnp.float32
    actions = sequence.actions.astype(action_dtype)
    previous_actions = jnp.concatenate(
        [jnp.zeros_like(actions[:1]), actions[:-1]],
        axis=0,
    )
    posterior = observe_dreamer_sequence(
        world_model_params,
        sequence.observations.astype(jnp.float32),
        previous_actions,
        sequence.is_first.astype(bool),
        config,
        key,
    )
    time_steps, num_sequences = sequence.observations.shape[:2]
    capacity_time = time_steps if capacity_time is None else int(capacity_time)
    if capacity_time < time_steps:
        raise ValueError("capacity_time cannot be smaller than the initial sequence")
    if capacity_time <= sequence_length:
        raise ValueError("Dreamer replay capacity must exceed sequence_length")

    def pad_time(array: jax.Array) -> jax.Array:
        padding = [(0, capacity_time - time_steps)] + [(0, 0)] * (array.ndim - 1)
        return jnp.pad(array, padding)

    sequence = JaxSequenceBatch(*(pad_time(array) for array in sequence))
    deterministic = pad_time(posterior["deterministic"])
    stochastic = pad_time(posterior["stochastic"])
    logits = pad_time(posterior["posterior_logits"])
    max_starts = capacity_time - sequence_length
    valid_starts = time_steps - sequence_length
    flat_indices = jnp.arange(max_starts * num_sequences, dtype=jnp.int32)
    valid_start_logits = jnp.where(
        flat_indices < valid_starts * num_sequences,
        0.0,
        -jnp.inf,
    )
    context_length = sequence_length + 1
    max_online_windows = (capacity_time + context_length - 1) // context_length
    online_times = jnp.arange(max_online_windows, dtype=jnp.int32) * context_length
    online_starts = (
        online_times[:, None] * num_sequences
        + jnp.arange(num_sequences, dtype=jnp.int32)[None]
    ).reshape(-1)
    online_windows = jnp.maximum(
        (time_steps - context_length) // context_length + 1,
        0,
    )
    online_count = jnp.asarray(online_windows * num_sequences, dtype=jnp.int32)
    sentinel = jnp.asarray(capacity_time * num_sequences, dtype=jnp.int32)
    online_starts = jnp.where(
        jnp.arange(online_starts.size, dtype=jnp.int32) < online_count,
        online_starts,
        sentinel,
    )
    return JaxDreamerReplay(
        sequence=sequence,
        deterministic=deterministic,
        stochastic=stochastic,
        logits=logits,
        valid_start_logits=valid_start_logits,
        online_starts=online_starts,
        online_count=online_count,
        online_cursor=jnp.zeros((), dtype=jnp.int32),
        size=jnp.asarray(time_steps, dtype=jnp.int32),
        write_index=jnp.asarray(time_steps % capacity_time, dtype=jnp.int32),
        total_inserted=jnp.asarray(time_steps, dtype=jnp.int32),
    )


def sample_dreamer_replay(
    replay: JaxDreamerReplay,
    key: jax.Array,
    *,
    sequence_length: int,
    batch_size: int,
) -> tuple[JaxDreamerReplay, DreamerReplaySample]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    capacity_time, num_sequences = replay.sequence.observations.shape[:2]
    context_length = sequence_length + 1
    oldest = replay.total_inserted - replay.size
    first_online_window = (oldest + context_length - 1) // context_length
    online_cursor = jnp.maximum(
        replay.online_cursor,
        first_online_window * num_sequences,
    )
    remaining = jnp.maximum(replay.online_count - online_cursor, 0)
    online_items = jnp.minimum(remaining, batch_size)
    online_indices = online_cursor + jnp.arange(batch_size, dtype=jnp.int32)
    online_sequence_indices = online_indices % num_sequences
    online_logical_starts = (online_indices // num_sequences) * context_length
    valid_starts = jnp.maximum(replay.size - sequence_length, 1)
    uniform_indices = jax.random.randint(
        key,
        (batch_size,),
        minval=0,
        maxval=valid_starts * num_sequences,
        dtype=jnp.int32,
    )
    uniform_sequence_indices = uniform_indices % num_sequences
    uniform_logical_starts = oldest + uniform_indices // num_sequences
    use_online = jnp.arange(batch_size, dtype=jnp.int32) < online_items
    sequence_indices = jnp.where(
        use_online,
        online_sequence_indices,
        uniform_sequence_indices,
    )
    logical_starts = jnp.where(
        use_online,
        online_logical_starts,
        uniform_logical_starts,
    )
    logical_time_indices = (
        logical_starts[None]
        + jnp.arange(
            context_length,
            dtype=jnp.int32,
        )[:, None]
    )
    physical_time_indices = logical_time_indices % capacity_time
    context_starts = physical_time_indices[0]

    def sample_field(array: jax.Array) -> jax.Array:
        return array[physical_time_indices, sequence_indices[None], ...]

    context = JaxSequenceBatch(*(sample_field(array) for array in replay.sequence))
    batch = JaxSequenceBatch(*(array[1:] for array in context))
    initial_state = RSSMState(
        deterministic=replay.deterministic[context_starts, sequence_indices],
        stochastic=replay.stochastic[context_starts, sequence_indices],
        logits=replay.logits[context_starts, sequence_indices],
    )
    time_indices = physical_time_indices[1:]
    replay = replay._replace(
        online_cursor=online_cursor + online_items,
    )
    return replay, DreamerReplaySample(
        batch=batch,
        initial_state=initial_state,
        previous_actions=context.actions[:-1],
        context_starts=context_starts,
        time_indices=time_indices,
        sequence_indices=sequence_indices,
        online_items=online_items,
    )


def append_dreamer_replay(
    replay: JaxDreamerReplay,
    sequence: JaxSequenceBatch,
    posterior: dict[str, jax.Array],
    *,
    sequence_length: int,
) -> JaxDreamerReplay:
    capacity_time, num_sequences = replay.sequence.observations.shape[:2]
    if sequence.observations.shape[1] != num_sequences:
        raise ValueError("appended replay sequence must preserve the environment axis")
    if sequence.observations.shape[0] > capacity_time:
        raise ValueError("appended replay sequence exceeds replay capacity")

    inputs = (
        *sequence,
        posterior["deterministic"],
        posterior["stochastic"],
        posterior["posterior_logits"],
    )

    def append_one(train_replay: JaxDreamerReplay, values):
        *record_values, deterministic, stochastic, logits = values
        record = JaxSequenceBatch(*record_values)
        index = train_replay.write_index
        replay_sequence = JaxSequenceBatch(
            *(
                buffer.at[index].set(value)
                for buffer, value in zip(train_replay.sequence, record, strict=True)
            )
        )
        return train_replay._replace(
            sequence=replay_sequence,
            deterministic=train_replay.deterministic.at[index].set(deterministic),
            stochastic=train_replay.stochastic.at[index].set(stochastic),
            logits=train_replay.logits.at[index].set(logits),
            size=jnp.minimum(train_replay.size + 1, capacity_time),
            write_index=(index + 1) % capacity_time,
            total_inserted=train_replay.total_inserted + 1,
        ), None

    replay, _ = jax.lax.scan(append_one, replay, inputs)
    max_starts = capacity_time - sequence_length
    valid_starts = jnp.maximum(replay.size - sequence_length, 0)
    flat_indices = jnp.arange(max_starts * num_sequences, dtype=jnp.int32)
    valid_start_logits = jnp.where(
        flat_indices < valid_starts * num_sequences,
        0.0,
        -jnp.inf,
    )
    context_length = sequence_length + 1
    online_windows = jnp.maximum(
        (replay.total_inserted - context_length) // context_length + 1,
        0,
    )
    online_count = online_windows * num_sequences
    max_online_windows = replay.online_starts.size // num_sequences
    first_window = jnp.maximum(online_windows - max_online_windows, 0)
    window_indices = first_window + jnp.arange(
        max_online_windows,
        dtype=jnp.int32,
    )
    online_starts = (
        window_indices[:, None] * context_length * num_sequences
        + jnp.arange(num_sequences, dtype=jnp.int32)[None]
    ).reshape(-1)
    sentinel = replay.total_inserted * num_sequences
    online_starts = jnp.where(
        window_indices.repeat(num_sequences) < online_windows,
        online_starts,
        sentinel,
    )
    return replay._replace(
        valid_start_logits=valid_start_logits,
        online_starts=online_starts,
        online_count=online_count.astype(jnp.int32),
    )


def update_dreamer_replay_latents(
    replay: JaxDreamerReplay,
    sample: DreamerReplaySample,
    posterior: dict[str, jax.Array],
) -> JaxDreamerReplay:
    sequence_indices = jnp.broadcast_to(
        sample.sequence_indices[None, :],
        sample.time_indices.shape,
    )
    return replay._replace(
        deterministic=replay.deterministic.at[
            sample.time_indices, sequence_indices
        ].set(posterior["deterministic"]),
        stochastic=replay.stochastic.at[sample.time_indices, sequence_indices].set(
            posterior["stochastic"]
        ),
        logits=replay.logits.at[sample.time_indices, sequence_indices].set(
            posterior["posterior_logits"]
        ),
    )
