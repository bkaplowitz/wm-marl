from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class JaxSequenceBatch(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    continues: jax.Array
    is_first: jax.Array
    is_terminal: jax.Array
    is_last: jax.Array


@dataclass(slots=True)
class WorldModelSequenceBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    continues: np.ndarray
    is_first: np.ndarray
    is_terminal: np.ndarray
    is_last: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations)
        self.actions = np.asarray(self.actions)
        self.rewards = np.asarray(self.rewards)
        self.continues = np.asarray(self.continues)
        self.is_first = np.asarray(self.is_first, dtype=bool)
        self.is_terminal = np.asarray(self.is_terminal, dtype=bool)
        self.is_last = np.asarray(
            self.is_terminal if self.is_last is None else self.is_last,
            dtype=bool,
        )
        self.metadata = dict(self.metadata or {})

        if self.observations.ndim < 2:
            raise ValueError(
                "observations must be time-major with shape (time, batch, ...)"
            )

        prefix = self.observations.shape[:2]
        self._require_prefix("actions", self.actions, prefix)
        self._require_prefix("rewards", self.rewards, prefix)
        self._require_prefix("continues", self.continues, prefix)
        self._require_prefix("is_first", self.is_first, prefix)
        self._require_prefix("is_terminal", self.is_terminal, prefix)
        self._require_prefix("is_last", self.is_last, prefix)

    @staticmethod
    def _require_prefix(name: str, array: np.ndarray, prefix: tuple[int, int]) -> None:
        if array.ndim < 2 or array.shape[:2] != prefix:
            raise ValueError(
                f"{name} must start with shape {prefix}, got {array.shape}"
            )

    @property
    def time_steps(self) -> int:
        return int(self.observations.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.observations.shape[1])

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.observations.shape[2:])

    @property
    def action_shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.actions.shape[2:])


def sequence_batch_to_jax(batch: WorldModelSequenceBatch) -> JaxSequenceBatch:
    return JaxSequenceBatch(
        observations=jnp.asarray(batch.observations, dtype=jnp.float32),
        actions=jnp.asarray(batch.actions),
        rewards=jnp.asarray(batch.rewards, dtype=jnp.float32),
        continues=jnp.asarray(batch.continues, dtype=jnp.float32),
        is_first=jnp.asarray(batch.is_first, dtype=bool),
        is_terminal=jnp.asarray(batch.is_terminal, dtype=bool),
        is_last=jnp.asarray(batch.is_last, dtype=bool),
    )


def sample_sequence_windows(
    batch: JaxSequenceBatch,
    key: jax.Array,
    *,
    sequence_length: int,
    batch_size: int,
    require_same_episode: bool = False,
    force_first: bool = True,
) -> JaxSequenceBatch:
    time_steps, num_sequences = batch.observations.shape[:2]
    if not 1 <= sequence_length <= time_steps:
        raise ValueError(
            f"sequence_length must be in [1, {time_steps}], got {sequence_length}"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    num_starts = time_steps - sequence_length + 1
    if require_same_episode:
        if sequence_length == 1:
            valid = jnp.ones((num_starts, num_sequences), dtype=bool)
        else:
            valid = jax.vmap(
                lambda start: (
                    ~jnp.any(
                        jax.lax.dynamic_slice_in_dim(
                            batch.is_first,
                            start + 1,
                            sequence_length - 1,
                            axis=0,
                        ),
                        axis=0,
                    )
                )
            )(jnp.arange(num_starts, dtype=jnp.int32))
        logits = jnp.where(valid.reshape(-1), 0.0, -jnp.inf)
        indices = jax.random.categorical(key, logits, shape=(batch_size,))
        starts = indices // num_sequences
        sequence_ids = indices % num_sequences
    else:
        start_key, sequence_key = jax.random.split(key)
        starts = jax.random.randint(
            start_key,
            (batch_size,),
            minval=0,
            maxval=num_starts,
        )
        sequence_ids = jax.random.randint(
            sequence_key,
            (batch_size,),
            minval=0,
            maxval=num_sequences,
        )

    def sample(array: jax.Array) -> jax.Array:
        selected = jnp.swapaxes(array[:, sequence_ids, ...], 0, 1)
        windows = jax.vmap(
            lambda values, start: jax.lax.dynamic_slice_in_dim(
                values,
                start,
                sequence_length,
                axis=0,
            )
        )(selected, starts)
        return jnp.swapaxes(windows, 0, 1)

    sampled = JaxSequenceBatch(*(sample(array) for array in batch))
    if force_first:
        sampled = sampled._replace(is_first=sampled.is_first.at[0].set(True))
    return sampled


def synthetic_observation_batch(
    *,
    time_steps: int,
    batch_size: int,
    observation_shape: tuple[int, ...],
    action_dim: int = 4,
) -> WorldModelSequenceBatch:
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")

    flat_size = int(np.prod(observation_shape, dtype=np.int64))
    base = np.arange(max(flat_size, 1), dtype=np.float32)
    base = base / np.float32(max(flat_size - 1, 1))
    base = base[:flat_size].reshape(observation_shape)

    time = np.arange(time_steps, dtype=np.int32)[:, None]
    batch = np.arange(batch_size, dtype=np.int32)[None, :]
    offsets = (time + batch).astype(np.float32) / np.float32(
        max(time_steps + batch_size - 2, 1)
    )
    observations = np.mod(
        base.reshape((1, 1, *observation_shape))
        + offsets.reshape((time_steps, batch_size, *(1,) * len(observation_shape))),
        np.float32(1.0),
    )
    actions = ((time + batch) % np.int32(action_dim)).astype(np.int32)
    rewards = np.zeros((time_steps, batch_size), dtype=np.float32)
    rewards[1:] = actions[:-1].astype(np.float32) / np.float32(max(action_dim - 1, 1))
    is_first = np.zeros((time_steps, batch_size), dtype=bool)
    is_first[0, :] = True
    is_terminal = np.zeros((time_steps, batch_size), dtype=bool)
    is_terminal[-1, :] = True
    continues = 1.0 - is_terminal.astype(np.float32)

    return WorldModelSequenceBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        continues=continues,
        is_first=is_first,
        is_terminal=is_terminal,
        metadata={
            "source": "synthetic_observation_batch",
            "observation_shape": tuple(observation_shape),
            "action_dim": action_dim,
        },
    )
