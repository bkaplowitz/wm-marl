"""Sequence replay for single-agent vector environments."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class ReplayBatch:
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    dones: jax.Array


class SequenceReplayBuffer:
    """Ring buffer that samples contiguous per-env trajectory chunks."""

    def __init__(
        self,
        *,
        capacity: int,
        num_envs: int,
        observation_shape: tuple[int, ...],
    ) -> None:
        if capacity < 2:
            raise ValueError("capacity must be >= 2")
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.observation_shape = tuple(int(dim) for dim in observation_shape)
        self.observations = np.zeros(
            (self.capacity, self.num_envs, *self.observation_shape),
            dtype=np.float32,
        )
        self.actions = np.zeros((self.capacity, self.num_envs), dtype=np.int32)
        self.rewards = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self._position = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def add_step(
        self,
        *,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        obs = np.asarray(observations, dtype=np.float32).reshape(
            (self.num_envs, *self.observation_shape)
        )
        self.observations[self._position] = obs
        self.actions[self._position] = np.asarray(actions, dtype=np.int32).reshape(
            (self.num_envs,)
        )
        self.rewards[self._position] = np.asarray(rewards, dtype=np.float32).reshape(
            (self.num_envs,)
        )
        self.dones[self._position] = np.asarray(dones, dtype=np.float32).reshape(
            (self.num_envs,)
        )
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def can_sample(self, *, chunk_length: int, max_horizon: int) -> bool:
        return self._size >= chunk_length + max_horizon

    def sample(
        self,
        rng: np.random.Generator,
        *,
        batch_size: int,
        chunk_length: int,
        max_horizon: int,
    ) -> ReplayBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if chunk_length < 1:
            raise ValueError("chunk_length must be >= 1")
        if max_horizon < 1:
            raise ValueError("max_horizon must be >= 1")
        sequence_length = chunk_length + max_horizon
        if self._size < sequence_length:
            raise ValueError(
                f"need at least {sequence_length} steps to sample, have {self._size}"
            )

        observations, actions, rewards, dones = self._ordered_arrays()
        max_start = self._size - sequence_length
        starts = rng.integers(0, max_start + 1, size=(batch_size,))
        envs = rng.integers(0, self.num_envs, size=(batch_size,))
        offsets = np.arange(sequence_length)
        indices = starts[:, None] + offsets[None, :]

        obs_batch = observations[indices, envs[:, None]]
        # actions/rewards/dones align with transitions out of obs[t], so one fewer
        # item than the observation sequence is needed.
        trans_indices = indices[:, :-1]
        action_batch = actions[trans_indices, envs[:, None]]
        reward_batch = rewards[trans_indices, envs[:, None]]
        done_batch = dones[trans_indices, envs[:, None]]
        return ReplayBatch(
            observations=jnp.asarray(obs_batch, dtype=jnp.float32),
            actions=jnp.asarray(action_batch, dtype=jnp.int32),
            rewards=jnp.asarray(reward_batch, dtype=jnp.float32),
            dones=jnp.asarray(done_batch, dtype=jnp.float32),
        )

    def _ordered_arrays(self):
        if self._size < self.capacity:
            return (
                self.observations[: self._size],
                self.actions[: self._size],
                self.rewards[: self._size],
                self.dones[: self._size],
            )
        order = np.concatenate(
            [
                np.arange(self._position, self.capacity),
                np.arange(0, self._position),
            ]
        )
        return (
            self.observations[order],
            self.actions[order],
            self.rewards[order],
            self.dones[order],
        )
