"""Sequence replay for single-agent vector environments."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from numpy.typing import ArrayLike


@struct.dataclass
class ReplayBatch:
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    dones: jax.Array


class SequenceReplayBuffer:
    """Ring buffer that samples contiguous per-env stream windows."""

    def __init__(
        self,
        *,
        capacity: int,
        num_envs: int,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...] = (),
        action_dtype: np.dtype | type = np.int32,
    ) -> None:
        if capacity < 2:
            raise ValueError("capacity must be >= 2")
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.observation_shape = tuple(int(dim) for dim in observation_shape)
        self.action_shape = tuple(int(dim) for dim in action_shape)
        self.action_dtype = np.dtype(action_dtype)
        self.observations = np.zeros(
            (self.capacity, self.num_envs, *self.observation_shape),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (self.capacity, self.num_envs, *self.action_shape),
            dtype=self.action_dtype,
        )
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
        self.actions[self._position] = np.asarray(
            actions,
            dtype=self.action_dtype,
        ).reshape((self.num_envs, *self.action_shape))
        self.rewards[self._position] = np.asarray(rewards, dtype=np.float32).reshape(
            (self.num_envs,)
        )
        self.dones[self._position] = np.asarray(dones, dtype=np.float32).reshape(
            (self.num_envs,)
        )
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_steps(
        self,
        *,
        observations: ArrayLike,
        actions: ArrayLike,
        rewards: ArrayLike,
        dones: ArrayLike,
    ) -> None:
        """Bulk write a ``[T, num_envs, ...]`` block, equivalent to T ``add_step`` calls."""
        obs = np.asarray(observations, dtype=np.float32).reshape(
            (-1, self.num_envs, *self.observation_shape)
        )
        steps = obs.shape[0]
        if steps == 0:
            return
        acts = np.asarray(actions, dtype=self.action_dtype).reshape(
            (steps, self.num_envs, *self.action_shape)
        )
        rews = np.asarray(rewards, dtype=np.float32).reshape((steps, self.num_envs))
        dons = np.asarray(dones, dtype=np.float32).reshape((steps, self.num_envs))
        # Only the trailing ``capacity`` rows survive a longer block; skip the
        # overwritten prefix so the fancy-indexed write has unique indices.
        offset = max(0, steps - self.capacity)
        indices = (self._position + np.arange(offset, steps)) % self.capacity
        self.observations[indices] = obs[offset:]
        self.actions[indices] = acts[offset:]
        self.rewards[indices] = rews[offset:]
        self.dones[indices] = dons[offset:]
        self._position = (self._position + steps) % self.capacity
        self._size = min(self._size + steps, self.capacity)

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

        max_start = self._size - sequence_length
        starts = rng.integers(0, max_start + 1, size=(batch_size,))
        envs = rng.integers(0, self.num_envs, size=(batch_size,))
        offsets = np.arange(sequence_length)
        # Map logical (oldest-first) indices to physical ring positions so the
        # gather touches only the sampled windows instead of reordering the
        # whole buffer.
        indices = starts[:, None] + offsets[None, :]
        if self._size == self.capacity:
            indices = (self._position + indices) % self.capacity

        obs_batch = self.observations[indices, envs[:, None]]
        # actions/rewards/dones align with transitions out of obs[t], so one fewer
        # item than the observation sequence is needed.
        trans_indices = indices[:, :-1]
        action_batch = self.actions[trans_indices, envs[:, None]]
        reward_batch = self.rewards[trans_indices, envs[:, None]]
        done_batch = self.dones[trans_indices, envs[:, None]]
        return ReplayBatch(
            observations=jnp.asarray(obs_batch, dtype=jnp.float32),
            actions=jnp.asarray(
                action_batch,
                dtype=(
                    jnp.int32
                    if np.issubdtype(self.action_dtype, np.integer)
                    else jnp.float32
                ),
            ),
            rewards=jnp.asarray(reward_batch, dtype=jnp.float32),
            dones=jnp.asarray(done_batch, dtype=jnp.float32),
        )
