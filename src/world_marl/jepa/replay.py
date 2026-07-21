"""Sequence replay for single-agent vector environments."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from world_marl.jepa.reproducibility import fingerprint_arrays


@struct.dataclass
class ReplayBatch:
    """Contiguous windows with explicit boundary and bootstrap semantics."""

    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    is_last: jax.Array
    is_terminal: jax.Array

    @property
    def dones(self) -> jax.Array:
        """Legacy boundary alias for diagnostics written before `is_last`."""

        return self.is_last


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
        self.is_last = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self.is_terminal = np.zeros(
            (self.capacity, self.num_envs),
            dtype=np.float32,
        )
        # Cuts mark collector-imposed stream boundaries, such as reset-rich
        # bootstrap segments. They are neither episode-end nor Bellman-terminal
        # targets; sampled sequences simply must not cross them.
        self.cuts = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self._position = 0
        self._size = 0
        self._cut_count = 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def cut_count(self) -> int:
        """Number of collector-imposed boundaries in the stored replay."""

        return self._cut_count

    @property
    def dones(self) -> np.ndarray:
        """Legacy boundary alias for diagnostics written before `is_last`."""

        return self.is_last

    def save_npz(self, path: str | Path) -> None:
        """Persist ordered replay contents for exact diagnostic reuse."""

        observations, actions, rewards, is_last, is_terminal = self._ordered_arrays()
        np.savez_compressed(
            Path(path),
            observations=observations,
            actions=actions,
            rewards=rewards,
            is_last=is_last,
            is_terminal=is_terminal,
            cuts=self._ordered_cuts(),
            capacity=np.asarray(self.capacity, dtype=np.int64),
            num_envs=np.asarray(self.num_envs, dtype=np.int64),
            observation_shape=np.asarray(self.observation_shape, dtype=np.int64),
            action_shape=np.asarray(self.action_shape, dtype=np.int64),
            action_dtype=np.asarray(str(self.action_dtype), dtype="U32"),
        )

    def fingerprint(self) -> str:
        """Return a stable digest of ordered replay contents and metadata."""

        observations, actions, rewards, is_last, is_terminal = self._ordered_arrays()
        arrays = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "num_envs": np.asarray(self.num_envs, dtype=np.int64),
            "observation_shape": np.asarray(self.observation_shape, dtype=np.int64),
            "action_shape": np.asarray(self.action_shape, dtype=np.int64),
        }
        # Preserve fingerprints for legacy and ordinary replays that contain
        # no collector-imposed cuts.
        if self._cut_count:
            arrays["cuts"] = self._ordered_cuts()
        return fingerprint_arrays(arrays)

    @classmethod
    def load_npz(
        cls,
        path: str | Path,
        *,
        capacity: int | None = None,
    ) -> "SequenceReplayBuffer":
        data = np.load(Path(path), allow_pickle=False)
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"])
        rewards = np.asarray(data["rewards"], dtype=np.float32)
        legacy_dones = (
            np.asarray(data["dones"], dtype=np.float32)
            if "dones" in data.files
            else None
        )
        is_last = (
            np.asarray(data["is_last"], dtype=np.float32)
            if "is_last" in data.files
            else legacy_dones
        )
        if is_last is None:
            raise ValueError("saved replay is missing is_last/dones")
        is_terminal = (
            np.asarray(data["is_terminal"], dtype=np.float32)
            if "is_terminal" in data.files
            else np.asarray(is_last, dtype=np.float32)
        )
        cuts = (
            np.asarray(data["cuts"], dtype=np.float32)
            if "cuts" in data.files
            else np.zeros_like(is_last)
        )

        if observations.ndim < 3:
            raise ValueError("saved replay observations must be [T, N, ...]")
        size = int(observations.shape[0])
        num_envs = int(observations.shape[1])
        observation_shape = tuple(int(dim) for dim in observations.shape[2:])
        action_shape = tuple(int(dim) for dim in actions.shape[2:])
        action_dtype = actions.dtype
        saved_capacity = (
            int(np.asarray(data["capacity"]).item())
            if "capacity" in data.files
            else size
        )
        buffer_capacity = (
            int(capacity) if capacity is not None else max(2, saved_capacity, size)
        )
        if buffer_capacity < size:
            raise ValueError(
                f"capacity {buffer_capacity} is smaller than saved replay size {size}"
            )

        replay = cls(
            capacity=buffer_capacity,
            num_envs=num_envs,
            observation_shape=observation_shape,
            action_shape=action_shape,
            action_dtype=action_dtype,
        )
        replay.observations[:size] = observations
        replay.actions[:size] = actions.astype(replay.action_dtype, copy=False)
        replay.rewards[:size] = rewards
        replay.is_last[:size] = is_last
        replay.is_terminal[:size] = is_terminal
        replay.cuts[:size] = cuts
        replay._size = size
        replay._position = size % replay.capacity
        replay._cut_count = int(np.count_nonzero(cuts))
        return replay

    def add_step(
        self,
        *,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        is_last: np.ndarray,
        is_terminal: np.ndarray,
        cuts: np.ndarray | None = None,
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
        self.is_last[self._position] = np.asarray(
            is_last,
            dtype=np.float32,
        ).reshape((self.num_envs,))
        self.is_terminal[self._position] = np.asarray(
            is_terminal,
            dtype=np.float32,
        ).reshape((self.num_envs,))
        cut_values = (
            np.zeros((self.num_envs,), dtype=np.float32)
            if cuts is None
            else np.asarray(cuts, dtype=np.float32).reshape((self.num_envs,))
        )
        self._cut_count -= int(np.count_nonzero(self.cuts[self._position]))
        self.cuts[self._position] = cut_values
        self._cut_count += int(np.count_nonzero(cut_values))
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
        starts, envs = self.sample_indices(
            rng,
            batch_size=batch_size,
            chunk_length=chunk_length,
            max_horizon=max_horizon,
        )
        return self.sample_from_indices(
            starts,
            envs,
            chunk_length=chunk_length,
            max_horizon=max_horizon,
        )

    def sample_indices(
        self,
        rng: np.random.Generator,
        *,
        batch_size: int,
        chunk_length: int,
        max_horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw logical sequence starts and environment indices."""

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
        if self._cut_count:
            for _ in range(1024):
                valid = self._starts_avoid_cuts(starts, envs, sequence_length)
                if np.all(valid):
                    break
                count = int(np.count_nonzero(~valid))
                starts[~valid] = rng.integers(0, max_start + 1, size=(count,))
                envs[~valid] = rng.integers(0, self.num_envs, size=(count,))
            else:
                raise ValueError(
                    "replay cuts leave no sampleable contiguous sequence for "
                    f"sequence length {sequence_length}"
                )
        return starts.astype(np.int64), envs.astype(np.int64)

    def episode_start_indices(
        self,
        *,
        max_age: int,
        chunk_length: int,
        max_horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return valid sequence starts near real or collector reset boundaries."""

        if max_age < 0:
            raise ValueError("max_age must be >= 0")
        if chunk_length < 1:
            raise ValueError("chunk_length must be >= 1")
        if max_horizon < 1:
            raise ValueError("max_horizon must be >= 1")
        sequence_length = chunk_length + max_horizon
        if self._size < sequence_length:
            raise ValueError(
                f"need at least {sequence_length} steps to sample, have {self._size}"
            )

        _, _, _, is_last, _ = self._ordered_arrays()
        cuts = self._ordered_cuts()
        boundaries = (is_last > 0.5) | (cuts > 0.5)
        max_start = self._size - sequence_length
        starts: list[np.ndarray] = []
        envs: list[np.ndarray] = []
        if self._size < self.capacity:
            initial_starts = np.arange(min(max_age, max_start) + 1, dtype=np.int64)
            starts.append(np.repeat(initial_starts, self.num_envs))
            envs.append(np.tile(np.arange(self.num_envs), initial_starts.size))
        boundary_steps, boundary_envs = np.nonzero(boundaries)
        if boundary_steps.size:
            offsets = np.arange(1, max_age + 2, dtype=np.int64)
            boundary_starts = (boundary_steps[:, None] + offsets[None, :]).reshape(-1)
            boundary_start_envs = np.repeat(boundary_envs, offsets.size)
            retained = boundary_starts <= max_start
            starts.append(boundary_starts[retained])
            envs.append(boundary_start_envs[retained])
        if not starts:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

        candidate_pairs = np.unique(
            np.stack([np.concatenate(starts), np.concatenate(envs)], axis=1),
            axis=0,
        )
        candidate_starts = candidate_pairs[:, 0]
        candidate_envs = candidate_pairs[:, 1]
        transition_count = sequence_length - 1
        boundary_prefix = np.concatenate(
            [
                np.zeros((1, self.num_envs), dtype=np.int64),
                np.cumsum(boundaries, axis=0, dtype=np.int64),
            ],
            axis=0,
        )
        boundary_counts = (
            boundary_prefix[
                candidate_starts + transition_count,
                candidate_envs,
            ]
            - boundary_prefix[
                candidate_starts,
                candidate_envs,
            ]
        )
        valid = boundary_counts == 0
        return (
            candidate_starts[valid].astype(np.int64),
            candidate_envs[valid].astype(np.int64),
        )

    def _starts_avoid_cuts(
        self,
        starts: np.ndarray,
        envs: np.ndarray,
        sequence_length: int,
    ) -> np.ndarray:
        transition_offsets = np.arange(sequence_length - 1, dtype=np.int64)
        indices = starts[:, None] + transition_offsets[None, :]
        if self._size == self.capacity:
            indices = (self._position + indices) % self.capacity
        return ~np.any(self.cuts[indices, envs[:, None]] > 0.5, axis=1)

    def sample_from_indices(
        self,
        starts: np.ndarray,
        envs: np.ndarray,
        *,
        chunk_length: int,
        max_horizon: int,
    ) -> ReplayBatch:
        """Materialize a batch from pre-generated logical replay indices."""

        if chunk_length < 1:
            raise ValueError("chunk_length must be >= 1")
        if max_horizon < 1:
            raise ValueError("max_horizon must be >= 1")
        sequence_length = chunk_length + max_horizon
        if self._size < sequence_length:
            raise ValueError(
                f"need at least {sequence_length} steps to sample, have {self._size}"
            )
        starts = np.asarray(starts, dtype=np.int64).reshape((-1,))
        envs = np.asarray(envs, dtype=np.int64).reshape((-1,))
        if starts.shape != envs.shape:
            raise ValueError("starts and envs must have the same shape")
        max_start = self._size - sequence_length
        if np.any(starts < 0) or np.any(starts > max_start):
            raise ValueError(f"starts must be in [0, {max_start}]")
        if np.any(envs < 0) or np.any(envs >= self.num_envs):
            raise ValueError(f"envs must be in [0, {self.num_envs})")

        offsets = np.arange(sequence_length)
        indices = starts[:, None] + offsets[None, :]
        # Map logical oldest-first indices to physical ring-buffer positions so
        # sampling does not reorder the whole replay on every batch.
        if self._size == self.capacity:
            indices = (self._position + indices) % self.capacity

        obs_batch = self.observations[indices, envs[:, None]]
        # Transition fields align with transitions out of obs[t], so one fewer
        # item than the observation sequence is needed.
        trans_indices = indices[:, :-1]
        action_batch = self.actions[trans_indices, envs[:, None]]
        reward_batch = self.rewards[trans_indices, envs[:, None]]
        is_last_batch = self.is_last[trans_indices, envs[:, None]]
        is_terminal_batch = self.is_terminal[trans_indices, envs[:, None]]
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
            is_last=jnp.asarray(is_last_batch, dtype=jnp.float32),
            is_terminal=jnp.asarray(is_terminal_batch, dtype=jnp.float32),
        )

    def _ordered_arrays(self):
        if self._size < self.capacity:
            return (
                self.observations[: self._size],
                self.actions[: self._size],
                self.rewards[: self._size],
                self.is_last[: self._size],
                self.is_terminal[: self._size],
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
            self.is_last[order],
            self.is_terminal[order],
        )

    def _ordered_cuts(self) -> np.ndarray:
        if self._size < self.capacity:
            return self.cuts[: self._size]
        order = np.concatenate(
            [
                np.arange(self._position, self.capacity),
                np.arange(0, self._position),
            ]
        )
        return self.cuts[order]
