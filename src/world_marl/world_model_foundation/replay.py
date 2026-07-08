from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(slots=True)
class WorldModelSequenceBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    continues: np.ndarray
    is_first: np.ndarray
    is_terminal: np.ndarray
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations)
        self.actions = np.asarray(self.actions)
        self.rewards = np.asarray(self.rewards)
        self.continues = np.asarray(self.continues)
        self.is_first = np.asarray(self.is_first, dtype=bool)
        self.is_terminal = np.asarray(self.is_terminal, dtype=bool)
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

    observations = np.empty(
        (time_steps, batch_size, *observation_shape), dtype=np.float32
    )
    for t in range(time_steps):
        for b in range(batch_size):
            offset = np.float32((t + b) / max(time_steps + batch_size - 2, 1))
            observations[t, b] = np.mod(base + offset, np.float32(1.0))

    time = np.arange(time_steps, dtype=np.int32)[:, None]
    batch = np.arange(batch_size, dtype=np.int32)[None, :]
    actions = ((time + batch) % np.int32(action_dim)).astype(np.int32)
    rewards = actions.astype(np.float32) / np.float32(max(action_dim - 1, 1))
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
