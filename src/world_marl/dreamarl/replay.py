"""Bounded sequence replay for explicit joint multi-agent transitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from world_marl.dreamarl.config import ReplayConfig
from world_marl.dreamarl.contracts import MultiAgentSequenceBatch


_ARRAY_FIELDS = (
    "observations",
    "next_observations",
    "actions",
    "rewards",
    "team_rewards",
    "is_first",
    "is_last",
    "is_terminal",
    "valid",
    "agent_alive",
    "next_agent_alive",
    "action_mask",
    "next_action_mask",
)


@dataclass(frozen=True, slots=True)
class ReplayMetadata:
    """Serializable replay geometry and cursor state."""

    num_envs: int
    num_agents: int
    lane_capacity: int
    size_per_lane: int
    cursor: int
    agent_ids: tuple[str, ...]


class JointSequenceReplay:
    """Circular replay that samples contiguous windows from one env lane.

    Capacity is measured in joint environment transitions across all lanes.
    Sampling never mixes environment lanes inside a sequence and preserves
    real terminal successors through the explicit next-observations field.
    """

    def __init__(self, config: ReplayConfig, *, seed: int) -> None:
        self.config = config
        self._rng = np.random.default_rng(seed)
        self._storage: dict[str, np.ndarray] = {}
        self._metadata: ReplayMetadata | None = None

    @property
    def initialized(self) -> bool:
        return self._metadata is not None

    @property
    def size(self) -> int:
        """Number of joint environment transitions currently retained."""

        if self._metadata is None:
            return 0
        return self._metadata.size_per_lane * self._metadata.num_envs

    @property
    def size_per_lane(self) -> int:
        if self._metadata is None:
            return 0
        return self._metadata.size_per_lane

    @property
    def can_sample(self) -> bool:
        return self.size_per_lane >= self.config.sequence_length

    def append(self, batch: MultiAgentSequenceBatch) -> None:
        """Append one synchronized collection chunk to every env lane."""

        if not self.initialized:
            self._initialize(batch)
        assert self._metadata is not None
        self._validate_append(batch)

        time_steps = batch.time_steps
        if time_steps >= self._metadata.lane_capacity:
            start = time_steps - self._metadata.lane_capacity
            source = {
                name: np.asarray(getattr(batch, name))[start:]
                for name in _ARRAY_FIELDS
            }
            for name, value in source.items():
                self._storage[name][...] = value
            self._metadata = ReplayMetadata(
                num_envs=self._metadata.num_envs,
                num_agents=self._metadata.num_agents,
                lane_capacity=self._metadata.lane_capacity,
                size_per_lane=self._metadata.lane_capacity,
                cursor=0,
                agent_ids=self._metadata.agent_ids,
            )
            return

        cursor = self._metadata.cursor
        first_count = min(time_steps, self._metadata.lane_capacity - cursor)
        second_count = time_steps - first_count
        for name in _ARRAY_FIELDS:
            source = np.asarray(getattr(batch, name))
            self._storage[name][cursor : cursor + first_count] = source[:first_count]
            if second_count:
                self._storage[name][:second_count] = source[first_count:]
        self._metadata = ReplayMetadata(
            num_envs=self._metadata.num_envs,
            num_agents=self._metadata.num_agents,
            lane_capacity=self._metadata.lane_capacity,
            size_per_lane=min(
                self._metadata.size_per_lane + time_steps,
                self._metadata.lane_capacity,
            ),
            cursor=(cursor + time_steps) % self._metadata.lane_capacity,
            agent_ids=self._metadata.agent_ids,
        )

    def sample(self, batch_size: int | None = None) -> MultiAgentSequenceBatch:
        """Sample deterministic-RNG contiguous time-major windows."""

        if not self.can_sample or self._metadata is None:
            raise RuntimeError(
                "replay does not yet contain one complete sequence window"
            )
        batch_size = batch_size or self.config.batch_size
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        length = self.config.sequence_length
        max_start = self._metadata.size_per_lane - length
        starts = self._rng.integers(0, max_start + 1, size=batch_size)
        lanes = self._rng.integers(0, self._metadata.num_envs, size=batch_size)
        oldest = (
            self._metadata.cursor - self._metadata.size_per_lane
        ) % self._metadata.lane_capacity
        offsets = np.arange(length)[None]
        physical = (
            oldest + starts[:, None] + offsets
        ) % self._metadata.lane_capacity

        sampled: dict[str, np.ndarray] = {}
        for name, storage in self._storage.items():
            batch_major = storage[physical, lanes[:, None]]
            sampled[name] = np.swapaxes(batch_major, 0, 1).copy()
        # A sampled window has no preceding context. This is a temporal cut,
        # not an environment terminal, so only is_first is changed.
        sampled["is_first"][0] = True
        return MultiAgentSequenceBatch(
            **sampled,
            agent_ids=self._metadata.agent_ids,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return an exact-resume snapshot, including sampler RNG state."""

        return {
            "metadata": self._metadata,
            "storage": {
                name: value.copy() for name, value in self._storage.items()
            },
            "rng_state": self._rng.bit_generator.state,
            "config": self.config,
        }

    def sample_template(
        self, batch_size: int | None = None
    ) -> MultiAgentSequenceBatch:
        """Sample without advancing RNG, for checkpoint shape restoration."""

        rng_state = self._rng.bit_generator.state
        try:
            return self.sample(batch_size)
        finally:
            self._rng.bit_generator.state = rng_state

    def save(self, directory: str | Path) -> Path:
        """Atomically persist replay arrays, geometry, and exact RNG state."""

        if self._metadata is None:
            raise RuntimeError("cannot checkpoint uninitialized replay")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        arrays_path = root / "replay.npz"
        arrays_temporary = root / ".replay.npz.tmp"
        with arrays_temporary.open("wb") as handle:
            np.savez_compressed(handle, **self._storage)
        arrays_temporary.replace(arrays_path)

        metadata_path = root / "replay.json"
        metadata_temporary = root / ".replay.json.tmp"
        payload = {
            "metadata": asdict(self._metadata),
            "rng_state": self._rng.bit_generator.state,
            "config": asdict(self.config),
        }
        metadata_temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        metadata_temporary.replace(metadata_path)
        return arrays_path

    def load(self, directory: str | Path) -> None:
        """Restore a replay checkpoint written by save."""

        root = Path(directory)
        payload = json.loads((root / "replay.json").read_text(encoding="utf-8"))
        if payload["config"] != asdict(self.config):
            raise ValueError("replay configuration does not match checkpoint")
        metadata = dict(payload["metadata"])
        metadata["agent_ids"] = tuple(metadata["agent_ids"])
        with np.load(root / "replay.npz", allow_pickle=False) as archive:
            storage = {name: archive[name].copy() for name in archive.files}
        self.load_state_dict(
            {
                "metadata": ReplayMetadata(**metadata),
                "storage": storage,
                "rng_state": payload["rng_state"],
                "config": self.config,
            }
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a snapshot produced by state_dict."""

        if state["config"] != self.config:
            raise ValueError("replay configuration does not match checkpoint")
        metadata = state["metadata"]
        if metadata is not None and not isinstance(metadata, ReplayMetadata):
            raise TypeError("checkpoint replay metadata has the wrong type")
        storage = {
            name: np.asarray(value).copy()
            for name, value in state["storage"].items()
        }
        if set(storage) not in (set(), set(_ARRAY_FIELDS)):
            raise ValueError("checkpoint replay fields are incomplete")
        self._metadata = metadata
        self._storage = storage
        self._rng.bit_generator.state = state["rng_state"]

    def _initialize(self, batch: MultiAgentSequenceBatch) -> None:
        lane_capacity = self.config.capacity // batch.num_envs
        if lane_capacity < self.config.sequence_length:
            raise ValueError(
                "replay capacity divided across env lanes must cover one sequence"
            )
        self._storage = {
            name: np.empty(
                (lane_capacity, *np.asarray(getattr(batch, name)).shape[1:]),
                dtype=np.asarray(getattr(batch, name)).dtype,
            )
            for name in _ARRAY_FIELDS
        }
        self._metadata = ReplayMetadata(
            num_envs=batch.num_envs,
            num_agents=batch.num_agents,
            lane_capacity=lane_capacity,
            size_per_lane=0,
            cursor=0,
            agent_ids=batch.agent_ids,
        )

    def _validate_append(self, batch: MultiAgentSequenceBatch) -> None:
        assert self._metadata is not None
        if batch.num_envs != self._metadata.num_envs:
            raise ValueError("number of replay environment lanes cannot change")
        if batch.num_agents != self._metadata.num_agents:
            raise ValueError("number of replay agent slots cannot change")
        if batch.agent_ids != self._metadata.agent_ids:
            raise ValueError("ordered replay agent identities cannot change")
        for name, storage in self._storage.items():
            value = np.asarray(getattr(batch, name))
            if value.shape[1:] != storage.shape[1:] or value.dtype != storage.dtype:
                raise ValueError(f"{name} geometry or dtype changed across append")
        if self._metadata.size_per_lane:
            previous = (self._metadata.cursor - 1) % self._metadata.lane_capacity
            expected_first = self._storage["is_last"][previous]
            if np.any(batch.is_first[0] != expected_first):
                raise ValueError(
                    "cross-chunk lifecycle mismatch: is_first must follow is_last"
                )
