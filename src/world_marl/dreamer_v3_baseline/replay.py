from __future__ import annotations

import copy
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from .config import ReplayConfig
from .networks import TensorSpace


_RESERVED_KEYS = frozenset({"consec", "stepid"})
_REPLAY_MODES = ("train", "report", "eval")
_REQUIRED_TRANSITIONS = {
    "is_first": ((), "bool"),
    "is_last": ((), "bool"),
    "is_terminal": ((), "bool"),
    "reward": ((), "float32"),
}
_METRIC_KEYS = (
    "inserted_items",
    "inserted_rows",
    "online_samples",
    "sample_calls",
    "sampled_sequences",
    "stale_online",
    "stale_updates",
    "uniform_samples",
    "update_calls",
    "updated_rows",
)


class _ReplayRestored(RuntimeError):
    pass


def _require_exact_keys(
    mapping: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} keys differ; missing={missing}, extra={extra}")


def _coerce_value(value: Any, space: TensorSpace, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.dtype(space.dtype))
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} cannot be converted to {space.dtype}") from error
    if result.shape != space.shape:
        raise ValueError(f"{name} shape {result.shape} != {space.shape}")
    return result


def _validate_latent_value(
    value: Any,
    space: TensorSpace,
    name: str,
) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.dtype(space.dtype) or result.shape != space.shape:
        raise ValueError(f"{name} has wrong latent dtype or shape")
    return result


def _space_signature(spaces: Mapping[str, TensorSpace]) -> dict[str, dict[str, Any]]:
    return {
        name: {"dtype": space.dtype, "shape": list(space.shape)}
        for name, space in sorted(spaces.items())
    }


@dataclass(frozen=True, order=True)
class ReplayKey:
    chunk_id: bytes
    offset: int

    def __post_init__(self) -> None:
        if type(self.chunk_id) is not bytes or len(self.chunk_id) != 16:
            raise ValueError("chunk_id must be exactly 16 bytes")
        if type(self.offset) is not int or not 0 <= self.offset < 2**32:
            raise ValueError("offset must be a uint32 integer")

    def to_step_id(self) -> np.ndarray:
        result = np.empty((20,), np.uint8)
        result[:16] = np.frombuffer(self.chunk_id, np.uint8)
        result[16:] = np.frombuffer(self.offset.to_bytes(4, "big"), np.uint8)
        return result

    @classmethod
    def from_step_id(cls, value: np.ndarray) -> ReplayKey:
        array = np.asarray(value)
        if array.dtype != np.uint8 or array.shape != (20,):
            raise ValueError("step id must have dtype uint8 and shape [20]")
        return cls(array[:16].tobytes(), int.from_bytes(array[16:].tobytes(), "big"))

    def state_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "offset": self.offset}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> ReplayKey:
        if not isinstance(state, Mapping):
            raise TypeError("ReplayKey state must be a mapping")
        _require_exact_keys(state, {"chunk_id", "offset"}, "ReplayKey state")
        return cls(state["chunk_id"], state["offset"])


class ReplayBatch:
    def __init__(
        self,
        data: Mapping[str, np.ndarray],
        step_ids: np.ndarray,
    ) -> None:
        if not isinstance(data, Mapping) or not data:
            raise ValueError("ReplayBatch data must be a nonempty mapping")
        ids = np.asarray(step_ids)
        if ids.dtype != np.uint8 or ids.ndim != 3 or ids.shape[-1] != 20:
            raise ValueError("ReplayBatch step_ids must be uint8[B,S,20]")
        copied = {}
        for name, value in sorted(data.items()):
            array = np.asarray(value)
            if array.ndim < 2 or array.shape[:2] != ids.shape[:2]:
                raise ValueError(f"ReplayBatch leaf {name!r} has wrong leading axes")
            copied[name] = array.copy()
        self.data = MappingProxyType(copied)
        self.step_ids = ids.copy()

    def __getitem__(self, index: Any) -> ReplayBatch:
        data = {name: value[index].copy() for name, value in self.data.items()}
        return ReplayBatch(data, self.step_ids[index].copy())

    def as_dict(self) -> dict[str, np.ndarray]:
        result = {name: value.copy() for name, value in self.data.items()}
        result["stepid"] = self.step_ids.copy()
        return result


class ReplayChunk:
    def __init__(
        self,
        chunk_id: bytes,
        size: int,
        transition_spaces: Mapping[str, TensorSpace],
        latent_spaces: Mapping[str, TensorSpace],
        *,
        owner_id: int | None = None,
    ) -> None:
        ReplayKey(chunk_id, 0)
        if type(size) is not int or size <= 0:
            raise ValueError("chunk size must be positive")
        if owner_id is not None and type(owner_id) is not int:
            raise TypeError("chunk owner id must be an integer or None")
        self.chunk_id = chunk_id
        self.size = size
        self.owner_id = owner_id
        self.transition_spaces = dict(sorted(transition_spaces.items()))
        self.latent_spaces = dict(sorted(latent_spaces.items()))
        if set(self.transition_spaces) & set(self.latent_spaces):
            raise ValueError("transition and latent chunk spaces must be disjoint")
        self.length = 0
        self.sealed = False
        self.successor_id: bytes | None = None
        self.transition_data: dict[str, np.ndarray] = {}
        self.latent_data: dict[str, np.ndarray] = {}

    def _allocate(self) -> None:
        if self.transition_data or self.latent_data:
            return
        self.transition_data = {
            name: np.zeros((self.size, *space.shape), np.dtype(space.dtype))
            for name, space in self.transition_spaces.items()
        }
        self.latent_data = {
            name: np.zeros((self.size, *space.shape), np.dtype(space.dtype))
            for name, space in self.latent_spaces.items()
        }

    def append(self, row: Mapping[str, Any]) -> ReplayKey:
        if self.sealed:
            raise RuntimeError("cannot append to a sealed replay chunk")
        if self.length >= self.size:
            raise RuntimeError("cannot append to a full replay chunk")
        names = set(self.transition_spaces) | set(self.latent_spaces)
        _require_exact_keys(row, names, "replay row")
        converted = {
            name: _coerce_value(row[name], space, name)
            for name, space in self.transition_spaces.items()
        }
        converted.update(
            {
                name: _validate_latent_value(row[name], space, name)
                for name, space in self.latent_spaces.items()
            }
        )
        self._allocate()
        index = self.length
        for name in self.transition_spaces:
            self.transition_data[name][index] = converted[name]
        for name in self.latent_spaces:
            self.latent_data[name][index] = converted[name]
        self.length += 1
        return ReplayKey(self.chunk_id, index)

    def seal(self, successor_id: bytes) -> None:
        ReplayKey(successor_id, 0)
        if self.sealed:
            raise RuntimeError("replay chunk is already sealed")
        if successor_id == self.chunk_id:
            raise ValueError("replay chunk cannot link to itself")
        self.successor_id = successor_id
        self.sealed = True
        for value in self.transition_data.values():
            value.flags.writeable = False

    def read(self, offset: int, length: int) -> dict[str, np.ndarray]:
        if type(offset) is not int or type(length) is not int:
            raise TypeError("chunk read bounds must be integers")
        if offset < 0 or length < 0 or offset + length > self.length:
            raise IndexError("chunk read is outside the valid prefix")
        return {
            name: value[offset : offset + length].copy()
            for name, value in {**self.transition_data, **self.latent_data}.items()
        }

    def update_context(
        self,
        offset: int,
        values: Mapping[str, np.ndarray],
    ) -> None:
        if type(offset) is not int or not 0 <= offset < self.length:
            raise IndexError("latent update offset is outside the valid prefix")
        _require_exact_keys(values, set(self.latent_spaces), "latent update")
        converted = {}
        for name, space in self.latent_spaces.items():
            value = np.asarray(values[name])
            if value.dtype != np.dtype(space.dtype) or value.shape != space.shape:
                raise ValueError(f"latent update {name!r} has wrong dtype or shape")
            converted[name] = value
        for name, value in converted.items():
            self.latent_data[name][offset] = value

    def state_dict(self) -> dict[str, Any]:
        transition = {
            name: (
                self.transition_data[name][: self.length].copy()
                if self.transition_data
                else np.zeros((0, *space.shape), np.dtype(space.dtype))
            )
            for name, space in self.transition_spaces.items()
        }
        latent = {
            name: (
                self.latent_data[name][: self.length].copy()
                if self.latent_data
                else np.zeros((0, *space.shape), np.dtype(space.dtype))
            )
            for name, space in self.latent_spaces.items()
        }
        return {
            "chunk_id": self.chunk_id,
            "latent": latent,
            "length": self.length,
            "owner_id": self.owner_id,
            "sealed": self.sealed,
            "size": self.size,
            "successor": self.successor_id,
            "transition": transition,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        transition_spaces: Mapping[str, TensorSpace],
        latent_spaces: Mapping[str, TensorSpace],
    ) -> ReplayChunk:
        expected = {
            "chunk_id",
            "latent",
            "length",
            "owner_id",
            "sealed",
            "size",
            "successor",
            "transition",
        }
        _require_exact_keys(state, expected, "replay chunk state")
        chunk = cls(
            state["chunk_id"],
            state["size"],
            transition_spaces,
            latent_spaces,
            owner_id=state["owner_id"],
        )
        length = state["length"]
        if type(length) is not int or not 0 <= length <= chunk.size:
            raise ValueError("invalid replay chunk length")
        _require_exact_keys(
            state["transition"], set(transition_spaces), "transition state"
        )
        _require_exact_keys(state["latent"], set(latent_spaces), "latent state")
        for namespace, spaces, target in (
            (state["transition"], transition_spaces, chunk.transition_data),
            (state["latent"], latent_spaces, chunk.latent_data),
        ):
            if length and not target:
                chunk._allocate()
                target = (
                    chunk.transition_data
                    if spaces is transition_spaces
                    else chunk.latent_data
                )
            for name, space in spaces.items():
                value = np.asarray(namespace[name])
                expected_shape = (length, *space.shape)
                if (
                    value.dtype != np.dtype(space.dtype)
                    or value.shape != expected_shape
                ):
                    raise ValueError(f"invalid replay chunk tensor {name!r}")
                if length:
                    target[name][:length] = value
        chunk.length = length
        sealed = state["sealed"]
        if type(sealed) is not bool:
            raise ValueError("invalid sealed flag")
        successor = state["successor"]
        if sealed:
            if successor is None or length != chunk.size:
                raise ValueError("sealed chunk must be full and have a successor")
            chunk.seal(successor)
        elif successor is not None or length == chunk.size:
            raise ValueError("open chunk must be nonfull without a successor")
        return chunk


class OnlineQueue:
    def __init__(self) -> None:
        self.keys: list[ReplayKey] = []

    def __len__(self) -> int:
        return len(self.keys)

    def push(self, key: ReplayKey) -> None:
        if not isinstance(key, ReplayKey):
            raise TypeError("online queue accepts ReplayKey values")
        self.keys.append(key)

    def pop(self) -> ReplayKey:
        if not self.keys:
            raise IndexError("online replay queue is empty")
        return self.keys.pop(0)

    def state_dict(self) -> dict[str, Any]:
        return {"keys": [key.state_dict() for key in self.keys]}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> OnlineQueue:
        _require_exact_keys(state, {"keys"}, "online queue state")
        result = cls()
        if not isinstance(state["keys"], list):
            raise ValueError("online queue keys must be a list")
        result.keys = [ReplayKey.from_state_dict(value) for value in state["keys"]]
        return result


class UniformSelector:
    def __init__(self, seed: int = 0) -> None:
        if type(seed) is not int:
            raise TypeError("selector seed must be an integer")
        self.indices: dict[int, int] = {}
        self.keys: list[int] = []
        self.rng = np.random.default_rng(seed)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self.keys)

    def insert(self, item_id: int) -> None:
        with self._lock:
            if type(item_id) is not int or item_id in self.indices:
                raise ValueError("selector item id must be a unique integer")
            self.indices[item_id] = len(self.keys)
            self.keys.append(item_id)

    def delete(self, item_id: int) -> None:
        with self._lock:
            if type(item_id) is not int:
                raise TypeError("selector item id must be an integer")
            if item_id not in self.indices:
                raise KeyError(item_id)
            index = self.indices.pop(item_id)
            final = self.keys.pop()
            if index < len(self.keys):
                self.keys[index] = final
                self.indices[final] = index

    def sample(self) -> int:
        with self._lock:
            if not self.keys:
                raise IndexError("uniform selector is empty")
            index = self.rng.integers(0, len(self.keys)).item()
            return self.keys[index]

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "bit_generator": type(self.rng.bit_generator).__name__,
                "indices": dict(self.indices),
                "keys": list(self.keys),
                "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> UniformSelector:
        _require_exact_keys(
            state,
            {"bit_generator", "indices", "keys", "rng_state"},
            "selector state",
        )
        if state["bit_generator"] != "PCG64":
            raise ValueError("replay selector requires PCG64")
        keys = state["keys"]
        indices = state["indices"]
        if (
            not isinstance(keys, list)
            or any(type(key) is not int for key in keys)
            or len(set(keys)) != len(keys)
            or not isinstance(indices, Mapping)
            or any(
                type(key) is not int or type(index) is not int
                for key, index in indices.items()
            )
            or indices != {key: index for index, key in enumerate(keys)}
        ):
            raise ValueError("invalid selector key/index state")
        result = cls()
        result.keys = list(keys)
        result.indices = dict(indices)
        rng_state = copy.deepcopy(state["rng_state"])
        if rng_state.get("bit_generator") != "PCG64":
            raise ValueError("invalid selector RNG state")
        try:
            result.rng.bit_generator.state = rng_state
        except (TypeError, ValueError) as error:
            raise ValueError("invalid selector RNG state") from error
        return result


class ConsecutiveStream:
    def __init__(
        self,
        source: Callable[[], ReplayBatch],
        *,
        sequence_length: int,
        consecutive: int,
        context: int,
    ) -> None:
        if any(
            type(value) is not int for value in (sequence_length, consecutive, context)
        ):
            raise TypeError("consecutive replay dimensions must be integers")
        if sequence_length <= 0 or consecutive <= 0 or context < 0:
            raise ValueError("invalid consecutive replay dimensions")
        self.source = source
        self.sequence_length = sequence_length
        self.consecutive = consecutive
        self.context = context
        self.raw_length = context + sequence_length * consecutive
        self.index = 0
        self.current: ReplayBatch | None = None

    def __iter__(self) -> ConsecutiveStream:
        return self

    def __next__(self) -> ReplayBatch:
        if self.index == self.consecutive:
            self.index = 0
            self.current = None
        if self.index == 0:
            current = self.source()
            if current.step_ids.shape[1] != self.raw_length:
                raise ValueError("raw replay batch length changed")
            self.current = current
        if self.current is None:
            raise RuntimeError("consecutive replay current batch is missing")
        start = self.index * self.sequence_length
        stop = start + self.sequence_length + self.context
        sliced = self.current[:, start:stop]
        data = dict(sliced.data)
        data["consec"] = np.full(sliced.step_ids.shape[:2], self.index, np.int32)
        result = ReplayBatch(data, sliced.step_ids)
        self.index += 1
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "current": None if self.current is None else self.current.as_dict(),
            "index": self.index,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        source: Callable[[], ReplayBatch],
        *,
        sequence_length: int,
        consecutive: int,
        context: int,
    ) -> ConsecutiveStream:
        _require_exact_keys(state, {"current", "index"}, "consecutive state")
        result = cls(
            source,
            sequence_length=sequence_length,
            consecutive=consecutive,
            context=context,
        )
        index = state["index"]
        if type(index) is not int or not 0 <= index <= consecutive:
            raise ValueError("invalid consecutive stream index")
        current = state["current"]
        if index > 0 and current is None:
            raise ValueError("active consecutive stream requires current batch")
        if index == 0 and current is not None:
            raise ValueError("inactive consecutive stream cannot retain a batch")
        if current is not None:
            if not isinstance(current, Mapping) or "stepid" not in current:
                raise ValueError("invalid consecutive current batch")
            data = {name: value for name, value in current.items() if name != "stepid"}
            result.current = ReplayBatch(data, current["stepid"])
            if result.current.step_ids.shape[1] != result.raw_length:
                raise ValueError("invalid consecutive current length")
        result.index = index
        return result


class ReplayWriter:
    def __init__(self, worker_id: int, replay: DreamerReplay) -> None:
        if type(worker_id) is not int:
            raise TypeError("worker id must be an integer")
        self.worker_id = worker_id
        self.replay = replay
        self.current_chunk_id: bytes | None = None
        self.pending: deque[ReplayKey] = deque()
        self.row_count = 0
        self.emitted_count = 0
        self.has_rows = False
        self.last_is_last = False
        self.chunk_history: list[bytes] = []

    @property
    def current_offset(self) -> int:
        if self.current_chunk_id is None:
            return 0
        return self.replay.chunks[self.current_chunk_id].length

    def add(self, row: Mapping[str, Any]) -> ReplayKey:
        with self.replay._lock:
            if self.replay.writers.get(self.worker_id) is not self:
                raise RuntimeError("replay writer is not active")
            return self.replay._add_for_writer(self, row)

    def state_dict(self) -> dict[str, Any]:
        return {
            "chunk_history": list(self.chunk_history),
            "current_chunk_id": self.current_chunk_id,
            "current_offset": self.current_offset,
            "emitted_count": self.emitted_count,
            "has_rows": self.has_rows,
            "last_is_last": self.last_is_last,
            "pending": [key.state_dict() for key in self.pending],
            "row_count": self.row_count,
            "worker_id": self.worker_id,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        replay: DreamerReplay,
    ) -> ReplayWriter:
        expected = {
            "chunk_history",
            "current_chunk_id",
            "current_offset",
            "emitted_count",
            "has_rows",
            "last_is_last",
            "pending",
            "row_count",
            "worker_id",
        }
        _require_exact_keys(state, expected, "writer state")
        result = cls(state["worker_id"], replay)
        history = state["chunk_history"]
        if not isinstance(history, list):
            raise ValueError("writer chunk history must be a list")
        history_numbers = []
        for chunk_id in history:
            if type(chunk_id) is not bytes or len(chunk_id) != 16:
                raise ValueError("invalid writer chunk history id")
            number = int.from_bytes(chunk_id, "big")
            if number <= 0:
                raise ValueError("invalid writer chunk history id")
            history_numbers.append(number)
        if any(
            current >= following
            for current, following in zip(
                history_numbers, history_numbers[1:], strict=False
            )
        ):
            raise ValueError("writer chunk history must be strictly increasing")
        result.chunk_history = list(history)
        result.current_chunk_id = state["current_chunk_id"]
        result.pending = deque(
            ReplayKey.from_state_dict(value) for value in state["pending"]
        )
        if type(state["current_offset"]) is not int or state["current_offset"] < 0:
            raise ValueError("invalid writer current offset")
        if type(state["row_count"]) is not int or state["row_count"] < 0:
            raise ValueError("invalid writer row count")
        if type(state["emitted_count"]) is not int or state["emitted_count"] < 0:
            raise ValueError("invalid writer emitted count")
        if (
            type(state["has_rows"]) is not bool
            or type(state["last_is_last"]) is not bool
        ):
            raise ValueError("invalid writer chronology state")
        result.row_count = state["row_count"]
        result.emitted_count = state["emitted_count"]
        result.has_rows = state["has_rows"]
        result.last_is_last = state["last_is_last"]
        result._restored_offset = state["current_offset"]
        return result


class DreamerReplay:
    SCHEMA_VERSION = 2

    def __init__(
        self,
        config: ReplayConfig,
        transition_spaces: Mapping[str, TensorSpace],
        latent_spaces: Mapping[str, TensorSpace],
        *,
        batch_size: int,
        consecutive: int = 1,
        seed: int = 0,
    ) -> None:
        if (
            config.uniform_fraction != 1.0
            or config.priority_fraction != 0.0
            or config.recency_fraction != 0.0
        ):
            raise ValueError("DreamerV3 replay only supports the uniform selector")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("replay batch size must be positive")
        if type(consecutive) is not int or consecutive <= 0:
            raise ValueError("replay consecutive count must be positive")
        self.config = config
        self.transition_spaces = dict(sorted(transition_spaces.items()))
        self.latent_spaces = dict(sorted(latent_spaces.items()))
        self._validate_spaces()
        self.batch_size = batch_size
        self.consecutive = consecutive
        self.raw_length = config.context + config.sequence_length * consecutive
        self.seed = seed
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._sample_locks = {mode: threading.Lock() for mode in _REPLAY_MODES}
        self._restore_epoch = 0
        self.chunks: dict[bytes, ReplayChunk] = {}
        self.refs: dict[bytes, int] = {}
        self.writers: dict[int, ReplayWriter] = {}
        self.items: dict[int, ReplayKey] = {}
        self.fifo: list[int] = []
        self.next_item_id = 0
        self.next_chunk_id = 1
        self.online_queue = OnlineQueue()
        self.selector = UniformSelector(seed)
        self._stream_timeouts: dict[str, float | None] = {
            mode: None for mode in _REPLAY_MODES
        }
        self.consecutive_streams = {
            mode: self._make_stream(mode) for mode in _REPLAY_MODES
        }
        self.consecutive_stream = self.consecutive_streams["train"]
        self._metrics = {name: 0 for name in _METRIC_KEYS}

    def _validate_spaces(self) -> None:
        if not self.transition_spaces:
            raise ValueError("replay transition spaces cannot be empty")
        overlap = set(self.transition_spaces) & set(self.latent_spaces)
        reserved = _RESERVED_KEYS & (
            set(self.transition_spaces) | set(self.latent_spaces)
        )
        if overlap or reserved:
            raise ValueError("replay schemas overlap or use reserved keys")
        for name, (shape, dtype) in _REQUIRED_TRANSITIONS.items():
            space = self.transition_spaces.get(name)
            if space is None or space.shape != shape or space.dtype != dtype:
                raise ValueError(f"invalid required replay space {name!r}")
        for name, space in self.latent_spaces.items():
            if space.dtype != "float32":
                raise ValueError(f"latent replay space {name!r} must be float32")

    def _make_stream(
        self,
        mode: str,
        state: Mapping[str, Any] | None = None,
    ) -> ConsecutiveStream:
        if mode not in _REPLAY_MODES:
            raise ValueError("replay mode must be train, report, or eval")

        def source() -> ReplayBatch:
            return self._sample_raw(
                mode,
                timeout=self._stream_timeouts[mode],
                expected_epoch=self._restore_epoch,
            )

        kwargs = {
            "sequence_length": self.config.sequence_length,
            "consecutive": self.consecutive,
            "context": self.config.context,
        }
        if state is None:
            return ConsecutiveStream(source, **kwargs)
        self._validate_consecutive_state(state)
        return ConsecutiveStream.from_state_dict(state, source, **kwargs)

    def _validate_consecutive_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("consecutive state must be a mapping")
        _require_exact_keys(state, {"current", "index"}, "consecutive state")
        current = state["current"]
        if current is None:
            return
        if not isinstance(current, Mapping):
            raise ValueError("invalid consecutive current batch")
        spaces = {**self.transition_spaces, **self.latent_spaces}
        _require_exact_keys(current, set(spaces) | {"stepid"}, "consecutive current")
        step_ids = np.asarray(current["stepid"])
        if step_ids.dtype != np.uint8 or step_ids.shape != (
            self.batch_size,
            self.raw_length,
            20,
        ):
            raise ValueError("invalid consecutive current step ids")
        for name, space in spaces.items():
            value = np.asarray(current[name])
            expected = (self.batch_size, self.raw_length, *space.shape)
            if value.dtype != np.dtype(space.dtype) or value.shape != expected:
                raise ValueError(f"invalid consecutive current tensor {name!r}")
        self._validate_consecutive_current(current)

    def _validate_consecutive_current(self, current: Mapping[str, Any]) -> None:
        is_first = np.asarray(current["is_first"])
        is_last = np.asarray(current["is_last"])
        is_terminal = np.asarray(current["is_terminal"])
        if not np.all(is_first[:, 0]):
            raise ValueError("consecutive current must begin with is_first")
        if np.any(is_terminal & ~is_last):
            raise ValueError("consecutive current terminal rows must be is_last")
        if not np.array_equal(is_last[:, :-1], is_first[:, 1:]):
            raise ValueError("invalid consecutive current episode boundaries")
        locations = self._history_locations()
        ids = np.asarray(current["stepid"])
        for batch_index in range(self.batch_size):
            keys = [ReplayKey.from_step_id(value) for value in ids[batch_index]]
            logical_rows: list[tuple[int, int]] = []
            for key in keys:
                location = locations.get(key.chunk_id)
                if location is None or key.offset >= self.config.chunk_size:
                    raise ValueError("consecutive current step id was never allocated")
                writer, ordinal = location
                logical_length = (
                    self.config.chunk_size
                    if ordinal < len(writer.chunk_history) - 1
                    else writer.row_count % self.config.chunk_size
                )
                if key.offset >= logical_length:
                    raise ValueError("consecutive current step id exceeds writer rows")
                logical_rows.append(
                    (writer.worker_id, ordinal * self.config.chunk_size + key.offset)
                )
            for previous, following in zip(
                logical_rows, logical_rows[1:], strict=False
            ):
                if previous[0] != following[0] or previous[1] + 1 != following[1]:
                    raise ValueError("consecutive current step ids are not consecutive")
            for position, key in enumerate(keys):
                if key.chunk_id not in self.chunks:
                    continue
                try:
                    resolved = self._resolve(key, self.raw_length - position)
                except KeyError as error:
                    raise ValueError(
                        "live consecutive current suffix is not resolvable"
                    ) from error
                if resolved != keys[position:]:
                    raise ValueError(
                        "live consecutive current suffix has invalid chronology"
                    )

    def __len__(self) -> int:
        with self._lock:
            return len(self.items)

    def _new_chunk(self, references: int, owner_id: int) -> ReplayChunk:
        writer = self.writers.get(owner_id)
        if writer is None or writer.worker_id != owner_id or writer.replay is not self:
            raise RuntimeError("cannot allocate a chunk without an active writer")
        self._preflight_chunk_ids(1)
        chunk_id = self.next_chunk_id.to_bytes(16, "big")
        self.next_chunk_id += 1
        chunk = ReplayChunk(
            chunk_id,
            self.config.chunk_size,
            self.transition_spaces,
            self.latent_spaces,
            owner_id=owner_id,
        )
        self.chunks[chunk_id] = chunk
        self.refs[chunk_id] = references
        writer.chunk_history.append(chunk_id)
        return chunk

    def _preflight_chunk_ids(self, count: int) -> None:
        if type(count) is not int or count <= 0:
            raise ValueError("chunk id reservation must be positive")
        if not 0 < self.next_chunk_id or self.next_chunk_id + count > 2**128:
            raise OverflowError("replay chunk id counter exhausted")

    def _inc_ref(self, chunk_id: bytes) -> None:
        if chunk_id not in self.refs:
            raise KeyError(chunk_id)
        self.refs[chunk_id] += 1

    def _dec_ref(self, chunk_id: bytes) -> None:
        if chunk_id not in self.refs or self.refs[chunk_id] <= 0:
            raise RuntimeError("invalid replay reference decrement")
        self.refs[chunk_id] -= 1
        if self.refs[chunk_id]:
            return
        chunk = self.chunks.pop(chunk_id)
        del self.refs[chunk_id]
        if chunk.successor_id is not None:
            self._dec_ref(chunk.successor_id)

    def _normalize_row(self, row: Mapping[str, Any]) -> dict[str, np.ndarray]:
        if not isinstance(row, Mapping):
            raise TypeError("replay row must be a mapping")
        names = set(self.transition_spaces) | set(self.latent_spaces)
        _require_exact_keys(row, names, "replay row")
        result = {
            name: _coerce_value(row[name], space, name)
            for name, space in self.transition_spaces.items()
        }
        result.update(
            {
                name: _validate_latent_value(row[name], space, name)
                for name, space in self.latent_spaces.items()
            }
        )
        return result

    def add(self, row: Mapping[str, Any], *, worker: int = 0) -> ReplayKey:
        if type(worker) is not int:
            raise TypeError("worker id must be an integer")
        with self._lock:
            writer = self.writers.get(worker)
            created = writer is None
            if writer is None:
                writer = ReplayWriter(worker, self)
                self.writers[worker] = writer
            try:
                return self._add_for_writer(writer, row)
            except Exception:
                if created and not writer.has_rows:
                    self.writers.pop(worker, None)
                raise

    def _add_for_writer(
        self,
        writer: ReplayWriter,
        row: Mapping[str, Any],
    ) -> ReplayKey:
        with self._lock:
            normalized = self._normalize_row(row)
            is_first = bool(normalized["is_first"])
            is_last = bool(normalized["is_last"])
            is_terminal = bool(normalized["is_terminal"])
            if not writer.has_rows and not is_first:
                raise ValueError("a replay worker's first row must be is_first")
            if is_terminal and not is_last:
                raise ValueError("is_terminal requires is_last")
            if writer.has_rows and writer.last_is_last and not is_first:
                raise ValueError("a row after is_last must be is_first")
            if writer.current_chunk_id is None:
                required_ids = 2 if self.config.chunk_size == 1 else 1
                self._preflight_chunk_ids(required_ids)
                writer.current_chunk_id = self._new_chunk(1, writer.worker_id).chunk_id
            chunk = self.chunks[writer.current_chunk_id]
            if chunk.length + 1 == chunk.size:
                self._preflight_chunk_ids(1)
            key = chunk.append(normalized)
            writer.pending.append(key)
            self._inc_ref(key.chunk_id)
            if chunk.length == chunk.size:
                successor = self._new_chunk(2, writer.worker_id)
                chunk.seal(successor.chunk_id)
                self._dec_ref(chunk.chunk_id)
                writer.current_chunk_id = successor.chunk_id
            emitted = None
            if len(writer.pending) >= self.raw_length:
                emitted = writer.pending.popleft()
                writer.emitted_count += 1
                self._insert_item(emitted)
            if (
                emitted is not None
                and self.config.online
                and writer.row_count % self.raw_length == 0
            ):
                self.online_queue.push(emitted)
            writer.row_count += 1
            writer.has_rows = True
            writer.last_is_last = is_last
            self._metrics["inserted_rows"] += 1
            return key

    def _insert_item(self, key: ReplayKey) -> None:
        while len(self.items) >= self.config.capacity:
            self._evict_item()
        item_id = self.next_item_id
        self.next_item_id += 1
        self.items[item_id] = key
        self.fifo.append(item_id)
        self.selector.insert(item_id)
        self._metrics["inserted_items"] += 1
        self._condition.notify_all()

    def _evict_item(self) -> None:
        if not self.fifo:
            raise RuntimeError("replay FIFO is empty during capacity eviction")
        item_id = self.fifo.pop(0)
        key = self.items.pop(item_id)
        self.selector.delete(item_id)
        self._dec_ref(key.chunk_id)

    def _resolve(self, start: ReplayKey, length: int) -> list[ReplayKey]:
        if start.chunk_id not in self.chunks:
            raise KeyError(start.chunk_id)
        result = []
        chunk_id = start.chunk_id
        offset = start.offset
        while len(result) < length:
            chunk = self.chunks.get(chunk_id)
            if chunk is None or not 0 <= offset < chunk.length:
                raise KeyError((chunk_id, offset))
            while offset < chunk.length and len(result) < length:
                result.append(ReplayKey(chunk_id, offset))
                offset += 1
            if len(result) < length:
                if chunk.successor_id is None:
                    raise KeyError("replay sequence has no linked successor")
                chunk_id = chunk.successor_id
                offset = 0
        return result

    def _read_keys(self, keys: list[ReplayKey]) -> dict[str, np.ndarray]:
        rows: dict[str, list[np.ndarray]] = {}
        for key in keys:
            values = self.chunks[key.chunk_id].read(key.offset, 1)
            for name, value in values.items():
                rows.setdefault(name, []).append(value[0])
        return {name: np.stack(values) for name, values in rows.items()}

    def sample_raw(
        self,
        mode: str = "train",
        *,
        timeout: float | None = None,
    ) -> ReplayBatch:
        return self._sample_raw(mode, timeout=timeout, expected_epoch=None)

    def _sample_raw(
        self,
        mode: str,
        *,
        timeout: float | None,
        expected_epoch: int | None,
    ) -> ReplayBatch:
        if mode not in _REPLAY_MODES:
            raise ValueError("replay mode must be train, report, or eval")
        if timeout is not None and timeout < 0:
            raise ValueError("replay timeout cannot be negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self.items:
                if expected_epoch is not None and expected_epoch != self._restore_epoch:
                    raise _ReplayRestored
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out waiting for replay data")
                self._condition.wait(remaining)
            if expected_epoch is not None and expected_epoch != self._restore_epoch:
                raise _ReplayRestored
            sequences = []
            sequence_ids = []
            for _ in range(self.batch_size):
                keys = None
                if mode == "train":
                    while self.online_queue.keys:
                        candidate = self.online_queue.pop()
                        try:
                            keys = self._resolve(candidate, self.raw_length)
                        except KeyError:
                            self._metrics["stale_online"] += 1
                            continue
                        self._metrics["online_samples"] += 1
                        break
                if keys is None:
                    item_id = self.selector.sample()
                    keys = self._resolve(self.items[item_id], self.raw_length)
                    if mode == "train":
                        self._metrics["uniform_samples"] += 1
                sequences.append(self._read_keys(keys))
                sequence_ids.append(np.stack([key.to_step_id() for key in keys]))
            names = set(sequences[0])
            if any(set(sequence) != names for sequence in sequences):
                raise RuntimeError("sampled replay schemas differ")
            data = {
                name: np.stack([sequence[name] for sequence in sequences])
                for name in sorted(names)
            }
            step_ids = np.stack(sequence_ids).astype(np.uint8, copy=False)
            data["is_first"] = data["is_first"].copy()
            data["is_last"] = data["is_last"].copy()
            data["is_first"][:, 0] = True
            next_first = np.zeros_like(data["is_first"])
            next_first[:, :-1] = data["is_first"][:, 1:]
            data["is_last"] |= next_first
            if mode == "train":
                self._metrics["sample_calls"] += 1
                self._metrics["sampled_sequences"] += self.batch_size
            return ReplayBatch(data, step_ids)

    def sample(
        self,
        mode: str = "train",
        *,
        timeout: float | None = None,
    ) -> ReplayBatch:
        if mode not in _REPLAY_MODES:
            raise ValueError("replay mode must be train, report, or eval")
        if timeout is not None and timeout < 0:
            raise ValueError("replay timeout cannot be negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        sample_lock = self._sample_locks[mode]
        if timeout is None:
            sample_lock.acquire()
            acquired = True
        else:
            acquired = sample_lock.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError("timed out waiting for consecutive replay stream")
        try:
            while True:
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                with self._lock:
                    self._stream_timeouts[mode] = remaining
                    try:
                        return next(self.consecutive_streams[mode])
                    except _ReplayRestored:
                        continue
        finally:
            sample_lock.release()

    def update_context(
        self,
        step_ids: np.ndarray,
        values: Mapping[str, np.ndarray],
    ) -> int:
        ids = np.asarray(step_ids)
        if ids.dtype != np.uint8 or ids.ndim != 3 or ids.shape[-1] != 20:
            raise ValueError("latent update step ids must be uint8[B,U,20]")
        _require_exact_keys(values, set(self.latent_spaces), "latent update")
        batch, length = ids.shape[:2]
        converted = {}
        for name, space in self.latent_spaces.items():
            value = np.asarray(values[name])
            expected = (batch, length, *space.shape)
            if value.dtype != np.dtype(space.dtype) or value.shape != expected:
                raise ValueError(f"latent update {name!r} has wrong dtype or shape")
            converted[name] = value
        with self._lock:
            plans: list[tuple[list[ReplayKey], int]] = []
            stale = 0
            for batch_index in range(batch):
                first = ReplayKey.from_step_id(ids[batch_index, 0])
                if first.chunk_id not in self.chunks:
                    stale += 1
                    continue
                try:
                    keys = self._resolve(first, length)
                except KeyError as error:
                    raise ValueError(
                        "latent update sequence is not resolvable"
                    ) from error
                supplied = [ReplayKey.from_step_id(value) for value in ids[batch_index]]
                if supplied != keys:
                    raise ValueError(
                        "latent update step ids do not match replay chronology"
                    )
                plans.append((keys, batch_index))
            for keys, batch_index in plans:
                for time_index, key in enumerate(keys):
                    self.chunks[key.chunk_id].update_context(
                        key.offset,
                        {
                            name: value[batch_index, time_index]
                            for name, value in converted.items()
                        },
                    )
            updated = sum(len(keys) for keys, _ in plans)
            self._metrics["update_calls"] += 1
            self._metrics["updated_rows"] += updated
            self._metrics["stale_updates"] += stale
            return updated

    def stats(self, *, reset: bool = False) -> dict[str, float | int]:
        with self._lock:
            result: dict[str, float | int] = dict(self._metrics)
            sampled = self._metrics["online_samples"] + self._metrics["uniform_samples"]
            result.update(
                {
                    "chunks": len(self.chunks),
                    "online_fraction": (
                        self._metrics["online_samples"] / sampled if sampled else 0.0
                    ),
                    "online_queue": len(self.online_queue),
                    "replay_ratio": (
                        self._metrics["sampled_sequences"]
                        * self.config.sequence_length
                        / self._metrics["inserted_rows"]
                        if self._metrics["inserted_rows"]
                        else 0.0
                    ),
                    "size": len(self.items),
                }
            )
            if reset:
                self._metrics = {name: 0 for name in _METRIC_KEYS}
            return result

    def _dimensions(self) -> dict[str, int]:
        return {
            "batch_size": self.batch_size,
            "consecutive": self.consecutive,
            "context": self.config.context,
            "raw_length": self.raw_length,
            "sequence_length": self.config.sequence_length,
        }

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "chunks": [
                    self.chunks[key].state_dict() for key in sorted(self.chunks)
                ],
                "config": asdict(self.config),
                "consecutive": {
                    mode: self.consecutive_streams[mode].state_dict()
                    for mode in _REPLAY_MODES
                },
                "dimensions": self._dimensions(),
                "fifo": list(self.fifo),
                "items": [
                    {"item_id": item_id, "key": key.state_dict()}
                    for item_id, key in self.items.items()
                ],
                "metrics": dict(self._metrics),
                "next_chunk_id": self.next_chunk_id,
                "next_item_id": self.next_item_id,
                "online_queue": self.online_queue.state_dict(),
                "refs": dict(self.refs),
                "schema_version": self.SCHEMA_VERSION,
                "selector": self.selector.state_dict(),
                "spaces": {
                    "latent": _space_signature(self.latent_spaces),
                    "transition": _space_signature(self.transition_spaces),
                },
                "writers": {
                    worker: writer.state_dict()
                    for worker, writer in sorted(self.writers.items())
                },
            }

    def _expected_state_keys(self) -> set[str]:
        return {
            "chunks",
            "config",
            "consecutive",
            "dimensions",
            "fifo",
            "items",
            "metrics",
            "next_chunk_id",
            "next_item_id",
            "online_queue",
            "refs",
            "schema_version",
            "selector",
            "spaces",
            "writers",
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("replay state must be a mapping")
        with self._lock:
            _require_exact_keys(state, self._expected_state_keys(), "replay state")
            if state["schema_version"] != self.SCHEMA_VERSION:
                raise ValueError("replay state schema version changed")
            if state["config"] != asdict(self.config):
                raise ValueError("replay state config mismatch")
            if state["dimensions"] != self._dimensions():
                raise ValueError("replay state dimensions mismatch")
            spaces = {
                "latent": _space_signature(self.latent_spaces),
                "transition": _space_signature(self.transition_spaces),
            }
            if state["spaces"] != spaces:
                raise ValueError("replay state space signature mismatch")
            candidate = DreamerReplay(
                self.config,
                self.transition_spaces,
                self.latent_spaces,
                batch_size=self.batch_size,
                consecutive=self.consecutive,
                seed=self.seed,
            )
            candidate.chunks = {}
            for chunk_state in state["chunks"]:
                chunk = ReplayChunk.from_state_dict(
                    chunk_state, self.transition_spaces, self.latent_spaces
                )
                if chunk.chunk_id in candidate.chunks:
                    raise ValueError("duplicate replay chunk id")
                candidate.chunks[chunk.chunk_id] = chunk
            for chunk in candidate.chunks.values():
                if (
                    chunk.successor_id is not None
                    and chunk.successor_id not in candidate.chunks
                ):
                    raise ValueError("replay chunk successor is missing")
            candidate._validate_acyclic_links()
            refs = state["refs"]
            if set(refs) != set(candidate.chunks) or any(
                type(value) is not int or value <= 0 for value in refs.values()
            ):
                raise ValueError("invalid replay refcount state")
            candidate.refs = dict(refs)
            next_chunk_id = state["next_chunk_id"]
            if type(next_chunk_id) is not int or not 1 <= next_chunk_id <= 2**128:
                raise ValueError("invalid next replay chunk id")
            candidate.next_chunk_id = next_chunk_id
            candidate.writers = {}
            writer_states = state["writers"]
            if not isinstance(writer_states, Mapping):
                raise TypeError("replay writers state must be a mapping")
            for worker_key, writer_state in writer_states.items():
                if type(worker_key) is not int:
                    raise ValueError("invalid replay writer id")
                writer = ReplayWriter.from_state_dict(writer_state, candidate)
                if (
                    type(writer.worker_id) is not int
                    or worker_key != writer.worker_id
                    or worker_key in candidate.writers
                ):
                    raise ValueError("invalid replay writer id")
                candidate.writers[worker_key] = writer
            candidate._validate_chunk_histories()
            candidate.items = {}
            item_keys: set[ReplayKey] = set()
            for item_state in state["items"]:
                _require_exact_keys(item_state, {"item_id", "key"}, "item state")
                item_id = item_state["item_id"]
                key = ReplayKey.from_state_dict(item_state["key"])
                if (
                    type(item_id) is not int
                    or item_id < 0
                    or item_id in candidate.items
                    or key in item_keys
                ):
                    raise ValueError("duplicate or invalid replay item id or key")
                candidate.items[item_id] = key
                item_keys.add(key)
            candidate.fifo = list(state["fifo"])
            if candidate.fifo != list(candidate.items):
                raise ValueError("replay FIFO/items disagreement")
            candidate.next_item_id = state["next_item_id"]
            expected_next_item_id = max(candidate.items) + 1 if candidate.items else 0
            if (
                type(candidate.next_item_id) is not int
                or candidate.next_item_id != expected_next_item_id
            ):
                raise ValueError("invalid next replay item id")
            expected_item_ids = list(
                range(
                    candidate.next_item_id - len(candidate.items),
                    candidate.next_item_id,
                )
            )
            if list(candidate.items) != expected_item_ids:
                raise ValueError("replay item ids are not a contiguous retained suffix")
            candidate.selector = UniformSelector.from_state_dict(state["selector"])
            if set(candidate.selector.keys) != set(candidate.items):
                raise ValueError("selector/items disagreement")
            candidate.online_queue = OnlineQueue.from_state_dict(state["online_queue"])
            candidate._validate_online_queue()
            metrics = state["metrics"]
            _require_exact_keys(metrics, set(_METRIC_KEYS), "replay metrics")
            if any(type(value) is not int or value < 0 for value in metrics.values()):
                raise ValueError("invalid replay metric state")
            if (
                metrics["sampled_sequences"]
                != candidate.batch_size * metrics["sample_calls"]
                or metrics["sampled_sequences"]
                != metrics["online_samples"] + metrics["uniform_samples"]
            ):
                raise ValueError("invalid replay sample metric identities")
            candidate._metrics = dict(metrics)
            candidate._validate_restored_state()
            for writer in candidate.writers.values():
                del writer._restored_offset
            consecutive_state = state["consecutive"]
            if not isinstance(consecutive_state, Mapping):
                raise TypeError("replay consecutive state must be a mapping")
            _require_exact_keys(
                consecutive_state,
                set(_REPLAY_MODES),
                "replay consecutive state",
            )
            streams = {
                mode: candidate._make_stream(mode, consecutive_state[mode])
                for mode in _REPLAY_MODES
            }
            self.chunks = candidate.chunks
            self.refs = candidate.refs
            self.writers = candidate.writers
            for writer in self.writers.values():
                writer.replay = self
            self.items = candidate.items
            self.fifo = candidate.fifo
            self.next_item_id = candidate.next_item_id
            self.next_chunk_id = candidate.next_chunk_id
            self.online_queue = candidate.online_queue
            self.selector = candidate.selector
            self._metrics = candidate._metrics
            self._stream_timeouts = {mode: None for mode in _REPLAY_MODES}
            self.consecutive_streams = {
                mode: self._make_stream(mode, streams[mode].state_dict())
                for mode in _REPLAY_MODES
            }
            self.consecutive_stream = self.consecutive_streams["train"]
            self._restore_epoch += 1
            self._condition.notify_all()

    def _validate_acyclic_links(self) -> None:
        predecessor_counts = {chunk_id: 0 for chunk_id in self.chunks}
        for chunk in self.chunks.values():
            if chunk.successor_id is not None:
                predecessor_counts[chunk.successor_id] += 1
                successor = self.chunks[chunk.successor_id]
                if chunk.owner_id != successor.owner_id:
                    raise ValueError("replay chunk link crosses writer ownership")
        if any(count > 1 for count in predecessor_counts.values()):
            raise ValueError("replay chunk graph merges writer chains")
        for start in self.chunks:
            seen = set()
            current = start
            while current is not None:
                if current in seen:
                    raise ValueError("replay chunk link cycle")
                seen.add(current)
                current = self.chunks[current].successor_id

    def _history_locations(self) -> dict[bytes, tuple[ReplayWriter, int]]:
        return {
            chunk_id: (writer, ordinal)
            for writer in self.writers.values()
            for ordinal, chunk_id in enumerate(writer.chunk_history)
        }

    def _validate_chunk_histories(self) -> None:
        locations: dict[bytes, tuple[ReplayWriter, int]] = {}
        numbers: list[int] = []
        for writer in self.writers.values():
            expected_chunks = (
                0
                if writer.row_count == 0
                else writer.row_count // self.config.chunk_size + 1
            )
            if len(writer.chunk_history) != expected_chunks:
                raise ValueError(
                    "writer row count or emitted cadence disagrees with chunk history"
                )
            if not writer.chunk_history:
                if writer.current_chunk_id is not None:
                    raise ValueError("empty writer has a current chunk")
                continue
            if writer.current_chunk_id != writer.chunk_history[-1]:
                raise ValueError("writer current chunk disagrees with history")
            for ordinal, chunk_id in enumerate(writer.chunk_history):
                if chunk_id in locations:
                    raise ValueError("writer chunk histories overlap")
                number = int.from_bytes(chunk_id, "big")
                if not 0 < number < self.next_chunk_id:
                    raise ValueError("writer chunk history id is out of range")
                locations[chunk_id] = (writer, ordinal)
                numbers.append(number)
        if len(locations) != self.next_chunk_id - 1:
            raise ValueError("writer chunk histories do not cover allocated ids")
        if numbers and (min(numbers) != 1 or max(numbers) != self.next_chunk_id - 1):
            raise ValueError("writer chunk histories are not contiguous")
        for chunk_id, chunk in self.chunks.items():
            location = locations.get(chunk_id)
            if location is None or location[0].worker_id != chunk.owner_id:
                raise ValueError("live replay chunk has invalid lifetime owner")
            if chunk.size != self.config.chunk_size:
                raise ValueError("restored replay chunk size differs from config")
            if chunk.sealed:
                if chunk.length != chunk.size or chunk.successor_id is None:
                    raise ValueError("sealed replay chunk must be full")
            elif chunk.length >= chunk.size or chunk_id != location[0].current_chunk_id:
                raise ValueError("open replay chunk has invalid geometry")
        for writer in self.writers.values():
            retained = [
                chunk_id for chunk_id in writer.chunk_history if chunk_id in self.chunks
            ]
            if not retained:
                if writer.chunk_history:
                    raise ValueError("writer has no retained current chunk")
                continue
            if retained != writer.chunk_history[-len(retained) :]:
                raise ValueError("retained writer chunks are not a history suffix")
            for chunk_id, successor_id in zip(retained, retained[1:], strict=False):
                if self.chunks[chunk_id].successor_id != successor_id:
                    raise ValueError("retained writer history link is invalid")
            if self.chunks[retained[-1]].successor_id is not None:
                raise ValueError("writer history does not end at current chunk")

    def _validate_online_queue(self) -> None:
        locations = self._history_locations()
        seen: set[ReplayKey] = set()
        last_rows: dict[int, int] = {}
        for key in self.online_queue.keys:
            if key in seen:
                raise ValueError("online queue keys must be unique")
            seen.add(key)
            location = locations.get(key.chunk_id)
            if location is None or key.offset >= self.config.chunk_size:
                raise ValueError("online queue key was never allocated")
            writer, ordinal = location
            absolute_row = ordinal * self.config.chunk_size + key.offset
            if (
                absolute_row >= writer.row_count
                or absolute_row + self.raw_length > writer.row_count
                or absolute_row % self.raw_length != 1 % self.raw_length
            ):
                raise ValueError("online queue key has invalid writer cadence")
            previous = last_rows.get(writer.worker_id)
            if previous is not None and absolute_row <= previous:
                raise ValueError("online queue writer order is invalid")
            last_rows[writer.worker_id] = absolute_row
            if key.chunk_id not in self.chunks:
                continue
            try:
                self._resolve(key, self.raw_length)
            except KeyError as error:
                raise ValueError(
                    "online queue key is not a resolvable replay sequence"
                ) from error

    def _validate_retained_chronology(self) -> None:
        for writer in self.writers.values():
            retained = [
                chunk_id for chunk_id in writer.chunk_history if chunk_id in self.chunks
            ]
            complete_history = bool(retained) and retained[0] == writer.chunk_history[0]
            previous_is_last = False
            first_retained_row = True
            for chunk_id in retained:
                chunk = self.chunks[chunk_id]
                for offset in range(chunk.length):
                    is_first = bool(chunk.transition_data["is_first"][offset])
                    is_last = bool(chunk.transition_data["is_last"][offset])
                    is_terminal = bool(chunk.transition_data["is_terminal"][offset])
                    if is_terminal and not is_last:
                        raise ValueError("invalid retained replay chronology")
                    if first_retained_row and complete_history and not is_first:
                        raise ValueError("invalid retained replay chronology")
                    if previous_is_last and not is_first:
                        raise ValueError("invalid retained replay chronology")
                    first_retained_row = False
                    previous_is_last = is_last

    def _recomputed_refs(self) -> dict[bytes, int]:
        refs = {chunk_id: 0 for chunk_id in self.chunks}
        for chunk in self.chunks.values():
            if chunk.successor_id is not None:
                refs[chunk.successor_id] += 1
        for writer in self.writers.values():
            if writer.current_chunk_id is not None:
                refs[writer.current_chunk_id] += 1
            for key in writer.pending:
                if key.chunk_id not in refs:
                    raise ValueError("writer pending key references a missing chunk")
                refs[key.chunk_id] += 1
        for key in self.items.values():
            if key.chunk_id not in refs:
                raise ValueError("replay item references a missing chunk")
            refs[key.chunk_id] += 1
        return refs

    def _validate_restored_state(self) -> None:
        self._validate_chunk_histories()
        self._validate_retained_chronology()
        writer_ids = set(self.writers)
        if any(chunk.owner_id not in writer_ids for chunk in self.chunks.values()):
            raise ValueError("replay chunk has no owning writer")
        current_ids = [
            writer.current_chunk_id
            for writer in self.writers.values()
            if writer.current_chunk_id is not None
        ]
        if len(current_ids) != len(set(current_ids)):
            raise ValueError("replay writers share a current chunk")
        terminal_ids = {
            chunk.chunk_id
            for chunk in self.chunks.values()
            if chunk.successor_id is None
        }
        if terminal_ids != set(current_ids):
            raise ValueError("replay chunk chains do not terminate at writer cursors")
        for writer in self.writers.values():
            restored_offset = getattr(writer, "_restored_offset", writer.current_offset)
            expected_pending = min(writer.row_count, self.raw_length - 1)
            expected_emitted = max(0, writer.row_count - self.raw_length + 1)
            if (
                len(writer.pending) != expected_pending
                or writer.emitted_count != expected_emitted
                or writer.has_rows != (writer.row_count > 0)
            ):
                raise ValueError("invalid writer cadence state")
            if writer.current_chunk_id is None:
                if (
                    restored_offset != 0
                    or writer.row_count != 0
                    or writer.emitted_count != 0
                    or writer.pending
                    or writer.last_is_last
                ):
                    raise ValueError("invalid empty writer state")
            else:
                chunk = self.chunks.get(writer.current_chunk_id)
                if (
                    chunk is None
                    or chunk.owner_id != writer.worker_id
                    or chunk.sealed
                    or restored_offset != chunk.length
                    or chunk.length != writer.row_count % self.config.chunk_size
                ):
                    raise ValueError("invalid current writer state")
                if chunk.length:
                    tail: ReplayKey | None = ReplayKey(chunk.chunk_id, chunk.length - 1)
                else:
                    predecessors = [
                        candidate
                        for candidate in self.chunks.values()
                        if candidate.successor_id == chunk.chunk_id
                    ]
                    if len(predecessors) > 1:
                        raise ValueError("invalid writer predecessor state")
                    tail = None
                    if predecessors:
                        predecessor = predecessors[0]
                        if not predecessor.length:
                            raise ValueError("invalid writer predecessor state")
                        tail = ReplayKey(predecessor.chunk_id, predecessor.length - 1)
                pending = list(writer.pending)
                if pending:
                    try:
                        resolved = self._resolve(pending[0], len(pending))
                    except KeyError as error:
                        raise ValueError(
                            "writer pending keys are not resolvable"
                        ) from error
                    if resolved != pending:
                        raise ValueError("writer pending keys are not consecutive")
                    if tail is not None and pending[-1] != tail:
                        raise ValueError("writer pending keys do not end at its cursor")
                    tail = pending[-1]
                if tail is not None:
                    tail_chunk = self.chunks[tail.chunk_id]
                    actual_last = bool(
                        tail_chunk.transition_data["is_last"][tail.offset]
                    )
                    if writer.last_is_last != actual_last:
                        raise ValueError("invalid writer episode chronology state")
        if sum(writer.emitted_count for writer in self.writers.values()) != (
            self.next_item_id
        ):
            raise ValueError("writer emitted counts disagree with next item id")
        if len(self.items) > self.config.capacity:
            raise ValueError("restored replay exceeds capacity")
        for key in self.items.values():
            self._resolve(key, self.raw_length)
        self._validate_online_queue()
        if self.refs != self._recomputed_refs():
            raise ValueError("restored replay refcounts are inconsistent")

    def validate(self) -> None:
        with self._lock:
            self._validate_acyclic_links()
            self._validate_restored_state()
            if self.fifo != list(self.items):
                raise ValueError("replay FIFO/items disagreement")
            if set(self.selector.keys) != set(self.items):
                raise ValueError("replay selector/items disagreement")


__all__ = [
    "ConsecutiveStream",
    "DreamerReplay",
    "OnlineQueue",
    "ReplayBatch",
    "ReplayChunk",
    "ReplayKey",
    "ReplayWriter",
    "UniformSelector",
]
