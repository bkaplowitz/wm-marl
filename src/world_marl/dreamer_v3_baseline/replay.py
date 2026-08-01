from __future__ import annotations

import copy
from collections import deque
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from .config import ReplayConfig, SequenceShapeConfig
from .networks import TensorSpace


Array = npt.NDArray[np.generic]


_MAX_CHUNK_ID = 2**128 - 1
_CHUNK_SENTINEL = 2**128
_MAX_ITEM_ID = 2**63 - 1
_ITEM_SENTINEL = 2**63
_MAX_COUNTER = int(np.iinfo(np.int64).max)
_MODES = ("train", "report")
_REQUIRED = {
    "is_first": ((), "bool"),
    "is_last": ((), "bool"),
    "is_terminal": ((), "bool"),
    "reward": ((), "float32"),
}


def _exact_mapping(
    value: object, keys: AbstractSet[object], label: str
) -> Mapping[Any, Any]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be a plain dictionary")
    if set(value) != keys:
        missing = sorted((str(x) for x in keys - set(value)))
        extra = sorted((str(x) for x in set(value) - keys))
        raise ValueError(f"{label} keys differ; missing={missing}, extra={extra}")
    return value


def _check_plain_tree(value: object, label: str, seen: set[int] | None = None) -> None:
    if seen is None:
        seen = set()
    if type(value) is dict:
        identity = id(value)
        if identity in seen:
            raise ValueError(f"{label} contains a mutable alias")
        seen.add(identity)
        for key, item in value.items():
            if type(key) not in (str, int, bytes):
                raise TypeError(f"{label} contains an invalid mapping key")
            _check_plain_tree(item, label, seen)
        return
    if type(value) is list:
        identity = id(value)
        if identity in seen:
            raise ValueError(f"{label} contains a mutable alias")
        seen.add(identity)
        for item in value:
            _check_plain_tree(item, label, seen)
        return
    if type(value) is np.ndarray:
        identity = id(value)
        if identity in seen:
            raise ValueError(f"{label} contains a mutable alias")
        seen.add(identity)
        if value.dtype.hasobject:
            raise TypeError(f"{label} contains an object array")
        return
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            raise TypeError(f"{label} contains an object scalar")
        return
    if value is None or type(value) in (str, bytes, bool, int, float):
        return
    raise TypeError(f"{label} contains a nonprimitive leaf")


def _array(
    value: object, space: TensorSpace, leading: tuple[int, ...], label: str
) -> Array:
    result = np.asarray(value)
    expected = (*leading, *space.shape)
    if result.dtype != np.dtype(space.dtype) or result.shape != expected:
        raise ValueError(f"{label} must have dtype {space.dtype} and shape {expected}")
    if not np.isfinite(result).all() and np.issubdtype(result.dtype, np.floating):
        raise ValueError(f"{label} must be finite")
    return result.copy()


def _immutable_array(value: object) -> Array:
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _counter(value: object, label: str) -> np.int64:
    if type(value) is not np.int64 or value < 0:
        raise ValueError(f"{label} must be an exact nonnegative np.int64 counter")
    return value


def _advance_counter(value: np.int64, amount: int, label: str) -> np.int64:
    if type(amount) is not int or amount < 0:
        raise ValueError(f"{label} counter increment is invalid")
    if int(value) > _MAX_COUNTER - amount:
        raise OverflowError(f"{label} counter exhausted")
    return np.int64(int(value) + amount)


def _same_array(left: object, right: Array) -> bool:
    return (
        type(left) is np.ndarray
        and left.shape == right.shape
        and left.dtype == right.dtype
        and left.strides == right.strides
        and left.flags.writeable == right.flags.writeable
        and np.array_equal(left, right)
    )


def _same_array_mapping(left: object, right: Mapping[str, Array]) -> bool:
    return (
        isinstance(left, Mapping)
        and set(left) == set(right)
        and all(_same_array(left[name], value) for name, value in right.items())
    )


def _copy_key(key: ReplayKey | None) -> ReplayKey | None:
    if key is None:
        return None
    return ReplayKey(key.chunk_id, key.offset)


def _pcg64_state(value: object) -> dict[str, object]:
    record = _exact_mapping(
        value,
        {"bit_generator", "has_uint32", "state", "uinteger"},
        "UniformSelector RNG state",
    )
    if type(record["bit_generator"]) is not str or record["bit_generator"] != "PCG64":
        raise ValueError("UniformSelector RNG state requires PCG64")
    inner = _exact_mapping(
        record["state"], {"inc", "state"}, "UniformSelector inner RNG state"
    )

    def checked(name: str, item: object, upper: int) -> int:
        if type(item) is not int or not 0 <= item < upper:
            raise ValueError(f"UniformSelector RNG state {name} is invalid")
        return item

    has_uint32 = checked("has_uint32", record["has_uint32"], 2)
    return {
        "bit_generator": "PCG64",
        "state": {
            "state": checked("state", inner["state"], 2**128),
            "inc": checked("inc", inner["inc"], 2**128),
        },
        "has_uint32": has_uint32,
        "uinteger": checked("uinteger", record["uinteger"], 2**32),
    }


def _space_state(spaces: Mapping[str, TensorSpace]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, space in sorted(spaces.items()):
        runtime_classes = space.classes
        classes: int | list[int] | None
        if runtime_classes is None or isinstance(runtime_classes, int):
            classes = runtime_classes
        else:
            classes = list(runtime_classes)
        result[name] = {
            "classes": classes,
            "dtype": space.dtype,
            "shape": list(space.shape),
        }
    return result


def _validate_space_state(
    state: object, spaces: Mapping[str, TensorSpace], label: str
) -> None:
    record = _exact_mapping(state, set(spaces), label)
    for name, space in spaces.items():
        item = _exact_mapping(
            record[name], {"classes", "dtype", "shape"}, f"{label} {name!r}"
        )
        if type(item["dtype"]) is not str or item["dtype"] != space.dtype:
            raise ValueError(f"{label} {name!r} dtype differs")
        shape = item["shape"]
        if (
            type(shape) is not list
            or any(type(size) is not int for size in shape)
            or shape != list(space.shape)
        ):
            raise ValueError(f"{label} {name!r} shape differs")
        classes = item["classes"]
        runtime_classes = space.classes
        expected_classes: int | list[int] | None
        if runtime_classes is None or isinstance(runtime_classes, int):
            expected_classes = runtime_classes
        else:
            expected_classes = list(runtime_classes)
        if expected_classes is None:
            valid_classes = classes is None
        elif isinstance(expected_classes, int):
            valid_classes = type(classes) is int and classes == expected_classes
        else:
            valid_classes = (
                type(classes) is list
                and len(classes) == len(expected_classes)
                and all(type(value) is int for value in classes)
                and classes == expected_classes
            )
        if not valid_classes:
            raise ValueError(f"{label} {name!r} classes differ")


@dataclass(frozen=True, order=True)
class ReplayKey:
    chunk_id: bytes
    offset: int

    def __post_init__(self) -> None:
        if type(self.chunk_id) is not bytes or len(self.chunk_id) != 16:
            raise ValueError("chunk_id must be exactly 16 bytes")
        if not 1 <= int.from_bytes(self.chunk_id, "big") <= _MAX_CHUNK_ID:
            raise ValueError("chunk_id is outside the allocatable domain")
        if type(self.offset) is not int or not 0 <= self.offset < 2**32:
            raise ValueError("offset must be a uint32 integer")

    def to_step_id(self) -> Array:
        return np.frombuffer(
            self.chunk_id + self.offset.to_bytes(4, "big"), np.uint8
        ).copy()

    @classmethod
    def from_step_id(cls, value: object) -> ReplayKey:
        array = np.asarray(value)
        if array.dtype != np.uint8 or array.shape != (20,):
            raise ValueError("step id must have dtype uint8 and shape [20]")
        return cls(array[:16].tobytes(), int.from_bytes(array[16:].tobytes(), "big"))

    def state_dict(self) -> dict[str, object]:
        return {"chunk_id": self.chunk_id, "offset": self.offset}

    @classmethod
    def from_state_dict(cls, state: object) -> ReplayKey:
        record = _exact_mapping(state, {"chunk_id", "offset"}, "ReplayKey state")
        return cls(record["chunk_id"], record["offset"])


class ReplayBatch:
    __slots__ = ("_locked", "data", "step_ids")

    _locked: bool
    data: Mapping[str, Array]
    step_ids: Array

    def __init__(self, data: Mapping[str, object], step_ids: object) -> None:
        self._locked = False
        if not isinstance(data, Mapping) or not data:
            raise ValueError("ReplayBatch data must be a nonempty mapping")
        ids = np.asarray(step_ids)
        if ids.dtype != np.uint8 or ids.ndim != 3 or ids.shape[-1] != 20:
            raise ValueError("ReplayBatch step_ids must be uint8[B,T,20]")
        copied: dict[str, Array] = {}
        for name, value in sorted(data.items()):
            if type(name) is not str:
                raise TypeError("ReplayBatch names must be exact strings")
            array = np.asarray(value)
            if array.ndim < 2 or array.shape[:2] != ids.shape[:2]:
                raise ValueError(f"ReplayBatch leaf {name!r} has wrong leading axes")
            if array.dtype.hasobject:
                raise TypeError("ReplayBatch cannot contain object arrays")
            copied[name] = _immutable_array(array)
        for sequence in ids:
            previous: ReplayKey | None = None
            seen: set[bytes] = set()
            for encoded in sequence:
                key = ReplayKey.from_step_id(encoded)
                if previous is not None:
                    if key.chunk_id == previous.chunk_id:
                        if key.offset != previous.offset + 1:
                            raise ValueError(
                                "ReplayBatch step ids are not chronological"
                            )
                    elif key.offset != 0 or key.chunk_id in seen:
                        raise ValueError("ReplayBatch step ids are not chronological")
                seen.add(key.chunk_id)
                previous = key
        self.data = MappingProxyType(copied)
        self.step_ids = _immutable_array(ids)
        self._locked = True

    def __setattr__(self, name: str, value: object) -> None:
        if name in ("data", "step_ids") and getattr(self, "_locked", False):
            raise AttributeError(f"ReplayBatch {name} is non-reassignable")
        object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, Array]:
        result = {name: value.copy() for name, value in self.data.items()}
        result["stepid"] = self.step_ids.copy()
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "data": {name: value.copy() for name, value in self.data.items()},
            "step_ids": self.step_ids.copy(),
        }

    @classmethod
    def from_state(
        cls,
        state: object,
        transition_spaces: Mapping[str, TensorSpace],
        latent_spaces: Mapping[str, TensorSpace],
        expected_batch_size: int,
        expected_time_length: int,
    ) -> ReplayBatch:
        _check_plain_tree(state, "ReplayBatch state")
        record = _exact_mapping(state, {"data", "step_ids"}, "ReplayBatch state")
        data = record["data"]
        if type(data) is not dict:
            raise TypeError("ReplayBatch data state must be a plain dictionary")
        expected = set(transition_spaces) | set(latent_spaces)
        if set(data) not in (expected, expected | {"consec"}):
            raise ValueError("ReplayBatch data schema differs")
        leading = (expected_batch_size, expected_time_length)
        copied = {
            name: _array(data[name], space, leading, f"ReplayBatch {name}")
            for name, space in {**transition_spaces, **latent_spaces}.items()
        }
        if "consec" in data:
            consec = np.asarray(data["consec"])
            if consec.dtype != np.int32 or consec.shape != leading:
                raise ValueError("ReplayBatch consec has wrong dtype or shape")
            copied["consec"] = consec.copy()
        ids = np.asarray(record["step_ids"])
        if ids.dtype != np.uint8 or ids.shape != (*leading, 20):
            raise ValueError("ReplayBatch step_ids have wrong dtype or shape")
        return cls(copied, ids)


def _same_batch(left: object, right: ReplayBatch) -> bool:
    return (
        type(left) is type(right)
        and _same_array_mapping(left.data, right.data)
        and _same_array(left.step_ids, right.step_ids)
    )


class ReplayChunk:
    def __init__(
        self,
        chunk_id: bytes,
        capacity: int,
        transition_spaces: Mapping[str, TensorSpace],
        latent_spaces: Mapping[str, TensorSpace],
        owner_id: int,
    ) -> None:
        ReplayKey(chunk_id, 0)
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("chunk capacity must be positive")
        if type(owner_id) is not int:
            raise TypeError("chunk owner must be an integer")
        self.chunk_id = chunk_id
        self.capacity = capacity
        self.owner_id = owner_id
        self.transition_spaces = dict(sorted(transition_spaces.items()))
        self.latent_spaces = dict(sorted(latent_spaces.items()))
        self.length = np.int64(0)
        self.successor_id: bytes | None = None
        self.transition_data = {
            name: np.zeros((capacity, *space.shape), np.dtype(space.dtype))
            for name, space in self.transition_spaces.items()
        }
        self.latent_data = {
            name: np.zeros((capacity, *space.shape), np.dtype(space.dtype))
            for name, space in self.latent_spaces.items()
        }

    @property
    def sealed(self) -> bool:
        return self.successor_id is not None

    def append(self, row: Mapping[str, Array]) -> ReplayKey:
        if self.sealed or self.length >= self.capacity:
            raise RuntimeError("cannot append to a closed replay chunk")
        next_length = _advance_counter(self.length, 1, "ReplayChunk length")
        index = int(self.length)
        for name in self.transition_spaces:
            self.transition_data[name][index] = row[name]
        for name in self.latent_spaces:
            self.latent_data[name][index] = row[name]
        key = ReplayKey(self.chunk_id, index)
        self.length = next_length
        return key

    def seal(self, successor_id: bytes) -> None:
        ReplayKey(successor_id, 0)
        if self.sealed or self.length != self.capacity or successor_id == self.chunk_id:
            raise RuntimeError("only a full open chunk can be linked")
        self.successor_id = successor_id

    def read(self, offset: int, length: int) -> dict[str, Array]:
        if type(offset) is not int or type(length) is not int:
            raise TypeError("chunk read bounds must be integers")
        if offset < 0 or length < 0 or offset + length > self.length:
            raise IndexError("chunk read is outside the live prefix")
        tensors = {**self.transition_data, **self.latent_data}
        return {
            name: value[offset : offset + length].copy()
            for name, value in tensors.items()
        }

    def update_context(self, offset: int, values: Mapping[str, Array]) -> None:
        if not 0 <= offset < self.length:
            raise IndexError("context update is outside the live prefix")
        for name, value in values.items():
            self.latent_data[name][offset] = value

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "chunk_id": self.chunk_id,
            "latent": {
                name: value[: int(self.length)].copy()
                for name, value in self.latent_data.items()
            },
            "length": self.length,
            "owner_id": self.owner_id,
            "successor": self.successor_id,
            "transition": {
                name: value[: int(self.length)].copy()
                for name, value in self.transition_data.items()
            },
        }

    @classmethod
    def from_state_dict(
        cls,
        state: object,
        transition_spaces: Mapping[str, TensorSpace],
        latent_spaces: Mapping[str, TensorSpace],
    ) -> ReplayChunk:
        _check_plain_tree(state, "ReplayChunk state")
        record = _exact_mapping(
            state,
            {
                "capacity",
                "chunk_id",
                "latent",
                "length",
                "owner_id",
                "successor",
                "transition",
            },
            "ReplayChunk state",
        )
        chunk = cls(
            record["chunk_id"],
            record["capacity"],
            transition_spaces,
            latent_spaces,
            record["owner_id"],
        )
        length = _counter(record["length"], "ReplayChunk length")
        if not 0 <= int(length) <= chunk.capacity:
            raise ValueError("invalid ReplayChunk length")
        transition = _exact_mapping(
            record["transition"], set(transition_spaces), "ReplayChunk transition"
        )
        latent = _exact_mapping(
            record["latent"], set(latent_spaces), "ReplayChunk latent"
        )
        for name, space in transition_spaces.items():
            chunk.transition_data[name][:length] = _array(
                transition[name], space, (int(length),), f"ReplayChunk {name}"
            )
        for name, space in latent_spaces.items():
            chunk.latent_data[name][:length] = _array(
                latent[name], space, (int(length),), f"ReplayChunk {name}"
            )
        chunk.length = length
        successor = record["successor"]
        if successor is not None:
            if length != chunk.capacity:
                raise ValueError("linked ReplayChunk must be full")
            chunk.seal(successor)
        elif length == chunk.capacity:
            raise ValueError("full ReplayChunk must have a successor")
        return chunk


class OnlineQueue:
    def __init__(self, maxlen: int) -> None:
        if type(maxlen) is not int or maxlen <= 0:
            raise ValueError("online queue size must be positive")
        self.maxlen = maxlen
        self.keys: list[ReplayKey] = []

    def __len__(self) -> int:
        return len(self.keys)

    def push(self, key: ReplayKey) -> None:
        if len(self.keys) == self.maxlen:
            self.keys.pop(0)
        self.keys.append(key)

    def pop(self) -> ReplayKey:
        if not self.keys:
            raise IndexError("online queue is empty")
        return self.keys.pop(0)

    def state_dict(self) -> dict[str, object]:
        return {
            "keys": [key.state_dict() for key in self.keys],
            "maxlen": self.maxlen,
        }

    @classmethod
    def from_state_dict(cls, state: object) -> OnlineQueue:
        _check_plain_tree(state, "OnlineQueue state")
        record = _exact_mapping(state, {"keys", "maxlen"}, "OnlineQueue state")
        if type(record["keys"]) is not list:
            raise TypeError("OnlineQueue keys must be a plain list")
        result = cls(record["maxlen"])
        result.keys = [ReplayKey.from_state_dict(item) for item in record["keys"]]
        if len(result.keys) > result.maxlen:
            raise ValueError("OnlineQueue exceeds maxlen")
        return result


class UniformSelector:
    def __init__(self, seed: int) -> None:
        if type(seed) is not int:
            raise TypeError("selector seed must be an integer")
        self.indices: dict[int, int] = {}
        self.keys: list[int] = []
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.keys)

    def insert(self, item_id: int) -> None:
        if (
            type(item_id) is not int
            or not 0 <= item_id <= _MAX_ITEM_ID
            or item_id in self.indices
        ):
            raise ValueError("selector item id must be a unique integer")
        self.indices[item_id] = len(self.keys)
        self.keys.append(item_id)

    def delete(self, item_id: int) -> None:
        if item_id not in self.indices:
            raise KeyError(item_id)
        index = self.indices.pop(item_id)
        final = self.keys.pop()
        if index < len(self.keys):
            self.keys[index] = final
            self.indices[final] = index

    def sample(self) -> int:
        if not self.keys:
            raise IndexError("uniform selector is empty")
        return self.keys[int(self.rng.integers(0, len(self.keys)))]

    def state_dict(self) -> dict[str, object]:
        return {
            "bit_generator": "PCG64",
            "indices": dict(self.indices),
            "keys": list(self.keys),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    @classmethod
    def from_state_dict(cls, state: object) -> UniformSelector:
        _check_plain_tree(state, "UniformSelector state")
        record = _exact_mapping(
            state,
            {"bit_generator", "indices", "keys", "rng_state"},
            "UniformSelector state",
        )
        if (
            type(record["bit_generator"]) is not str
            or record["bit_generator"] != "PCG64"
        ):
            raise ValueError("UniformSelector requires PCG64")
        keys = record["keys"]
        indices = record["indices"]
        if (
            type(keys) is not list
            or any(
                type(item) is not int or not 0 <= item <= _MAX_ITEM_ID for item in keys
            )
            or len(set(keys)) != len(keys)
            or type(indices) is not dict
            or any(
                type(item_id) is not int
                or not 0 <= item_id <= _MAX_ITEM_ID
                or type(index) is not int
                or not 0 <= index < len(keys)
                for item_id, index in indices.items()
            )
            or indices != {item: index for index, item in enumerate(keys)}
        ):
            raise ValueError("invalid UniformSelector key/index state")
        result = cls(0)
        result.keys = list(keys)
        result.indices = dict(indices)
        rng_state = _pcg64_state(record["rng_state"])
        try:
            result.rng.bit_generator.state = rng_state
        except (KeyError, OverflowError, TypeError, ValueError) as error:
            raise ValueError("invalid UniformSelector RNG state") from error
        return result


class ConsecutiveStream:
    def __init__(self, sequence_length: int, consecutive: int, context: int) -> None:
        if type(sequence_length) is not int or sequence_length <= 0:
            raise ValueError("stream sequence length must be positive")
        if type(consecutive) is not int or consecutive <= 0:
            raise ValueError("stream consecutive count must be positive")
        if type(context) is not int or context < 0:
            raise ValueError("stream context must be nonnegative")
        self.sequence_length = sequence_length
        self.consecutive = consecutive
        self.context = context
        self.index = np.int64(0)
        self.current: ReplayBatch | None = None

    @property
    def raw_length(self) -> int:
        return self.context + self.sequence_length * self.consecutive

    def state_dict(self) -> dict[str, object]:
        return {
            "current": None if self.current is None else self.current.state_dict(),
            "index": self.index,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: object,
        transition_spaces: Mapping[str, TensorSpace],
        latent_spaces: Mapping[str, TensorSpace],
        expected_batch_size: int,
        sequence_length: int,
        consecutive: int,
        context: int,
    ) -> ConsecutiveStream:
        _check_plain_tree(state, "ConsecutiveStream state")
        record = _exact_mapping(state, {"current", "index"}, "ConsecutiveStream state")
        result = cls(sequence_length, consecutive, context)
        index = _counter(record["index"], "ConsecutiveStream index")
        if not 0 <= int(index) <= consecutive:
            raise ValueError("invalid ConsecutiveStream index")
        current = record["current"]
        if current is None and index:
            raise ValueError("active ConsecutiveStream needs a current batch")
        if current is not None and index == 0:
            raise ValueError("active ConsecutiveStream index must be positive")
        if current is not None:
            current_record = _exact_mapping(
                current, {"data", "step_ids"}, "ConsecutiveStream current"
            )
            current_data = current_record["data"]
            if type(current_data) is not dict:
                raise TypeError(
                    "ConsecutiveStream current data must be a plain dictionary"
                )
            if "consec" in current_data:
                raise ValueError("ConsecutiveStream raw current cannot contain consec")
            result.current = ReplayBatch.from_state(
                current_record,
                transition_spaces,
                latent_spaces,
                expected_batch_size,
                result.raw_length,
            )
        result.index = index
        return result


class ReplayWriter:
    def __init__(self, worker_id: int, raw_length: int) -> None:
        if type(worker_id) is not int:
            raise TypeError("worker id must be an integer")
        self.worker_id = worker_id
        self.current_chunk_id: bytes | None = None
        self.suffix: list[ReplayKey] = []
        self.online_phase = np.int64(0)
        self.retained_rows = np.int64(0)
        self.has_rows = False
        self.last_is_last = False
        self.raw_length = raw_length

    def state_dict(self) -> dict[str, object]:
        return {
            "current_chunk_id": self.current_chunk_id,
            "has_rows": self.has_rows,
            "last_is_last": self.last_is_last,
            "online_phase": self.online_phase,
            "retained_rows": self.retained_rows,
            "suffix": [key.state_dict() for key in self.suffix],
            "worker_id": self.worker_id,
        }

    @classmethod
    def from_state_dict(cls, state: object, raw_length: int) -> ReplayWriter:
        _check_plain_tree(state, "ReplayWriter state")
        record = _exact_mapping(
            state,
            {
                "current_chunk_id",
                "has_rows",
                "last_is_last",
                "online_phase",
                "retained_rows",
                "suffix",
                "worker_id",
            },
            "ReplayWriter state",
        )
        result = cls(record["worker_id"], raw_length)
        current = record["current_chunk_id"]
        if current is not None:
            ReplayKey(current, 0)
        result.current_chunk_id = current
        if type(record["suffix"]) is not list:
            raise TypeError("ReplayWriter suffix must be a plain list")
        result.suffix = [ReplayKey.from_state_dict(item) for item in record["suffix"]]
        phase = _counter(record["online_phase"], "ReplayWriter online phase")
        if not 0 <= int(phase) < raw_length:
            raise ValueError("invalid ReplayWriter online phase")
        retained_rows = _counter(record["retained_rows"], "ReplayWriter retained rows")
        if (
            type(record["has_rows"]) is not bool
            or type(record["last_is_last"]) is not bool
        ):
            raise TypeError("invalid ReplayWriter boundary flags")
        result.online_phase = phase
        result.retained_rows = retained_rows
        result.has_rows = record["has_rows"]
        result.last_is_last = record["last_is_last"]
        return result


@dataclass(frozen=True, slots=True)
class _AddPlan:
    _owner: object
    version: np.int64
    worker: int
    row: Mapping[str, Array]
    emits_item: bool
    report_ready: ReplayKey | None


@dataclass(frozen=True, slots=True)
class _PreparedAdd:
    public: _AddPlan
    version: np.int64
    worker: int
    row_view: Mapping[str, Array]
    row: Mapping[str, Array]
    new_writer: ReplayWriter | None
    new_chunks: tuple[ReplayChunk, ...]
    emits_item: bool
    report_ready_view: ReplayKey | None
    report_ready: ReplayKey | None


@dataclass(frozen=True, slots=True)
class _SamplePlan:
    _owner: object
    version: np.int64
    mode: str
    batch: ReplayBatch
    index: np.int64
    queue: tuple[ReplayKey, ...]
    online: int
    uniform: int
    stale: int


@dataclass(frozen=True, slots=True)
class _PreparedSample:
    public: _SamplePlan
    version: np.int64
    mode: str
    batch_view: ReplayBatch
    batch: ReplayBatch
    current: ReplayBatch
    index: np.int64
    queue: list[ReplayKey]
    queue_view: tuple[ReplayKey, ...]
    rng: np.random.Generator
    online: int
    uniform: int
    stale: int


class DreamerReplay:
    SCHEMA_VERSION = 4

    def __init__(
        self,
        config: ReplayConfig,
        sequence_shape: SequenceShapeConfig,
        transition_spaces: Mapping[str, TensorSpace],
        latent_spaces: Mapping[str, TensorSpace],
    ) -> None:
        if (
            type(config) is not ReplayConfig
            or type(sequence_shape) is not SequenceShapeConfig
        ):
            raise TypeError("replay requires resolved replay and sequence configs")
        self.config = config
        self.sequence_shape = sequence_shape
        self.transition_spaces = dict(sorted(transition_spaces.items()))
        self.latent_spaces = dict(sorted(latent_spaces.items()))
        self._validate_spaces()
        self.raw_length = sequence_shape.raw_length
        self.report_raw_length = sequence_shape.report_raw_length
        self.chunks: dict[bytes, ReplayChunk] = {}
        self.refs: dict[bytes, np.int64] = {}
        self.writers: dict[int, ReplayWriter] = {}
        self.items: dict[int, ReplayKey] = {}
        self.fifo: deque[int] = deque()
        self.next_chunk_id = 1
        self.next_item_id = 0
        self.online_queue = OnlineQueue(config.online_queue_size)
        self.selector = UniformSelector(seed=0)
        self._item_ids_by_key: dict[ReplayKey, int] = {}
        self._report_pending: set[int] = set()
        self.streams = {
            "train": ConsecutiveStream(
                sequence_shape.sequence_length,
                sequence_shape.consecutive,
                sequence_shape.context,
            ),
            "report": ConsecutiveStream(
                sequence_shape.report_length,
                sequence_shape.report_consecutive,
                sequence_shape.context,
            ),
        }
        self.__plan_owner = object()
        self.__prepared_add: dict[int, _PreparedAdd] = {}
        self.__prepared_sample: dict[int, _PreparedSample] = {}
        self.__consumed_add: _AddPlan | None = None
        self.__consumed_sample: _SamplePlan | None = None
        self._version = np.int64(0)
        self._metrics: dict[str, np.int64] = {
            "inserted_rows": np.int64(0),
            "inserted_items": np.int64(0),
            "sample_calls": np.int64(0),
            "sampled_sequences": np.int64(0),
            "online_samples": np.int64(0),
            "uniform_samples": np.int64(0),
            "stale_online": np.int64(0),
            "update_calls": np.int64(0),
            "updated_rows": np.int64(0),
            "stale_updates": np.int64(0),
        }

    def _preflight_counters(self, updates: Mapping[str, int]) -> None:
        for name, amount in updates.items():
            _advance_counter(self._metrics[name], amount, f"Replay metric {name}")
        _advance_counter(self._version, 1, "Replay version")

    def _commit_counters(self, updates: Mapping[str, int]) -> None:
        for name, amount in updates.items():
            self._metrics[name] = _advance_counter(
                self._metrics[name], amount, f"Replay metric {name}"
            )
        self._version = _advance_counter(self._version, 1, "Replay version")

    def _validate_spaces(self) -> None:
        if not self.transition_spaces:
            raise ValueError("transition spaces cannot be empty")
        if set(self.transition_spaces) & set(self.latent_spaces):
            raise ValueError("transition and latent spaces overlap")
        if {"consec", "stepid"} & (
            set(self.transition_spaces) | set(self.latent_spaces)
        ):
            raise ValueError("replay spaces use reserved keys")
        for name, (shape, dtype) in _REQUIRED.items():
            space = self.transition_spaces.get(name)
            if space is None or space.shape != shape or space.dtype != dtype:
                raise ValueError(f"invalid required replay space {name!r}")

    @property
    def _action_keys(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.transition_spaces
            if name == "action" or name.startswith("action/")
        )

    def __len__(self) -> int:
        return len(self.items)

    def _normalize_row(self, row: object) -> dict[str, Array]:
        if not isinstance(row, Mapping):
            raise TypeError("replay row must be a mapping")
        expected = set(self.transition_spaces) | set(self.latent_spaces)
        if set(row) != expected:
            raise ValueError("replay row schema differs")
        normalized = {
            name: _array(row[name], space, (), f"replay row {name}")
            for name, space in {**self.transition_spaces, **self.latent_spaces}.items()
        }
        if bool(normalized["is_terminal"]) and not bool(normalized["is_last"]):
            raise ValueError("is_terminal requires is_last")
        if bool(normalized["is_last"]) and any(
            np.any(normalized[name] != 0) for name in self._action_keys
        ):
            raise ValueError("final replay row action must be zero")
        return normalized

    def _required_chunk_allocations(self, writer: ReplayWriter | None) -> int:
        if writer is None or writer.current_chunk_id is None:
            return 2 if self.config.chunk_size == 1 else 1
        current = self.chunks[writer.current_chunk_id]
        return 1 if current.length + 1 == current.capacity else 0

    def _preflight_chunk_allocations(self, count: int) -> None:
        if count == 0:
            return
        if self.next_chunk_id > _MAX_CHUNK_ID - count + 1:
            raise OverflowError("replay chunk id counter exhausted")
        for value in range(self.next_chunk_id, self.next_chunk_id + count):
            candidate = value.to_bytes(16, "big")
            if candidate in self.chunks or any(
                key.chunk_id == candidate for key in self.online_queue.keys
            ):
                raise RuntimeError("replay chunk id collision")

    def prepare_add(self, row: object, *, worker: int = 0) -> _AddPlan:
        if type(worker) is not int:
            raise TypeError("worker id must be an integer")
        private = {
            name: _immutable_array(value)
            for name, value in self._normalize_row(row).items()
        }
        private_row = MappingProxyType(private)
        public_row = MappingProxyType(
            {name: _immutable_array(value) for name, value in private.items()}
        )
        writer = self.writers.get(worker)
        if writer is None:
            if not bool(private["is_first"]):
                raise ValueError("a replay worker's first row must be is_first")
        elif writer.last_is_last and not bool(private["is_first"]):
            raise ValueError("a row after is_last must be is_first")
        required = self._required_chunk_allocations(writer)
        self._preflight_chunk_allocations(required)
        emits = writer is not None and len(writer.suffix) + 1 >= self.raw_length
        if writer is None:
            emits = self.raw_length == 1
        self._preflight_counters({"inserted_rows": 1, "inserted_items": int(emits)})
        if emits and self.next_item_id > _MAX_ITEM_ID:
            raise OverflowError("replay item id counter exhausted")
        if emits and self.next_item_id in self.items:
            raise RuntimeError("replay item id collision")
        if emits and len(self.items) >= self.config.capacity:
            if not self.fifo or self.fifo[0] not in self.items:
                raise RuntimeError("replay eviction state is invalid")
        suffix = [] if writer is None else writer.suffix
        report_ready = None
        if len(suffix) + 1 >= self.report_raw_length:
            if self.report_raw_length == 1:
                if writer is None or writer.current_chunk_id is None:
                    report_ready = ReplayKey(self.next_chunk_id.to_bytes(16, "big"), 0)
                else:
                    current = self.chunks[writer.current_chunk_id]
                    report_ready = ReplayKey(current.chunk_id, int(current.length))
            else:
                report_ready = suffix[-(self.report_raw_length - 1)]
        private_ready = _copy_key(report_ready)
        public_ready = _copy_key(report_ready)
        new_writer = ReplayWriter(worker, self.raw_length) if writer is None else None
        new_chunks = tuple(
            ReplayChunk(
                (self.next_chunk_id + index).to_bytes(16, "big"),
                self.config.chunk_size,
                self.transition_spaces,
                self.latent_spaces,
                worker,
            )
            for index in range(required)
        )
        for chunk in new_chunks:
            for value in (*chunk.transition_data.values(), *chunk.latent_data.values()):
                value.flags.writeable = False
        public = _AddPlan(
            self.__plan_owner,
            self._version,
            worker,
            public_row,
            emits,
            public_ready,
        )
        if self.__prepared_add:
            self.__consumed_add = next(iter(self.__prepared_add.values())).public
            self.__prepared_add.clear()
        self.__prepared_add[id(public)] = _PreparedAdd(
            public,
            self._version,
            worker,
            public_row,
            private_row,
            new_writer,
            new_chunks,
            emits,
            public_ready,
            private_ready,
        )
        return public

    def _install_chunk(self, chunk: ReplayChunk, refs: int) -> ReplayChunk:
        if int.from_bytes(chunk.chunk_id, "big") != self.next_chunk_id:
            raise RuntimeError("staged replay chunk is out of order")
        for value in (*chunk.transition_data.values(), *chunk.latent_data.values()):
            value.flags.writeable = True
        self.next_chunk_id += 1
        self.chunks[chunk.chunk_id] = chunk
        self.refs[chunk.chunk_id] = np.int64(refs)
        return chunk

    def _inc_ref(self, chunk_id: bytes) -> None:
        self.refs[chunk_id] = _advance_counter(
            self.refs[chunk_id], 1, "Replay reference count"
        )

    def _dec_ref(self, chunk_id: bytes) -> None:
        if self.refs[chunk_id] <= 0:
            raise RuntimeError("Replay reference count underflow")
        self.refs[chunk_id] = np.int64(int(self.refs[chunk_id]) - 1)
        if self.refs[chunk_id]:
            return
        chunk = self.chunks.pop(chunk_id)
        del self.refs[chunk_id]
        if chunk.successor_id is not None:
            self._dec_ref(chunk.successor_id)

    def _evict_one(self) -> None:
        item_id = self.fifo.popleft()
        key = self.items.pop(item_id)
        del self._item_ids_by_key[key]
        self._report_pending.discard(item_id)
        self.selector.delete(item_id)
        self._dec_ref(key.chunk_id)

    def _insert_item(self, key: ReplayKey) -> None:
        if len(self.items) == self.config.capacity:
            self._evict_one()
        item_id = self.next_item_id
        self.next_item_id += 1
        self.items[item_id] = key
        self.fifo.append(item_id)
        self._item_ids_by_key[key] = item_id
        if self.report_raw_length > self.raw_length:
            self._report_pending.add(item_id)
        self.selector.insert(item_id)
        self._inc_ref(key.chunk_id)

    def _take_add_plan(self, plan: _AddPlan) -> _PreparedAdd:
        key = id(plan)
        prepared = self.__prepared_add.get(key)
        if prepared is None:
            if self.__consumed_add is plan:
                raise RuntimeError("replay add plan was already consumed")
            if type(plan) is not _AddPlan:
                raise RuntimeError("invalid replay add plan")
            try:
                owner = plan._owner
                version = plan.version
            except Exception as error:
                raise RuntimeError("replay add plan was already consumed") from error
            if owner is not self.__plan_owner:
                raise RuntimeError("replay add plan has the wrong owner")
            if version != self._version:
                raise RuntimeError("stale replay add plan")
            raise RuntimeError("replay add plan was already consumed")
        self.__prepared_add.pop(key)
        self.__consumed_add = plan
        if prepared.public is not plan:
            raise RuntimeError("replay add plan identity was tampered")
        if type(plan) is not _AddPlan:
            raise RuntimeError("invalid replay add plan")
        try:
            valid = (
                plan._owner is self.__plan_owner
                and plan.version == prepared.version
                and plan.worker == prepared.worker
                and plan.row is prepared.row_view
                and _same_array_mapping(plan.row, prepared.row)
                and plan.emits_item is prepared.emits_item
                and plan.report_ready is prepared.report_ready_view
                and plan.report_ready == prepared.report_ready
            )
        except Exception as error:
            raise RuntimeError("replay add plan transaction was rejected") from error
        if not valid:
            raise RuntimeError("replay add plan was tampered")
        if prepared.version != self._version:
            raise RuntimeError("stale replay add plan")
        return prepared

    def commit_add(self, plan: _AddPlan) -> ReplayKey:
        prepared = self._take_add_plan(plan)
        writer = self.writers.get(prepared.worker)
        if writer is None:
            if prepared.new_writer is None:
                raise RuntimeError("replay add plan is missing its writer")
            writer = prepared.new_writer
            self.writers[prepared.worker] = writer
        staged = iter(prepared.new_chunks)
        if writer.current_chunk_id is None:
            writer.current_chunk_id = self._install_chunk(next(staged), 1).chunk_id
        chunk = self.chunks[writer.current_chunk_id]
        key = chunk.append(prepared.row)
        writer.suffix.append(key)
        self._inc_ref(key.chunk_id)
        if chunk.length == chunk.capacity:
            successor = self._install_chunk(next(staged), 2)
            chunk.seal(successor.chunk_id)
            self._dec_ref(chunk.chunk_id)
            writer.current_chunk_id = successor.chunk_id
        emitted: ReplayKey | None = None
        if len(writer.suffix) >= self.raw_length:
            emitted = writer.suffix[-self.raw_length]
            self._insert_item(emitted)
            if writer.online_phase == 0:
                self.online_queue.push(emitted)
        if prepared.report_ready is not None:
            ready_item = self._item_ids_by_key.get(prepared.report_ready)
            if ready_item is not None:
                self._report_pending.discard(ready_item)
        writer.online_phase = np.int64((int(writer.online_phase) + 1) % self.raw_length)
        suffix_capacity = max(self.raw_length, self.report_raw_length) - 1
        writer.retained_rows = np.int64(
            min(int(writer.retained_rows) + 1, suffix_capacity)
        )
        while len(writer.suffix) > suffix_capacity:
            self._dec_ref(writer.suffix.pop(0).chunk_id)
        writer.has_rows = True
        writer.last_is_last = bool(prepared.row["is_last"])
        self._commit_counters(
            {"inserted_rows": 1, "inserted_items": int(prepared.emits_item)}
        )
        return key

    def add(self, row: object, *, worker: int = 0) -> ReplayKey:
        return self.commit_add(self.prepare_add(row, worker=worker))

    def _resolve(self, start: ReplayKey, length: int) -> list[ReplayKey]:
        result: list[ReplayKey] = []
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
                    raise KeyError("replay sequence has no successor")
                chunk_id = chunk.successor_id
                offset = 0
        return result

    def _read(self, keys: list[ReplayKey]) -> dict[str, Array]:
        result: dict[str, list[Array]] = {}
        for key in keys:
            for name, value in self.chunks[key.chunk_id].read(key.offset, 1).items():
                result.setdefault(name, []).append(value[0])
        return {name: np.stack(values) for name, values in result.items()}

    def _raw_plan(
        self, mode: str
    ) -> tuple[ReplayBatch, list[ReplayKey], np.random.Generator, int, int, int]:
        length = self.raw_length if mode == "train" else self.report_raw_length
        queue = deque(
            ReplayKey(key.chunk_id, key.offset) for key in self.online_queue.keys
        )
        rng = np.random.default_rng(0)
        rng.bit_generator.state = copy.deepcopy(self.selector.rng.bit_generator.state)
        excluded = self._report_pending if mode == "report" else set()
        available = len(self.selector) - len(excluded)
        if available <= 0 and not (mode == "train" and queue):
            raise LookupError("replay does not contain a complete batch")
        blocked_positions = sorted(
            self.selector.indices[item_id] for item_id in excluded
        )
        sequences: list[dict[str, Array]] = []
        ids: list[Array] = []
        online = uniform = stale = 0
        for _ in range(self.sequence_shape.batch_size):
            keys: list[ReplayKey] | None = None
            if mode == "train":
                while queue:
                    candidate = queue.popleft()
                    try:
                        keys = self._resolve(candidate, length)
                    except KeyError:
                        stale += 1
                        continue
                    online += 1
                    break
            if keys is None:
                if available <= 0:
                    raise LookupError("replay does not contain a complete batch")
                index = int(rng.integers(0, available))
                for blocked in blocked_positions:
                    if blocked > index:
                        break
                    index += 1
                item_id = self.selector.keys[index]
                keys = self._resolve(self.items[item_id], length)
                uniform += 1
            sequences.append(self._read(keys))
            ids.append(np.stack([key.to_step_id() for key in keys]))
        data = {
            name: np.stack([sequence[name] for sequence in sequences])
            for name in sorted(sequences[0])
        }
        data["is_first"] = data["is_first"].copy()
        data["is_last"] = data["is_last"].copy()
        data["is_first"][:, 0] = True
        data["is_last"][:, :-1] = np.logical_or(
            data["is_last"][:, :-1], data["is_first"][:, 1:]
        )
        return (
            ReplayBatch(data, np.stack(ids).astype(np.uint8, copy=False)),
            list(queue),
            rng,
            online,
            uniform,
            stale,
        )

    def _can_resolve(self, key: ReplayKey, length: int) -> bool:
        try:
            self._resolve(key, length)
        except KeyError:
            return False
        return True

    def prepare_sample(self, mode: str) -> _SamplePlan:
        if mode not in _MODES:
            raise ValueError("replay mode must be train or report")
        stream = self.streams[mode]
        index = stream.index
        current = stream.current
        queue = [ReplayKey(key.chunk_id, key.offset) for key in self.online_queue.keys]
        rng = np.random.default_rng(0)
        rng.bit_generator.state = copy.deepcopy(self.selector.rng.bit_generator.state)
        online = uniform = stale = 0
        if current is None or index >= stream.consecutive:
            current, queue, rng, online, uniform, stale = self._raw_plan(mode)
            index = np.int64(0)
        else:
            current = ReplayBatch(current.data, current.step_ids)
        self._preflight_counters(
            {
                "sample_calls": 1,
                "sampled_sequences": self.sequence_shape.batch_size,
                "online_samples": online,
                "uniform_samples": uniform,
                "stale_online": stale,
            }
        )
        start = int(index) * stream.sequence_length
        stop = start + stream.sequence_length + stream.context
        ids = current.step_ids[:, start:stop]
        data = {name: value[:, start:stop] for name, value in current.data.items()}
        data["consec"] = np.full(ids.shape[:2], index, np.int32)
        batch = ReplayBatch(data, ids)
        batch_view = ReplayBatch(batch.data, batch.step_ids)
        private_queue = [ReplayKey(key.chunk_id, key.offset) for key in queue]
        queue_view = tuple(ReplayKey(key.chunk_id, key.offset) for key in private_queue)
        next_index = _advance_counter(index, 1, "ConsecutiveStream index")
        plan = _SamplePlan(
            self.__plan_owner,
            self._version,
            mode,
            batch_view,
            next_index,
            queue_view,
            online,
            uniform,
            stale,
        )
        if self.__prepared_sample:
            self.__consumed_sample = next(iter(self.__prepared_sample.values())).public
            self.__prepared_sample.clear()
        self.__prepared_sample[id(plan)] = _PreparedSample(
            plan,
            self._version,
            mode,
            batch_view,
            batch,
            current,
            next_index,
            private_queue,
            queue_view,
            rng,
            online,
            uniform,
            stale,
        )
        return plan

    def _take_sample_plan(self, plan: _SamplePlan) -> _PreparedSample:
        key = id(plan)
        prepared = self.__prepared_sample.get(key)
        if prepared is None:
            if self.__consumed_sample is plan:
                raise RuntimeError("replay sample plan was already consumed")
            if type(plan) is not _SamplePlan:
                raise RuntimeError("invalid replay sample plan")
            try:
                owner = plan._owner
                version = plan.version
            except Exception as error:
                raise RuntimeError("replay sample plan was already consumed") from error
            if owner is not self.__plan_owner:
                raise RuntimeError("replay sample plan has the wrong owner")
            if version != self._version:
                raise RuntimeError("stale replay sample plan")
            raise RuntimeError("replay sample plan was already consumed")
        self.__prepared_sample.pop(key)
        self.__consumed_sample = plan
        if prepared.public is not plan:
            raise RuntimeError("replay sample plan identity was tampered")
        if type(plan) is not _SamplePlan:
            raise RuntimeError("invalid replay sample plan")
        try:
            valid = (
                plan._owner is self.__plan_owner
                and plan.version == prepared.version
                and plan.mode == prepared.mode
                and plan.batch is prepared.batch_view
                and _same_batch(plan.batch, prepared.batch)
                and plan.index == prepared.index
                and plan.queue is prepared.queue_view
                and len(plan.queue) == len(prepared.queue)
                and all(
                    public == private
                    for public, private in zip(plan.queue, prepared.queue, strict=True)
                )
                and plan.online == prepared.online
                and plan.uniform == prepared.uniform
                and plan.stale == prepared.stale
            )
        except Exception as error:
            raise RuntimeError("replay sample plan transaction was rejected") from error
        if not valid:
            raise RuntimeError("replay sample plan was tampered")
        if prepared.version != self._version:
            raise RuntimeError("stale replay sample plan")
        return prepared

    def commit_sample(self, plan: _SamplePlan) -> ReplayBatch:
        prepared = self._take_sample_plan(plan)
        self.online_queue.keys = prepared.queue
        self.selector.rng = prepared.rng
        stream = self.streams[prepared.mode]
        stream.current = prepared.current
        stream.index = prepared.index
        self._commit_counters(
            {
                "sample_calls": 1,
                "sampled_sequences": self.sequence_shape.batch_size,
                "online_samples": prepared.online,
                "uniform_samples": prepared.uniform,
                "stale_online": prepared.stale,
            }
        )
        return prepared.batch

    def sample(self, mode: str = "train") -> ReplayBatch:
        return self.commit_sample(self.prepare_sample(mode))

    def can_sample_batch(self, mode: str) -> bool:
        if mode not in _MODES:
            raise ValueError("replay mode must be train or report")
        stream = self.streams[mode]
        if stream.current is not None and stream.index < stream.consecutive:
            return True
        available = len(self.selector)
        if mode == "report":
            return available > len(self._report_pending)
        if available:
            return True
        required = self.sequence_shape.batch_size
        resolved = 0
        for key in self.online_queue.keys:
            if self._can_resolve(key, self.raw_length):
                resolved += 1
                if resolved >= required:
                    return True
        return False

    def update_context(self, step_ids: object, values: Mapping[str, object]) -> int:
        ids = np.asarray(step_ids)
        if ids.dtype != np.uint8 or ids.ndim != 3 or ids.shape[-1] != 20:
            raise ValueError("context step ids must be uint8[B,T,20]")
        if set(values) != set(self.latent_spaces):
            raise ValueError("context update schema differs")
        leading = ids.shape[:2]
        converted = {
            name: _array(values[name], space, leading, f"context {name}")
            for name, space in self.latent_spaces.items()
        }
        plans: list[tuple[list[ReplayKey], int]] = []
        stale = 0
        for batch_index, encoded in enumerate(ids):
            supplied = [ReplayKey.from_step_id(value) for value in encoded]
            if supplied[0].chunk_id not in self.chunks:
                stale += 1
                continue
            resolved = self._resolve(supplied[0], len(supplied))
            if resolved != supplied:
                raise ValueError("context step ids do not follow replay chronology")
            plans.append((resolved, batch_index))
        updated = sum(len(keys) for keys, _ in plans)
        self._preflight_counters(
            {
                "update_calls": 1,
                "updated_rows": updated,
                "stale_updates": stale,
            }
        )
        for keys, batch_index in plans:
            for time_index, key in enumerate(keys):
                self.chunks[key.chunk_id].update_context(
                    key.offset,
                    {
                        name: value[batch_index, time_index]
                        for name, value in converted.items()
                    },
                )
        self._commit_counters(
            {
                "update_calls": 1,
                "updated_rows": updated,
                "stale_updates": stale,
            }
        )
        return updated

    def stats(self, *, reset: bool = False) -> dict[str, int | float]:
        result: dict[str, int | float] = {
            name: int(value) for name, value in self._metrics.items()
        }
        result.update(
            {
                "chunks": len(self.chunks),
                "online_queue": len(self.online_queue),
                "size": len(self.items),
            }
        )
        if reset:
            next_version = _advance_counter(self._version, 1, "Replay version")
            self._metrics = {name: np.int64(0) for name in self._metrics}
            self._version = next_version
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "chunks": [self.chunks[key].state_dict() for key in sorted(self.chunks)],
            "config": self.config.state_dict(),
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
            "sequence_shape": self.sequence_shape.state_dict(),
            "spaces": {
                "latent": _space_state(self.latent_spaces),
                "transition": _space_state(self.transition_spaces),
            },
            "streams": {mode: self.streams[mode].state_dict() for mode in _MODES},
            "version": self._version,
            "writers": {
                worker: writer.state_dict()
                for worker, writer in sorted(self.writers.items())
            },
        }

    @classmethod
    def from_state_dict(
        cls,
        state: object,
        replay_config: ReplayConfig,
        sequence_shape: SequenceShapeConfig,
        transition_spaces: Mapping[str, TensorSpace],
        latent_spaces: Mapping[str, TensorSpace],
    ) -> DreamerReplay:
        _check_plain_tree(state, "DreamerReplay state")
        keys = {
            "chunks",
            "config",
            "fifo",
            "items",
            "metrics",
            "next_chunk_id",
            "next_item_id",
            "online_queue",
            "refs",
            "schema_version",
            "selector",
            "sequence_shape",
            "spaces",
            "streams",
            "version",
            "writers",
        }
        record = _exact_mapping(state, keys, "DreamerReplay state")
        if (
            type(record["schema_version"]) is not int
            or record["schema_version"] != cls.SCHEMA_VERSION
        ):
            raise ValueError("DreamerReplay schema version differs")
        restored_config = ReplayConfig.from_state(record["config"])
        if restored_config != replay_config:
            raise ValueError("DreamerReplay config differs")
        restored_sequence_shape = SequenceShapeConfig.from_state(
            record["sequence_shape"]
        )
        if restored_sequence_shape != sequence_shape:
            raise ValueError("DreamerReplay sequence shape differs")
        spaces = _exact_mapping(
            record["spaces"], {"latent", "transition"}, "DreamerReplay spaces"
        )
        _validate_space_state(
            spaces["latent"], latent_spaces, "DreamerReplay latent spaces"
        )
        _validate_space_state(
            spaces["transition"],
            transition_spaces,
            "DreamerReplay transition spaces",
        )
        result = cls(replay_config, sequence_shape, transition_spaces, latent_spaces)
        if type(record["chunks"]) is not list:
            raise TypeError("DreamerReplay chunks must be a plain list")
        result.chunks = {}
        for item in record["chunks"]:
            chunk_record = _exact_mapping(
                item,
                {
                    "capacity",
                    "chunk_id",
                    "latent",
                    "length",
                    "owner_id",
                    "successor",
                    "transition",
                },
                "ReplayChunk state",
            )
            if chunk_record["capacity"] != replay_config.chunk_size:
                raise ValueError("ReplayChunk capacity differs from replay config")
            chunk = ReplayChunk.from_state_dict(item, transition_spaces, latent_spaces)
            if chunk.chunk_id in result.chunks:
                raise ValueError("duplicate DreamerReplay chunk")
            result.chunks[chunk.chunk_id] = chunk
        refs = record["refs"]
        if type(refs) is not dict:
            raise TypeError("DreamerReplay refs must be a plain dictionary")
        result.refs = dict(refs)
        writers = record["writers"]
        if type(writers) is not dict:
            raise TypeError("DreamerReplay writers must be a plain dictionary")
        result.writers = {
            worker: ReplayWriter.from_state_dict(item, result.raw_length)
            for worker, item in writers.items()
        }
        if type(record["items"]) is not list:
            raise TypeError("DreamerReplay items must be a plain list")
        result.items = {}
        for item in record["items"]:
            item_record = _exact_mapping(item, {"item_id", "key"}, "replay item")
            item_id = item_record["item_id"]
            if (
                type(item_id) is not int
                or not 0 <= item_id <= _MAX_ITEM_ID
                or item_id in result.items
            ):
                raise ValueError("invalid replay item id")
            result.items[item_id] = ReplayKey.from_state_dict(item_record["key"])
        if type(record["fifo"]) is not list:
            raise TypeError("DreamerReplay FIFO must be a plain list")
        result.fifo = deque(record["fifo"])
        result.next_chunk_id = record["next_chunk_id"]
        result.next_item_id = record["next_item_id"]
        result.online_queue = OnlineQueue.from_state_dict(record["online_queue"])
        result.selector = UniformSelector.from_state_dict(record["selector"])
        stream_states = _exact_mapping(record["streams"], set(_MODES), "replay streams")
        result.streams = {
            "train": ConsecutiveStream.from_state_dict(
                stream_states["train"],
                transition_spaces,
                latent_spaces,
                sequence_shape.batch_size,
                sequence_shape.sequence_length,
                sequence_shape.consecutive,
                sequence_shape.context,
            ),
            "report": ConsecutiveStream.from_state_dict(
                stream_states["report"],
                transition_spaces,
                latent_spaces,
                sequence_shape.batch_size,
                sequence_shape.report_length,
                sequence_shape.report_consecutive,
                sequence_shape.context,
            ),
        }
        metrics = record["metrics"]
        if type(metrics) is not dict or set(metrics) != set(result._metrics):
            raise ValueError("DreamerReplay metrics differ")
        result._metrics = {
            name: _counter(value, f"DreamerReplay metric {name}")
            for name, value in metrics.items()
        }
        result._version = _counter(record["version"], "DreamerReplay version")
        result._rebuild_runtime_indices()
        result.validate()
        return result

    def _rebuild_runtime_indices(self) -> None:
        self._item_ids_by_key = {}
        for item_id, key in self.items.items():
            if key in self._item_ids_by_key:
                raise ValueError("DreamerReplay item start keys must be unique")
            self._item_ids_by_key[key] = item_id
        self._report_pending = {
            item_id
            for item_id, key in self.items.items()
            if not self._can_resolve(key, self.report_raw_length)
        }

    def _recomputed_refs(self) -> dict[bytes, np.int64]:
        refs = {chunk_id: np.int64(0) for chunk_id in self.chunks}
        for chunk in self.chunks.values():
            if chunk.successor_id is not None:
                if chunk.successor_id not in refs:
                    raise ValueError("DreamerReplay chunk successor is missing")
                refs[chunk.successor_id] = _advance_counter(
                    refs[chunk.successor_id], 1, "Replay reference count"
                )
        for writer in self.writers.values():
            if writer.current_chunk_id is not None:
                if writer.current_chunk_id not in refs:
                    raise ValueError("DreamerReplay writer chunk is missing")
                refs[writer.current_chunk_id] = _advance_counter(
                    refs[writer.current_chunk_id], 1, "Replay reference count"
                )
            for key in writer.suffix:
                if key.chunk_id not in refs:
                    raise ValueError("DreamerReplay writer suffix is missing")
                refs[key.chunk_id] = _advance_counter(
                    refs[key.chunk_id], 1, "Replay reference count"
                )
        for key in self.items.values():
            if key.chunk_id not in refs:
                raise ValueError("DreamerReplay item chunk is missing")
            refs[key.chunk_id] = _advance_counter(
                refs[key.chunk_id], 1, "Replay reference count"
            )
        return refs

    def _validate_stored_key_domain(self, key: ReplayKey, label: str) -> None:
        if int.from_bytes(key.chunk_id, "big") >= self.next_chunk_id:
            raise ValueError(f"{label} chunk was not previously allocated")
        if key.offset >= self.config.chunk_size:
            raise ValueError(f"{label} offset exceeds replay chunk geometry")

    def _validate_online_queue(self) -> None:
        if len(set(self.online_queue.keys)) != len(self.online_queue.keys):
            raise ValueError("DreamerReplay online queue contains duplicate keys")
        for key in self.online_queue.keys:
            self._validate_stored_key_domain(key, "DreamerReplay online queue key")
            if key.chunk_id not in self.chunks:
                continue
            try:
                resolved = self._resolve(key, self.raw_length)
            except KeyError as error:
                raise ValueError(
                    "DreamerReplay live online queue key cannot resolve"
                ) from error
            if len(resolved) != self.raw_length or resolved[0] != key:
                raise ValueError("DreamerReplay live online queue key differs")

    def _validate_retained_streams(self) -> None:
        for mode, stream in self.streams.items():
            if stream.current is None:
                continue
            for batch_index, encoded in enumerate(stream.current.step_ids):
                keys = [ReplayKey.from_step_id(step_id) for step_id in encoded]
                for key in keys:
                    self._validate_stored_key_domain(
                        key, f"DreamerReplay {mode} stream step id"
                    )
                for previous, key in zip(keys, keys[1:], strict=False):
                    if key.chunk_id == previous.chunk_id:
                        valid = key.offset == previous.offset + 1
                    else:
                        valid = (
                            previous.offset == self.config.chunk_size - 1
                            and key.offset == 0
                            and int.from_bytes(key.chunk_id, "big")
                            > int.from_bytes(previous.chunk_id, "big")
                        )
                    if not valid:
                        raise ValueError(
                            f"DreamerReplay {mode} stream chronology differs"
                        )
                first_live = next(
                    (
                        index
                        for index, key in enumerate(keys)
                        if key.chunk_id in self.chunks
                    ),
                    None,
                )
                if first_live is None:
                    continue
                try:
                    resolved = self._resolve(keys[first_live], len(keys) - first_live)
                except KeyError as error:
                    raise ValueError(
                        f"DreamerReplay live {mode} stream suffix cannot resolve"
                    ) from error
                if resolved != keys[first_live:]:
                    raise ValueError(
                        f"DreamerReplay live {mode} stream chronology differs"
                    )
                for position in range(first_live, len(keys)):
                    key = keys[position]
                    chunk = self.chunks[key.chunk_id]
                    for name in self.transition_spaces:
                        expected = chunk.transition_data[name][key.offset]
                        if name == "is_first" and position == 0:
                            expected = np.asarray(True, np.bool_)
                        elif name == "is_last" and position < len(keys) - 1:
                            next_key = keys[position + 1]
                            next_first = self.chunks[next_key.chunk_id].transition_data[
                                "is_first"
                            ][next_key.offset]
                            expected = np.logical_or(expected, next_first)
                        actual = stream.current.data[name][batch_index, position]
                        if (
                            np.asarray(actual).tobytes()
                            != np.asarray(expected).tobytes()
                        ):
                            raise ValueError(
                                f"DreamerReplay live {mode} stream transition "
                                f"{name!r} differs"
                            )

    def _validate_writer_local_item_order(self) -> None:
        previous: dict[int, ReplayKey] = {}
        for item_id in sorted(self.items):
            key = self.items[item_id]
            owner = self.chunks[key.chunk_id].owner_id
            prior = previous.get(owner)
            if prior is not None:
                try:
                    expected = self._resolve(prior, 2)[1]
                except KeyError as error:
                    raise ValueError(
                        "DreamerReplay retained item start has no next row"
                    ) from error
                if key != expected:
                    raise ValueError(
                        "DreamerReplay writer-local item starts are not consecutive"
                    )
            previous[owner] = key

    def validate(self) -> None:
        if (
            type(self.next_chunk_id) is not int
            or not 1 <= self.next_chunk_id <= _CHUNK_SENTINEL
        ):
            raise ValueError("invalid DreamerReplay chunk cursor")
        if (
            type(self.next_item_id) is not int
            or not 0 <= self.next_item_id <= _ITEM_SENTINEL
        ):
            raise ValueError("invalid DreamerReplay item cursor")
        if any(
            type(item_id) is not int or not 0 <= item_id <= _MAX_ITEM_ID
            for item_id in self.items
        ):
            raise ValueError("invalid DreamerReplay item identity")
        if any(
            type(item_id) is not int or not 0 <= item_id <= _MAX_ITEM_ID
            for item_id in self.fifo
        ):
            raise ValueError("invalid DreamerReplay FIFO identity")
        if list(self.fifo) != list(self.items):
            raise ValueError("DreamerReplay FIFO and items differ")
        if not self.items and (
            self.next_item_id != 0
            or self.online_queue.keys
            or any(stream.current is not None for stream in self.streams.values())
            or self.selector.rng.bit_generator.state
            != np.random.default_rng(0).bit_generator.state
        ):
            raise ValueError(
                "DreamerReplay empty item owner must have a fresh cursor and no "
                "online queue keys, active streams, or advanced selector RNG"
            )
        expected_item_ids = list(
            range(self.next_item_id - len(self.items), self.next_item_id)
        )
        if list(self.fifo) != expected_item_ids:
            raise ValueError(
                "DreamerReplay item identities are not a contiguous suffix"
            )
        if set(self.selector.keys) != set(self.items):
            raise ValueError("DreamerReplay selector and items differ")
        if len(self.items) > self.config.capacity:
            raise ValueError("DreamerReplay exceeds capacity")
        if set(self.refs) != set(self.chunks) or self.refs != self._recomputed_refs():
            raise ValueError("DreamerReplay reference counts differ")
        if any(
            count <= 0 or type(count) is not np.int64 for count in self.refs.values()
        ):
            raise ValueError("invalid DreamerReplay reference count")
        predecessors = {chunk_id: 0 for chunk_id in self.chunks}
        for chunk_id, chunk in self.chunks.items():
            if chunk.capacity != self.config.chunk_size:
                raise ValueError("DreamerReplay chunk capacity differs")
            if type(chunk.owner_id) is not int or chunk.owner_id not in self.writers:
                raise ValueError("DreamerReplay chunk owner is missing")
            if chunk.successor_id is not None:
                successor = self.chunks.get(chunk.successor_id)
                if successor is None:
                    raise ValueError("DreamerReplay chunk successor is missing")
                if successor.owner_id != chunk.owner_id:
                    raise ValueError("DreamerReplay successor owner differs")
                if int.from_bytes(chunk.successor_id, "big") <= int.from_bytes(
                    chunk_id, "big"
                ):
                    raise ValueError("DreamerReplay chunk links must increase")
                predecessors[chunk.successor_id] += 1
                if predecessors[chunk.successor_id] > 1:
                    raise ValueError(
                        "DreamerReplay chunk successor has multiple parents"
                    )
                if (
                    chunk.length
                    and successor.length
                    and bool(chunk.transition_data["is_last"][chunk.length - 1])
                    and not bool(successor.transition_data["is_first"][0])
                ):
                    raise ValueError("DreamerReplay row after last must be first")
            terminal = chunk.transition_data["is_terminal"][: chunk.length]
            last = chunk.transition_data["is_last"][: chunk.length]
            if np.any(terminal & ~last):
                raise ValueError("DreamerReplay terminal row must be last")
            for index in np.flatnonzero(last):
                if any(
                    np.any(chunk.transition_data[name][index] != 0)
                    for name in self._action_keys
                ):
                    raise ValueError("DreamerReplay last row action must be zero")
            if chunk.length > 1:
                previous_last = last[:-1]
                next_first = chunk.transition_data["is_first"][1 : chunk.length]
                if np.any(previous_last & ~next_first):
                    raise ValueError("DreamerReplay row after last must be first")
        live_items_by_owner = {worker: 0 for worker in self.writers}
        for key in self.items.values():
            live_items_by_owner[self.chunks[key.chunk_id].owner_id] += 1
        expected_chunk_cursor = (
            max(int.from_bytes(chunk_id, "big") for chunk_id in self.chunks) + 1
            if self.chunks
            else 1
        )
        if self.next_chunk_id != expected_chunk_cursor:
            raise ValueError("DreamerReplay chunk cursor is not the exact successor")
        for worker, writer in self.writers.items():
            if type(worker) is not int or worker != writer.worker_id:
                raise ValueError("DreamerReplay writer identity differs")
            if not writer.has_rows or writer.current_chunk_id is None:
                raise ValueError("DreamerReplay published writer has no rows")
            suffix_capacity = max(self.raw_length, self.report_raw_length) - 1
            if len(writer.suffix) > suffix_capacity:
                raise ValueError("DreamerReplay writer suffix is unbounded")
            if (
                type(writer.retained_rows) is not np.int64
                or not 0 <= int(writer.retained_rows) <= suffix_capacity
            ):
                raise ValueError("DreamerReplay writer retained rows are invalid")
            if len(writer.suffix) != int(writer.retained_rows):
                raise ValueError("DreamerReplay writer retained rows differ")
            live_items = live_items_by_owner[worker]
            if live_items and int(writer.retained_rows) < min(
                live_items + self.raw_length - 1, suffix_capacity
            ):
                raise ValueError(
                    "DreamerReplay writer retained rows are below the live-item "
                    "lower bound"
                )
            current = self.chunks.get(writer.current_chunk_id)
            if current is None or current.owner_id != worker or current.sealed:
                raise ValueError("DreamerReplay writer current chunk is invalid")
            open_chunks = [
                chunk
                for chunk in self.chunks.values()
                if chunk.owner_id == worker and not chunk.sealed
            ]
            if open_chunks != [current]:
                raise ValueError("DreamerReplay writer must own one open chunk")
            if writer.suffix:
                if self._resolve(writer.suffix[0], len(writer.suffix)) != writer.suffix:
                    raise ValueError("DreamerReplay writer suffix is not consecutive")
                if any(
                    self.chunks[key.chunk_id].owner_id != worker
                    for key in writer.suffix
                ):
                    raise ValueError("DreamerReplay writer suffix owner differs")
            tail: ReplayKey | None = None
            if current.length:
                tail = ReplayKey(current.chunk_id, int(current.length) - 1)
            else:
                parents = [
                    chunk
                    for chunk in self.chunks.values()
                    if chunk.successor_id == current.chunk_id
                ]
                if len(parents) > 1:
                    raise ValueError("DreamerReplay current chunk has multiple parents")
                if parents:
                    tail = ReplayKey(parents[0].chunk_id, int(parents[0].length) - 1)
            if writer.suffix and writer.suffix[-1] != tail:
                raise ValueError("DreamerReplay writer suffix does not end at tail")
            if suffix_capacity and not writer.suffix:
                raise ValueError("DreamerReplay writer is missing its retained tail")
            if len(writer.suffix) < suffix_capacity and writer.online_phase != np.int64(
                len(writer.suffix) % self.raw_length
            ):
                raise ValueError("DreamerReplay early writer phase differs from rows")
            if tail is not None:
                tail_chunk = self.chunks[tail.chunk_id]
                tail_is_last = bool(tail_chunk.transition_data["is_last"][tail.offset])
                if writer.last_is_last != tail_is_last:
                    raise ValueError(
                        "DreamerReplay writer boundary flag differs from tail"
                    )
        for item in self.items.values():
            self._resolve(item, self.raw_length)
        self._validate_writer_local_item_order()
        expected_by_key = {key: item_id for item_id, key in self.items.items()}
        if self._item_ids_by_key != expected_by_key:
            raise ValueError("DreamerReplay item reverse index differs")
        expected_pending = {
            item_id
            for item_id, key in self.items.items()
            if not self._can_resolve(key, self.report_raw_length)
        }
        if self._report_pending != expected_pending:
            raise ValueError("DreamerReplay report eligibility differs")
        if self.online_queue.maxlen != self.config.online_queue_size:
            raise ValueError("DreamerReplay online queue size differs")
        self._validate_online_queue()
        self._validate_retained_streams()


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
