from __future__ import annotations

import copy
import gc
import inspect
from argparse import Namespace
from collections.abc import Callable, Mapping
from collections import deque
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn, cast

import numpy as np
import numpy.typing as npt
import pytest

from world_marl.dreamer_v3_baseline.config import (
    DreamerProfile,
    ObservationMode,
    ReplayConfig,
    SequenceShapeConfig,
)
from world_marl.dreamer_v3_baseline.fixture_generator import _parse_args
from world_marl.dreamer_v3_baseline.networks import TensorSpace
from world_marl.dreamer_v3_baseline.oracle import OracleManifest
from world_marl.dreamer_v3_baseline.replay import (
    ConsecutiveStream,
    DreamerReplay,
    OnlineQueue,
    ReplayBatch,
    ReplayChunk,
    ReplayKey,
    ReplayWriter,
    UniformSelector,
)


Array = npt.NDArray[np.generic]


class _PCG64String(str):
    pass


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "dreamer_v3"
OFFICIAL_CHECKOUT = Path("/private/tmp/danijar-dreamerv3-20260713")
REVISIONS = {
    DreamerProfile.PAPER: "bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01",
    DreamerProfile.UPSTREAM_CURRENT: "e3f02248693a79dc8b0ebd62c93683888ddaccfe",
}


def _transition_spaces() -> dict[str, TensorSpace]:
    return {
        "action": TensorSpace((2,), "float32"),
        "is_first": TensorSpace((), "bool"),
        "is_last": TensorSpace((), "bool"),
        "is_terminal": TensorSpace((), "bool"),
        "obs": TensorSpace((1,), "float32"),
        "reward": TensorSpace((), "float32"),
    }


def _latent_spaces() -> dict[str, TensorSpace]:
    return {
        "dyn/deter": TensorSpace((2,), "float32"),
        "dyn/stoch": TensorSpace((1, 2), "float32"),
    }


def _sequence(
    *,
    batch: int = 1,
    length: int = 2,
    context: int = 1,
    consecutive: int = 1,
    report_length: int = 2,
    report_consecutive: int = 1,
) -> SequenceShapeConfig:
    return SequenceShapeConfig(
        batch_size=batch,
        sequence_length=length,
        context=context,
        consecutive=consecutive,
        report_length=report_length,
        report_consecutive=report_consecutive,
    )


def _replay(
    *,
    capacity: int = 20,
    chunk_size: int = 3,
    online_queue_size: int = 20,
    sequence: SequenceShapeConfig | None = None,
) -> DreamerReplay:
    return DreamerReplay(
        ReplayConfig(
            capacity=capacity,
            chunk_size=chunk_size,
            online_queue_size=online_queue_size,
        ),
        sequence or _sequence(),
        _transition_spaces(),
        _latent_spaces(),
    )


def _row(
    value: int,
    *,
    first: bool = False,
    last: bool = False,
    terminal: bool = False,
    action: tuple[float, float] | None = None,
) -> dict[str, object]:
    if action is None:
        action = (0.0, 0.0) if last else (value + 1.25, -value - 1.5)
    return {
        "action": np.asarray(action, np.float32),
        "dyn/deter": np.asarray([100 + value, 200 + value], np.float32),
        "dyn/stoch": np.asarray([[300 + value, 400 + value]], np.float32),
        "is_first": first,
        "is_last": last,
        "is_terminal": terminal,
        "obs": np.asarray([value], np.float32),
        "reward": np.float32(value),
    }


def _add(replay: DreamerReplay, count: int, *, worker: int = 0) -> list[ReplayKey]:
    return [
        replay.add(_row(index, first=index == 0), worker=worker)
        for index in range(count)
    ]


def _tree_equal(left: object, right: object) -> None:
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _tree_equal(left[key], right[key])
    elif isinstance(left, list):
        assert isinstance(right, list)
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right, strict=True):
            _tree_equal(lhs, rhs)
    else:
        assert left == right


def _restore(replay: DreamerReplay, state: object) -> DreamerReplay:
    return DreamerReplay.from_state_dict(
        state,
        replay.config,
        replay.sequence_shape,
        _transition_spaces(),
        _latent_spaces(),
    )


def _replace_chunk_id(value: object, old: bytes, new: bytes) -> object:
    if type(value) is dict:
        return {
            new if key == old else key: _replace_chunk_id(item, old, new)
            for key, item in value.items()
        }
    if type(value) is list:
        return [_replace_chunk_id(item, old, new) for item in value]
    return new if type(value) is bytes and value == old else value


def _replace_item_id(state: dict[str, object], new: object) -> None:
    items = cast(list[dict[str, object]], state["items"])
    old = items[0]["item_id"]
    items[0]["item_id"] = new
    state["fifo"] = [
        new if item == old else item for item in cast(list[object], state["fifo"])
    ]
    selector = cast(dict[str, object], state["selector"])
    selector["keys"] = [
        new if item == old else item for item in cast(list[object], selector["keys"])
    ]
    indices = cast(dict[object, object], selector["indices"])
    selector["indices"] = {
        new if item == old else item: index for item, index in indices.items()
    }


def _recompute_serialized_refs(state: dict[str, object]) -> None:
    chunks = cast(list[dict[str, object]], state["chunks"])
    refs = {cast(bytes, chunk["chunk_id"]): np.int64(0) for chunk in chunks}
    for chunk in chunks:
        successor = chunk["successor"]
        if successor is not None:
            refs[cast(bytes, successor)] += 1
    writers = cast(dict[object, dict[str, object]], state["writers"])
    for writer in writers.values():
        current = writer["current_chunk_id"]
        if current is not None:
            refs[cast(bytes, current)] += 1
        for item in cast(list[dict[str, object]], writer["suffix"]):
            refs[cast(bytes, item["chunk_id"])] += 1
    for item in cast(list[dict[str, object]], state["items"]):
        key = cast(dict[str, object], item["key"])
        refs[cast(bytes, key["chunk_id"])] += 1
    state["refs"] = refs


def _value_at(replay: DreamerReplay, key: ReplayKey) -> int:
    return int(replay.chunks[key.chunk_id].read(key.offset, 1)["obs"][0, 0])


def _ordinary_counter_paths(state: dict[str, object]) -> list[tuple[object, ...]]:
    paths: list[tuple[object, ...]] = []
    for index, _ in enumerate(cast(list[object], state["chunks"])):
        paths.append(("chunks", index, "length"))
    for chunk_id in cast(dict[bytes, object], state["refs"]):
        paths.append(("refs", chunk_id))
    for mode in ("train", "report"):
        paths.append(("streams", mode, "index"))
    for worker in cast(dict[object, object], state["writers"]):
        paths.append(("writers", worker, "online_phase"))
        paths.append(("writers", worker, "retained_rows"))
    for name in cast(dict[str, object], state["metrics"]):
        paths.append(("metrics", name))
    paths.append(("version",))
    return paths


def _path_value(state: object, path: tuple[object, ...]) -> object:
    current = state
    for part in path:
        current = cast(Any, current)[part]
    return current


def _set_path(state: object, path: tuple[object, ...], value: object) -> None:
    current = state
    for part in path[:-1]:
        current = cast(Any, current)[part]
    cast(Any, current)[path[-1]] = value


def _active_stream_replay(mode: str, *, partial_live: bool) -> DreamerReplay:
    sequence = _sequence(
        length=1,
        context=0,
        consecutive=2,
        report_length=1,
        report_consecutive=2,
    )
    replay = _replay(capacity=1, chunk_size=2, sequence=sequence)
    _add(replay, 3)
    replay.sample(mode)
    if partial_live:
        replay.add(_row(3))
    current = replay.streams[mode].current
    assert current is not None
    keys = [ReplayKey.from_step_id(value) for value in current.step_ids[0]]
    assert (keys[0].chunk_id in replay.chunks) is not partial_live
    assert keys[1].chunk_id in replay.chunks
    return replay


def _serialized_stream_current(
    state: dict[str, object], mode: str
) -> dict[str, object]:
    streams = cast(dict[str, dict[str, object]], state["streams"])
    current = streams[mode]["current"]
    assert type(current) is dict
    return cast(dict[str, object], current)


def _replace_stream_step_key(
    current: dict[str, object], position: int, key: ReplayKey
) -> None:
    step_ids = cast(Array, current["step_ids"])
    step_ids[0, position] = key.to_step_id()


def _restore_with_spaces(
    replay: DreamerReplay,
    state: object,
    transition_spaces: Mapping[str, TensorSpace],
    latent_spaces: Mapping[str, TensorSpace],
) -> DreamerReplay:
    return DreamerReplay.from_state_dict(
        state,
        replay.config,
        replay.sequence_shape,
        transition_spaces,
        latent_spaces,
    )


_SPACE_IDENTITY_CASES = (
    ("continuous", TensorSpace((2,), "float32"), None),
    ("scalar_discrete", TensorSpace((), "int32", classes=3), 3),
    ("uniform_vector", TensorSpace((2,), "int32", classes=3), [3, 3]),
    (
        "uniform_multidimensional",
        TensorSpace((2, 2), "int32", classes=4),
        [4, 4, 4, 4],
    ),
    (
        "nonuniform_vector",
        TensorSpace((2,), "int32", classes=(2, 3)),
        [2, 3],
    ),
    (
        "nonuniform_multidimensional",
        TensorSpace((2, 2), "int32", classes=cast(Any, ((2, 3), (4, 5)))),
        [2, 3, 4, 5],
    ),
)


def _space_identity_replay(
    owner: str, space: TensorSpace
) -> tuple[
    DreamerReplay,
    dict[str, TensorSpace],
    dict[str, TensorSpace],
    dict[str, object],
]:
    transition_spaces = _transition_spaces()
    latent_spaces = _latent_spaces()
    name = f"{owner}/probe"
    if owner == "transition":
        transition_spaces[name] = space
    else:
        latent_spaces[name] = space
    replay = DreamerReplay(
        ReplayConfig(capacity=4, chunk_size=2, online_queue_size=4),
        _sequence(length=1, context=0),
        transition_spaces,
        latent_spaces,
    )
    row = _row(0, first=True)
    row[name] = np.zeros(space.shape, np.dtype(space.dtype))
    replay.add(row)
    return replay, transition_spaces, latent_spaces, row


def _assert_no_tuple(value: object) -> None:
    assert type(value) is not tuple
    if type(value) is dict:
        for item in value.values():
            _assert_no_tuple(item)
    elif type(value) is list:
        for item in value:
            _assert_no_tuple(item)


def test_public_constructor_and_inverse_signatures_are_closed() -> None:
    assert list(inspect.signature(DreamerReplay).parameters) == [
        "config",
        "sequence_shape",
        "transition_spaces",
        "latent_spaces",
    ]
    assert list(inspect.signature(DreamerReplay.from_state_dict).parameters) == [
        "state",
        "replay_config",
        "sequence_shape",
        "transition_spaces",
        "latent_spaces",
    ]
    assert list(inspect.signature(ReplayBatch.from_state).parameters) == [
        "state",
        "transition_spaces",
        "latent_spaces",
        "expected_batch_size",
        "expected_time_length",
    ]


@pytest.mark.parametrize(
    ("owner", "name"),
    [(ReplayBatch, "__getitem__"), (DreamerReplay, "sample_raw")],
)
def test_replay_omits_undeclared_dead_convenience_apis(
    owner: type[object], name: str
) -> None:
    assert name not in owner.__dict__


def test_raw_lengths_come_only_from_sequence_shape_and_round_trip() -> None:
    sequence = _sequence(
        length=3,
        context=2,
        consecutive=2,
        report_length=2,
        report_consecutive=2,
    )
    replay = _replay(sequence=sequence)
    assert replay.raw_length == 8
    assert replay.report_raw_length == 6
    restored = DreamerReplay.from_state_dict(
        replay.state_dict(),
        replay.config,
        sequence,
        _transition_spaces(),
        _latent_spaces(),
    )
    assert restored.raw_length == 8
    assert restored.report_raw_length == 6


def test_mode_raw_lengths_differ_from_returned_trimmed_lengths() -> None:
    sequence = _sequence(
        length=3,
        context=2,
        consecutive=2,
        report_length=2,
        report_consecutive=3,
    )
    replay = _replay(sequence=sequence)
    _add(replay, 10)
    assert replay.raw_length == 8
    assert replay.report_raw_length == 8
    assert replay.sample("train").step_ids.shape[1] == 5
    assert replay.sample("report").step_ids.shape[1] == 4


@pytest.mark.parametrize(
    ("last", "terminal", "action", "accepted"),
    [
        (False, False, (4.5, -3.0), True),
        (True, True, (0.0, 0.0), True),
        (True, False, (0.0, 0.0), True),
        (True, True, (0.0, 1.0), False),
        (True, False, (-1.0, 0.0), False),
    ],
)
def test_action_is_next_call_unbounded_and_final_zero_is_transactional(
    last: bool,
    terminal: bool,
    action: tuple[float, float],
    accepted: bool,
) -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    row = _row(0, first=True, last=last, terminal=terminal, action=action)
    row_before = copy.deepcopy(row)
    before = replay.state_dict()
    if accepted:
        key = replay.add(row)
        np.testing.assert_array_equal(
            replay.chunks[key.chunk_id].read(key.offset, 1)["action"][0], action
        )
    else:
        with pytest.raises(ValueError, match="final replay row action must be zero"):
            replay.add(row)
        _tree_equal(replay.state_dict(), before)
    _tree_equal(row, row_before)


def test_row_boundary_chronology_is_transactional() -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    before = replay.state_dict()
    with pytest.raises(ValueError, match="first row"):
        replay.add(_row(0, first=False))
    _tree_equal(replay.state_dict(), before)


def test_transition_row_table_preserves_action_and_auto_reset_order() -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    rows = [
        _row(0, first=True, action=(1.0, 2.0)),
        _row(1, last=True, terminal=True, action=(0.0, 0.0)),
        _row(2, first=True, action=(3.0, 4.0)),
        _row(3, last=True, terminal=False, action=(0.0, 0.0)),
        _row(4, first=True, action=(5.0, 6.0)),
    ]
    keys = [replay.add(row) for row in rows]
    for key, expected in zip(keys, rows, strict=True):
        stored = replay.chunks[key.chunk_id].read(key.offset, 1)
        for name in _transition_spaces():
            np.testing.assert_array_equal(stored[name][0], np.asarray(expected[name]))
    assert [
        bool(replay.chunks[key.chunk_id].read(key.offset, 1)["is_first"][0])
        for key in keys
    ] == [
        True,
        False,
        True,
        False,
        True,
    ]
    replay.add(_row(0, first=True, last=True))
    before = replay.state_dict()
    with pytest.raises(ValueError, match="after is_last"):
        replay.add(_row(1, first=False))
    _tree_equal(replay.state_dict(), before)


def test_online_old_length_phase_and_global_fifo_across_writers() -> None:
    replay = _replay(sequence=_sequence(length=2, context=1))
    for index in range(8):
        replay.add(_row(index, first=index == 0), worker=1)
        replay.add(_row(100 + index, first=index == 0), worker=2)
    assert [_value_at(replay, key) for key in replay.online_queue.keys] == [
        1,
        101,
        4,
        104,
    ]
    assert replay.writers[1].online_phase == 2
    assert replay.writers[2].online_phase == 2

    every = _replay(sequence=_sequence(length=1, context=0))
    _add(every, 4)
    assert [_value_at(every, key) for key in every.online_queue.keys] == [0, 1, 2, 3]


def test_train_online_first_report_uniform_and_streams_are_independent() -> None:
    replay = _replay(
        sequence=_sequence(
            length=2,
            context=1,
            consecutive=2,
            report_length=2,
            report_consecutive=2,
        )
    )
    _add(replay, 12)
    queued = list(replay.online_queue.keys)
    train0 = replay.sample("train")
    assert train0.data["consec"].tolist() == [[0, 0, 0]]
    assert len(replay.online_queue.keys) < len(queued)
    queue_after_train = list(replay.online_queue.keys)
    report0 = replay.sample("report")
    assert report0.data["consec"].tolist() == [[0, 0, 0]]
    assert replay.online_queue.keys == queue_after_train
    train1 = replay.sample("train")
    report1 = replay.sample("report")
    assert train1.data["consec"].tolist() == [[1, 1, 1]]
    assert report1.data["consec"].tolist() == [[1, 1, 1]]


def test_readiness_is_pure_for_both_modes_and_requires_a_complete_raw_batch() -> None:
    replay = _replay(sequence=_sequence(batch=3, length=2, context=1))
    assert replay.can_sample_batch("train") is False
    assert replay.can_sample_batch("report") is False
    _add(replay, 2)
    before = replay.state_dict()
    assert replay.can_sample_batch("train") is False
    assert replay.can_sample_batch("report") is False
    _tree_equal(replay.state_dict(), before)
    replay.add(_row(2))
    before = replay.state_dict()
    assert replay.can_sample_batch("train") is True
    assert replay.can_sample_batch("report") is True
    _tree_equal(replay.state_dict(), before)
    with pytest.raises(ValueError, match="train or report"):
        replay.can_sample_batch("eval")


def test_readiness_covers_zero_all_batch_and_independent_partial_streams() -> None:
    replay = _replay(
        sequence=_sequence(
            batch=3,
            length=2,
            context=1,
            consecutive=2,
            report_length=2,
            report_consecutive=2,
        )
    )
    assert not replay.can_sample_batch("train")
    _add(replay, 5)
    assert replay.can_sample_batch("train")
    assert replay.can_sample_batch("report")
    replay.sample("train")
    before = replay.state_dict()
    assert replay.can_sample_batch("train")
    assert replay.can_sample_batch("report")
    _tree_equal(replay.state_dict(), before)
    replay.sample("report")
    assert replay.streams["train"].index == 1
    assert replay.streams["report"].index == 1


def test_readiness_never_constructs_rng_or_reads_a_batch(monkeypatch) -> None:
    replay = _replay(sequence=_sequence(batch=4, length=2, context=1))
    _add(replay, 3)
    monkeypatch.setattr(
        replay,
        "_raw_plan",
        lambda mode: (_ for _ in ()).throw(AssertionError(f"sampled {mode}")),
    )
    monkeypatch.setattr(
        replay,
        "_read",
        lambda keys: (_ for _ in ()).throw(AssertionError(f"read {keys}")),
    )
    assert replay.can_sample_batch("train")
    assert replay.can_sample_batch("report")


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("insufficient_live", False),
        ("enough_live", True),
        ("stale_does_not_count", False),
        ("selector_replacement", True),
    ],
)
def test_train_readiness_proves_every_batch_row_without_mutation(
    case: str, expected: bool
) -> None:
    sequence = _sequence(batch=2, length=1, context=0)
    replay = _replay(sequence=sequence)
    if case == "selector_replacement":
        replay.add(_row(0, first=True))
        replay.online_queue.keys.clear()
    else:
        replay.add(_row(0, first=True))
        replay.add(_row(1))
        live = list(replay.online_queue.keys)
        replay.items.clear()
        replay.fifo.clear()
        replay.selector.keys.clear()
        replay.selector.indices.clear()
        replay._item_ids_by_key.clear()
        replay._report_pending.clear()
        if case == "insufficient_live":
            replay.online_queue.keys = live[:1]
        elif case == "enough_live":
            replay.online_queue.keys = live[:2]
        else:
            replay.online_queue.keys = [
                ReplayKey((99).to_bytes(16, "big"), 0),
                live[0],
            ]
    before = replay.state_dict()

    assert replay.can_sample_batch("train") is expected
    _tree_equal(replay.state_dict(), before)
    if expected:
        plan = replay.prepare_sample("train")
        assert plan.batch.step_ids.shape == (2, 1, 20)
    else:
        with pytest.raises(LookupError, match="complete batch"):
            replay.prepare_sample("train")
    _tree_equal(replay.state_dict(), before)


def test_report_longer_than_train_uses_bounded_pending_starts() -> None:
    replay = _replay(
        capacity=200,
        sequence=_sequence(
            batch=2,
            length=2,
            context=0,
            consecutive=1,
            report_length=5,
            report_consecutive=1,
        ),
    )
    _add(replay, 30)

    class NoCapacityIteration(dict[int, ReplayKey]):
        def items(self) -> NoReturn:
            raise AssertionError("sample preparation scanned replay capacity")

        def __iter__(self) -> NoReturn:
            raise AssertionError("sample preparation scanned replay capacity")

    replay.items = NoCapacityIteration(replay.items)
    batch = replay.prepare_sample("report").batch
    assert batch.step_ids.shape == (2, 5, 20)


def test_prepare_commit_plans_are_bounded_and_reject_stale_commit() -> None:
    replay = _replay(capacity=2, sequence=_sequence(length=1, context=0))
    first = replay.prepare_add(_row(0, first=True), worker=0)
    replay.commit_add(first)
    stale = replay.prepare_add(_row(1), worker=0)
    replay.add(_row(1), worker=0)
    before = replay.state_dict()
    with pytest.raises(RuntimeError, match="stale"):
        replay.commit_add(stale)
    _tree_equal(replay.state_dict(), before)
    assert not hasattr(replay.writers[0], "chunk_history")
    replay.validate = lambda: (_ for _ in ()).throw(AssertionError("validate called"))
    for index in range(2, 100):
        replay.add(_row(index), worker=0)
    assert len(replay.items) == 2
    assert len(replay.chunks) <= 4


def test_add_plan_row_is_irreversibly_immutable_before_commit() -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    before = replay.state_dict()
    plan = replay.prepare_add(_row(0, first=True, last=True, terminal=True))

    with pytest.raises(ValueError, match="WRITEABLE"):
        plan.row["action"].flags.writeable = True

    _tree_equal(replay.state_dict(), before)
    np.testing.assert_array_equal(plan.row["action"], np.zeros(2, np.float32))


def test_add_plan_hides_mutable_staging_and_rejects_scalar_tampering() -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    before = replay.state_dict()
    plan = replay.prepare_add(_row(0, first=True))

    assert not hasattr(plan, "_new_writer")
    assert not hasattr(plan, "_new_chunks")
    assert not hasattr(plan, "_token")
    object.__setattr__(plan, "worker", 7)
    with pytest.raises(RuntimeError, match="tampered"):
        replay.commit_add(plan)

    _tree_equal(replay.state_dict(), before)


def test_add_plan_is_owner_bound_and_single_use() -> None:
    left = _replay(sequence=_sequence(length=1, context=0))
    right = _replay(sequence=_sequence(length=1, context=0))
    left_before = left.state_dict()
    right_before = right.state_dict()
    plan = left.prepare_add(_row(0, first=True))

    with pytest.raises(RuntimeError, match="owner"):
        right.commit_add(plan)
    _tree_equal(left.state_dict(), left_before)
    _tree_equal(right.state_dict(), right_before)

    left.commit_add(plan)
    committed = left.state_dict()
    with pytest.raises(RuntimeError, match="consumed"):
        left.commit_add(plan)
    _tree_equal(left.state_dict(), committed)


@pytest.mark.parametrize("kind", ["add", "sample"])
def test_local_plan_reclassification_rejects_and_consumes(kind: str) -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    plan: Any
    commit: Callable[[Any], object]
    if kind == "add":
        plan = replay.prepare_add(_row(0, first=True))
        commit = replay.commit_add
        pending = getattr(replay, "_DreamerReplay__prepared_add")
    else:
        replay.add(_row(0, first=True))
        plan = replay.prepare_sample("train")
        commit = replay.commit_sample
        pending = getattr(replay, "_DreamerReplay__prepared_sample")
    base = type(plan)
    reclassified = type("ReclassifiedPlan", (base,), {"__slots__": ()})
    before = replay.state_dict()

    object.__setattr__(plan, "__class__", reclassified)
    with pytest.raises(RuntimeError, match="invalid"):
        commit(plan)
    _tree_equal(replay.state_dict(), before)
    assert len(pending) == 0

    object.__setattr__(plan, "__class__", base)
    with pytest.raises(RuntimeError, match="consumed"):
        commit(plan)
    _tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize("kind", ["add", "sample"])
def test_consumed_plan_reclassification_stays_consumed(kind: str) -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    plan: Any
    commit: Callable[[Any], object]
    if kind == "add":
        plan = replay.prepare_add(_row(0, first=True))
        commit = replay.commit_add
    else:
        replay.add(_row(0, first=True))
        plan = replay.prepare_sample("train")
        commit = replay.commit_sample
    commit(plan)
    committed = replay.state_dict()
    base = type(plan)
    reclassified = type("ConsumedReclassifiedPlan", (base,), {"__slots__": ()})

    object.__setattr__(plan, "__class__", reclassified)
    with pytest.raises(RuntimeError, match="consumed"):
        commit(plan)
    _tree_equal(replay.state_dict(), committed)


@pytest.mark.parametrize("kind", ["add", "sample"])
def test_cross_replay_reclassified_plan_preserves_source_plan(kind: str) -> None:
    source = _replay(sequence=_sequence(length=1, context=0))
    target = _replay(sequence=_sequence(length=1, context=0))
    plan: Any
    source_commit: Callable[[Any], object]
    target_commit: Callable[[Any], object]
    if kind == "add":
        plan = source.prepare_add(_row(0, first=True))
        source_commit = source.commit_add
        target_commit = target.commit_add
        pending = getattr(source, "_DreamerReplay__prepared_add")
    else:
        source.add(_row(0, first=True))
        target.add(_row(0, first=True))
        plan = source.prepare_sample("train")
        source_commit = source.commit_sample
        target_commit = target.commit_sample
        pending = getattr(source, "_DreamerReplay__prepared_sample")
    source_before = source.state_dict()
    target_before = target.state_dict()
    base = type(plan)
    reclassified = type("ForeignReclassifiedPlan", (base,), {"__slots__": ()})

    object.__setattr__(plan, "__class__", reclassified)
    with pytest.raises(RuntimeError, match="invalid"):
        target_commit(plan)
    _tree_equal(source.state_dict(), source_before)
    _tree_equal(target.state_dict(), target_before)
    assert len(pending) == 1

    object.__setattr__(plan, "__class__", base)
    source_commit(plan)
    assert len(pending) == 0


@pytest.mark.parametrize("mutation", ["replace", "delete"])
def test_add_plan_owner_rejection_consumes_local_plan(mutation: str) -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    plan = replay.prepare_add(_row(0, first=True))
    owner = plan._owner
    before = replay.state_dict()

    if mutation == "replace":
        object.__setattr__(plan, "_owner", object())
    else:
        object.__delattr__(plan, "_owner")
    with pytest.raises(RuntimeError):
        replay.commit_add(plan)
    _tree_equal(replay.state_dict(), before)

    object.__setattr__(plan, "_owner", owner)
    with pytest.raises(RuntimeError, match="consumed"):
        replay.commit_add(plan)
    _tree_equal(replay.state_dict(), before)


def test_add_plan_comparison_exception_is_normalized_and_consumes_plan() -> None:
    replay = _replay(
        sequence=_sequence(
            length=1,
            context=0,
            report_length=2,
            report_consecutive=1,
        )
    )
    replay.add(_row(0, first=True))
    plan = replay.prepare_add(_row(1))
    assert plan.report_ready is not None
    offset = plan.report_ready.offset
    before = replay.state_dict()

    object.__delattr__(plan.report_ready, "offset")
    with pytest.raises(RuntimeError, match="rejected"):
        replay.commit_add(plan)
    _tree_equal(replay.state_dict(), before)

    object.__setattr__(plan.report_ready, "offset", offset)
    with pytest.raises(RuntimeError, match="consumed"):
        replay.commit_add(plan)
    _tree_equal(replay.state_dict(), before)


def test_add_preparations_replace_abandoned_staging_with_one_pending_plan() -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    oldest = replay.prepare_add(_row(0, first=True))
    newest = oldest
    for _ in range(128):
        newest = replay.prepare_add(_row(0, first=True))
    gc.collect()

    pending = getattr(replay, "_DreamerReplay__prepared_add")
    assert len(pending) == 1
    with pytest.raises(RuntimeError, match="consumed"):
        replay.commit_add(oldest)
    assert len(pending) == 1
    key = replay.commit_add(newest)
    assert _value_at(replay, key) == 0
    assert len(pending) == 0


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("_owner", lambda plan: object()),
        ("version", lambda plan: plan.version + 1),
        ("worker", lambda plan: plan.worker + 1),
        ("row", lambda plan: {}),
        ("emits_item", lambda plan: not plan.emits_item),
        ("report_ready", lambda plan: ReplayKey((99).to_bytes(16, "big"), 0)),
    ],
)
def test_add_plan_rejects_every_public_field_replacement_atomically(
    field: str, replacement: Callable[[Any], object]
) -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    plan = replay.prepare_add(_row(0, first=True))
    before = replay.state_dict()

    object.__setattr__(plan, field, replacement(plan))
    with pytest.raises(RuntimeError):
        replay.commit_add(plan)

    _tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize("mutation", ["shape", "dtype"])
def test_add_plan_rejects_nested_row_metadata_and_consumes_plan(
    mutation: str,
) -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    plan = replay.prepare_add(_row(0, first=True))
    before = replay.state_dict()

    if mutation == "shape":
        plan.row["action"].shape = (2, 1)
    else:
        object.__setattr__(plan.row["action"], "dtype", np.dtype(np.uint32))
    with pytest.raises(RuntimeError, match="tampered"):
        replay.commit_add(plan)
    _tree_equal(replay.state_dict(), before)
    with pytest.raises(RuntimeError, match="consumed"):
        replay.commit_add(plan)

    valid = replay.prepare_add(_row(0, first=True))
    key = replay.commit_add(valid)
    np.testing.assert_array_equal(
        replay.chunks[key.chunk_id].read(key.offset, 1)["obs"], [[0.0]]
    )


def test_add_plan_rejects_nested_report_ready_mutation_atomically() -> None:
    replay = _replay(
        sequence=_sequence(
            length=1,
            context=0,
            report_length=2,
            report_consecutive=1,
        )
    )
    replay.add(_row(0, first=True))
    plan = replay.prepare_add(_row(1))
    assert plan.report_ready is not None
    before = replay.state_dict()

    object.__setattr__(plan.report_ready, "offset", plan.report_ready.offset + 1)
    with pytest.raises(RuntimeError, match="tampered"):
        replay.commit_add(plan)
    _tree_equal(replay.state_dict(), before)
    with pytest.raises(RuntimeError, match="consumed"):
        replay.commit_add(plan)

    valid = replay.prepare_add(_row(1))
    key = replay.commit_add(valid)
    assert _value_at(replay, key) == 1


def test_sample_plan_hides_queue_rng_and_rejects_tampering() -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    replay.add(_row(0, first=True))
    plan = replay.prepare_sample("train")
    before = replay.state_dict()

    assert not hasattr(plan, "current")
    assert not hasattr(plan, "_token")
    for name, replacement in (("_queue_state", []), ("_rng", np.random.default_rng(9))):
        with pytest.raises(AttributeError):
            object.__setattr__(plan, name, replacement)
    with pytest.raises(ValueError, match="WRITEABLE"):
        plan.batch.step_ids.flags.writeable = True
    object.__setattr__(plan, "index", plan.index + 1)
    with pytest.raises(RuntimeError, match="tampered"):
        replay.commit_sample(plan)

    _tree_equal(replay.state_dict(), before)


def test_sample_plan_is_owner_bound_stale_safe_and_single_use() -> None:
    left = _replay(sequence=_sequence(length=1, context=0))
    right = _replay(sequence=_sequence(length=1, context=0))
    left.add(_row(0, first=True))
    right.add(_row(0, first=True))
    plan = left.prepare_sample("train")
    left_before = left.state_dict()
    right_before = right.state_dict()

    with pytest.raises(RuntimeError, match="owner"):
        right.commit_sample(plan)
    _tree_equal(left.state_dict(), left_before)
    _tree_equal(right.state_dict(), right_before)

    left.commit_sample(plan)
    committed = left.state_dict()
    with pytest.raises(RuntimeError, match="consumed"):
        left.commit_sample(plan)
    _tree_equal(left.state_dict(), committed)

    stale = right.prepare_sample("train")
    right.add(_row(1))
    advanced = right.state_dict()
    with pytest.raises(RuntimeError, match="stale"):
        right.commit_sample(stale)
    _tree_equal(right.state_dict(), advanced)


@pytest.mark.parametrize("mutation", ["replace", "delete"])
def test_sample_plan_owner_rejection_consumes_local_plan(mutation: str) -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    replay.add(_row(0, first=True))
    plan = replay.prepare_sample("train")
    owner = plan._owner
    before = replay.state_dict()

    if mutation == "replace":
        object.__setattr__(plan, "_owner", object())
    else:
        object.__delattr__(plan, "_owner")
    with pytest.raises(RuntimeError):
        replay.commit_sample(plan)
    _tree_equal(replay.state_dict(), before)

    object.__setattr__(plan, "_owner", owner)
    with pytest.raises(RuntimeError, match="consumed"):
        replay.commit_sample(plan)
    _tree_equal(replay.state_dict(), before)


def test_sample_plan_comparison_exception_is_normalized_and_consumes_plan() -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    replay.add(_row(0, first=True))
    plan = replay.prepare_sample("train")
    data = plan.batch.data
    before = replay.state_dict()

    object.__delattr__(plan.batch, "data")
    with pytest.raises(RuntimeError, match="rejected"):
        replay.commit_sample(plan)
    _tree_equal(replay.state_dict(), before)

    object.__setattr__(plan.batch, "data", data)
    with pytest.raises(RuntimeError, match="consumed"):
        replay.commit_sample(plan)
    _tree_equal(replay.state_dict(), before)


def test_sample_preparations_replace_abandoned_staging_with_one_pending_plan() -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    replay.add(_row(0, first=True))
    oldest = replay.prepare_sample("train")
    newest = oldest
    for _ in range(128):
        newest = replay.prepare_sample("train")
    gc.collect()

    pending = getattr(replay, "_DreamerReplay__prepared_sample")
    assert len(pending) == 1
    with pytest.raises(RuntimeError, match="consumed"):
        replay.commit_sample(oldest)
    assert len(pending) == 1
    batch = replay.commit_sample(newest)
    assert batch.step_ids.shape == (1, 1, 20)
    assert len(pending) == 0


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("_owner", lambda plan: object()),
        ("version", lambda plan: plan.version + 1),
        ("mode", lambda plan: "report"),
        ("batch", lambda plan: ReplayBatch(plan.batch.data, plan.batch.step_ids)),
        ("index", lambda plan: plan.index + 1),
        ("queue", lambda plan: (ReplayKey((99).to_bytes(16, "big"), 0),)),
        ("online", lambda plan: plan.online + 1),
        ("uniform", lambda plan: plan.uniform + 1),
        ("stale", lambda plan: plan.stale + 1),
    ],
)
def test_sample_plan_rejects_every_public_field_replacement_atomically(
    field: str, replacement: Callable[[Any], object]
) -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    replay.add(_row(0, first=True))
    plan = replay.prepare_sample("train")
    before = replay.state_dict()

    object.__setattr__(plan, field, replacement(plan))
    with pytest.raises(RuntimeError):
        replay.commit_sample(plan)

    _tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize("field", ["step_ids", "data"])
def test_replay_batch_attributes_are_non_reassignable(field: str) -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    replay.add(_row(0, first=True))
    plan = replay.prepare_sample("train")
    before = replay.state_dict()
    replacement: object
    if field == "step_ids":
        replacement = np.full(plan.batch.step_ids.shape, 255, np.uint8)
    else:
        replacement = MappingProxyType({})

    with pytest.raises(AttributeError):
        setattr(plan.batch, field, replacement)
    _tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize("field", ["step_ids", "data"])
def test_sample_plan_rejects_object_level_batch_replacement_atomically(
    field: str,
) -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    replay.add(_row(0, first=True))
    plan = replay.prepare_sample("train")
    before = replay.state_dict()
    replacement: object
    if field == "step_ids":
        replacement = np.full(plan.batch.step_ids.shape, 255, np.uint8)
    else:
        replacement = MappingProxyType({})

    object.__setattr__(plan.batch, field, replacement)
    with pytest.raises(RuntimeError, match="tampered"):
        replay.commit_sample(plan)
    _tree_equal(replay.state_dict(), before)
    with pytest.raises(RuntimeError, match="consumed"):
        replay.commit_sample(plan)


@pytest.mark.parametrize("mutation", ["shape", "dtype"])
def test_sample_plan_rejects_nested_batch_leaf_metadata_atomically(
    mutation: str,
) -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    replay.add(_row(0, first=True))
    plan = replay.prepare_sample("train")
    before = replay.state_dict()

    if mutation == "shape":
        plan.batch.data["obs"].shape = (1, 1)
    else:
        object.__setattr__(plan.batch.data["obs"], "dtype", np.dtype(np.uint32))
    with pytest.raises(RuntimeError, match="tampered"):
        replay.commit_sample(plan)
    _tree_equal(replay.state_dict(), before)
    with pytest.raises(RuntimeError, match="consumed"):
        replay.commit_sample(plan)


def test_sample_plan_rejects_nested_queue_key_mutation_atomically() -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    replay.add(_row(0, first=True))
    replay.add(_row(1))
    plan = replay.prepare_sample("train")
    assert plan.queue
    before = replay.state_dict()

    object.__setattr__(plan.queue[0], "offset", plan.queue[0].offset + 99)
    with pytest.raises(RuntimeError, match="tampered"):
        replay.commit_sample(plan)
    _tree_equal(replay.state_dict(), before)
    with pytest.raises(RuntimeError, match="consumed"):
        replay.commit_sample(plan)


def test_sample_commit_returns_detached_trusted_batch_and_advances_once() -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    replay.add(_row(0, first=True))
    plan = replay.prepare_sample("train")
    expected = plan.batch.state_dict()
    before = replay.state_dict()

    batch = replay.commit_sample(plan)

    assert batch is not plan.batch
    _tree_equal(batch.state_dict(), expected)
    before_metrics = cast(dict[str, np.int64], before["metrics"])
    assert replay.stats()["sample_calls"] == int(before_metrics["sample_calls"]) + 1
    assert replay.state_dict()["version"] == int(cast(np.int64, before["version"])) + 1


def test_prepare_is_bounded_and_commit_does_not_construct_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = _replay(capacity=2, chunk_size=2, sequence=_sequence(length=1, context=0))
    replay.add(_row(0, first=True))

    class NoChunkIteration(dict[bytes, ReplayChunk]):
        def __iter__(self) -> NoReturn:
            raise AssertionError("add preflight scanned replay capacity")

    replay.chunks = NoChunkIteration(replay.chunks)
    add_plan = replay.prepare_add(_row(1))
    with pytest.raises(ValueError):
        add_plan.row["obs"][0] = -1
    monkeypatch.setattr(
        "world_marl.dreamer_v3_baseline.replay.ReplayChunk",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("commit allocated chunk")
        ),
    )
    replay.commit_add(add_plan)

    sample_plan = replay.prepare_sample("train")
    assert not hasattr(sample_plan, "rng")
    assert not hasattr(sample_plan, "rng_state")
    assert isinstance(sample_plan.queue, tuple)
    assert not hasattr(sample_plan.queue, "__setitem__")
    with pytest.raises(ValueError):
        sample_plan.batch.step_ids[0, 0, 0] = 1
    monkeypatch.setattr(
        "world_marl.dreamer_v3_baseline.replay.ReplayBatch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("commit allocated batch")
        ),
    )
    expected = sample_plan.batch.state_dict()
    committed = replay.commit_sample(sample_plan)
    assert committed is not sample_plan.batch
    _tree_equal(committed.state_dict(), expected)
    assert isinstance(replay.fifo, deque)


def test_writer_phase_initial_partial_and_restore() -> None:
    sequence = _sequence(length=2, context=1)
    replay = _replay(sequence=sequence)
    assert replay.writers == {}
    replay.add(_row(0, first=True), worker=4)
    replay.add(_row(1), worker=4)
    assert replay.writers[4].online_phase == 2
    restored = DreamerReplay.from_state_dict(
        replay.state_dict(),
        replay.config,
        sequence,
        _transition_spaces(),
        _latent_spaces(),
    )
    assert restored.writers[4].online_phase == 2
    assert restored.writers[4].state_dict() == replay.writers[4].state_dict()


@pytest.mark.parametrize(
    "corruption",
    [
        "next_chunk_gap",
        "unearned_chunk_sentinel",
        "next_item_gap",
        "unearned_item_sentinel",
        "noncontiguous_fifo_items",
    ],
)
def test_restore_rejects_deterministic_identity_cursor_gaps(
    corruption: str,
) -> None:
    replay = _replay(
        capacity=2,
        chunk_size=2,
        sequence=_sequence(length=1, context=0),
    )
    _add(replay, 3)
    state = replay.state_dict()
    before = replay.state_dict()

    if corruption == "next_chunk_gap":
        state["next_chunk_id"] = cast(int, state["next_chunk_id"]) + 1
    elif corruption == "unearned_chunk_sentinel":
        state["next_chunk_id"] = 2**128
    elif corruption == "next_item_gap":
        state["next_item_id"] = cast(int, state["next_item_id"]) + 1
    elif corruption == "unearned_item_sentinel":
        state["next_item_id"] = 2**63
    else:
        items = cast(list[dict[str, object]], state["items"])
        old = items[0]["item_id"]
        assert old == 1
        items[0]["item_id"] = 0
        state["fifo"] = [
            0 if item == old else item for item in cast(list[object], state["fifo"])
        ]
        selector = cast(dict[str, object], state["selector"])
        selector["keys"] = [
            0 if item == old else item for item in cast(list[object], selector["keys"])
        ]
        indices = cast(dict[object, object], selector["indices"])
        selector["indices"] = {
            0 if item == old else item: index for item, index in indices.items()
        }

    with pytest.raises(ValueError):
        _restore(replay, state)
    _tree_equal(replay.state_dict(), before)


def test_restore_rejects_unearned_early_writer_phase() -> None:
    replay = _replay(sequence=_sequence(length=2, context=1))
    replay.add(_row(0, first=True))
    state = replay.state_dict()
    before = replay.state_dict()
    writer = cast(dict[object, dict[str, object]], state["writers"])[0]
    assert len(cast(list[object], writer["suffix"])) == 1
    writer["online_phase"] = np.int64(0)

    with pytest.raises(ValueError):
        _restore(replay, state)
    _tree_equal(replay.state_dict(), before)


def test_restore_accepts_saturated_writer_suffix_with_hidden_phase_history() -> None:
    replay = _replay(sequence=_sequence(length=2, context=1))
    _add(replay, 5)
    state = replay.state_dict()
    writer = cast(dict[object, dict[str, object]], state["writers"])[0]
    assert len(cast(list[object], writer["suffix"])) == 2
    writer["online_phase"] = np.int64(0)

    restored = _restore(replay, state)

    _tree_equal(restored.state_dict(), state)


def test_writer_retained_rows_restore_boundary_and_exact_continuations() -> None:
    accepted_shortening: list[str] = []
    sequence = _sequence(length=3, context=0, report_length=3)
    replay = _replay(chunk_size=10, sequence=sequence)
    _add(replay, 4)
    state = replay.state_dict()
    writer = cast(dict[object, dict[str, object]], state["writers"])[0]
    assert len(cast(list[object], writer["suffix"])) == 2
    assert list(replay.items) == [0, 1]
    assert replay.next_item_id == 2

    malformed = copy.deepcopy(state)
    malformed_writer = cast(dict[object, dict[str, object]], malformed["writers"])[0]
    malformed_writer["suffix"] = [
        copy.deepcopy(cast(list[object], malformed_writer["suffix"])[-1])
    ]
    if "retained_rows" in malformed_writer:
        malformed_writer["retained_rows"] = np.int64(1)
    _recompute_serialized_refs(malformed)
    checkpoint_before = copy.deepcopy(malformed)
    replay_before = replay.state_dict()

    try:
        shortened = _restore(replay, malformed)
    except ValueError:
        pass
    else:
        _tree_equal(malformed, checkpoint_before)
        _tree_equal(replay.state_dict(), replay_before)
        uninterrupted = _restore(replay, state)
        uninterrupted.add(_row(4))
        shortened.add(_row(4))
        assert list(uninterrupted.items) == [0, 1, 2]
        assert list(shortened.items) == [0, 1]
        assert uninterrupted.next_item_id == 3
        assert shortened.next_item_id == 2
        assert not np.array_equal(
            uninterrupted.sample("report").data["obs"],
            shortened.sample("report").data["obs"],
        )
        accepted_shortening.append("equal")

    _tree_equal(malformed, checkpoint_before)
    _tree_equal(replay.state_dict(), replay_before)

    fresh_writer = ReplayWriter(7, replay.raw_length)
    assert type(fresh_writer.retained_rows) is np.int64
    assert fresh_writer.retained_rows == 0
    assert DreamerReplay.SCHEMA_VERSION == 4
    assert set(writer) == {
        "current_chunk_id",
        "has_rows",
        "last_is_last",
        "online_phase",
        "retained_rows",
        "suffix",
        "worker_id",
    }

    restored = _restore(replay, state)
    assert replay.add(_row(4)) == restored.add(_row(4))
    assert list(replay.items) == list(restored.items) == [0, 1, 2]
    assert replay.next_item_id == restored.next_item_id == 3
    assert replay.online_queue.keys == restored.online_queue.keys
    _tree_equal(replay.selector.state_dict(), restored.selector.state_dict())
    _tree_equal(replay.state_dict(), restored.state_dict())

    unequal = _replay(
        chunk_size=3,
        sequence=_sequence(length=2, context=0, report_length=5),
    )
    _add(unequal, 6)
    unequal_state = unequal.state_dict()
    unequal_writer = cast(dict[object, dict[str, object]], unequal_state["writers"])[0]
    assert len(cast(list[object], unequal_writer["suffix"])) == 4
    assert unequal_writer["retained_rows"] == np.int64(4)
    unequal_malformed = copy.deepcopy(unequal_state)
    unequal_malformed_writer = cast(
        dict[object, dict[str, object]], unequal_malformed["writers"]
    )[0]
    unequal_malformed_writer["suffix"] = copy.deepcopy(
        cast(list[object], unequal_malformed_writer["suffix"])[-2:]
    )
    unequal_malformed_writer["retained_rows"] = np.int64(2)
    _recompute_serialized_refs(unequal_malformed)
    unequal_checkpoint_before = copy.deepcopy(unequal_malformed)
    unequal_before = unequal.state_dict()
    try:
        unequal_shortened = _restore(unequal, unequal_malformed)
    except ValueError:
        pass
    else:
        unequal_uninterrupted = _restore(unequal, unequal_state)
        unequal_uninterrupted.add(_row(6))
        unequal_shortened.add(_row(6))
        assert unequal_uninterrupted._report_pending == {3, 4, 5}
        assert unequal_shortened._report_pending == {2, 3, 4, 5}
        assert not np.array_equal(
            unequal_uninterrupted.sample("report").data["obs"],
            unequal_shortened.sample("report").data["obs"],
        )
        accepted_shortening.append("unequal")
    _tree_equal(unequal_malformed, unequal_checkpoint_before)
    _tree_equal(unequal.state_dict(), unequal_before)

    unequal_restored = _restore(unequal, unequal_state)
    assert unequal.add(_row(6)) == unequal_restored.add(_row(6))
    _tree_equal(unequal.state_dict(), unequal_restored.state_dict())

    unsaturated = _replay(
        chunk_size=2,
        sequence=_sequence(batch=2, length=3, context=0, report_length=5),
    )
    unsaturated.add(_row(0, first=True))
    unsaturated.add(_row(1))
    unsaturated_state = unsaturated.state_dict()
    unsaturated_writer = cast(
        dict[object, dict[str, object]], unsaturated_state["writers"]
    )[0]
    assert unsaturated.items == {}
    assert unsaturated_writer["retained_rows"] == np.int64(2)
    unsaturated_restored = _restore(unsaturated, unsaturated_state)
    assert unsaturated.add(_row(2)) == unsaturated_restored.add(_row(2))
    _tree_equal(unsaturated.state_dict(), unsaturated_restored.state_dict())

    multi = _replay(chunk_size=10, sequence=sequence)
    for index in range(4):
        multi.add(_row(index, first=index == 0), worker=0)
        multi.add(_row(100 + index, first=index == 0), worker=1)
    multi_state = multi.state_dict()
    multi_malformed = copy.deepcopy(multi_state)
    multi_writers = cast(dict[object, dict[str, object]], multi_malformed["writers"])
    writer_one_before = copy.deepcopy(multi_writers[1])
    multi_writers[0]["suffix"] = [
        copy.deepcopy(cast(list[object], multi_writers[0]["suffix"])[-1])
    ]
    multi_writers[0]["retained_rows"] = np.int64(1)
    _recompute_serialized_refs(multi_malformed)
    multi_checkpoint_before = copy.deepcopy(multi_malformed)
    multi_before = multi.state_dict()
    try:
        multi_shortened = _restore(multi, multi_malformed)
    except ValueError:
        pass
    else:
        multi_uninterrupted = _restore(multi, multi_state)
        _tree_equal(multi_shortened.writers[1].state_dict(), writer_one_before)
        multi_uninterrupted.add(_row(4), worker=0)
        multi_shortened.add(_row(4), worker=0)
        assert list(multi_uninterrupted.items) == [0, 1, 2, 3, 4]
        assert list(multi_shortened.items) == [0, 1, 2, 3]
        assert multi_uninterrupted.next_item_id == 5
        assert multi_shortened.next_item_id == 4
        _tree_equal(multi_shortened.writers[1].state_dict(), writer_one_before)
        accepted_shortening.append("multi_writer")
    _tree_equal(multi_malformed, multi_checkpoint_before)
    _tree_equal(multi.state_dict(), multi_before)

    multi_restored = _restore(multi, multi_state)
    for worker, value in ((0, 4), (1, 104)):
        assert multi.add(_row(value), worker=worker) == multi_restored.add(
            _row(value), worker=worker
        )
    _tree_equal(multi.state_dict(), multi_restored.state_dict())

    evicted = _replay(
        capacity=1,
        chunk_size=2,
        sequence=_sequence(length=3, context=0, report_length=5),
    )
    _add(evicted, 8)
    assert len(evicted.items) == 1
    assert evicted.writers[0].retained_rows == np.int64(4)
    evicted_restored = _restore(evicted, evicted.state_dict())
    assert evicted.add(_row(8)) == evicted_restored.add(_row(8))
    _tree_equal(evicted.state_dict(), evicted_restored.state_dict())

    for invalid in (
        2,
        True,
        np.int32(2),
        np.int64(-1),
        np.int64(3),
        np.int64(1),
    ):
        corrupted = copy.deepcopy(state)
        corrupted_writer = cast(dict[object, dict[str, object]], corrupted["writers"])[
            0
        ]
        corrupted_writer["retained_rows"] = invalid
        corrupted_before = copy.deepcopy(corrupted)
        source = _restore(replay, replay_before)
        source_before = source.state_dict()
        with pytest.raises((TypeError, ValueError)):
            _restore(source, corrupted)
        _tree_equal(corrupted, corrupted_before)
        _tree_equal(source.state_dict(), source_before)

    if accepted_shortening:
        pytest.fail(
            "coherently shortened retained rows restored: "
            + ", ".join(accepted_shortening)
        )

    for corruption in ("old_schema", "missing_field", "extra_field"):
        corrupted = copy.deepcopy(state)
        corrupted_writer = cast(dict[object, dict[str, object]], corrupted["writers"])[
            0
        ]
        if corruption == "old_schema":
            corrupted["schema_version"] = 3
        elif corruption == "missing_field":
            del corrupted_writer["retained_rows"]
        else:
            corrupted_writer["extra"] = None
        corrupted_before = copy.deepcopy(corrupted)
        source = _restore(replay, replay_before)
        source_before = source.state_dict()
        with pytest.raises((TypeError, ValueError)):
            _restore(source, corrupted)
        _tree_equal(corrupted, corrupted_before)
        _tree_equal(source.state_dict(), source_before)


@pytest.mark.parametrize("next_item_id", [1, 0], ids=["emitted_cursor", "reset_cursor"])
def test_restore_rejects_queue_only_state_transactionally(next_item_id: int) -> None:
    replay = _replay(sequence=_sequence(batch=2, length=1, context=0))
    replay.add(_row(0, first=True))
    state = replay.state_dict()
    cast(list[object], state["items"]).clear()
    cast(list[object], state["fifo"]).clear()
    selector = cast(dict[str, object], state["selector"])
    cast(list[object], selector["keys"]).clear()
    cast(dict[object, object], selector["indices"]).clear()
    state["next_item_id"] = next_item_id
    _recompute_serialized_refs(state)
    checkpoint_before = copy.deepcopy(state)
    replay_before = replay.state_dict()

    try:
        restored = _restore(replay, state)
    except ValueError as error:
        assert "empty item owner" in str(error)
    else:
        assert restored.items == {}
        assert len(restored.online_queue.keys) == 1
        assert restored.can_sample_batch("train") is True
        with pytest.raises(LookupError, match="complete batch"):
            restored.prepare_sample("train")
        _tree_equal(state, checkpoint_before)
        _tree_equal(replay.state_dict(), replay_before)
        pytest.fail("queue-only replay state restored with lying batch readiness")

    _tree_equal(state, checkpoint_before)
    _tree_equal(replay.state_dict(), replay_before)


def test_restore_accepts_pre_first_item_writer_state_and_next_add_is_exact() -> None:
    sequence = _sequence(batch=2, length=3, context=0)
    replay = _replay(chunk_size=2, sequence=sequence)
    replay.add(_row(0, first=True))
    replay.add(_row(1))
    state = replay.state_dict()
    checkpoint_before = copy.deepcopy(state)
    assert replay.items == {}
    assert replay.online_queue.keys == []
    assert replay.next_item_id == 0
    assert replay.writers[0].suffix
    assert replay.chunks

    restored = _restore(replay, state)

    _tree_equal(restored.state_dict(), checkpoint_before)
    _tree_equal(state, checkpoint_before)
    assert replay.add(_row(2)) == restored.add(_row(2))
    _tree_equal(restored.state_dict(), replay.state_dict())


def test_empty_owner_selector_rng_restore_boundary_and_valid_continuations() -> None:
    sequence = _sequence(length=1, context=0, report_length=1)
    replay = _replay(sequence=sequence)
    fresh_state = replay.state_dict()
    advanced = UniformSelector(0)
    for item_id in range(4):
        advanced.insert(item_id)
    for _ in range(7):
        advanced.sample()
    for item_id in range(4):
        advanced.delete(item_id)
    advanced_state = copy.deepcopy(fresh_state)
    advanced_state["selector"] = advanced.state_dict()
    checkpoint_before = copy.deepcopy(advanced_state)
    replay_before = replay.state_dict()

    try:
        malformed = _restore(replay, advanced_state)
    except ValueError as error:
        assert "empty item owner" in str(error)
    else:
        _tree_equal(advanced_state, checkpoint_before)
        _tree_equal(replay.state_dict(), replay_before)
        replay.add(_row(0, first=True))
        malformed.add(_row(0, first=True))
        assert int(replay.sample("report").data["obs"][0, 0, 0]) == 0
        assert int(malformed.sample("report").data["obs"][0, 0, 0]) == 0
        for index in range(1, 5):
            replay.add(_row(index))
            malformed.add(_row(index))
        fresh_draws = [
            int(replay.sample("report").data["obs"][0, 0, 0]) for _ in range(16)
        ]
        advanced_draws = [
            int(malformed.sample("report").data["obs"][0, 0, 0]) for _ in range(16)
        ]
        assert advanced_draws != fresh_draws
        pytest.fail("advanced empty selector RNG state restored")

    _tree_equal(advanced_state, checkpoint_before)
    _tree_equal(replay.state_dict(), replay_before)
    restored = _restore(replay, fresh_state)
    _tree_equal(restored.state_dict(), fresh_state)
    assert replay.add(_row(0, first=True)) == restored.add(_row(0, first=True))
    _tree_equal(
        replay.sample("report").state_dict(), restored.sample("report").state_dict()
    )
    for index in range(1, 5):
        assert replay.add(_row(index)) == restored.add(_row(index))
    for _ in range(16):
        _tree_equal(
            replay.sample("report").state_dict(),
            restored.sample("report").state_dict(),
        )
    _tree_equal(restored.state_dict(), replay.state_dict())

    nonempty = _replay(sequence=sequence)
    _add(nonempty, 5)
    for _ in range(7):
        nonempty.sample("report")
    resumed = _restore(nonempty, nonempty.state_dict())
    _tree_equal(
        nonempty.sample("report").state_dict(), resumed.sample("report").state_dict()
    )
    _tree_equal(resumed.state_dict(), nonempty.state_dict())

    pre_item = _replay(
        chunk_size=2,
        sequence=_sequence(batch=2, length=3, context=0),
    )
    pre_item.add(_row(0, first=True))
    pre_item.add(_row(1))
    pre_item_resumed = _restore(pre_item, pre_item.state_dict())
    assert pre_item.add(_row(2)) == pre_item_resumed.add(_row(2))
    _tree_equal(pre_item_resumed.state_dict(), pre_item.state_dict())


def test_selector_complete_pcg64_state_resumes_at_exact_next_draw() -> None:
    selector = UniformSelector(0)
    for item_id in range(5):
        selector.insert(item_id)
    for _ in range(7):
        selector.sample()
    state = selector.state_dict()
    restored = UniformSelector.from_state_dict(state)
    assert [selector.sample() for _ in range(20)] == [
        restored.sample() for _ in range(20)
    ]
    assert state["bit_generator"] == "PCG64"
    rng_state = cast(dict[str, object], state["rng_state"])
    assert rng_state["bit_generator"] == "PCG64"


@pytest.mark.parametrize("leaf", ["selector", "rng_state"])
@pytest.mark.parametrize("alias", [np.str_("PCG64"), _PCG64String("PCG64")])
def test_selector_rejects_nonexact_pcg64_identity_strings_transactionally(
    leaf: str, alias: str
) -> None:
    selector = UniformSelector(7)
    for item_id in range(3):
        selector.insert(item_id)
    selector.sample()
    state = selector.state_dict()
    selector_before = selector.state_dict()
    if leaf == "selector":
        state["bit_generator"] = alias
    else:
        cast(dict[str, object], state["rng_state"])["bit_generator"] = alias
    checkpoint_before = copy.deepcopy(state)

    try:
        restored = UniformSelector.from_state_dict(state)
    except (TypeError, ValueError):
        pass
    else:
        restored_state = restored.state_dict()
        restored_identity = (
            restored_state["bit_generator"]
            if leaf == "selector"
            else cast(dict[str, object], restored_state["rng_state"])["bit_generator"]
        )
        assert type(restored_identity) is str
        assert restored_identity == "PCG64"
        _tree_equal(state, checkpoint_before)
        _tree_equal(selector.state_dict(), selector_before)
        pytest.fail(f"{leaf} accepted and normalized a nonexact PCG64 string")

    _tree_equal(state, checkpoint_before)
    _tree_equal(selector.state_dict(), selector_before)


def test_selector_exact_builtin_pcg64_identity_continues_next_draw() -> None:
    selector = UniformSelector(11)
    for item_id in range(4):
        selector.insert(item_id)
    for _ in range(5):
        selector.sample()
    state = selector.state_dict()
    before = copy.deepcopy(state)
    assert type(state["bit_generator"]) is str
    rng_state = cast(dict[str, object], state["rng_state"])
    assert type(rng_state["bit_generator"]) is str

    restored = UniformSelector.from_state_dict(state)

    assert [selector.sample() for _ in range(12)] == [
        restored.sample() for _ in range(12)
    ]
    _tree_equal(state, before)


def test_checkpoint_ordinary_counter_inventory_is_exact_np_int64() -> None:
    replay = _replay(
        sequence=_sequence(
            length=1,
            context=0,
            consecutive=2,
            report_length=1,
            report_consecutive=2,
        )
    )
    _add(replay, 4)
    replay.sample("train")
    state = replay.state_dict()
    paths = _ordinary_counter_paths(state)
    assert {path[-1] for path in paths if path[0] == "metrics"} == {
        "inserted_rows",
        "inserted_items",
        "sample_calls",
        "sampled_sequences",
        "online_samples",
        "uniform_samples",
        "stale_online",
        "update_calls",
        "updated_rows",
        "stale_updates",
    }
    assert any(path[0] == "chunks" for path in paths)
    assert any(path[0] == "refs" for path in paths)
    assert {path[1] for path in paths if path[0] == "streams"} == {
        "train",
        "report",
    }
    assert any(path[0] == "writers" for path in paths)
    assert ("version",) in paths
    for path in paths:
        assert type(_path_value(state, path)) is np.int64, path


@pytest.mark.parametrize(
    "invalid",
    [
        0,
        False,
        np.int32(0),
        np.asarray(0, np.int64),
        np.asarray([0], np.int64),
        np.int64(-1),
        2**63,
    ],
)
def test_restore_rejects_every_noncanonical_ordinary_counter(
    invalid: object,
) -> None:
    replay = _replay(
        sequence=_sequence(
            length=1,
            context=0,
            consecutive=2,
            report_length=1,
            report_consecutive=2,
        )
    )
    _add(replay, 4)
    replay.sample("train")
    state = replay.state_dict()
    before = replay.state_dict()

    for path in _ordinary_counter_paths(state):
        corrupted = copy.deepcopy(state)
        _set_path(corrupted, path, invalid)
        with pytest.raises((TypeError, ValueError)):
            _restore(replay, corrupted)
        _tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize(
    ("family", "counter"),
    [
        ("add", "inserted_rows"),
        ("add", "inserted_items"),
        ("add", "version"),
        ("sample_online", "sample_calls"),
        ("sample_online", "sampled_sequences"),
        ("sample_online", "online_samples"),
        ("sample_uniform", "uniform_samples"),
        ("sample_stale", "stale_online"),
        ("sample_online", "version"),
        ("context_update", "update_calls"),
        ("context_update", "updated_rows"),
        ("context_stale", "stale_updates"),
        ("context_update", "version"),
        ("reset_stats", "version"),
    ],
)
def test_replay_counter_overflow_preflight_is_owner_local_and_atomic(
    family: str,
    counter: str,
) -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    if family == "add":

        def operation() -> object:
            return replay.add(_row(0, first=True))

    elif family.startswith("sample"):
        replay.add(_row(0, first=True))
        replay.add(_row(1))
        if family == "sample_uniform":
            replay.online_queue.keys.clear()
        elif family == "sample_stale":
            replay.online_queue.keys.insert(0, ReplayKey((99).to_bytes(16, "big"), 0))

        def operation() -> object:
            return replay.sample("train")

    elif family.startswith("context"):
        replay.add(_row(0, first=True))
        batch = replay.sample("train")
        step_ids = batch.step_ids.copy()
        if family == "context_stale":
            step_ids[0, 0, :16] = np.frombuffer((99).to_bytes(16, "big"), np.uint8)
        values = {
            "dyn/deter": np.full((*step_ids.shape[:2], 2), 9, np.float32),
            "dyn/stoch": np.full((*step_ids.shape[:2], 1, 2), 8, np.float32),
        }

        def operation() -> object:
            return replay.update_context(step_ids, values)

    else:
        replay._metrics["inserted_rows"] = np.int64(7)

        def operation() -> object:
            return replay.stats(reset=True)

    if counter == "version":
        replay._version = np.int64(np.iinfo(np.int64).max)
    else:
        replay._metrics[counter] = np.int64(np.iinfo(np.int64).max)
    before = replay.state_dict()

    with pytest.raises(OverflowError, match="counter"):
        operation()

    _tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("has_uint32",), True),
        (("has_uint32",), -1),
        (("has_uint32",), 2),
        (("uinteger",), True),
        (("uinteger",), 0.0),
        (("uinteger",), -1),
        (("uinteger",), 2**32),
        (("state", "state"), True),
        (("state", "state"), -1),
        (("state", "state"), 2**128),
        (("state", "inc"), True),
        (("state", "inc"), -1),
        (("state", "inc"), 2**128),
    ],
)
def test_selector_rejects_out_of_schema_pcg64_numbers(
    path: tuple[str, ...], value: object
) -> None:
    state = UniformSelector(0).state_dict()
    target = cast(dict[str, object], state["rng_state"])
    for name in path[:-1]:
        target = cast(dict[str, object], target[name])
    target[path[-1]] = value

    with pytest.raises(ValueError, match="RNG state"):
        UniformSelector.from_state_dict(state)


def test_selector_rejects_missing_and_extra_pcg64_fields() -> None:
    base = UniformSelector(0).state_dict()
    rng = cast(dict[str, object], base["rng_state"])
    inner = cast(dict[str, object], rng["state"])

    for key in tuple(rng):
        state = copy.deepcopy(base)
        del cast(dict[str, object], state["rng_state"])[key]
        with pytest.raises(ValueError, match="RNG state"):
            UniformSelector.from_state_dict(state)
    for key in tuple(inner):
        state = copy.deepcopy(base)
        nested = cast(
            dict[str, object], cast(dict[str, object], state["rng_state"])["state"]
        )
        del nested[key]
        with pytest.raises(ValueError, match="RNG state"):
            UniformSelector.from_state_dict(state)
    for target_path in ((), ("state",)):
        state = copy.deepcopy(base)
        target = cast(dict[str, object], state["rng_state"])
        for name in target_path:
            target = cast(dict[str, object], target[name])
        target["extra"] = 0
        with pytest.raises(ValueError, match="RNG state"):
            UniformSelector.from_state_dict(state)

    state = copy.deepcopy(base)
    cast(dict[str, object], state["rng_state"])["state"] = []
    with pytest.raises(TypeError, match="RNG state"):
        UniformSelector.from_state_dict(state)


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate",
        "next_chunk",
        "future_chunk",
        "chunk_size_offset",
        "live_unresolvable",
    ],
)
def test_restore_rejects_unreachable_online_queue_keys(corruption: str) -> None:
    replay = _replay(
        chunk_size=3,
        sequence=_sequence(length=2, context=0),
    )
    _add(replay, 3)
    state = replay.state_dict()
    before = replay.state_dict()
    queue = cast(dict[str, object], state["online_queue"])
    keys = cast(list[dict[str, object]], queue["keys"])
    assert len(keys) == 1
    if corruption == "duplicate":
        keys.append(copy.deepcopy(keys[0]))
    elif corruption in ("next_chunk", "future_chunk"):
        chunk_value = cast(int, state["next_chunk_id"])
        if corruption == "future_chunk":
            chunk_value += 1
        keys[0] = ReplayKey(chunk_value.to_bytes(16, "big"), 0).state_dict()
    elif corruption == "chunk_size_offset":
        keys[0]["offset"] = replay.config.chunk_size
    else:
        writer = cast(dict[object, dict[str, object]], state["writers"])[0]
        keys[0] = ReplayKey(cast(bytes, writer["current_chunk_id"]), 0).state_dict()

    with pytest.raises(ValueError):
        _restore(replay, state)

    _tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize("mode", ["train", "report"])
@pytest.mark.parametrize("corruption", ["future_chunk", "chunk_size_offset"])
def test_restore_rejects_unreachable_retained_stream_step_ids(
    mode: str, corruption: str
) -> None:
    sequence = _sequence(
        length=1,
        context=0,
        consecutive=2,
        report_length=1,
        report_consecutive=2,
    )
    replay = _replay(chunk_size=3, sequence=sequence)
    _add(replay, 4)
    replay.sample(mode)
    state = replay.state_dict()
    before = replay.state_dict()
    streams = cast(dict[str, dict[str, object]], state["streams"])
    current = cast(dict[str, object], streams[mode]["current"])
    step_ids = cast(Array, current["step_ids"])
    if corruption == "future_chunk":
        chunk_id = cast(int, state["next_chunk_id"]).to_bytes(16, "big")
        offsets = range(step_ids.shape[1])
    else:
        chunks = cast(list[dict[str, object]], state["chunks"])
        chunk_id = cast(bytes, chunks[0]["chunk_id"])
        offsets = range(replay.config.chunk_size, replay.config.chunk_size + 2)
    step_ids[0] = np.stack(
        [ReplayKey(chunk_id, offset).to_step_id() for offset in offsets]
    )

    with pytest.raises(ValueError):
        _restore(replay, state)

    _tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize("mode", ["train", "report"])
@pytest.mark.parametrize(
    "corruption", ["unrelated_live", "live_empty", "live_out_of_prefix"]
)
def test_restore_rejects_partial_live_stream_identity_redirection(
    mode: str, corruption: str
) -> None:
    if corruption == "live_out_of_prefix":
        sequence = _sequence(
            length=1,
            context=0,
            consecutive=3,
            report_length=1,
            report_consecutive=3,
        )
        replay = _replay(capacity=1, chunk_size=2, sequence=sequence)
        _add(replay, 4)
        replay.sample(mode)
        replay.add(_row(4))
        replay.add(_row(100, first=True), worker=1)
    else:
        replay = _active_stream_replay(mode, partial_live=True)
    state = replay.state_dict()
    before = replay.state_dict()
    current = _serialized_stream_current(state, mode)

    if corruption == "unrelated_live":
        replay.add(_row(100, first=True), worker=1)
        replay.add(_row(101), worker=1)
        state = replay.state_dict()
        before = replay.state_dict()
        current = _serialized_stream_current(state, mode)
        unrelated = next(
            chunk
            for chunk in replay.chunks.values()
            if chunk.owner_id == 1 and chunk.length == chunk.capacity
        )
        forged = ReplayKey(unrelated.chunk_id, 0)
    elif corruption == "live_empty":
        empty = next(chunk for chunk in replay.chunks.values() if chunk.length == 0)
        forged = ReplayKey(empty.chunk_id, 0)
    else:
        partial = next(
            chunk
            for chunk in replay.chunks.values()
            if chunk.owner_id == 1 and chunk.length == 1
        )
        _replace_stream_step_key(current, 1, ReplayKey(partial.chunk_id, 0))
        forged = ReplayKey(partial.chunk_id, 1)

    position = 2 if corruption == "live_out_of_prefix" else 1
    _replace_stream_step_key(current, position, forged)

    with pytest.raises(ValueError):
        _restore(replay, state)

    _tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize("mode", ["train", "report"])
@pytest.mark.parametrize("corruption", ["mid_chunk_rollover", "decreasing_chunk"])
def test_restore_rejects_fully_evicted_stream_identity_chronology(
    mode: str, corruption: str
) -> None:
    sequence = _sequence(
        length=2,
        context=0,
        consecutive=2,
        report_length=2,
        report_consecutive=2,
    )
    replay = _replay(capacity=1, chunk_size=3, sequence=sequence)
    _add(replay, 5)
    replay.sample(mode)
    current_batch = replay.streams[mode].current
    assert current_batch is not None
    original_ids = {
        ReplayKey.from_step_id(value).chunk_id
        for value in current_batch.step_ids.reshape(-1, 20)
    }
    for index in range(5, 20):
        replay.add(_row(index))
    assert original_ids.isdisjoint(replay.chunks)
    state = replay.state_dict()
    before = replay.state_dict()
    current = _serialized_stream_current(state, mode)
    first = (1).to_bytes(16, "big")
    second = (2).to_bytes(16, "big")
    if corruption == "mid_chunk_rollover":
        keys = [
            ReplayKey(first, 0),
            ReplayKey(second, 0),
            ReplayKey(second, 1),
            ReplayKey(second, 2),
        ]
    else:
        keys = [
            ReplayKey(second, 1),
            ReplayKey(second, 2),
            ReplayKey(first, 0),
            ReplayKey(first, 1),
        ]
    cast(Array, current["step_ids"])[0] = np.stack([key.to_step_id() for key in keys])

    with pytest.raises(ValueError):
        _restore(replay, state)

    _tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize("mode", ["train", "report"])
@pytest.mark.parametrize(
    "partial_live", [False, True], ids=["fully_live", "partial_live"]
)
@pytest.mark.parametrize(
    "leaf", ["obs", "action", "reward", "is_terminal", "is_first", "is_last"]
)
def test_restore_rejects_live_stream_immutable_transition_corruption(
    mode: str, partial_live: bool, leaf: str
) -> None:
    replay = _active_stream_replay(mode, partial_live=partial_live)
    state = replay.state_dict()
    before = replay.state_dict()
    current = _serialized_stream_current(state, mode)
    data = cast(dict[str, Array], current["data"])
    position = 1 if partial_live else 0
    if leaf in ("obs", "action"):
        data[leaf][0, position, 0] += np.asarray(1000, data[leaf].dtype)
    elif leaf == "reward":
        data[leaf][0, position] += np.float32(1000)
    else:
        data[leaf][0, position] = not bool(data[leaf][0, position])

    with pytest.raises(ValueError):
        _restore(replay, state)

    _tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize("mode", ["train", "report"])
def test_restore_accepts_unmodified_partial_live_stream_exact_next_slice(
    mode: str,
) -> None:
    replay = _active_stream_replay(mode, partial_live=True)
    restored = _restore(replay, replay.state_dict())

    _tree_equal(restored.state_dict(), replay.state_dict())
    _tree_equal(restored.sample(mode).state_dict(), replay.sample(mode).state_dict())
    _tree_equal(restored.state_dict(), replay.state_dict())


@pytest.mark.parametrize("mode", ["train", "report"])
def test_restore_rejects_slice_only_consec_in_active_raw_stream(
    mode: str,
) -> None:
    replay = _active_stream_replay(mode, partial_live=False)
    state = replay.state_dict()
    before = replay.state_dict()
    current = _serialized_stream_current(state, mode)
    data = cast(dict[str, Array], current["data"])
    step_ids = cast(Array, current["step_ids"])
    data["consec"] = np.zeros(step_ids.shape[:2], np.int32)

    try:
        restored = _restore(replay, state)
    except (TypeError, ValueError) as error:
        assert "consec" in str(error)
    else:
        restored_current = restored.streams[mode].current
        assert restored_current is not None
        assert "consec" in restored_current.data
        _tree_equal(replay.state_dict(), before)
        pytest.fail("active raw stream accepted and preserved slice-only consec")

    _tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize("mode", ["train", "report"])
def test_restore_accepts_partial_live_stream_with_skipped_successor_id(
    mode: str,
) -> None:
    sequence = _sequence(
        length=1,
        context=0,
        consecutive=2,
        report_length=1,
        report_consecutive=2,
    )
    replay = _replay(capacity=1, chunk_size=2, sequence=sequence)
    replay.add(_row(0, first=True), worker=0)
    replay.add(_row(100, first=True), worker=1)
    replay.add(_row(1), worker=0)
    replay.add(_row(2), worker=0)
    replay.sample(mode)
    current = replay.streams[mode].current
    assert current is not None
    keys = [ReplayKey.from_step_id(value) for value in current.step_ids[0]]
    assert [int.from_bytes(key.chunk_id, "big") for key in keys] == [1, 3]
    replay.add(_row(3), worker=0)
    assert keys[0].chunk_id not in replay.chunks
    assert keys[1].chunk_id in replay.chunks
    restored = _restore(replay, replay.state_dict())

    _tree_equal(restored.sample(mode).state_dict(), replay.sample(mode).state_dict())
    _tree_equal(restored.state_dict(), replay.state_dict())


@pytest.mark.parametrize("mode", ["train", "report"])
def test_restore_accepts_partial_live_stream_latent_writeback_divergence(
    mode: str,
) -> None:
    replay = _active_stream_replay(mode, partial_live=True)
    current = replay.streams[mode].current
    assert current is not None
    live_step_id = current.step_ids[:, 1:2]
    replay.update_context(
        live_step_id,
        {
            "dyn/deter": np.full((1, 1, 2), -7, np.float32),
            "dyn/stoch": np.full((1, 1, 1, 2), -8, np.float32),
        },
    )
    assert not np.array_equal(
        current.data["dyn/deter"][:, 1:2],
        np.full((1, 1, 2), -7, np.float32),
    )
    restored = _restore(replay, replay.state_dict())

    _tree_equal(restored.sample(mode).state_dict(), replay.sample(mode).state_dict())
    _tree_equal(restored.state_dict(), replay.state_dict())


@pytest.mark.parametrize("mode", ["train", "report"])
def test_restore_accepts_live_stream_sampling_annotations(mode: str) -> None:
    sequence = _sequence(
        length=1,
        context=0,
        consecutive=2,
        report_length=1,
        report_consecutive=2,
    )
    replay = _replay(capacity=1, chunk_size=3, sequence=sequence)
    replay.add(_row(0, first=True))
    replay.add(_row(1))
    replay.add(_row(2, first=True))
    replay.sample(mode)
    current = replay.streams[mode].current
    assert current is not None
    keys = [ReplayKey.from_step_id(value) for value in current.step_ids[0]]
    stored_first = replay.chunks[keys[0].chunk_id].transition_data["is_first"]
    stored_last = replay.chunks[keys[0].chunk_id].transition_data["is_last"]
    assert not bool(stored_first[keys[0].offset])
    assert not bool(stored_last[keys[0].offset])
    assert bool(current.data["is_first"][0, 0])
    assert bool(current.data["is_last"][0, 0])

    restored = _restore(replay, replay.state_dict())

    _tree_equal(restored.state_dict(), replay.state_dict())


@pytest.mark.parametrize("corruption", ["swapped", "skipped"])
def test_restore_rejects_wrong_writer_local_item_start_order(
    corruption: str,
) -> None:
    replay = _replay(
        capacity=3,
        chunk_size=10,
        sequence=_sequence(length=1, context=0),
    )
    _add(replay, 6)
    state = replay.state_dict()
    before = replay.state_dict()
    items = cast(list[dict[str, object]], state["items"])
    assert [cast(dict[str, object], item["key"])["offset"] for item in items] == [
        3,
        4,
        5,
    ]
    if corruption == "swapped":
        items[0]["key"], items[2]["key"] = items[2]["key"], items[0]["key"]
    else:
        first_key = cast(dict[str, object], items[0]["key"])
        first_key["offset"] = 2

    with pytest.raises(ValueError):
        _restore(replay, state)

    _tree_equal(replay.state_dict(), before)


def test_restore_accepts_interleaved_multi_writer_item_start_order() -> None:
    replay = _replay(
        capacity=10,
        chunk_size=8,
        sequence=_sequence(length=1, context=0),
    )
    for index in range(3):
        replay.add(_row(index, first=index == 0), worker=0)
        replay.add(_row(100 + index, first=index == 0), worker=1)
    starts = list(replay.items.values())
    assert starts != sorted(starts)

    restored = _restore(replay, replay.state_dict())

    _tree_equal(restored.state_dict(), replay.state_dict())


def test_restore_accepts_and_drains_evicted_stale_online_queue_exactly() -> None:
    replay = _replay(
        capacity=1,
        chunk_size=1,
        online_queue_size=20,
        sequence=_sequence(length=1, context=0),
    )
    _add(replay, 6)
    stale = [
        key for key in replay.online_queue.keys if key.chunk_id not in replay.chunks
    ]
    assert len(stale) >= 2
    restored = _restore(replay, replay.state_dict())

    _tree_equal(restored.state_dict(), replay.state_dict())
    _tree_equal(
        restored.sample("train").state_dict(), replay.sample("train").state_dict()
    )
    assert restored.stats()["stale_online"] == len(stale)
    assert replay.stats()["stale_online"] == len(stale)
    _tree_equal(restored.state_dict(), replay.state_dict())


@pytest.mark.parametrize("mode", ["train", "report"])
def test_restore_accepts_retained_stream_after_source_chunks_are_evicted(
    mode: str,
) -> None:
    sequence = _sequence(
        length=1,
        context=0,
        consecutive=2,
        report_length=1,
        report_consecutive=2,
    )
    replay = _replay(capacity=1, chunk_size=2, sequence=sequence)
    _add(replay, 3)
    replay.sample(mode)
    retained = replay.streams[mode].current
    assert retained is not None
    retained_chunk_ids = {
        ReplayKey.from_step_id(step_id).chunk_id
        for step_id in retained.step_ids.reshape(-1, 20)
    }
    for index in range(3, 12):
        replay.add(_row(index))
    assert retained_chunk_ids.isdisjoint(replay.chunks)
    restored = _restore(replay, replay.state_dict())

    _tree_equal(restored.state_dict(), replay.state_dict())
    _tree_equal(restored.sample(mode).state_dict(), replay.sample(mode).state_dict())
    _tree_equal(restored.state_dict(), replay.state_dict())


@pytest.mark.parametrize("mode", ["train", "report"])
def test_restore_rejects_empty_item_owner_with_active_retained_stream_transactionally(
    mode: str,
) -> None:
    sequence = _sequence(
        length=1,
        context=0,
        consecutive=2,
        report_length=1,
        report_consecutive=2,
    )
    replay = _replay(capacity=1, chunk_size=2, sequence=sequence)
    _add(replay, 3)
    replay.sample(mode)
    retained = replay.streams[mode].current
    assert retained is not None
    retained_chunk_ids = {
        ReplayKey.from_step_id(step_id).chunk_id
        for step_id in retained.step_ids.reshape(-1, 20)
    }
    for index in range(3, 12):
        replay.add(_row(index))
    assert retained_chunk_ids.isdisjoint(replay.chunks)
    state = replay.state_dict()
    cast(list[object], state["items"]).clear()
    cast(list[object], state["fifo"]).clear()
    selector = cast(dict[str, object], state["selector"])
    cast(list[object], selector["keys"]).clear()
    cast(dict[object, object], selector["indices"]).clear()
    cast(list[object], cast(dict[str, object], state["online_queue"])["keys"]).clear()
    state["next_item_id"] = 0
    _recompute_serialized_refs(state)
    assert all(
        count > 0 for count in cast(dict[bytes, np.int64], state["refs"]).values()
    )
    checkpoint_before = copy.deepcopy(state)
    replay_before = replay.state_dict()

    try:
        restored = _restore(replay, state)
    except ValueError as error:
        assert "empty item owner" in str(error)
    else:
        assert restored.can_sample_batch(mode) is True
        next_slice = restored.sample(mode)
        assert next_slice.step_ids.shape == (1, 1, 20)
        assert next_slice.data["consec"].tolist() == [[1]]
        _tree_equal(state, checkpoint_before)
        _tree_equal(replay.state_dict(), replay_before)
        pytest.fail(f"{mode} restored an empty item owner with an active stream")

    _tree_equal(state, checkpoint_before)
    _tree_equal(replay.state_dict(), replay_before)


def test_complete_state_is_fresh_transactional_and_resumes_exact_behavior() -> None:
    sequence = _sequence(
        batch=2,
        length=2,
        context=1,
        consecutive=2,
        report_length=2,
        report_consecutive=2,
    )
    replay = _replay(capacity=5, chunk_size=3, sequence=sequence)
    _add(replay, 14)
    replay.sample("report")
    replay.sample("train")
    raw = replay.sample("report")
    replay.update_context(
        raw.step_ids[:, 1:],
        {
            "dyn/deter": np.full((2, 2, 2), 9, np.float32),
            "dyn/stoch": np.full((2, 2, 1, 2), 8, np.float32),
        },
    )
    state = replay.state_dict()
    restored = DreamerReplay.from_state_dict(
        state,
        replay.config,
        sequence,
        _transition_spaces(),
        _latent_spaces(),
    )
    _tree_equal(restored.state_dict(), state)
    chunks = cast(list[dict[str, object]], state["chunks"])
    transition = cast(dict[str, Array], chunks[0]["transition"])
    transition["obs"][0, 0] = -999
    restored_chunks = cast(list[dict[str, object]], restored.state_dict()["chunks"])
    restored_transition = cast(dict[str, Array], restored_chunks[0]["transition"])
    assert restored_transition["obs"][0, 0] != -999
    for index in range(14, 22):
        assert replay.add(_row(index)) == restored.add(_row(index))
    for mode in ("train", "report", "train", "report"):
        _tree_equal(
            replay.sample(mode).state_dict(), restored.sample(mode).state_dict()
        )
    _tree_equal(replay.state_dict(), restored.state_dict())

    broken = restored.state_dict()
    broken["next_item_id"] = 2**63 + 1
    before = restored.state_dict()
    with pytest.raises(ValueError):
        DreamerReplay.from_state_dict(
            broken,
            restored.config,
            sequence,
            _transition_spaces(),
            _latent_spaces(),
        )
    _tree_equal(restored.state_dict(), before)


def _identity_record_replay() -> tuple[
    DreamerReplay, dict[str, TensorSpace], dict[str, TensorSpace]
]:
    transition_spaces = {
        **_transition_spaces(),
        "disc": TensorSpace((), "int32", classes=3),
    }
    latent_spaces = _latent_spaces()
    sequence = _sequence(
        batch=1,
        length=1,
        context=0,
        consecutive=1,
        report_length=1,
        report_consecutive=1,
    )
    replay = DreamerReplay(
        ReplayConfig(capacity=1, chunk_size=1, online_queue_size=1),
        sequence,
        transition_spaces,
        latent_spaces,
    )
    row = _row(0, first=True)
    row["disc"] = np.int32(1)
    replay.add(row)
    return replay, transition_spaces, latent_spaces


@pytest.mark.parametrize("owner", ["transition", "latent"])
@pytest.mark.parametrize(
    ("case_name", "space", "expected_classes"),
    _SPACE_IDENTITY_CASES,
    ids=[case[0] for case in _SPACE_IDENTITY_CASES],
)
def test_space_state_projects_classes_to_primitive_owner_records(
    owner: str,
    case_name: str,
    space: TensorSpace,
    expected_classes: object,
) -> None:
    del case_name
    replay, _, _, _ = _space_identity_replay(owner, space)
    state = replay.state_dict()
    spaces = cast(dict[str, dict[str, dict[str, object]]], state["spaces"])
    classes = spaces[owner][f"{owner}/probe"]["classes"]

    assert type(classes) is type(expected_classes)
    assert classes == expected_classes
    if type(classes) is list:
        assert all(type(value) is int for value in classes)
    _assert_no_tuple(spaces)


@pytest.mark.parametrize("owner", ["transition", "latent"])
@pytest.mark.parametrize(
    ("case_name", "space", "expected_classes"),
    _SPACE_IDENTITY_CASES,
    ids=[case[0] for case in _SPACE_IDENTITY_CASES],
)
def test_space_state_is_own_inverse_with_exact_next_state(
    owner: str,
    case_name: str,
    space: TensorSpace,
    expected_classes: object,
) -> None:
    del case_name
    replay, transition_spaces, latent_spaces, row = _space_identity_replay(owner, space)
    state = replay.state_dict()
    canonical_state = copy.deepcopy(state)

    restored = _restore_with_spaces(replay, state, transition_spaces, latent_spaces)

    assert restored.transition_spaces == transition_spaces
    assert restored.latent_spaces == latent_spaces
    _tree_equal(restored.state_dict(), canonical_state)
    spaces = cast(dict[str, dict[str, dict[str, object]]], state["spaces"])
    classes = spaces[owner][f"{owner}/probe"]["classes"]
    if type(expected_classes) is list:
        cast(list[int], classes)[0] += 100
        _tree_equal(restored.state_dict(), canonical_state)
    next_row = copy.deepcopy(row)
    next_row["is_first"] = False
    assert replay.add(next_row) == restored.add(next_row)
    _tree_equal(restored.state_dict(), replay.state_dict())


@pytest.mark.parametrize("owner", ["transition", "latent"])
@pytest.mark.parametrize(
    ("case_name", "space", "expected_classes"),
    _SPACE_IDENTITY_CASES,
    ids=[case[0] for case in _SPACE_IDENTITY_CASES],
)
def test_space_state_subtree_round_trips_through_public_flax(
    owner: str,
    case_name: str,
    space: TensorSpace,
    expected_classes: object,
) -> None:
    del case_name, expected_classes
    from flax import serialization

    replay, _, _, _ = _space_identity_replay(owner, space)
    spaces = cast(dict[str, object], replay.state_dict()["spaces"])

    encoded = serialization.msgpack_serialize(spaces)
    restored = serialization.msgpack_restore(encoded)

    _tree_equal(restored, spaces)


@pytest.mark.parametrize(
    "corruption",
    [
        "tuple",
        "ndarray",
        "nested_list",
        "numpy_integer",
        "bool",
        "wrong_length",
        "wrong_order",
        "wrong_value",
    ],
)
def test_restore_rejects_noncanonical_shaped_discrete_classes_transactionally(
    corruption: str,
) -> None:
    space = TensorSpace((2, 2), "int32", classes=cast(Any, ((2, 3), (4, 5))))
    replay, transition_spaces, latent_spaces, _ = _space_identity_replay(
        "transition", space
    )
    state = replay.state_dict()
    before = replay.state_dict()
    spaces = cast(dict[str, dict[str, dict[str, object]]], state["spaces"])
    record = spaces["transition"]["transition/probe"]
    replacements: dict[str, object] = {
        "tuple": (2, 3, 4, 5),
        "ndarray": np.asarray([2, 3, 4, 5], np.int64),
        "nested_list": [[2, 3], [4, 5]],
        "numpy_integer": [2, 3, 4, np.int64(5)],
        "bool": [2, 3, 4, True],
        "wrong_length": [2, 3, 4],
        "wrong_order": [3, 2, 4, 5],
        "wrong_value": [2, 3, 4, 6],
    }
    record["classes"] = replacements[corruption]

    with pytest.raises((TypeError, ValueError)):
        _restore_with_spaces(replay, state, transition_spaces, latent_spaces)

    _tree_equal(replay.state_dict(), before)


def test_restore_rejects_shaped_discrete_class_container_alias_transactionally() -> (
    None
):
    transition_spaces = {
        **_transition_spaces(),
        "transition/probe": TensorSpace((2,), "int32", classes=(2, 3)),
    }
    latent_spaces = {
        **_latent_spaces(),
        "latent/probe": TensorSpace((2,), "int32", classes=(2, 3)),
    }
    replay = DreamerReplay(
        ReplayConfig(capacity=1, chunk_size=1, online_queue_size=1),
        _sequence(length=1, context=0),
        transition_spaces,
        latent_spaces,
    )
    state = replay.state_dict()
    before = replay.state_dict()
    spaces = cast(dict[str, dict[str, dict[str, object]]], state["spaces"])
    shared = spaces["transition"]["transition/probe"]["classes"]
    spaces["latent"]["latent/probe"]["classes"] = shared

    with pytest.raises((TypeError, ValueError)):
        _restore_with_spaces(replay, state, transition_spaces, latent_spaces)

    _tree_equal(replay.state_dict(), before)


def test_restore_accepts_exact_canonical_replay_identity_records() -> None:
    replay, transition_spaces, latent_spaces = _identity_record_replay()
    state = replay.state_dict()

    restored = _restore_with_spaces(replay, state, transition_spaces, latent_spaces)

    _tree_equal(restored.state_dict(), state)


@pytest.mark.parametrize(
    "alias",
    [
        "schema_numpy",
        "schema_bool",
        "config_numpy",
        "config_bool",
        "sequence_numpy",
        "sequence_bool",
        "sequence_false_context",
        "space_shape_tuple",
        "space_shape_numpy",
        "space_shape_bool",
        "space_dtype_numpy",
        "space_classes_numpy",
        "space_classes_bool",
        "spaces_container",
        "spaces_extra_key",
        "space_group_extra_name",
        "space_record_container",
        "space_record_extra_key",
    ],
)
def test_restore_rejects_noncanonical_equal_replay_identity_aliases(
    alias: str,
) -> None:
    replay, transition_spaces, latent_spaces = _identity_record_replay()
    state = replay.state_dict()
    before = replay.state_dict()
    spaces = cast(dict[str, object], state["spaces"])
    transition = cast(dict[str, dict[str, object]], spaces["transition"])
    obs = transition["obs"]

    if alias == "schema_numpy":
        state["schema_version"] = np.int64(replay.SCHEMA_VERSION)
    elif alias == "schema_bool":
        state["schema_version"] = True
    elif alias == "config_numpy":
        cast(dict[str, object], state["config"])["chunk_size"] = np.int64(1)
    elif alias == "config_bool":
        cast(dict[str, object], state["config"])["capacity"] = True
    elif alias == "sequence_numpy":
        cast(dict[str, object], state["sequence_shape"])["report_length"] = np.int64(1)
    elif alias == "sequence_bool":
        cast(dict[str, object], state["sequence_shape"])["batch_size"] = True
    elif alias == "sequence_false_context":
        cast(dict[str, object], state["sequence_shape"])["context"] = False
    elif alias == "space_shape_tuple":
        obs["shape"] = (1,)
    elif alias == "space_shape_numpy":
        obs["shape"] = [np.int64(1)]
    elif alias == "space_shape_bool":
        obs["shape"] = [True]
    elif alias == "space_dtype_numpy":
        obs["dtype"] = np.str_("float32")
    elif alias == "space_classes_numpy":
        transition["disc"]["classes"] = np.int64(3)
    elif alias == "space_classes_bool":
        transition["disc"]["classes"] = True
    elif alias == "spaces_container":
        state["spaces"] = [spaces]
    elif alias == "spaces_extra_key":
        spaces["extra"] = {}
    elif alias == "space_group_extra_name":
        transition["extra"] = copy.deepcopy(obs)
    elif alias == "space_record_container":
        transition["obs"] = cast(Any, [copy.deepcopy(obs)])
    else:
        obs["extra"] = None

    with pytest.raises((TypeError, ValueError)):
        _restore_with_spaces(replay, state, transition_spaces, latent_spaces)

    _tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize(
    "identity", ["zero_chunk", "negative_item", "bool_item", "upper_item"]
)
def test_restore_rejects_out_of_domain_identities_atomically(identity: str) -> None:
    replay = _replay(sequence=_sequence(length=1, context=0))
    replay.add(_row(0, first=True))
    state = replay.state_dict()
    before = replay.state_dict()

    if identity == "zero_chunk":
        chunks = cast(list[dict[str, object]], state["chunks"])
        old = cast(bytes, chunks[0]["chunk_id"])
        state = cast(dict[str, object], _replace_chunk_id(state, old, bytes(16)))
    elif identity == "negative_item":
        _replace_item_id(state, -1)
    elif identity == "bool_item":
        _replace_item_id(state, True)
        state["next_item_id"] = 2
    else:
        _replace_item_id(state, 2**63)
        state["next_item_id"] = 2**63

    with pytest.raises((TypeError, ValueError)):
        _restore(replay, state)
    _tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize("corruption", ["capacity", "owner"])
def test_restore_binds_chunk_geometry_to_trusted_replay(
    corruption: str,
) -> None:
    replay = _replay(chunk_size=3, sequence=_sequence(length=1, context=0))
    replay.add(_row(0, first=True))
    state = replay.state_dict()
    before = replay.state_dict()
    chunk = cast(list[dict[str, object]], state["chunks"])[0]
    if corruption == "capacity":
        chunk["capacity"] = 4
    else:
        chunk["owner_id"] = 99

    with pytest.raises(ValueError):
        _restore(replay, state)
    _tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize(
    "corruption", ["sealed_current", "suffix_not_tail", "cross_owner_successor"]
)
def test_restore_rejects_impossible_writer_chunk_relationships(
    corruption: str,
) -> None:
    replay = _replay(chunk_size=2, sequence=_sequence(length=2, context=0))
    _add(replay, 3)
    state = replay.state_dict()
    before = replay.state_dict()
    chunks = cast(list[dict[str, object]], state["chunks"])
    writer = cast(dict[object, dict[str, object]], state["writers"])[0]
    first_id = cast(bytes, chunks[0]["chunk_id"])
    if corruption == "sealed_current":
        writer["current_chunk_id"] = first_id
    elif corruption == "suffix_not_tail":
        writer["suffix"] = [{"chunk_id": first_id, "offset": 0}]
    else:
        chunks[0]["owner_id"] = 1
    _recompute_serialized_refs(state)

    with pytest.raises(ValueError):
        _restore(replay, state)
    _tree_equal(replay.state_dict(), before)


def test_restore_rejects_impossible_consecutive_stream_cursor() -> None:
    sequence = _sequence(length=1, context=0, consecutive=2)
    replay = _replay(sequence=sequence)
    _add(replay, 2)
    replay.sample("train")
    state = replay.state_dict()
    before = replay.state_dict()
    streams = cast(dict[str, dict[str, object]], state["streams"])
    assert streams["train"]["current"] is not None
    streams["train"]["index"] = 0

    with pytest.raises(ValueError):
        _restore(replay, state)
    _tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize(
    "corruption", ["last_flag", "has_rows", "forged_terminal_tail", "false_last"]
)
def test_restore_reconciles_writer_flags_with_retained_tail(corruption: str) -> None:
    replay = _replay(chunk_size=3, sequence=_sequence(length=2, context=0))
    replay.add(_row(0, first=True))
    if corruption == "false_last":
        replay.add(_row(1))
    else:
        replay.add(_row(1, last=True, terminal=True))
    state = replay.state_dict()
    before = replay.state_dict()
    writer = cast(dict[object, dict[str, object]], state["writers"])[0]
    chunk = cast(list[dict[str, object]], state["chunks"])[0]
    transition = cast(dict[str, Array], chunk["transition"])
    if corruption == "last_flag":
        writer["last_is_last"] = False
    elif corruption == "has_rows":
        writer["has_rows"] = False
    elif corruption == "forged_terminal_tail":
        transition["is_last"][-1] = False
        writer["last_is_last"] = False
    else:
        writer["last_is_last"] = True

    with pytest.raises(ValueError):
        _restore(replay, state)
    _tree_equal(replay.state_dict(), before)


def test_restore_rejects_cross_chunk_row_after_last_without_first() -> None:
    replay = _replay(chunk_size=2, sequence=_sequence(length=2, context=0))
    replay.add(_row(0, first=True))
    replay.add(_row(1, last=True, terminal=True))
    replay.add(_row(2, first=True))
    state = replay.state_dict()
    before = replay.state_dict()
    chunks = cast(list[dict[str, object]], state["chunks"])
    transition = cast(dict[str, Array], chunks[1]["transition"])
    transition["is_first"][0] = False

    with pytest.raises(ValueError):
        _restore(replay, state)
    _tree_equal(replay.state_dict(), before)


def test_identity_final_allocations_and_exhausted_sentinels_round_trip() -> None:
    sequence = _sequence(length=1, context=0)
    replay = _replay(sequence=sequence, chunk_size=2, capacity=2)
    replay.next_chunk_id = 2**128 - 1
    key = replay.add(_row(0, first=True))
    assert int.from_bytes(key.chunk_id, "big") == 2**128 - 1
    assert replay.next_chunk_id == 2**128
    before = replay.state_dict()
    with pytest.raises(OverflowError):
        replay.add(_row(1))
    _tree_equal(replay.state_dict(), before)
    assert replay.next_chunk_id == 2**128

    item_replay = _replay(sequence=sequence, capacity=2)
    item_replay.next_item_id = 2**63 - 1
    item_replay.add(_row(0, first=True))
    assert item_replay.next_item_id == 2**63
    item_before = item_replay.state_dict()
    with pytest.raises(OverflowError):
        item_replay.add(_row(1))
    _tree_equal(item_replay.state_dict(), item_before)
    restored_item = _restore(item_replay, item_before)
    assert restored_item.next_item_id == 2**63
    _tree_equal(restored_item.state_dict(), item_before)
    restored = DreamerReplay.from_state_dict(
        before,
        replay.config,
        sequence,
        _transition_spaces(),
        _latent_spaces(),
    )
    assert restored.next_chunk_id == 2**128


def test_chunk_collision_preflight_rejects_at_rollover_without_mutation() -> None:
    replay = _replay(sequence=_sequence(length=1, context=0), chunk_size=2, capacity=2)
    replay.add(_row(0, first=True))
    replay.next_chunk_id = 1
    before = replay.state_dict()
    with pytest.raises(RuntimeError, match="collision"):
        replay.add(_row(1))
    _tree_equal(replay.state_dict(), before)

    item_replay = _replay(sequence=_sequence(length=1, context=0), capacity=2)
    item_replay.add(_row(0, first=True))
    item_replay.next_item_id = 0
    item_before = item_replay.state_dict()
    with pytest.raises(RuntimeError, match="item id collision"):
        item_replay.add(_row(1))
    _tree_equal(item_replay.state_dict(), item_before)


def test_checkpoint_immediately_before_rollover_and_item_allocation_is_exact() -> None:
    sequence = _sequence(
        batch=2,
        length=2,
        context=0,
        consecutive=1,
        report_length=2,
        report_consecutive=1,
    )
    replay = _replay(sequence=sequence, chunk_size=2, capacity=8)
    replay.add(_row(0, first=True))
    restored = DreamerReplay.from_state_dict(
        replay.state_dict(),
        replay.config,
        sequence,
        _transition_spaces(),
        _latent_spaces(),
    )
    for index in range(1, 8):
        assert replay.add(_row(index)) == restored.add(_row(index))
    _tree_equal(replay.state_dict(), restored.state_dict())
    for mode in ("train", "report"):
        original_batch = replay.sample(mode)
        restored_batch = restored.sample(mode)
        _tree_equal(original_batch.state_dict(), restored_batch.state_dict())
        values = {
            "dyn/deter": np.full(
                original_batch.step_ids.shape[:2] + (2,), 9, np.float32
            ),
            "dyn/stoch": np.full(
                original_batch.step_ids.shape[:2] + (1, 2), 8, np.float32
            ),
        }
        assert replay.update_context(
            original_batch.step_ids, values
        ) == restored.update_context(restored_batch.step_ids, values)
    _tree_equal(replay.state_dict(), restored.state_dict())


def test_replay_batch_and_component_state_records_copy_and_validate() -> None:
    ids = np.stack(
        [[ReplayKey((1).to_bytes(16, "big"), index).to_step_id() for index in range(2)]]
    )
    data = {
        **{
            name: np.zeros((1, 2, *space.shape), np.dtype(space.dtype))
            for name, space in _transition_spaces().items()
        },
        **{
            name: np.zeros((1, 2, *space.shape), np.dtype(space.dtype))
            for name, space in _latent_spaces().items()
        },
        "consec": np.zeros((1, 2), np.int32),
    }
    batch = ReplayBatch(data, ids)
    state = batch.state_dict()
    restored = ReplayBatch.from_state(
        state, _transition_spaces(), _latent_spaces(), 1, 2
    )
    _tree_equal(restored.state_dict(), state)
    state_data = cast(dict[str, Array], state["data"])
    state_data["obs"][0, 0, 0] = 7
    assert restored.data["obs"][0, 0, 0] == 0

    queue = OnlineQueue(2)
    queue.push(ReplayKey((1).to_bytes(16, "big"), 0))
    assert (
        OnlineQueue.from_state_dict(queue.state_dict()).state_dict()
        == queue.state_dict()
    )
    assert ConsecutiveStream.from_state_dict
    assert ReplayChunk.from_state_dict
    assert ReplayWriter.from_state_dict


def test_fixture_generator_replay_parser() -> None:
    args = _parse_args(
        [
            "replay",
            "--profile",
            "paper",
            "--observation-mode",
            "proprio",
            "--reference-checkout",
            str(OFFICIAL_CHECKOUT),
            "--source-revision",
            REVISIONS[DreamerProfile.PAPER],
            "--output-dir",
            str(FIXTURE_DIR),
        ]
    )
    assert args.profile == "paper"
    assert args.observation_mode == "proprio"
    assert args.handler.__name__ == "generate_replay"


@pytest.mark.parametrize("profile", tuple(DreamerProfile))
def test_replay_fixture_exact_official_arrays_and_deterministic_generation(
    profile: DreamerProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_marl.dreamer_v3_baseline.fixture_generator import generate_replay

    monkeypatch.setenv(
        "GIT_DIR",
        "/Users/bkaplowitz/Developer/work/wm-marl/.git/worktrees/"
        "wm-marl-dreamer-v3-parity-port",
    )
    monkeypatch.setenv(
        "GIT_OBJECT_DIRECTORY",
        "/private/tmp/danijar-dreamerv3-20260713/.git/objects",
    )
    monkeypatch.setenv(
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "/Users/bkaplowitz/Developer/work/wm-marl/.git/objects",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    common = {
        "profile": profile.value,
        "observation_mode": ObservationMode.PROPRIO.value,
        "reference_checkout": OFFICIAL_CHECKOUT,
        "source_revision": REVISIONS[profile],
    }
    generate_replay(Namespace(**common, output_dir=first))
    generate_replay(Namespace(**common, output_dir=second))
    stem = f"{profile.value}-proprio-replay"
    for suffix in ("npz", "manifest.json"):
        assert (first / f"{stem}.{suffix}").read_bytes() == (
            second / f"{stem}.{suffix}"
        ).read_bytes()
        assert (first / f"{stem}.{suffix}").read_bytes() == (
            FIXTURE_DIR / f"{stem}.{suffix}"
        ).read_bytes()
    manifest = OracleManifest.load(
        first / f"{stem}.manifest.json", fixture_path=first / f"{stem}.npz"
    )
    assert manifest.official_commit == REVISIONS[profile]
    assert manifest.source_spec == "replay"
    with np.load(first / f"{stem}.npz", allow_pickle=False) as arrays:
        assert sorted(arrays.files) == list(arrays.files)
        remaining = set(arrays.files)

        def expected(name: str) -> Array:
            remaining.remove(name)
            return arrays[name]

        item_replay = _replay(
            chunk_size=3,
            sequence=_sequence(length=2, context=1),
        )
        _add(item_replay, 10)
        starts = list(item_replay.items.values())
        np.testing.assert_array_equal(
            np.asarray(
                [int.from_bytes(key.chunk_id, "big") for key in starts],
                np.int64,
            ),
            expected("items.start_chunks"),
        )
        np.testing.assert_array_equal(
            np.asarray([key.offset for key in starts], np.int32),
            expected("items.start_offsets"),
        )
        np.testing.assert_array_equal(
            np.asarray(
                [_value_at(item_replay, key) for key in item_replay.online_queue.keys],
                np.int64,
            ),
            expected("online.starts"),
        )

        selector = UniformSelector(seed=7)
        for item_id in (0, 1, 2):
            selector.insert(item_id)
        draws = np.asarray([selector.sample() for _ in range(6)], np.int64)
        selector_state = selector.state_dict()
        restored_selector = UniformSelector.from_state_dict(selector_state)
        continuation = np.asarray([selector.sample() for _ in range(6)], np.int64)
        resumed = np.asarray([restored_selector.sample() for _ in range(6)], np.int64)
        np.testing.assert_array_equal(draws, expected("selector.draws"))
        np.testing.assert_array_equal(continuation, expected("selector.continuation"))
        np.testing.assert_array_equal(resumed, expected("selector.resumed"))

        stream_replay = _replay(sequence=_sequence(length=2, context=1, consecutive=2))
        _add(stream_replay, 5)
        slice0 = stream_replay.commit_sample(stream_replay.prepare_sample("train"))
        raw = stream_replay.streams["train"].current
        assert raw is not None
        slice1 = stream_replay.commit_sample(stream_replay.prepare_sample("train"))
        np.testing.assert_array_equal(
            np.asarray(raw.data["obs"].reshape(-1), np.int32),
            expected("stream.raw"),
        )
        np.testing.assert_array_equal(
            np.asarray(slice0.data["obs"].reshape(-1), np.int32),
            expected("stream.slice0"),
        )
        np.testing.assert_array_equal(
            np.asarray(slice1.data["obs"].reshape(-1), np.int32),
            expected("stream.slice1"),
        )
        np.testing.assert_array_equal(
            slice0.data["consec"].reshape(-1), expected("stream.consec0")
        )
        np.testing.assert_array_equal(
            slice1.data["consec"].reshape(-1), expected("stream.consec1")
        )
        assert not remaining
