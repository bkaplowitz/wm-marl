from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import world_marl.dreamer_v3_baseline as dreamer_package
import world_marl.dreamer_v3_baseline.replay as replay_module
import world_marl.jepa.replay as jepa_replay
from world_marl.dreamer_v3_baseline.config import (
    DreamerProfile,
    ReplayConfig,
)
from world_marl.dreamer_v3_baseline.networks import TensorSpace
from world_marl.dreamer_v3_baseline.oracle import (
    OracleHarness,
    OracleManifest,
    official_revision,
    profile_overrides,
)
from world_marl.dreamer_v3_baseline.replay_oracle import (
    REPLAY_SOURCE_SPEC,
    run_replay_case,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "dreamer_v3"
OFFICIAL_CHECKOUT = Path(
    os.environ.get(
        "DREAMERV3_ORACLE_CHECKOUT",
        "/private/tmp/danijar-dreamerv3-20260713",
    )
)
SOURCE_HASHES = {
    "dreamerv3/configs.yaml": (
        "9dff9c7062e3e33951cb54c6dd4b598aaf7e56e18e2cff39c812eaa797bcfcfc"
    ),
    "embodied/core/chunk.py": (
        "427b537afa75079a9f0dfd933ec49e6f71173371388418176c16c038be005c65"
    ),
    "embodied/core/replay.py": (
        "ea70f6f0494c0520acd357190c5c78bee46b0365dd950231ca522ddb6dbdd027"
    ),
    "embodied/core/selectors.py": (
        "b8cb69021e8f79888df88fb5c195e8ff19e3d54065bb08bd3586fd9cbe6655d3"
    ),
    "embodied/core/streams.py": (
        "8c75583d15be013a22058721f8f055c9c7ccaca95360d6ed100f60dc29ee19e5"
    ),
}


def _official_arrays(
    profile: DreamerProfile = DreamerProfile.PAPER,
) -> dict[str, np.ndarray]:
    path = FIXTURE_DIR / f"{profile.value}-proprio-replay.npz"
    with np.load(path, allow_pickle=False) as fixture:
        return {name: fixture[name] for name in fixture.files}


def _selector_rng_bytes(selector: UniformSelector) -> np.ndarray:
    payload = json.dumps(selector.state_dict()["rng_state"], sort_keys=True).encode()
    return np.frombuffer(payload, np.uint8).copy()


ReplayKey = replay_module.ReplayKey
ReplayBatch = replay_module.ReplayBatch
ReplayChunk = replay_module.ReplayChunk
ReplayWriter = replay_module.ReplayWriter
OnlineQueue = replay_module.OnlineQueue
UniformSelector = replay_module.UniformSelector
ConsecutiveStream = replay_module.ConsecutiveStream
DreamerReplay = replay_module.DreamerReplay


def _transition_spaces() -> dict[str, TensorSpace]:
    return {
        "action": TensorSpace((), "float32"),
        "is_first": TensorSpace((), "bool"),
        "is_last": TensorSpace((), "bool"),
        "is_terminal": TensorSpace((), "bool"),
        "obs": TensorSpace((2,), "uint8"),
        "reward": TensorSpace((), "float32"),
        "value": TensorSpace((), "int32"),
    }


def _latent_spaces() -> dict[str, TensorSpace]:
    return {
        "dyn/deter": TensorSpace((2,), "float32"),
        "dyn/stoch": TensorSpace((1, 2), "float32"),
    }


def _row(
    index: int,
    *,
    first: bool | None = None,
    last: bool | None = None,
    terminal: bool = False,
) -> dict[str, object]:
    return {
        "action": float(index),
        "dyn/deter": np.asarray([index + 1000, index + 1001], np.float32),
        "dyn/stoch": np.asarray([[index + 2000, index + 2001]], np.float32),
        "is_first": index == 0 if first is None else first,
        "is_last": False if last is None else last,
        "is_terminal": terminal,
        "obs": [index, index + 1],
        "reward": float(index),
        "value": index,
    }


def _config(
    *,
    capacity: int = 20,
    chunk_size: int = 3,
    online: bool = True,
    context: int = 1,
    sequence_length: int = 2,
) -> ReplayConfig:
    return ReplayConfig(
        capacity=capacity,
        chunk_size=chunk_size,
        online=online,
        context=context,
        sequence_length=sequence_length,
    )


def _replay(
    *,
    capacity: int = 20,
    chunk_size: int = 3,
    online: bool = True,
    context: int = 1,
    sequence_length: int = 2,
    consecutive: int = 1,
    batch_size: int = 1,
    seed: int = 7,
) -> DreamerReplay:
    return DreamerReplay(
        _config(
            capacity=capacity,
            chunk_size=chunk_size,
            online=online,
            context=context,
            sequence_length=sequence_length,
        ),
        _transition_spaces(),
        _latent_spaces(),
        batch_size=batch_size,
        consecutive=consecutive,
        seed=seed,
    )


def _add_rows(
    replay: DreamerReplay,
    count: int,
    *,
    worker: int = 0,
    natural_last: bool = False,
) -> list[ReplayKey]:
    keys = []
    for index in range(count):
        keys.append(
            replay.add(
                _row(
                    index,
                    first=index in (0, 4, 8),
                    last=natural_last and index in (3, 7),
                ),
                worker=worker,
            )
        )
    return keys


def _decode(ids: np.ndarray) -> list[ReplayKey]:
    return [ReplayKey.from_step_id(value) for value in ids.reshape(-1, 20)]


def _resolve_keys(
    replay: DreamerReplay,
    start: ReplayKey,
    length: int,
) -> list[ReplayKey]:
    keys = []
    chunk_id = start.chunk_id
    offset = start.offset
    while len(keys) < length:
        chunk = replay.chunks[chunk_id]
        while offset < chunk.length and len(keys) < length:
            keys.append(ReplayKey(chunk_id, offset))
            offset += 1
        if len(keys) < length:
            assert chunk.successor_id is not None
            chunk_id = chunk.successor_id
            offset = 0
    return keys


def _recomputed_refs(replay: DreamerReplay) -> dict[bytes, int]:
    refs = {chunk_id: 0 for chunk_id in replay.chunks}
    for chunk in replay.chunks.values():
        if chunk.successor_id is not None:
            refs[chunk.successor_id] += 1
    for writer in replay.writers.values():
        if writer.current_chunk_id is not None:
            refs[writer.current_chunk_id] += 1
        for key in writer.pending:
            refs[key.chunk_id] += 1
    for key in replay.items.values():
        refs[key.chunk_id] += 1
    return refs


def _assert_tree_equal(left, right) -> None:
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for x, y in zip(left, right, strict=True):
            _assert_tree_equal(x, y)
    else:
        assert left == right


def _assert_restore_rejected_without_mutation(
    replay: DreamerReplay,
    broken: dict[str, object],
) -> None:
    before = replay.state_dict()
    with pytest.raises((TypeError, ValueError)):
        replay.load_state_dict(broken)
    _assert_tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize("profile", tuple(DreamerProfile))
def test_replay_fixture_manifest_source_and_exact_official_arrays(profile) -> None:
    stem = f"{profile.value}-proprio-replay"
    fixture_path = FIXTURE_DIR / f"{stem}.npz"
    manifest_path = FIXTURE_DIR / f"{stem}.manifest.json"
    manifest = OracleManifest.load(manifest_path, fixture_path=fixture_path)
    request = json.loads(manifest.generator_request)
    assert manifest.source_spec == REPLAY_SOURCE_SPEC.name == "replay"
    assert dict(manifest.official_file_hashes) == SOURCE_HASHES
    assert manifest.official_commit == official_revision(profile)
    assert dict(manifest.overrides) == dict(profile_overrides(profile))
    assert request["official_commit"] == official_revision(profile)
    assert request["cases"]["primary"]["raw_length"] == 5
    assert request["cases"]["primary"]["collection_latent_bases"] == {
        "dyn/deter": 1000,
        "dyn/stoch": 2000,
    }
    assert request["cases"]["primary"]["latent_update_bases"] == {
        "dyn/deter": 100,
        "dyn/stoch": 200,
    }
    assert request["cases"]["capacity"] == {
        "batch": 1,
        "capacity": 3,
        "chunk_size": 3,
        "collection_latent_bases": {"dyn/deter": 1000, "dyn/stoch": 2000},
        "first_steps": [0, 4, 8],
        "last_steps": [3, 7],
        "online": True,
        "raw_length": 4,
        "seed": 7,
        "selector_checkpoints": [6, 8],
        "selector_draws": 12,
        "steps": 10,
        "terminal_steps": [],
    }
    assert list(request["row_schema"]) == sorted(request["row_schema"])
    assert request["row_schema"]["dyn/stoch"] == {
        "dtype": "float32",
        "shape": [1, 2],
    }
    assert request["runtime"]["elements_mode"] == "debug"
    assert request["runtime"]["elements_version"] == "3.22.0"
    assert request["runtime"]["numpy_version"] == "1.26.4"
    assert request["runtime"]["worker_mode"] == "isolated-ast-exec"
    assert request["runtime"]["elements_helper_hashes"] == {
        "elements/checkpoint.py": (
            "80f2fe99141d5bd1c96d9c5f32502cfff4fb5e56571595a092c6aace12f42e92"
        ),
        "elements/rwlock.py": (
            "020866b2d6d1216c3d9e8019d641bd5f073ea90d2cf5646361a722b0e156b9be"
        ),
        "elements/uuid.py": (
            "e8829a81c80058e4f130a5821acd9db62de5f72fbb02ad4b10262ccfa3006d6a"
        ),
    }
    assert set(request["runtime"]["shim_hashes"]) == {
        "Limiters",
        "RWLock",
        "Section",
        "Timer",
        "UUID",
        "timestamp",
    }
    assert request["compute_dtype"] == manifest.dtype == "float32"
    assert request["uuid_mode"] == "debug-counter"
    with np.load(fixture_path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["raw.train_values"], [[1, 2, 3, 4, 5]])
        np.testing.assert_array_equal(arrays["raw.report_values"], [[6, 7, 8, 9, 10]])
        np.testing.assert_array_equal(arrays["consecutive0.values"], [[6, 7, 8]])
        np.testing.assert_array_equal(arrays["consecutive1.values"], [[8, 9, 10]])
        np.testing.assert_array_equal(arrays["capacity.selector_keys"], [4, 5, 6])
        np.testing.assert_array_equal(
            arrays["capacity.selector_draws"], [6, 5, 6, 6, 5, 6, 6, 4, 4, 4, 4, 6]
        )


@pytest.mark.parametrize("profile", tuple(DreamerProfile))
def test_replay_fixture_regeneration_is_byte_deterministic(
    profile, tmp_path: Path
) -> None:
    if not (OFFICIAL_CHECKOUT / ".git").exists():
        pytest.skip("explicit DreamerV3 oracle checkout is unavailable")
    first = OracleHarness(OFFICIAL_CHECKOUT, tmp_path / "first")
    second = OracleHarness(OFFICIAL_CHECKOUT, tmp_path / "second")
    first_npz, first_manifest = run_replay_case(first, profile)
    second_npz, second_manifest = run_replay_case(second, profile)
    committed_stem = f"{profile.value}-proprio-replay"
    assert first_npz.read_bytes() == second_npz.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert (
        first_npz.read_bytes() == (FIXTURE_DIR / f"{committed_stem}.npz").read_bytes()
    )
    assert (
        first_manifest.read_bytes()
        == (FIXTURE_DIR / f"{committed_stem}.manifest.json").read_bytes()
    )


@pytest.mark.parametrize("profile", tuple(DreamerProfile))
def test_recorded_replay_worker_replays_all_arrays_and_attestations(profile) -> None:
    stem = f"{profile.value}-proprio-replay"
    fixture_path = FIXTURE_DIR / f"{stem}.npz"
    manifest_path = FIXTURE_DIR / f"{stem}.manifest.json"
    manifest = OracleManifest.load(manifest_path, fixture_path=fixture_path)
    request = json.loads(manifest.generator_request)
    assert tuple(manifest.generator_command)[-1] == "_worker"
    assert (
        Path(manifest.generator_command[0]).resolve() == Path(sys.executable).resolve()
    )
    assert request["python_executable"] == str(
        Path(manifest.generator_command[0]).resolve()
    )
    assert request["elements_package_dir"].endswith("site-packages/elements")
    assert request["elements_dist_info"].endswith("elements-3.22.0.dist-info")
    completed = subprocess.run(
        manifest.generator_command,
        cwd=request["official_checkout"],
        input=manifest.generator_request,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    repeated = json.loads(
        subprocess.run(
            manifest.generator_command,
            cwd=request["official_checkout"],
            input=manifest.generator_request,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    assert repeated["arrays"] == payload["arrays"]
    assert repeated["runtime"] == payload["runtime"]
    assert repeated["source_attestation"] == payload["source_attestation"]
    assert int(repeated["worker_pid"]) != os.getpid()
    assert payload["runtime"] == request["runtime"]
    assert int(payload["worker_pid"]) > 0
    assert int(payload["worker_pid"]) != os.getpid()
    assert payload["source_attestation"]["classes"] == [
        "Chunk",
        "Consec",
        "Replay",
        "Uniform",
    ]
    assert payload["source_attestation"]["bindings"] == [
        "chunk.elements",
        "chunk.numpy",
        "consec.numpy",
        "replay.chunk",
        "replay.elements",
        "replay.numpy",
        "replay.uniform",
        "uniform.numpy",
    ]
    expected_origins = {
        "Chunk.append": "embodied/core/chunk.py",
        "Chunk.update": "embodied/core/chunk.py",
        "Consec.__next__": "embodied/core/streams.py",
        "Replay._sample": "embodied/core/replay.py",
        "Replay.add": "embodied/core/replay.py",
        "Replay.sample": "embodied/core/replay.py",
        "Replay.update": "embodied/core/replay.py",
        "Uniform.__call__": "embodied/core/selectors.py",
        "Uniform.__delitem__": "embodied/core/selectors.py",
        "Uniform.__setitem__": "embodied/core/selectors.py",
    }
    assert payload["source_attestation"]["method_origins"] == {
        name: f"{official_revision(profile)}:{path}"
        for name, path in expected_origins.items()
    }
    assert payload["source_attestation"]["native_module_violations"] == []
    with np.load(fixture_path, allow_pickle=False) as expected:
        expected_names = {
            "capacity.chunk_ids",
            "capacity.fifo",
            "capacity.intermediate6",
            "capacity.intermediate8",
            "capacity.item_ids",
            "capacity.queue_chunks",
            "capacity.queue_offsets",
            "capacity.refs",
            "capacity.rng_after_online",
            "capacity.rng_before_online",
            "capacity.selector_draws",
            "capacity.selector_keys",
            "capacity.start_chunks",
            "capacity.start_offsets",
            "capacity.train_values",
            "collection.deter",
            "collection.stoch",
            "consecutive0.consec",
            "consecutive0.first",
            "consecutive0.last",
            "consecutive0.stepid",
            "consecutive0.values",
            "consecutive1.consec",
            "consecutive1.first",
            "consecutive1.last",
            "consecutive1.stepid",
            "consecutive1.values",
            "raw.item_ids",
            "raw.online_chunks",
            "raw.online_offsets",
            "raw.report_deter",
            "raw.report_first",
            "raw.report_last",
            "raw.report_stepid",
            "raw.report_stoch",
            "raw.report_terminal",
            "raw.report_values",
            "raw.start_chunks",
            "raw.start_offsets",
            "raw.train_values",
            "source_config.batch_length",
            "source_config.chunk_size",
            "source_config.context",
            "source_config.online",
            "source_config.uniform",
            "writeback.deter",
            "writeback.logical_values",
            "writeback.stoch",
        }
        assert set(expected.files) == set(payload["arrays"]) == expected_names
        assert len(expected.files) == len(payload["arrays"]) == 48
        assert expected.files == sorted(payload["arrays"])
        for name in expected.files:
            spec = payload["arrays"][name]
            actual = np.asarray(spec["values"], dtype=spec["dtype"])
            np.testing.assert_array_equal(actual, expected[name], err_msg=name)


@pytest.mark.parametrize("profile", tuple(DreamerProfile))
def test_replay_oracle_worker_never_imports_native_replay(profile) -> None:
    stem = f"{profile.value}-proprio-replay"
    manifest = OracleManifest.load(
        FIXTURE_DIR / f"{stem}.manifest.json",
        fixture_path=FIXTURE_DIR / f"{stem}.npz",
    )
    command = (
        manifest.generator_command[0],
        "-X",
        "importtime",
        *manifest.generator_command[1:],
    )
    completed = subprocess.run(
        command,
        cwd=OFFICIAL_CHECKOUT,
        input=manifest.generator_request,
        check=True,
        capture_output=True,
        text=True,
    )
    imported = [
        line.rsplit("|", 1)[-1].strip()
        for line in completed.stderr.splitlines()
        if line.startswith("import time:")
    ]
    assert "world_marl.dreamer_v3_baseline.replay" not in imported
    assert "world_marl.dreamer_v3_baseline.replay_oracle" not in imported


@pytest.mark.parametrize(
    ("module_name", "module_path"),
    [
        ("world_marl.dreamer_v3_baseline.replay", None),
        ("world_marl.dreamer_v3_baseline.replay_oracle", None),
        (
            "untrusted_replay_alias",
            Path(replay_module.__file__).resolve(),
        ),
    ],
)
def test_replay_oracle_worker_rejects_injected_native_modules(
    module_name: str,
    module_path: Path | None,
) -> None:
    manifest = OracleManifest.load(
        FIXTURE_DIR / "paper-proprio-replay.manifest.json",
        fixture_path=FIXTURE_DIR / "paper-proprio-replay.npz",
    )
    oracle_path = Path(manifest.generator_command[1]).resolve()
    lines = [
        "import runpy, sys",
        "from types import ModuleType",
        f"module = ModuleType({module_name!r})",
    ]
    if module_path is not None:
        lines.append(f"module.__file__ = {str(module_path)!r}")
    lines.extend(
        [
            f"sys.modules[{module_name!r}] = module",
            f"sys.argv = [{str(oracle_path)!r}, '_worker']",
            f"runpy.run_path({str(oracle_path)!r}, run_name='__main__')",
        ]
    )
    completed = subprocess.run(
        [manifest.generator_command[0], "-c", "\n".join(lines)],
        cwd=OFFICIAL_CHECKOUT,
        input=manifest.generator_request,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "forbidden native replay modules loaded" in completed.stderr


def test_replay_key_exact_encoding_order_and_fresh_copy() -> None:
    low = ReplayKey(bytes.fromhex("00" * 15 + "01"), 0x01020304)
    high = ReplayKey(bytes.fromhex("00" * 15 + "02"), 0)
    encoded = low.to_step_id()
    assert encoded.dtype == np.uint8
    assert encoded.shape == (20,)
    assert encoded[:16].tobytes() == low.chunk_id
    assert encoded[16:].tolist() == [1, 2, 3, 4]
    assert ReplayKey.from_step_id(encoded) == low
    assert len({low, ReplayKey(low.chunk_id, low.offset)}) == 1
    assert ReplayKey(low.chunk_id, low.offset - 1) < low
    assert low < high
    encoded[:] = 0
    np.testing.assert_array_equal(low.to_step_id()[16:], [1, 2, 3, 4])
    assert ReplayKey.from_state_dict(low.state_dict()) == low
    for broken in (
        {"chunk_id": b"short", "offset": 0},
        {"chunk_id": bytes(16), "offset": -1},
        {"chunk_id": bytes(16), "offset": 2**32},
        {"chunk_id": bytes(16)},
        {"chunk_id": bytes(16), "offset": 0, "extra": 1},
        {"chunk_id": bytes(16), "offset": np.int64(0)},
    ):
        with pytest.raises((TypeError, ValueError)):
            ReplayKey.from_state_dict(broken)
    for broken in (None, [], "not-a-state"):
        with pytest.raises((TypeError, ValueError)):
            ReplayKey.from_state_dict(broken)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: ReplayKey(b"short", 0),
        lambda: ReplayKey(bytearray(16), 0),
        lambda: ReplayKey("0" * 16, 0),
        lambda: ReplayKey(bytes(16), -1),
        lambda: ReplayKey(bytes(16), 1.0),
        lambda: ReplayKey(bytes(16), True),
        lambda: ReplayKey(bytes(16), 2**32),
        lambda: ReplayKey.from_step_id(np.zeros((19,), np.uint8)),
        lambda: ReplayKey.from_step_id(np.zeros((20,), np.int8)),
        lambda: ReplayKey.from_step_id(np.zeros((1, 20), np.uint8)),
    ],
)
def test_replay_key_rejects_invalid_identity(operation) -> None:
    with pytest.raises((TypeError, ValueError)):
        operation()


def test_replay_batch_validates_copies_slices_and_as_dict() -> None:
    data = {
        "value": np.arange(10, dtype=np.int32).reshape(2, 5),
        "is_first": np.zeros((2, 5), bool),
    }
    step_ids = np.zeros((2, 5, 20), np.uint8)
    batch = ReplayBatch(data, step_ids)
    data["value"][:] = -1
    step_ids[:] = 9
    np.testing.assert_array_equal(batch.data["value"], np.arange(10).reshape(2, 5))
    assert not np.any(batch.step_ids)
    sliced = batch[:, 1:4]
    assert sliced.step_ids.shape == (2, 3, 20)
    payload = batch.as_dict()
    assert set(payload) == {"value", "is_first", "stepid"}
    payload["value"][:] = -2
    assert batch.data["value"][0, 0] == 0
    with pytest.raises(ValueError):
        ReplayBatch({"x": np.zeros((2, 4))}, np.zeros((2, 5, 20), np.uint8))


def test_row_schema_coercion_chronology_and_no_key_drift() -> None:
    replay = _replay(sequence_length=1, context=0, online=False)
    first = replay.add(_row(0, first=True, last=True, terminal=True))
    second = replay.add(_row(1, first=True))
    third = replay.add(_row(2, first=True))  # abandoned reset is valid.
    assert first.offset == 0 and second.offset == 1 and third.offset == 2
    batch = replay.sample_raw("report", timeout=0.1)
    assert batch.data["obs"].dtype == np.uint8
    assert batch.data["action"].dtype == np.float32
    sampled_index = int(batch.data["value"][0, 0])
    np.testing.assert_array_equal(
        batch.data["dyn/deter"][0, 0],
        [sampled_index + 1000, sampled_index + 1001],
    )
    np.testing.assert_array_equal(
        batch.data["dyn/stoch"][0, 0],
        [[sampled_index + 2000, sampled_index + 2001]],
    )
    with pytest.raises(ValueError, match="first"):
        _replay().add(_row(0, first=False))
    with pytest.raises(ValueError, match="terminal"):
        _replay().add(_row(0, first=True, last=False, terminal=True))
    invalid_next = _replay(sequence_length=1, context=0)
    invalid_next.add(_row(0, first=True, last=True))
    before_invalid_next = invalid_next.state_dict()
    with pytest.raises(ValueError, match="first"):
        invalid_next.add(_row(1, first=False))
    _assert_tree_equal(invalid_next.state_dict(), before_invalid_next)
    nonempty = _replay(sequence_length=1, context=0)
    nonempty.add(_row(0))
    before_nonempty = nonempty.state_dict()
    malformed_nonempty = _row(1, first=False)
    malformed_nonempty["extra"] = 1
    with pytest.raises(ValueError):
        nonempty.add(malformed_nonempty)
    _assert_tree_equal(nonempty.state_dict(), before_nonempty)
    for mutate in (
        lambda row: row.pop("reward"),
        lambda row: row.pop("dyn/deter"),
        lambda row: row.update(extra=1),
        lambda row: row.update(**{"dyn/extra": np.zeros((), np.float32)}),
        lambda row: row.update(stepid=np.zeros(20, np.uint8)),
        lambda row: row.update(consec=np.zeros((), np.int32)),
        lambda row: row.update(obs=[1, 2, 3]),
        lambda row: row.update(**{"dyn/deter": np.zeros((1,), np.float32)}),
        lambda row: row.update(**{"dyn/stoch": np.zeros((1, 2), np.float64)}),
        lambda row: row.update(is_terminal=[False]),
        lambda row: row.update(reward="not-a-number"),
    ):
        target = _replay()
        row = _row(0)
        mutate(row)
        before = target.state_dict()
        with pytest.raises((TypeError, ValueError)):
            target.add(row)
        _assert_tree_equal(target.state_dict(), before)


def test_writer_identity_and_exact_worker_ids_are_enforced_without_mutation() -> None:
    replay = _replay(sequence_length=1, context=0, online=False)
    detached = ReplayWriter(0, replay)
    before = replay.state_dict()
    with pytest.raises(RuntimeError, match="active"):
        detached.add(_row(0))
    _assert_tree_equal(replay.state_dict(), before)

    replay.add(_row(0), worker=0)
    stale = replay.writers[0]
    replay.load_state_dict(replay.state_dict())
    before = replay.state_dict()
    with pytest.raises(RuntimeError, match="active"):
        stale.add(_row(1, first=False))
    _assert_tree_equal(replay.state_dict(), before)

    for worker in (True, False, 1.0, np.int64(1)):
        target = _replay(sequence_length=1, context=0, online=False)
        target.add(_row(0), worker=int(worker))
        before = target.state_dict()
        with pytest.raises(TypeError, match="worker"):
            target.add(_row(1, first=False), worker=worker)
        _assert_tree_equal(target.state_dict(), before)


def test_chunk_fixed_storage_seal_copy_read_and_latent_only_update() -> None:
    chunk_id = (1).to_bytes(16, "big")
    successor = (2).to_bytes(16, "big")
    chunk = ReplayChunk(chunk_id, 2, _transition_spaces(), _latent_spaces())
    assert chunk.transition_data == {}
    assert chunk.latent_data == {}
    empty_state = chunk.state_dict()
    assert all(value.shape[0] == 0 for value in empty_state["transition"].values())
    assert all(value.shape[0] == 0 for value in empty_state["latent"].values())
    empty_restored = ReplayChunk.from_state_dict(
        empty_state, _transition_spaces(), _latent_spaces()
    )
    assert empty_restored.transition_data == {}
    assert empty_restored.latent_data == {}
    first = chunk.append(_row(0))
    assert all(value.shape[0] == 2 for value in chunk.transition_data.values())
    assert all(value.shape[0] == 2 for value in chunk.latent_data.values())
    np.testing.assert_array_equal(chunk.latent_data["dyn/deter"][0], [1000, 1001])
    np.testing.assert_array_equal(chunk.latent_data["dyn/stoch"][0], [[2000, 2001]])
    second = chunk.append(_row(1, first=False))
    assert first == ReplayKey(chunk_id, 0)
    assert second == ReplayKey(chunk_id, 1)
    assert chunk.length == chunk.size == 2
    with pytest.raises((IndexError, RuntimeError, ValueError)):
        chunk.append(_row(2, first=False))
    chunk.seal(successor)
    assert chunk.sealed and chunk.successor_id == successor
    assert all(not value.flags.writeable for value in chunk.transition_data.values())
    assert all(value.flags.writeable for value in chunk.latent_data.values())
    sealed_state = chunk.state_dict()
    with pytest.raises((RuntimeError, ValueError)):
        chunk.seal((3).to_bytes(16, "big"))
    _assert_tree_equal(chunk.state_dict(), sealed_state)
    read = chunk.read(0, 2)
    read["value"][:] = 99
    assert chunk.read(0, 2)["value"].tolist() == [0, 1]
    chunk.update_context(
        1,
        {
            "dyn/deter": np.asarray([5, 6], np.float32),
            "dyn/stoch": np.asarray([[7, 8]], np.float32),
        },
    )
    np.testing.assert_array_equal(chunk.read(1, 1)["dyn/deter"], [[5, 6]])
    with pytest.raises((KeyError, ValueError)):
        chunk.update_context(0, {"reward": np.asarray(4, np.float32)})
    for offset, length in ((-1, 1), (0, 3), (2, 1)):
        with pytest.raises((IndexError, ValueError)):
            chunk.read(offset, length)
    before_bad_update = chunk.state_dict()
    with pytest.raises((IndexError, ValueError)):
        chunk.update_context(
            2,
            {
                "dyn/deter": np.asarray([1, 2], np.float32),
                "dyn/stoch": np.asarray([[3, 4]], np.float32),
            },
        )
    _assert_tree_equal(chunk.state_dict(), before_bad_update)
    with pytest.raises((RuntimeError, ValueError)):
        chunk.append(_row(2, first=False))
    _assert_tree_equal(chunk.state_dict(), before_bad_update)
    partial = ReplayChunk(
        (9).to_bytes(16, "big"), 2, _transition_spaces(), _latent_spaces()
    )
    partial.append(_row(0))
    with pytest.raises((TypeError, ValueError)):
        partial.seal(b"short")
    partial.seal((10).to_bytes(16, "big"))
    partial_sealed = partial.state_dict()
    with pytest.raises((RuntimeError, ValueError)):
        partial.append(_row(1, first=False))
    _assert_tree_equal(partial.state_dict(), partial_sealed)
    restored = ReplayChunk.from_state_dict(
        chunk.state_dict(), _transition_spaces(), _latent_spaces()
    )
    assert restored.sealed and restored.successor_id == successor
    assert not restored.transition_data["value"].flags.writeable


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update(**{"dyn/deter": np.zeros((1,), np.float32)}),
        lambda row: row.update(**{"dyn/stoch": np.zeros((1, 2), np.float64)}),
    ],
)
@pytest.mark.parametrize("populated", [False, True], ids=["fresh", "partial"])
def test_chunk_latent_append_validation_is_transactional(
    mutate,
    populated: bool,
) -> None:
    chunk = ReplayChunk(
        (17).to_bytes(16, "big"),
        3,
        _transition_spaces(),
        _latent_spaces(),
    )
    if populated:
        chunk.append(_row(0))
    before = chunk.state_dict()
    transition_allocated = bool(chunk.transition_data)
    latent_allocated = bool(chunk.latent_data)
    row = _row(1, first=False)
    mutate(row)
    with pytest.raises((TypeError, ValueError)):
        chunk.append(row)
    assert chunk.length == int(populated)
    assert bool(chunk.transition_data) is transition_allocated
    assert bool(chunk.latent_data) is latent_allocated
    _assert_tree_equal(chunk.state_dict(), before)


def test_interleaved_workers_keep_independent_streams_links_and_resets() -> None:
    replay = _replay(sequence_length=2, context=0, online=False, chunk_size=2)
    histories = {10: [], 20: []}
    for index in range(5):
        histories[10].append(replay.add(_row(index, first=index in (0, 3)), worker=10))
        histories[20].append(
            replay.add(_row(100 + index, first=index in (0, 2)), worker=20)
        )
    assert set(replay.writers) == {10, 20}
    assert all(isinstance(writer, ReplayWriter) for writer in replay.writers.values())
    starts = list(replay.items.values())
    assert len(starts) == 8
    worker_chunks = {
        worker: {key.chunk_id for key in histories[worker]} for worker in histories
    }
    assert worker_chunks[10].isdisjoint(worker_chunks[20])
    for worker, writer in replay.writers.items():
        assert writer.row_count == 5, worker
        assert len(writer.pending) == 1
    for worker, chunk_ids in worker_chunks.items():
        for chunk_id in chunk_ids:
            chunk = replay.chunks[chunk_id]
            if chunk.successor_id is not None:
                assert chunk.successor_id in chunk_ids, worker
    for start in replay.items.values():
        owners = [worker for worker, keys in histories.items() if start in keys]
        assert len(owners) == 1
        resolved = _resolve_keys(replay, start, 2)
        assert all(key.chunk_id in worker_chunks[owners[0]] for key in resolved)
    assert all(replay.refs[chunk_id] > 0 for chunk_id in replay.chunks)
    replay.validate()


def test_exact_valid_starts_cross_chunks_and_episode_boundaries() -> None:
    official = _official_arrays()
    replay = _replay(sequence_length=4, context=1, online=False)
    _add_rows(replay, 11)
    expected = [
        ReplayKey(int(chunk).to_bytes(16, "big"), int(offset))
        for chunk, offset in zip(
            official["raw.start_chunks"],
            official["raw.start_offsets"],
            strict=True,
        )
    ]
    assert list(replay.items.values()) == expected
    np.testing.assert_array_equal(list(replay.items), official["raw.item_ids"])
    assert len(replay) == len(official["raw.item_ids"])
    for start in replay.items.values():
        resolved = _resolve_keys(replay, start, 5)
        assert resolved[0] == start and len(resolved) == 5
    assert ReplayKey((3).to_bytes(16, "big"), 1) not in replay.items.values()


def test_policy_collection_latents_match_exact_official_storage_and_restore() -> None:
    official = _official_arrays()
    replay = _replay(sequence_length=4, context=1, online=False)
    _add_rows(replay, 11)
    ordered = [replay.chunks[key] for key in sorted(replay.chunks)]
    stored_deter = np.concatenate(
        [chunk.read(0, chunk.length)["dyn/deter"] for chunk in ordered]
    )
    stored_stoch = np.concatenate(
        [chunk.read(0, chunk.length)["dyn/stoch"] for chunk in ordered]
    )
    np.testing.assert_array_equal(stored_deter, official["collection.deter"])
    np.testing.assert_array_equal(stored_stoch, official["collection.stoch"])
    assert np.any(stored_deter) and np.any(stored_stoch)

    sampled = replay.sample_raw("report", timeout=0.1)
    np.testing.assert_array_equal(
        sampled.data["dyn/deter"], official["raw.report_deter"]
    )
    np.testing.assert_array_equal(
        sampled.data["dyn/stoch"], official["raw.report_stoch"]
    )

    restored = _replay(sequence_length=4, context=1, online=False)
    restored.load_state_dict(replay.state_dict())
    _assert_tree_equal(restored.state_dict(), replay.state_dict())
    restored_chunks = [restored.chunks[key] for key in sorted(restored.chunks)]
    np.testing.assert_array_equal(
        np.concatenate(
            [chunk.read(0, chunk.length)["dyn/deter"] for chunk in restored_chunks]
        ),
        official["collection.deter"],
    )
    np.testing.assert_array_equal(
        np.concatenate(
            [chunk.read(0, chunk.length)["dyn/stoch"] for chunk in restored_chunks]
        ),
        official["collection.stoch"],
    )


def test_exact_online_phase_fifo_per_worker_and_nonconsuming_report() -> None:
    official = _official_arrays()
    replay = _replay(sequence_length=4, context=1, online=True)
    _add_rows(replay, 11)
    expected = [
        ReplayKey(int(chunk).to_bytes(16, "big"), int(offset))
        for chunk, offset in zip(
            official["raw.online_chunks"],
            official["raw.online_offsets"],
            strict=True,
        )
    ]
    assert replay.online_queue.keys == expected
    before = replay.online_queue.state_dict()
    replay.sample_raw("report", timeout=0.1)
    replay.sample_raw("eval", timeout=0.1)
    assert replay.online_queue.state_dict() == before
    train0 = replay.sample_raw("train", timeout=0.1)
    train1 = replay.sample_raw("train", timeout=0.1)
    np.testing.assert_array_equal(train0.data["value"], official["raw.train_values"])
    np.testing.assert_array_equal(train1.data["value"], official["raw.report_values"])
    assert len(replay.online_queue) == 0

    multi = _replay(sequence_length=3, context=0, online=True)
    histories = {1: [], 2: []}
    for index in range(7):
        histories[1].append(multi.add(_row(index, first=index == 0), worker=1))
        histories[2].append(multi.add(_row(100 + index, first=index == 0), worker=2))
    assert multi.online_queue.keys == [
        histories[1][1],
        histories[2][1],
        histories[1][4],
        histories[2][4],
    ]


def test_online_first_batch_stale_skip_uniform_fallback_without_rng_draw() -> None:
    official = _official_arrays()
    replay = _replay(
        capacity=3,
        sequence_length=3,
        context=1,
        online=True,
        batch_size=3,
    )
    _add_rows(replay, 10, natural_last=True)
    assert replay.online_queue.keys[0].chunk_id == (1).to_bytes(16, "big")
    assert replay.online_queue.keys[1] == ReplayKey((2).to_bytes(16, "big"), 2)
    np.testing.assert_array_equal(
        _selector_rng_bytes(replay.selector), official["capacity.rng_before_online"]
    )
    first = replay.sample_raw("train", timeout=0.1)
    np.testing.assert_array_equal(
        first.data["value"][0:1], official["capacity.train_values"]
    )
    item_starts = {
        int(item_id): ReplayKey(int(chunk).to_bytes(16, "big"), int(offset))
        for item_id, chunk, offset in zip(
            official["capacity.item_ids"],
            official["capacity.start_chunks"],
            official["capacity.start_offsets"],
            strict=True,
        )
    }
    fallback_starts = [
        ReplayKey.from_step_id(first.step_ids[index, 0]) for index in (1, 2)
    ]
    assert fallback_starts == [
        item_starts[int(item_id)] for item_id in official["capacity.selector_draws"][:2]
    ]
    official_rng_after_online = official["capacity.rng_after_online"]
    np.testing.assert_array_equal(
        official["capacity.rng_before_online"], official_rng_after_online
    )
    assert not np.array_equal(
        _selector_rng_bytes(replay.selector), official_rng_after_online
    )  # Only the two uniform fallback rows advance PCG64.


def test_uniform_selector_source_draw_swap_pop_and_statistics() -> None:
    official = _official_arrays()
    selector = UniformSelector(7)
    for item_id in (0, 1, 2):
        selector.insert(item_id)
    selector.delete(0)
    assert selector.keys == [2, 1]
    selector.insert(3)
    np.testing.assert_array_equal(selector.keys, official["capacity.intermediate6"])
    selector.delete(1)
    selector.insert(4)
    selector.delete(2)
    selector.insert(5)
    np.testing.assert_array_equal(selector.keys, official["capacity.intermediate8"])

    exact = UniformSelector(7)
    for item_id in [4, 5, 6]:
        exact.insert(item_id)
    np.testing.assert_array_equal(
        [exact.sample() for _ in range(12)], official["capacity.selector_draws"]
    )
    restored = UniformSelector.from_state_dict(exact.state_dict())
    assert [restored.sample() for _ in range(20)] == [exact.sample() for _ in range(20)]

    statistical = UniformSelector(11)
    for item_id in range(7):
        statistical.insert(item_id)
    counts = np.bincount([statistical.sample() for _ in range(70_000)], minlength=7)
    np.testing.assert_array_equal(
        counts, [9963, 10015, 9842, 10058, 10122, 9978, 10022]
    )
    expected = 10_000
    chi_square = np.square(counts - expected).sum() / expected
    assert chi_square == pytest.approx(4.5774)
    assert np.max(np.abs(counts - expected) / expected) < 0.03


def test_uniform_selector_duplicate_insert_is_atomic_across_threads() -> None:
    selector = UniformSelector(7)
    barrier = threading.Barrier(16)
    outcomes: list[str] = []
    outcome_lock = threading.Lock()

    def insert() -> None:
        barrier.wait()
        try:
            selector.insert(23)
        except ValueError:
            outcome = "duplicate"
        else:
            outcome = "inserted"
        with outcome_lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=insert) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
    assert all(not thread.is_alive() for thread in threads)
    assert outcomes.count("inserted") == 1
    assert outcomes.count("duplicate") == 15
    assert selector.keys == [23]
    assert selector.indices == {23: 0}


def test_capacity_fifo_refs_deletion_stale_queue_and_capacity_one() -> None:
    official = _official_arrays()
    replay = _replay(capacity=3, sequence_length=3, context=1, online=True)
    _add_rows(replay, 10, natural_last=True)
    np.testing.assert_array_equal(list(replay.items), official["capacity.item_ids"])
    np.testing.assert_array_equal(replay.fifo, official["capacity.fifo"])
    np.testing.assert_array_equal(
        replay.selector.keys, official["capacity.selector_keys"]
    )
    np.testing.assert_array_equal(
        [int.from_bytes(value, "big") for value in replay.chunks],
        official["capacity.chunk_ids"],
    )
    np.testing.assert_array_equal(
        [replay.refs[cid] for cid in replay.chunks], official["capacity.refs"]
    )
    queue_chunks = np.asarray(
        [int.from_bytes(key.chunk_id, "big") for key in replay.online_queue.keys]
    )
    queue_offsets = np.asarray([key.offset for key in replay.online_queue.keys])
    np.testing.assert_array_equal(queue_chunks, official["capacity.queue_chunks"])
    np.testing.assert_array_equal(queue_offsets, official["capacity.queue_offsets"])
    assert replay.refs == _recomputed_refs(replay)
    for start in replay.items.values():
        assert len(_resolve_keys(replay, start, 4)) == 4
    assert replay.online_queue.keys[0].chunk_id not in replay.chunks
    assert replay.next_item_id == 7

    one = _replay(capacity=1, sequence_length=1, context=0, online=False)
    _add_rows(one, 6)
    assert len(one) == 1
    assert one.fifo == [5]
    assert one.selector.keys == [5]
    assert one.next_item_id == 6


def test_online_queue_round_trip_preserves_stale_order() -> None:
    queue = OnlineQueue()
    keys = [ReplayKey(index.to_bytes(16, "big"), index) for index in (1, 2, 3)]
    for key in keys:
        queue.push(key)
    assert queue.pop() == keys[0]
    restored = OnlineQueue.from_state_dict(queue.state_dict())
    assert restored.keys == keys[1:]


def test_restore_rejects_live_unresolvable_online_keys_but_accepts_evicted_stale() -> (
    None
):
    replay = _replay(capacity=3, sequence_length=3, context=1, online=True)
    _add_rows(replay, 10, natural_last=True)
    before = replay.state_dict()
    assert before["online_queue"]["keys"][0]["chunk_id"] not in replay.chunks

    restored = _replay(capacity=3, sequence_length=3, context=1, online=True)
    restored.load_state_dict(before)
    _assert_tree_equal(restored.state_dict(), before)

    live_index = next(
        index
        for index, value in enumerate(before["online_queue"]["keys"])
        if value["chunk_id"] in replay.chunks
    )
    for corrupt in (
        lambda key: key.update(offset=999),
        lambda key: key.update(
            chunk_id=replay.writers[0].current_chunk_id,
            offset=replay.writers[0].current_offset - 1,
        ),
    ):
        broken = copy.deepcopy(before)
        corrupt(broken["online_queue"]["keys"][live_index])
        with pytest.raises((TypeError, ValueError)):
            replay.load_state_dict(broken)
        _assert_tree_equal(replay.state_dict(), before)


def test_raw_batch_annotation_copy_terminal_and_stepid_chronology() -> None:
    official = _official_arrays()
    replay = _replay(sequence_length=4, context=1, online=True)
    _add_rows(replay, 11)
    replay.sample_raw("train", timeout=0.1)
    stored = {
        chunk_id: chunk.read(0, chunk.length)
        for chunk_id, chunk in replay.chunks.items()
    }
    report = replay.sample_raw("report", timeout=0.1)
    np.testing.assert_array_equal(report.data["value"], official["raw.report_values"])
    np.testing.assert_array_equal(report.data["is_first"], official["raw.report_first"])
    np.testing.assert_array_equal(report.data["is_last"], official["raw.report_last"])
    np.testing.assert_array_equal(
        report.data["is_terminal"], official["raw.report_terminal"]
    )
    np.testing.assert_array_equal(report.step_ids, official["raw.report_stepid"])
    assert "is_online" not in report.data
    keys = _decode(report.step_ids)
    assert keys == [
        ReplayKey((3).to_bytes(16, "big"), 0),
        ReplayKey((3).to_bytes(16, "big"), 1),
        ReplayKey((3).to_bytes(16, "big"), 2),
        ReplayKey((4).to_bytes(16, "big"), 0),
        ReplayKey((4).to_bytes(16, "big"), 1),
    ]
    for value in report.data.values():
        value[...] = False if value.dtype == np.bool_ else 0
    report.step_ids[:] = 0
    payload = report.as_dict()
    for value in payload.values():
        value[...] = 1
    for chunk_id, expected in stored.items():
        actual = replay.chunks[chunk_id].read(0, replay.chunks[chunk_id].length)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def _raw_batch() -> ReplayBatch:
    official = _official_arrays()
    data = {
        "is_first": official["raw.report_first"],
        "is_last": official["raw.report_last"],
        "is_terminal": official["raw.report_terminal"],
        "value": official["raw.report_values"],
    }
    return ReplayBatch(data, official["raw.report_stepid"])


def test_consecutive_exact_slices_overlap_fetch_and_complete_state_round_trip() -> None:
    official = _official_arrays()
    calls = []

    def source() -> ReplayBatch:
        calls.append(1)
        return _raw_batch()

    stream = ConsecutiveStream(
        source,
        sequence_length=2,
        consecutive=2,
        context=1,
    )
    first = next(stream)
    saved_mid = stream.state_dict()
    second = next(stream)
    for prefix, batch in (("consecutive0", first), ("consecutive1", second)):
        np.testing.assert_array_equal(batch.data["value"], official[f"{prefix}.values"])
        np.testing.assert_array_equal(
            batch.data["consec"], official[f"{prefix}.consec"]
        )
        np.testing.assert_array_equal(
            batch.data["is_first"], official[f"{prefix}.first"]
        )
        np.testing.assert_array_equal(batch.data["is_last"], official[f"{prefix}.last"])
        np.testing.assert_array_equal(batch.step_ids, official[f"{prefix}.stepid"])
    assert first.data["consec"].dtype == np.int32
    np.testing.assert_array_equal(first.step_ids[:, -1], second.step_ids[:, 0])
    assert len(calls) == 1
    assert stream.index == 2

    restored_mid = ConsecutiveStream.from_state_dict(
        saved_mid,
        source,
        sequence_length=2,
        consecutive=2,
        context=1,
    )
    resumed = next(restored_mid)
    np.testing.assert_array_equal(resumed.as_dict()["value"], second.as_dict()["value"])
    assert len(calls) == 1
    complete = ConsecutiveStream.from_state_dict(
        stream.state_dict(),
        source,
        sequence_length=2,
        consecutive=2,
        context=1,
    )
    next(complete)
    assert len(calls) == 2
    for kwargs in (
        {"sequence_length": 0, "consecutive": 2, "context": 1},
        {"sequence_length": 2, "consecutive": 0, "context": 1},
        {"sequence_length": 2, "consecutive": 2, "context": -1},
    ):
        with pytest.raises(ValueError):
            ConsecutiveStream(source, **kwargs)
    for kwargs in (
        {"sequence_length": True, "consecutive": 2, "context": 1},
        {"sequence_length": 2.0, "consecutive": 2, "context": 1},
        {"sequence_length": np.int64(2), "consecutive": 2, "context": 1},
        {"sequence_length": 2.5, "consecutive": 2, "context": 1},
        {"sequence_length": 2, "consecutive": False, "context": 1},
        {"sequence_length": 2, "consecutive": 2.0, "context": 1},
        {"sequence_length": 2, "consecutive": np.int64(2), "context": 1},
        {"sequence_length": 2, "consecutive": 2.5, "context": 1},
        {"sequence_length": 2, "consecutive": 2, "context": True},
        {"sequence_length": 2, "consecutive": 2, "context": 1.0},
        {"sequence_length": 2, "consecutive": 2, "context": np.int64(1)},
        {"sequence_length": 2, "consecutive": 2, "context": 1.5},
    ):
        with pytest.raises(TypeError):
            ConsecutiveStream(source, **kwargs)

    aggregate = _replay(
        sequence_length=2,
        context=1,
        consecutive=2,
        online=False,
    )
    _add_rows(aggregate, 11)
    next(aggregate.consecutive_stream)
    rng_after_fetch = _selector_rng_bytes(aggregate.selector)
    next(aggregate.consecutive_stream)
    np.testing.assert_array_equal(
        _selector_rng_bytes(aggregate.selector), rng_after_fetch
    )


def _stored_leaf(replay: DreamerReplay, key: ReplayKey, name: str) -> np.ndarray:
    return replay.chunks[key.chunk_id].read(key.offset, 1)[name][0]


def test_latent_writeback_same_cross_chunk_post_context_and_atomic_validation() -> None:
    official = _official_arrays()
    replay = _replay(sequence_length=4, context=1, online=True)
    _add_rows(replay, 11)
    replay.sample_raw("train", timeout=0.1)
    report = replay.sample_raw("report", timeout=0.1)
    ids = report.step_ids[:, 1:].copy()
    deter = np.arange(8, dtype=np.float32).reshape(1, 4, 2) + 100
    stoch = np.arange(8, dtype=np.float32).reshape(1, 4, 1, 2) + 200
    ids_before = ids.copy()
    deter_before = deter.copy()
    stoch_before = stoch.copy()
    context_deter = report.data["dyn/deter"][0, 0].copy()
    context_stoch = report.data["dyn/stoch"][0, 0].copy()
    assert (
        replay.update_context(
            ids,
            {"dyn/deter": deter, "dyn/stoch": stoch},
        )
        == 4
    )
    np.testing.assert_array_equal(ids, ids_before)
    np.testing.assert_array_equal(deter, deter_before)
    np.testing.assert_array_equal(stoch, stoch_before)
    keys = _decode(report.step_ids)
    np.testing.assert_array_equal(
        _stored_leaf(replay, keys[0], "dyn/deter"), context_deter
    )
    np.testing.assert_array_equal(
        _stored_leaf(replay, keys[0], "dyn/stoch"), context_stoch
    )
    for index, key in enumerate(keys[1:]):
        np.testing.assert_array_equal(
            _stored_leaf(replay, key, "dyn/deter"), deter[0, index]
        )
        np.testing.assert_array_equal(
            _stored_leaf(replay, key, "dyn/stoch"), stoch[0, index]
        )
    ordered_chunks = [replay.chunks[key] for key in sorted(replay.chunks)]
    np.testing.assert_array_equal(
        np.concatenate(
            [chunk.read(0, chunk.length)["dyn/deter"] for chunk in ordered_chunks]
        ),
        official["writeback.deter"],
    )
    np.testing.assert_array_equal(
        np.concatenate(
            [chunk.read(0, chunk.length)["dyn/stoch"] for chunk in ordered_chunks]
        ),
        official["writeback.stoch"],
    )
    np.testing.assert_array_equal(
        np.concatenate(
            [chunk.read(0, chunk.length)["value"] for chunk in ordered_chunks]
        ),
        official["writeback.logical_values"],
    )

    before = replay.state_dict()
    malformed = ids.copy()
    malformed[0, -1] = ReplayKey((99).to_bytes(16, "big"), 0).to_step_id()
    with pytest.raises(ValueError, match="step"):
        replay.update_context(
            malformed,
            {"dyn/deter": deter, "dyn/stoch": stoch},
        )
    _assert_tree_equal(replay.state_dict(), before)
    for values in (
        {"dyn/deter": deter},
        {"dyn/deter": deter.astype(np.float64), "dyn/stoch": stoch},
        {"dyn/deter": deter[..., :1], "dyn/stoch": stoch},
        {"dyn/deter": deter, "dyn/stoch": stoch, "reward": np.zeros((1, 4))},
    ):
        with pytest.raises((TypeError, ValueError)):
            replay.update_context(ids, values)
    for bad_ids in (
        ids.astype(np.int8),
        ids[..., :19],
        ids[0],
    ):
        with pytest.raises((TypeError, ValueError)):
            replay.update_context(
                bad_ids,
                {"dyn/deter": deter, "dyn/stoch": stoch},
            )

    same = _replay(
        chunk_size=4,
        sequence_length=2,
        context=0,
        online=False,
    )
    _add_rows(same, 3)
    same_batch = same.sample_raw("report", timeout=0.1)
    same_keys = _decode(same_batch.step_ids)
    assert len({key.chunk_id for key in same_keys}) == 1
    same.update_context(
        same_batch.step_ids,
        {
            "dyn/deter": np.asarray([[[1, 2], [3, 4]]], np.float32),
            "dyn/stoch": np.asarray([[[[5, 6]], [[7, 8]]]], np.float32),
        },
    )
    np.testing.assert_array_equal(_stored_leaf(same, same_keys[1], "dyn/deter"), [3, 4])


def test_stale_first_writeback_skips_whole_row_without_partial_mutation() -> None:
    replay = _replay(capacity=1, sequence_length=2, context=0, online=False)
    initial = _add_rows(replay, 3)
    stale_ids = np.stack([key.to_step_id() for key in initial[:2]])[None]
    _add_rows(replay, 10)
    before = replay.state_dict()
    result = replay.update_context(
        stale_ids,
        {
            "dyn/deter": np.ones((1, 2, 2), np.float32),
            "dyn/stoch": np.ones((1, 2, 1, 2), np.float32),
        },
    )
    assert result == 0
    after = replay.state_dict()
    assert after["metrics"]["stale_updates"] == before["metrics"]["stale_updates"] + 1
    before["metrics"]["stale_updates"] += 1
    before["metrics"]["update_calls"] += 1
    _assert_tree_equal(after, before)
    restored = _replay(capacity=1, sequence_length=2, context=0, online=False)
    restored.load_state_dict(after)
    assert restored.stats()["stale_updates"] == 1
    assert restored.stats(reset=True)["stale_updates"] == 1
    assert restored.stats()["stale_updates"] == 0


def _complete_resume_scenario() -> DreamerReplay:
    replay = _replay(
        capacity=3,
        chunk_size=3,
        sequence_length=2,
        context=1,
        consecutive=2,
        online=True,
        seed=7,
    )
    _add_rows(replay, 11, natural_last=True)
    raw = replay.sample_raw("report", timeout=0.1)
    replay.update_context(
        raw.step_ids[:, 1:3],
        {
            "dyn/deter": np.asarray([[[9, 10], [11, 12]]], np.float32),
            "dyn/stoch": np.asarray([[[[13, 14]], [[15, 16]]]], np.float32),
        },
    )
    replay.sample_raw("report", timeout=0.1)
    replay.sample("report", timeout=0.1)
    assert replay.writers[0].current_chunk_id is not None
    assert replay.chunks[replay.writers[0].current_chunk_id].length == 2
    assert len(replay.online_queue) == 2
    assert replay.online_queue.keys[0].chunk_id not in replay.chunks
    assert replay.online_queue.keys[1].chunk_id in replay.chunks
    stream = replay.consecutive_streams["report"]
    assert stream.index == 1
    assert stream.current is not None
    assert any(
        np.any(chunk.read(0, chunk.length)["dyn/deter"])
        for chunk in replay.chunks.values()
    )
    return replay


def _evicted_consecutive_current_scenario() -> DreamerReplay:
    replay = _replay(
        capacity=1,
        chunk_size=2,
        sequence_length=2,
        context=0,
        consecutive=2,
        online=False,
    )
    _add_rows(replay, 5)
    replay.sample("report", timeout=0.1)
    current = replay.consecutive_streams["report"].current
    assert current is not None
    retained_ids = {key.chunk_id for key in _decode(current.step_ids)}
    for index in range(5, 25):
        replay.add(_row(index, first=False))
    assert retained_ids.isdisjoint(replay.chunks)
    return replay


def test_writer_chunk_history_persists_interleaved_allocations_and_round_trips() -> (
    None
):
    replay = _replay(
        chunk_size=2,
        sequence_length=1,
        context=0,
        online=False,
    )
    replay.add(_row(0), worker=7)
    replay.add(_row(100, first=True), worker=8)
    replay.add(_row(1, first=False), worker=7)
    replay.add(_row(101, first=False), worker=8)
    state = replay.state_dict()
    assert state["schema_version"] == 2
    assert [
        int.from_bytes(value, "big") for value in state["writers"][7]["chunk_history"]
    ] == [1, 3]
    assert [
        int.from_bytes(value, "big") for value in state["writers"][8]["chunk_history"]
    ] == [2, 4]
    restored = _replay(
        chunk_size=2,
        sequence_length=1,
        context=0,
        online=False,
    )
    restored.load_state_dict(state)
    _assert_tree_equal(restored.state_dict(), state)


def test_chunk_allocation_requires_active_writer_without_partial_mutation() -> None:
    replay = _replay(
        chunk_size=2,
        sequence_length=1,
        context=0,
        online=False,
    )
    before = replay.state_dict()
    with pytest.raises(RuntimeError, match="active writer"):
        replay._new_chunk(1, 7)
    _assert_tree_equal(replay.state_dict(), before)


def test_chunk_size_one_history_includes_empty_successor_and_binds_row_count() -> None:
    replay = _replay(
        chunk_size=1,
        sequence_length=1,
        context=0,
        online=False,
    )
    replay.add(_row(0))
    state = replay.state_dict()
    history = state["writers"][0]["chunk_history"]
    assert [int.from_bytes(value, "big") for value in history] == [1, 2]
    assert state["writers"][0]["current_chunk_id"] == history[-1]
    current = next(
        chunk for chunk in state["chunks"] if chunk["chunk_id"] == history[-1]
    )
    assert current["length"] == 0
    replay.add(_row(1, first=False))
    state = replay.state_dict()
    history = state["writers"][0]["chunk_history"]
    assert [int.from_bytes(value, "big") for value in history] == [1, 2, 3]
    assert len(history) == state["writers"][0]["row_count"] + 1
    restored = _replay(
        chunk_size=1,
        sequence_length=1,
        context=0,
        online=False,
    )
    restored.load_state_dict(state)
    _assert_tree_equal(restored.state_dict(), state)
    replay.add(_row(2, first=False))
    restored.add(_row(2, first=False))
    _assert_tree_equal(restored.state_dict(), replay.state_dict())
    state = restored.state_dict()
    broken = copy.deepcopy(state)
    broken["writers"][0]["chunk_history"].pop(0)
    _assert_restore_rejected_without_mutation(restored, broken)


@pytest.mark.parametrize(
    "corruption",
    ["missing", "duplicate", "reversed", "nonbytes", "zero", "shared"],
)
def test_restore_rejects_corrupt_lifetime_chunk_history_without_mutation(
    corruption: str,
) -> None:
    replay = _replay(
        chunk_size=2,
        sequence_length=1,
        context=0,
        online=False,
    )
    for index in range(4):
        replay.add(_row(index, first=index == 0), worker=7)
        replay.add(_row(100 + index, first=index == 0), worker=8)
    broken = copy.deepcopy(replay.state_dict())
    left = broken["writers"][7]["chunk_history"]
    right = broken["writers"][8]["chunk_history"]
    if corruption == "missing":
        left.pop(0)
    elif corruption == "duplicate":
        left[1] = left[0]
    elif corruption == "reversed":
        left.reverse()
    elif corruption == "nonbytes":
        left[0] = np.bytes_(left[0])
    elif corruption == "zero":
        left[0] = bytes(16)
    else:
        right[0] = left[0]
    _assert_restore_rejected_without_mutation(replay, broken)


def test_capacity_one_idle_writer_with_evicted_predecessor_resumes_exactly() -> None:
    replay = _replay(
        capacity=1,
        chunk_size=2,
        sequence_length=1,
        context=0,
        online=False,
    )
    replay.add(_row(0), worker=1)
    replay.add(_row(1, first=False), worker=1)
    replay.add(_row(100, first=True), worker=2)
    idle = replay.writers[1]
    assert idle.current_chunk_id is not None
    assert replay.chunks[idle.current_chunk_id].length == 0
    assert not any(
        chunk.successor_id == idle.current_chunk_id for chunk in replay.chunks.values()
    )
    state = replay.state_dict()
    restored = _replay(
        capacity=1,
        chunk_size=2,
        sequence_length=1,
        context=0,
        online=False,
    )
    restored.load_state_dict(state)
    _assert_tree_equal(restored.state_dict(), state)
    replay.add(_row(2, first=False), worker=1)
    restored.add(_row(2, first=False), worker=1)
    _assert_tree_equal(restored.state_dict(), replay.state_dict())


def test_idle_writer_with_evicted_terminal_predecessor_requires_first() -> None:
    replay = _replay(
        capacity=1,
        chunk_size=2,
        sequence_length=1,
        context=0,
        online=False,
    )
    replay.add(_row(0), worker=1)
    replay.add(_row(1, first=False, last=True), worker=1)
    replay.add(_row(100, first=True), worker=2)
    state = replay.state_dict()
    assert state["writers"][1]["last_is_last"] is True
    restored = _replay(
        capacity=1,
        chunk_size=2,
        sequence_length=1,
        context=0,
        online=False,
    )
    restored.load_state_dict(state)
    before = restored.state_dict()
    with pytest.raises(ValueError, match="after is_last"):
        restored.add(_row(2, first=False), worker=1)
    _assert_tree_equal(restored.state_dict(), before)
    replay.add(_row(2, first=True), worker=1)
    restored.add(_row(2, first=True), worker=1)
    _assert_tree_equal(restored.state_dict(), replay.state_dict())


def test_restore_rejects_opposing_per_writer_counter_drift_without_mutation() -> None:
    replay = _replay(
        capacity=30,
        chunk_size=3,
        sequence_length=2,
        context=0,
        online=True,
    )
    for index in range(7):
        replay.add(_row(index, first=index == 0), worker=1)
        replay.add(_row(100 + index, first=index == 0), worker=2)
    pristine = replay.state_dict()
    clone = _replay(
        capacity=30,
        chunk_size=3,
        sequence_length=2,
        context=0,
        online=True,
    )
    clone.load_state_dict(pristine)
    broken = copy.deepcopy(pristine)
    writers = broken["writers"]
    writers[1]["row_count"] += replay.config.chunk_size
    writers[1]["emitted_count"] += replay.config.chunk_size
    writers[2]["row_count"] -= replay.config.chunk_size
    writers[2]["emitted_count"] -= replay.config.chunk_size
    _assert_restore_rejected_without_mutation(replay, broken)
    for index in range(7, 11):
        left = replay.add(_row(index, first=False), worker=1)
        right = clone.add(_row(index, first=False), worker=1)
        assert left == right
        left = replay.add(_row(100 + index, first=False), worker=2)
        right = clone.add(_row(100 + index, first=False), worker=2)
        assert left == right
    actual = replay.sample_raw("train", timeout=0.1)
    expected = clone.sample_raw("train", timeout=0.1)
    _assert_tree_equal(actual.as_dict(), expected.as_dict())
    _assert_tree_equal(replay.state_dict(), clone.state_dict())


@pytest.mark.parametrize("corruption", ["size_mismatch", "zero_live"])
def test_restore_rejects_invalid_live_chunk_geometry_without_mutation(
    corruption: str,
) -> None:
    replay = _replay(
        capacity=20,
        chunk_size=3,
        sequence_length=1,
        context=0,
        online=False,
    )
    _add_rows(replay, 5)
    broken = copy.deepcopy(replay.state_dict())
    if corruption == "size_mismatch":
        sealed = next(chunk for chunk in broken["chunks"] if chunk["sealed"])
        sealed["size"] += 1
    else:
        zero = bytes(16)
        old = broken["chunks"][0]["chunk_id"]
        assert len(broken["chunks"]) == 2
        first = broken["chunks"][0]
        successor = first["successor"]
        first["chunk_id"] = zero
        broken["refs"][zero] = broken["refs"].pop(old)
        for item in broken["items"]:
            if item["key"]["chunk_id"] == old:
                item["key"]["chunk_id"] = zero
        first["successor"] = successor
        for writer in broken["writers"].values():
            writer["chunk_history"] = [
                zero if chunk_id == old else chunk_id
                for chunk_id in writer["chunk_history"]
            ]
            for pending in writer["pending"]:
                if pending["chunk_id"] == old:
                    pending["chunk_id"] = zero
        broken["chunks"].sort(key=lambda chunk: chunk["chunk_id"])
    _assert_restore_rejected_without_mutation(replay, broken)


@pytest.mark.parametrize("corruption", ["sealed_nonfull", "open_full"])
def test_chunk_restore_rejects_noncanonical_seal_geometry(corruption: str) -> None:
    chunk = ReplayChunk(
        (1).to_bytes(16, "big"),
        3,
        _transition_spaces(),
        _latent_spaces(),
        owner_id=0,
    )
    chunk.append({name: value for name, value in _row(0).items()})
    chunk.append({name: value for name, value in _row(1, first=False).items()})
    state = chunk.state_dict()
    if corruption == "sealed_nonfull":
        state["sealed"] = True
        state["successor"] = (2).to_bytes(16, "big")
    else:
        chunk.append({name: value for name, value in _row(2, first=False).items()})
        state = chunk.state_dict()
    with pytest.raises(ValueError, match="chunk"):
        ReplayChunk.from_state_dict(state, _transition_spaces(), _latent_spaces())


def test_per_writer_root_chronology_is_independent_of_unrelated_eviction() -> None:
    replay = _replay(
        capacity=3,
        chunk_size=2,
        sequence_length=1,
        context=0,
        online=False,
    )
    for index in range(5):
        replay.add(_row(index, first=index == 0), worker=2)
    replay.add(_row(100, first=True), worker=1)
    state = replay.state_dict()
    restored = _replay(
        capacity=3,
        chunk_size=2,
        sequence_length=1,
        context=0,
        online=False,
    )
    restored.load_state_dict(state)
    _assert_tree_equal(restored.state_dict(), state)
    broken = copy.deepcopy(state)
    owner_root = next(chunk for chunk in broken["chunks"] if chunk["owner_id"] == 1)
    owner_root["transition"]["is_first"][0] = False
    _assert_restore_rejected_without_mutation(replay, broken)


def test_restore_rejects_duplicate_item_replay_keys_without_mutation() -> None:
    replay = _replay(
        capacity=20,
        chunk_size=3,
        sequence_length=1,
        context=0,
        online=False,
    )
    _add_rows(replay, 4)
    broken = copy.deepcopy(replay.state_dict())
    assert (
        broken["items"][0]["key"]["chunk_id"] == broken["items"][1]["key"]["chunk_id"]
    )
    broken["items"][1]["key"] = copy.deepcopy(broken["items"][0]["key"])
    _assert_restore_rejected_without_mutation(replay, broken)


@pytest.mark.parametrize(
    "corruption",
    [
        "zero",
        "future",
        "bad_offset",
        "wrong_phase",
        "duplicate",
        "duplicate_live",
        "reverse",
    ],
)
def test_restore_rejects_invalid_or_reordered_stale_online_queue_keys(
    corruption: str,
) -> None:
    replay = _complete_resume_scenario()
    broken = copy.deepcopy(replay.state_dict())
    queue = broken["online_queue"]["keys"]
    assert queue[0]["chunk_id"] not in replay.chunks
    if corruption == "zero":
        queue[0] = {"chunk_id": bytes(16), "offset": 0}
    elif corruption == "future":
        queue[0] = {
            "chunk_id": replay.next_chunk_id.to_bytes(16, "big"),
            "offset": 0,
        }
    elif corruption == "bad_offset":
        queue[0]["offset"] = replay.config.chunk_size
    elif corruption == "wrong_phase":
        queue[0]["offset"] = 2
    elif corruption == "duplicate":
        queue.append(copy.deepcopy(queue[0]))
    elif corruption == "duplicate_live":
        queue.append(copy.deepcopy(queue[-1]))
    else:
        queue.reverse()
    _assert_restore_rejected_without_mutation(replay, broken)


def test_restore_rejects_live_online_queue_key_at_wrong_writer_phase() -> None:
    replay = _replay(
        capacity=20,
        chunk_size=4,
        sequence_length=2,
        context=0,
        online=True,
    )
    _add_rows(replay, 8)
    broken = copy.deepcopy(replay.state_dict())
    key = broken["online_queue"]["keys"][0]
    assert key["offset"] == 1
    key["offset"] = 2
    _assert_restore_rejected_without_mutation(replay, broken)


def test_online_queue_restore_allows_cross_writer_absolute_order_interleaving() -> None:
    replay = _replay(
        capacity=20,
        chunk_size=4,
        sequence_length=2,
        context=0,
        online=True,
    )
    for index in range(5):
        replay.add(_row(index, first=index == 0), worker=1)
    for index in range(3):
        replay.add(_row(100 + index, first=index == 0), worker=2)
    state = replay.state_dict()
    histories = {
        chunk_id: (worker, ordinal)
        for worker, writer in state["writers"].items()
        for ordinal, chunk_id in enumerate(writer["chunk_history"])
    }
    positions = [
        (
            histories[value["chunk_id"]][0],
            histories[value["chunk_id"]][1] * replay.config.chunk_size
            + value["offset"],
        )
        for value in state["online_queue"]["keys"]
    ]
    assert positions == [(1, 1), (1, 3), (2, 1)]
    restored = _replay(
        capacity=20,
        chunk_size=4,
        sequence_length=2,
        context=0,
        online=True,
    )
    restored.load_state_dict(state)
    _assert_tree_equal(restored.state_dict(), state)


def test_online_queue_raw_length_one_phase_round_trips() -> None:
    replay = _replay(
        capacity=20,
        chunk_size=2,
        sequence_length=1,
        context=0,
        online=True,
    )
    _add_rows(replay, 3)
    state = replay.state_dict()
    assert len(state["online_queue"]["keys"]) == 3
    restored = _replay(
        capacity=20,
        chunk_size=2,
        sequence_length=1,
        context=0,
        online=True,
    )
    restored.load_state_dict(state)
    _assert_tree_equal(restored.state_dict(), state)


@pytest.mark.parametrize(
    "corruption",
    [
        "leading",
        "terminal",
        "boundary",
        "boundary_reverse",
        "future_stepid",
        "bad_offset_stepid",
        "duplicate_stepid",
    ],
)
def test_restore_rejects_semantically_invalid_consecutive_current(
    corruption: str,
) -> None:
    replay = _complete_resume_scenario()
    broken = copy.deepcopy(replay.state_dict())
    current = broken["consecutive"]["report"]["current"]
    if corruption == "leading":
        current["is_first"][0, 0] = False
    elif corruption == "terminal":
        current["is_terminal"][0, 0] = True
        current["is_last"][0, 0] = False
    elif corruption == "boundary":
        current["is_last"][0, 0] = True
        current["is_first"][0, 1] = False
    elif corruption == "boundary_reverse":
        current["is_last"][0, 0] = False
        current["is_first"][0, 1] = True
    elif corruption == "future_stepid":
        current["stepid"][0, 0] = ReplayKey(
            replay.next_chunk_id.to_bytes(16, "big"), 0
        ).to_step_id()
    elif corruption == "bad_offset_stepid":
        key = ReplayKey.from_step_id(current["stepid"][0, 0])
        current["stepid"][0, 0] = ReplayKey(
            key.chunk_id, replay.config.chunk_size
        ).to_step_id()
    else:
        current["stepid"][0, 1] = current["stepid"][0, 2]
    _assert_restore_rejected_without_mutation(replay, broken)


def test_consecutive_current_with_fully_evicted_backing_round_trips() -> None:
    replay = _evicted_consecutive_current_scenario()
    state = replay.state_dict()
    restored = _replay(
        capacity=1,
        chunk_size=2,
        sequence_length=2,
        context=0,
        consecutive=2,
        online=False,
    )
    restored.load_state_dict(state)
    _assert_tree_equal(restored.state_dict(), state)


def test_restore_rejects_nonconsecutive_stepids_with_evicted_backing() -> None:
    replay = _evicted_consecutive_current_scenario()
    broken = copy.deepcopy(replay.state_dict())
    stepids = broken["consecutive"]["report"]["current"]["stepid"]
    stepids[:, [0, 1]] = stepids[:, [1, 0]]
    _assert_restore_rejected_without_mutation(replay, broken)


@pytest.mark.parametrize("corruption", ["calls", "source_sum"])
def test_restore_rejects_impossible_sample_metric_identities(
    corruption: str,
) -> None:
    replay = _replay(
        sequence_length=1,
        context=0,
        online=False,
        batch_size=2,
    )
    _add_rows(replay, 4)
    replay.sample_raw("train", timeout=0.1)
    broken = copy.deepcopy(replay.state_dict())
    if corruption == "calls":
        broken["metrics"]["sample_calls"] += 1
    else:
        broken["metrics"]["uniform_samples"] -= 1
    _assert_restore_rejected_without_mutation(replay, broken)


def test_zeroed_sample_metrics_after_stats_reset_round_trip() -> None:
    replay = _replay(
        sequence_length=1,
        context=0,
        online=False,
        batch_size=2,
    )
    _add_rows(replay, 4)
    replay.sample_raw("train", timeout=0.1)
    replay.stats(reset=True)
    state = replay.state_dict()
    assert all(value == 0 for value in state["metrics"].values())
    restored = _replay(
        sequence_length=1,
        context=0,
        online=False,
        batch_size=2,
    )
    restored.load_state_dict(state)
    _assert_tree_equal(restored.state_dict(), state)


@pytest.mark.parametrize("worker_key", [False, np.int64(0)])
def test_restore_rejects_non_exact_outer_writer_key_types(worker_key) -> None:
    replay = _complete_resume_scenario()
    broken = copy.deepcopy(replay.state_dict())
    writer = broken["writers"].pop(0)
    broken["writers"][worker_key] = writer
    _assert_restore_rejected_without_mutation(replay, broken)


@pytest.mark.parametrize("item_id", [False, np.int64(0)])
def test_uniform_selector_delete_rejects_non_exact_integer_aliases(item_id) -> None:
    selector = UniformSelector(7)
    selector.insert(0)
    before = selector.state_dict()
    with pytest.raises((TypeError, ValueError)):
        selector.delete(item_id)
    _assert_tree_equal(selector.state_dict(), before)


@pytest.mark.parametrize(
    "indices",
    [{False: 0}, {0: False}, {np.int64(0): 0}, {0: np.int64(0)}],
)
def test_uniform_selector_restore_rejects_non_exact_index_types(indices) -> None:
    selector = UniformSelector(7)
    selector.insert(0)
    state = selector.state_dict()
    state["indices"] = indices
    with pytest.raises((TypeError, ValueError)):
        UniformSelector.from_state_dict(state)


def test_complete_persistence_exact_resume_future_ids_rng_and_stream_current() -> None:
    original = _complete_resume_scenario()
    state = original.state_dict()
    pristine = copy.deepcopy(state)

    def fresh() -> DreamerReplay:
        return _replay(
            capacity=3,
            chunk_size=3,
            sequence_length=2,
            context=1,
            consecutive=2,
            online=True,
            seed=7,
        )

    restored = [fresh(), fresh()]
    for replay in restored:
        replay.load_state_dict(state)
        _assert_tree_equal(replay.state_dict(), pristine)
    state["chunks"][0]["transition"]["value"][:] = -999
    state["selector"]["keys"].clear()
    _assert_tree_equal(original.state_dict(), pristine)
    for replay in restored:
        _assert_tree_equal(replay.state_dict(), pristine)

    replicas = [original, *restored]

    def assert_replicas() -> None:
        for replay in replicas[1:]:
            _assert_tree_equal(replicas[0].state_dict(), replay.state_dict())

    resumed = [replay.sample("report", timeout=0.1) for replay in replicas]
    for batch in resumed[1:]:
        _assert_tree_equal(resumed[0].as_dict(), batch.as_dict())
    assert_replicas()

    online = [replay.sample_raw("train", timeout=0.1) for replay in replicas]
    for batch in online[1:]:
        _assert_tree_equal(online[0].as_dict(), batch.as_dict())
    assert_replicas()
    for replay, batch in zip(replicas, online, strict=True):
        replay.update_context(
            batch.step_ids[:, 1:3],
            {
                "dyn/deter": np.asarray([[[21, 22], [23, 24]]], np.float32),
                "dyn/stoch": np.asarray([[[[25, 26]], [[27, 28]]]], np.float32),
            },
        )
    assert_replicas()
    for _ in range(2):
        group = [replay.sample("report", timeout=0.1) for replay in replicas]
        for batch in group[1:]:
            _assert_tree_equal(group[0].as_dict(), batch.as_dict())
        assert_replicas()
    for index in range(11, 19):
        kwargs = {"first": index in (12, 16), "last": index in (11, 15)}
        keys = [replay.add(_row(index, **kwargs)) for replay in replicas]
        assert keys[1:] == [keys[0], keys[0]]
        assert_replicas()
        if index % 2:
            batches = [replay.sample_raw("report", timeout=0.1) for replay in replicas]
            for batch in batches[1:]:
                _assert_tree_equal(batches[0].as_dict(), batch.as_dict())
            assert_replicas()
    for replay in restored:
        _assert_tree_equal(original.state_dict(), replay.state_dict())


def test_restored_writer_discards_load_only_offset_and_accepts_future_add() -> None:
    original = _complete_resume_scenario()
    restored = _replay(
        capacity=3,
        chunk_size=3,
        sequence_length=2,
        context=1,
        consecutive=2,
        online=True,
        seed=7,
    )
    restored.load_state_dict(original.state_dict())
    assert not hasattr(restored.writers[0], "_restored_offset")
    restored.add(_row(11, first=False))
    restored.validate()


def test_chunk_id_exhaustion_is_preflighted_without_partial_append() -> None:
    empty = _replay(chunk_size=3, sequence_length=2, context=0, online=False)
    empty_state = empty.state_dict()
    empty_restored = _replay(chunk_size=3, sequence_length=2, context=0, online=False)
    empty_restored.load_state_dict(empty_state)
    _assert_tree_equal(empty_restored.state_dict(), empty_state)

    replay = _replay(chunk_size=3, sequence_length=2, context=0, online=False)
    replay.add(_row(0))
    replay.add(_row(1, first=False))
    replay.next_chunk_id = 2**128
    before = replay.state_dict()
    with pytest.raises(OverflowError):
        replay.add(_row(2, first=False))
    _assert_tree_equal(replay.state_dict(), before)

    fresh = _replay(chunk_size=1, sequence_length=1, context=0, online=False)
    fresh.next_chunk_id = 2**128 - 1
    fresh_before = fresh.state_dict()
    with pytest.raises(OverflowError):
        fresh.add(_row(0))
    _assert_tree_equal(fresh.state_dict(), fresh_before)

    sentinel = _replay(chunk_size=3, sequence_length=2, context=0, online=False)
    sentinel.next_chunk_id = 2**128 - 1
    key = sentinel.add(_row(0))
    assert key.chunk_id == (2**128 - 1).to_bytes(16, "big")
    assert sentinel.next_chunk_id == 2**128
    exhausted_state = sentinel.state_dict()
    restored = _replay(chunk_size=3, sequence_length=2, context=0, online=False)
    restored_before = restored.state_dict()
    with pytest.raises(ValueError, match="histor"):
        restored.load_state_dict(exhausted_state)
    _assert_tree_equal(restored.state_dict(), restored_before)
    sentinel.add(_row(1, first=False))
    before_fill = sentinel.state_dict()
    with pytest.raises(OverflowError):
        sentinel.add(_row(2, first=False))
    _assert_tree_equal(sentinel.state_dict(), before_fill)


def test_restore_rejects_malformed_consecutive_current_without_mutation() -> None:
    replay = _complete_resume_scenario()
    before = replay.state_dict()

    def remove_reward(current) -> None:
        current.pop("reward")

    def wrong_value_dtype(current) -> None:
        current["value"] = current["value"].astype(np.float64)

    def wrong_reward_trailing_shape(current) -> None:
        current["reward"] = current["reward"][..., None]

    def wrong_batch_size(current) -> None:
        for name, value in tuple(current.items()):
            current[name] = np.concatenate([value, value], axis=0)

    for corrupt in (
        remove_reward,
        wrong_value_dtype,
        wrong_reward_trailing_shape,
        wrong_batch_size,
    ):
        broken = copy.deepcopy(before)
        corrupt(broken["consecutive"]["report"]["current"])
        with pytest.raises((TypeError, ValueError)):
            replay.load_state_dict(broken)
        _assert_tree_equal(replay.state_dict(), before)

    broken = copy.deepcopy(before)
    broken["consecutive"]["report"]["index"] = 0
    with pytest.raises((TypeError, ValueError)):
        replay.load_state_dict(broken)
    _assert_tree_equal(replay.state_dict(), before)


def test_successful_nonempty_restore_notifies_existing_sample_waiter() -> None:
    source = _replay(sequence_length=2, context=0, online=False)
    _add_rows(source, 4)
    target = _replay(sequence_length=2, context=0, online=False)
    started = threading.Event()
    outcome: list[ReplayBatch] = []
    errors: list[BaseException] = []

    def sample() -> None:
        started.set()
        try:
            outcome.append(target.sample_raw("report", timeout=2.0))
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)

    thread = threading.Thread(target=sample)
    thread.start()
    assert started.wait(1)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with target._condition:
            if target._condition._waiters:
                break
        threading.Event().wait(0.005)
    with target._condition:
        assert len(target._condition._waiters) == 1
    target.load_state_dict(source.state_dict())
    thread.join(0.5)
    assert not thread.is_alive()
    assert not errors
    assert outcome and outcome[0].step_ids.shape == (1, 2, 20)


def test_restore_rebinds_blocked_sample_to_live_consecutive_stream() -> None:
    source = _replay(
        sequence_length=1,
        context=0,
        consecutive=2,
        online=False,
    )
    _add_rows(source, 3)
    first_source = source.sample("report", timeout=0.1)
    np.testing.assert_array_equal(first_source.data["consec"], [[0]])
    assert source.consecutive_streams["report"].index == 1
    saved = source.state_dict()
    serial = _replay(
        sequence_length=1,
        context=0,
        consecutive=2,
        online=False,
    )
    serial.load_state_dict(saved)
    expected = serial.sample("report", timeout=0.1)
    np.testing.assert_array_equal(expected.data["consec"], [[1]])
    target = _replay(
        sequence_length=1,
        context=0,
        consecutive=2,
        online=False,
    )
    outcome: list[ReplayBatch] = []
    errors: list[BaseException] = []

    def sample() -> None:
        try:
            outcome.append(target.sample("report", timeout=2.0))
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)

    thread = threading.Thread(target=sample)
    thread.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with target._condition:
            if target._condition._waiters:
                break
        threading.Event().wait(0.005)
    with target._condition:
        assert len(target._condition._waiters) == 1
    target.load_state_dict(saved)
    thread.join(0.5)
    assert not thread.is_alive()
    assert not errors
    assert outcome and outcome[0].step_ids.shape == (1, 1, 20)
    _assert_tree_equal(outcome[0].as_dict(), expected.as_dict())
    assert target.consecutive_streams["report"].index == 2
    _assert_tree_equal(target.state_dict(), serial.state_dict())
    second = target.sample("report", timeout=0.1)
    serial_second = serial.sample("report", timeout=0.1)
    np.testing.assert_array_equal(second.data["consec"], [[0]])
    _assert_tree_equal(second.as_dict(), serial_second.as_dict())
    _assert_tree_equal(target.state_dict(), serial.state_dict())


@pytest.mark.parametrize("corruption", ["splice", "merge"])
def test_restore_rejects_cross_worker_chain_splice_and_merge_without_mutation(
    corruption: str,
) -> None:
    replay = _replay(
        capacity=20,
        chunk_size=2,
        sequence_length=2,
        context=0,
        online=False,
    )
    for index in range(5):
        replay.add(_row(index, first=index == 0), worker=1)
        replay.add(_row(100 + index, first=index == 0), worker=2)
    before = replay.state_dict()
    broken = copy.deepcopy(before)
    sealed = [chunk for chunk in broken["chunks"] if chunk["sealed"]]
    left = min(sealed, key=lambda chunk: int(chunk["transition"]["value"][0]))
    right = max(sealed, key=lambda chunk: int(chunk["transition"]["value"][0]))
    left_successor = left["successor"]
    right_successor = right["successor"]
    assert left_successor != right_successor
    if corruption == "splice":
        left["successor"], right["successor"] = right_successor, left_successor
    else:
        left["successor"] = right_successor
        broken["refs"][left_successor] -= 1
        broken["refs"][right_successor] += 1
    with pytest.raises((TypeError, ValueError)):
        replay.load_state_dict(broken)
    _assert_tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize(
    "corrupt_writer",
    [
        lambda writer: writer["pending"].reverse(),
        lambda writer: writer.update(row_count=writer["row_count"] + 1),
        lambda writer: writer.update(emitted_count=writer["emitted_count"] + 1),
        lambda writer: writer.update(has_rows=not writer["has_rows"]),
        lambda writer: writer.update(last_is_last=not writer["last_is_last"]),
    ],
)
def test_restore_rejects_corrupt_writer_cadence_and_chronology_without_mutation(
    corrupt_writer,
) -> None:
    replay = _complete_resume_scenario()
    before = replay.state_dict()
    broken = copy.deepcopy(before)
    corrupt_writer(next(iter(broken["writers"].values())))
    with pytest.raises((TypeError, ValueError)):
        replay.load_state_dict(broken)
    _assert_tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize("populated", [False, True], ids=["empty", "nonempty"])
def test_restore_rejects_exact_next_item_id_gap_without_mutation(
    populated: bool,
) -> None:
    replay = _replay(sequence_length=1, context=0, online=False)
    if populated:
        _add_rows(replay, 4)
    before = replay.state_dict()
    broken = copy.deepcopy(before)
    broken["next_item_id"] += 1
    with pytest.raises(ValueError, match="next replay item"):
        replay.load_state_dict(broken)
    _assert_tree_equal(replay.state_dict(), before)


def test_restore_preserves_exact_item_counter_after_capacity_eviction() -> None:
    replay = _replay(
        capacity=1,
        sequence_length=1,
        context=0,
        online=False,
    )
    _add_rows(replay, 6)
    state = replay.state_dict()
    assert state["fifo"] == [5]
    assert state["next_item_id"] == 6
    assert sum(value["emitted_count"] for value in state["writers"].values()) == 6
    restored = _replay(
        capacity=1,
        sequence_length=1,
        context=0,
        online=False,
    )
    restored.load_state_dict(state)
    restored.add(_row(6, first=False))
    assert restored.fifo == [6]
    assert restored.next_item_id == 7


def test_restore_rejects_coordinated_writer_counter_drift_without_mutation() -> None:
    replay = _complete_resume_scenario()
    before = replay.state_dict()
    broken = copy.deepcopy(before)
    writer = next(iter(broken["writers"].values()))
    writer["row_count"] += replay.config.chunk_size
    writer["emitted_count"] += replay.config.chunk_size
    with pytest.raises(ValueError, match="emitted"):
        replay.load_state_dict(broken)
    _assert_tree_equal(replay.state_dict(), before)


def test_restore_rejects_coherent_negative_item_id_without_mutation() -> None:
    replay = _complete_resume_scenario()
    before = replay.state_dict()
    broken = copy.deepcopy(before)
    old = broken["items"][0]["item_id"]
    broken["items"][0]["item_id"] = -1
    broken["fifo"][broken["fifo"].index(old)] = -1
    key_index = broken["selector"]["keys"].index(old)
    broken["selector"]["keys"][key_index] = -1
    broken["selector"]["indices"] = {
        key: index for index, key in enumerate(broken["selector"]["keys"])
    }
    with pytest.raises(ValueError, match="item id"):
        replay.load_state_dict(broken)
    _assert_tree_equal(replay.state_dict(), before)


def test_restore_rejects_noncontiguous_retained_item_suffix_without_mutation() -> None:
    replay = _replay(
        capacity=2,
        sequence_length=1,
        context=0,
        online=False,
    )
    _add_rows(replay, 6)
    before = replay.state_dict()
    assert [item["item_id"] for item in before["items"]] == [4, 5]
    broken = copy.deepcopy(before)
    broken["items"][0]["item_id"] = 3
    broken["fifo"][0] = 3
    broken["selector"]["keys"][broken["selector"]["keys"].index(4)] = 3
    broken["selector"]["indices"] = {
        key: index for index, key in enumerate(broken["selector"]["keys"])
    }
    with pytest.raises(ValueError, match="item"):
        replay.load_state_dict(broken)
    _assert_tree_equal(replay.state_dict(), before)


@pytest.mark.parametrize(
    "corruption",
    [
        "root_missing_first",
        "terminal_without_last",
        "tail_terminal_without_last",
        "current_tail_after_last",
        "within",
        "cross",
    ],
)
def test_restore_rejects_corrupt_retained_transition_chronology_without_mutation(
    corruption: str,
) -> None:
    replay = _replay(
        capacity=20,
        chunk_size=3,
        sequence_length=2,
        context=0,
        online=False,
    )
    _add_rows(replay, 5)
    before = replay.state_dict()
    broken = copy.deepcopy(before)
    nonempty = [chunk for chunk in broken["chunks"] if chunk["length"]]
    if corruption == "root_missing_first":
        nonempty[0]["transition"]["is_first"][0] = False
    elif corruption == "terminal_without_last":
        nonempty[0]["transition"]["is_terminal"][0] = True
        nonempty[0]["transition"]["is_last"][0] = False
    elif corruption == "tail_terminal_without_last":
        nonempty[-1]["transition"]["is_terminal"][-1] = True
        nonempty[-1]["transition"]["is_last"][-1] = False
    elif corruption == "current_tail_after_last":
        nonempty[-1]["transition"]["is_last"][-2] = True
        nonempty[-1]["transition"]["is_first"][-1] = False
    elif corruption == "within":
        nonempty[0]["transition"]["is_last"][0] = True
        nonempty[0]["transition"]["is_first"][1] = False
    else:
        predecessor = next(
            chunk for chunk in nonempty if chunk["successor"] is not None
        )
        successor = next(
            chunk for chunk in nonempty if chunk["chunk_id"] == predecessor["successor"]
        )
        predecessor["transition"]["is_last"][-1] = True
        successor["transition"]["is_first"][0] = False
    with pytest.raises(ValueError, match="chronology"):
        replay.load_state_dict(broken)
    _assert_tree_equal(replay.state_dict(), before)


def test_restore_accepts_evicted_prefix_whose_retained_root_is_not_first() -> None:
    replay = _complete_resume_scenario()
    state = replay.state_dict()
    successor_ids = {
        chunk["successor"]
        for chunk in state["chunks"]
        if chunk["successor"] is not None
    }
    root = next(
        chunk
        for chunk in state["chunks"]
        if chunk["chunk_id"] not in successor_ids and chunk["length"]
    )
    assert bool(root["transition"]["is_first"][0]) is False
    restored = _replay(
        capacity=3,
        chunk_size=3,
        sequence_length=2,
        context=1,
        consecutive=2,
        online=True,
        seed=7,
    )
    restored.load_state_dict(state)
    _assert_tree_equal(restored.state_dict(), state)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda state: state.update(schema_version=999),
        lambda state: state["config"].update(capacity=999),
        lambda state: state["config"].update(sequence_length=999),
        lambda state: state["dimensions"].update(batch_size=999),
        lambda state: state["dimensions"].update(sequence_length=999),
        lambda state: state["dimensions"].update(consecutive=999),
        lambda state: state["dimensions"].update(context=999),
        lambda state: state["dimensions"].update(raw_length=999),
        lambda state: state["spaces"]["transition"]["value"].update(dtype="float64"),
        lambda state: state["spaces"]["transition"]["value"].update(shape=[1]),
        lambda state: state["selector"]["keys"].append(999),
        lambda state: state["fifo"].reverse(),
        lambda state: state["refs"].update({next(iter(state["refs"])): 999}),
        lambda state: state["chunks"][0].update(
            successor=state["chunks"][0]["chunk_id"]
        ),
        lambda state: state["consecutive"]["report"].update(current=None, index=1),
        lambda state: state["chunks"].append(copy.deepcopy(state["chunks"][0])),
        lambda state: state["chunks"].pop(),
        lambda state: state.update(next_chunk_id=1),
        lambda state: state.update(next_chunk_id=state["next_chunk_id"] + 7),
        lambda state: state.update(next_chunk_id=2**128),
        lambda state: state.update(next_chunk_id=2**128 + 1),
        lambda state: next(iter(state["writers"].values())).update(
            current_offset=2**32
        ),
        lambda state: state["selector"].update(bit_generator="MT19937"),
        lambda state: state["selector"]["rng_state"].update(bit_generator="MT19937"),
        lambda state: state["items"][0]["key"].update(offset=2**32),
        lambda state: state["items"].append(copy.deepcopy(state["items"][0])),
        lambda state: state["items"].pop(),
        lambda state: state["chunks"][0]["transition"].update(
            value=state["chunks"][0]["transition"]["value"].astype(np.float64)
        ),
        lambda state: state["chunks"][0]["transition"].update(
            value=state["chunks"][0]["transition"]["value"][:0]
        ),
        lambda state: state["chunks"][0].update(sealed=False),
        lambda state: next(iter(state["writers"].values()))["pending"].append(
            {"chunk_id": bytes(16), "offset": 0}
        ),
    ],
)
def test_corrupt_restore_rejected_without_partial_live_mutation(corrupt) -> None:
    replay = _complete_resume_scenario()
    before = replay.state_dict()
    broken = copy.deepcopy(before)
    corrupt(broken)
    with pytest.raises((TypeError, ValueError)):
        replay.load_state_dict(broken)
    _assert_tree_equal(replay.state_dict(), before)


def test_restore_rejects_array_only_and_chunk_only_legacy_state() -> None:
    replay = _complete_resume_scenario()
    before = replay.state_dict()
    for broken in (
        {"arrays": {"value": np.arange(3)}},
        {"chunks": copy.deepcopy(before["chunks"])},
    ):
        with pytest.raises((TypeError, ValueError)):
            replay.load_state_dict(broken)
        _assert_tree_equal(replay.state_dict(), before)


def test_empty_sample_timeout_blocks_then_notifies_without_partial_batch() -> None:
    replay = _replay(sequence_length=2, context=0, batch_size=2, online=False)
    with pytest.raises(TimeoutError):
        replay.sample_raw("train", timeout=0.01)
    outcome: list[ReplayBatch] = []
    errors: list[BaseException] = []

    def sample() -> None:
        try:
            outcome.append(replay.sample_raw("train", timeout=1.0))
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)

    thread = threading.Thread(target=sample)
    thread.start()
    replay.add(_row(0))
    assert thread.is_alive()
    replay.add(_row(1, first=False))
    thread.join(2)
    assert not thread.is_alive() and not errors
    assert outcome[0].step_ids.shape == (2, 2, 20)


def test_blocking_sample_calls_are_serialized_without_blocking_add() -> None:
    replay = _replay(
        sequence_length=1,
        context=0,
        consecutive=2,
        batch_size=1,
        online=False,
    )
    results: dict[str, ReplayBatch] = {}
    errors: list[BaseException] = []
    report_started = threading.Event()

    def consume(mode: str) -> None:
        if mode == "report":
            report_started.set()
        try:
            results[mode] = replay.sample(mode, timeout=2.0)
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)

    train = threading.Thread(target=consume, args=("train",))
    train.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with replay._condition:
            if replay._condition._waiters:
                break
        threading.Event().wait(0.005)
    report = threading.Thread(target=consume, args=("report",))
    report.start()
    assert report_started.wait(1)
    threading.Event().wait(0.05)
    with replay._condition:
        waiter_count = len(replay._condition._waiters)
    replay.add(_row(0))
    replay.add(_row(1, first=False))
    train.join(2)
    report.join(2)
    assert waiter_count == 2
    assert not train.is_alive() and not report.is_alive()
    assert not errors
    assert set(results) == {"train", "report"}
    assert results["train"].step_ids.shape == (1, 1, 20)
    assert results["report"].step_ids.shape == (1, 1, 20)
    np.testing.assert_array_equal(results["train"].data["consec"], [[0]])
    np.testing.assert_array_equal(results["report"].data["consec"], [[0]])
    assert replay.stats()["sample_calls"] == 1


def test_sample_timeout_budget_includes_stream_lock_wait() -> None:
    replay = _replay(
        sequence_length=1,
        context=0,
        consecutive=1,
        batch_size=1,
        online=False,
    )
    first_results: list[ReplayBatch] = []
    second_errors: list[BaseException] = []

    first = threading.Thread(
        target=lambda: first_results.append(replay.sample("report"))
    )
    first.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with replay._condition:
            if replay._condition._waiters:
                break
        threading.Event().wait(0.005)
    with replay._condition:
        assert len(replay._condition._waiters) == 1

    def finite_sample() -> None:
        try:
            replay.sample("report", timeout=0.05)
        except BaseException as error:  # pragma: no cover - asserted below.
            second_errors.append(error)

    second = threading.Thread(target=finite_sample)
    started = time.monotonic()
    second.start()
    second.join(0.2)
    completed_within_budget = not second.is_alive()
    elapsed = time.monotonic() - started
    replay.add(_row(0))
    first.join(1)
    second.join(1)
    assert completed_within_budget
    assert elapsed < 0.2
    assert len(second_errors) == 1
    assert isinstance(second_errors[0], TimeoutError)
    assert first_results and first_results[0].step_ids.shape == (1, 1, 20)


@pytest.mark.parametrize("nontrain_mode", ["report", "eval"])
def test_nontrain_stream_cannot_supply_train_slice_or_bypass_online_accounting(
    nontrain_mode: str,
) -> None:
    replay = _replay(
        sequence_length=1,
        context=0,
        consecutive=2,
        online=True,
    )
    _add_rows(replay, 5)
    queue_before = replay.online_queue.state_dict()
    stats_before = replay.stats()

    nontrain0 = replay.sample(nontrain_mode, timeout=0.1)
    np.testing.assert_array_equal(nontrain0.data["consec"], [[0]])
    np.testing.assert_array_equal(nontrain0.data["value"], [[3]])
    assert replay.online_queue.state_dict() == queue_before
    assert replay.stats() == stats_before

    train0 = replay.sample("train", timeout=0.1)
    np.testing.assert_array_equal(train0.data["consec"], [[0]])
    np.testing.assert_array_equal(train0.data["value"], [[1]])
    assert len(replay.online_queue) == len(queue_before["keys"]) - 1
    stats = replay.stats()
    assert stats["sample_calls"] == 1
    assert stats["sampled_sequences"] == 1
    assert stats["online_samples"] == 1
    assert stats["uniform_samples"] == 0

    queue_after = replay.online_queue.state_dict()
    nontrain1 = replay.sample(nontrain_mode, timeout=0.1)
    train1 = replay.sample("train", timeout=0.1)
    np.testing.assert_array_equal(nontrain1.data["consec"], [[1]])
    np.testing.assert_array_equal(train1.data["consec"], [[1]])
    np.testing.assert_array_equal(nontrain1.data["value"], [[4]])
    np.testing.assert_array_equal(train1.data["value"], [[2]])
    assert replay.online_queue.state_dict() == queue_after
    assert replay.stats() == stats


@pytest.mark.parametrize("nontrain_mode", ["report", "eval"])
def test_train_stream_retained_slice_survives_interleaved_nontrain_mode(
    nontrain_mode: str,
) -> None:
    replay = _replay(
        sequence_length=1,
        context=0,
        consecutive=2,
        online=True,
    )
    _add_rows(replay, 5)
    train0 = replay.sample("train", timeout=0.1)
    np.testing.assert_array_equal(train0.data["consec"], [[0]])
    np.testing.assert_array_equal(train0.data["value"], [[1]])
    queue_after_train = replay.online_queue.state_dict()
    stats_after_train = replay.stats()

    nontrain0 = replay.sample(nontrain_mode, timeout=0.1)
    np.testing.assert_array_equal(nontrain0.data["consec"], [[0]])
    np.testing.assert_array_equal(nontrain0.data["value"], [[3]])
    assert replay.online_queue.state_dict() == queue_after_train
    assert replay.stats() == stats_after_train

    train1 = replay.sample("train", timeout=0.1)
    np.testing.assert_array_equal(train1.data["consec"], [[1]])
    np.testing.assert_array_equal(train1.data["value"], [[2]])
    assert replay.online_queue.state_dict() == queue_after_train
    assert replay.stats() == stats_after_train


def test_all_mode_consecutive_streams_resume_exactly_and_independently() -> None:
    replay = _replay(
        sequence_length=1,
        context=0,
        consecutive=2,
        online=True,
    )
    _add_rows(replay, 6)
    for mode in ("train", "report", "eval"):
        batch = replay.sample(mode, timeout=0.1)
        np.testing.assert_array_equal(batch.data["consec"], [[0]])
        assert replay.consecutive_streams[mode].index == 1
    state = replay.state_dict()

    restored = _replay(
        sequence_length=1,
        context=0,
        consecutive=2,
        online=True,
    )
    restored.load_state_dict(state)
    for mode in ("eval", "train", "report"):
        expected = replay.sample(mode, timeout=0.1)
        actual = restored.sample(mode, timeout=0.1)
        np.testing.assert_array_equal(actual.data["consec"], [[1]])
        _assert_tree_equal(actual.as_dict(), expected.as_dict())
    _assert_tree_equal(restored.state_dict(), replay.state_dict())


def test_concurrent_add_sample_update_snapshot_preserves_invariants() -> None:
    replay = _replay(
        capacity=20,
        chunk_size=3,
        sequence_length=2,
        context=1,
        batch_size=2,
        online=True,
    )
    _add_rows(replay, 8, natural_last=True)
    errors: list[BaseException] = []

    def guarded(function) -> None:
        try:
            function()
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)

    def producer() -> None:
        for index in range(8, 48):
            replay.add(
                _row(
                    index,
                    first=index % 4 == 0,
                    last=index % 4 == 3,
                )
            )

    def consumer() -> None:
        for _ in range(30):
            batch = replay.sample_raw("train", timeout=1.0)
            replay.update_context(
                batch.step_ids[:, 1:],
                {
                    "dyn/deter": np.zeros((2, 2, 2), np.float32),
                    "dyn/stoch": np.zeros((2, 2, 1, 2), np.float32),
                },
            )

    def snapshotter() -> None:
        for _ in range(30):
            state = replay.state_dict()
            clone = _replay(
                capacity=20,
                chunk_size=3,
                sequence_length=2,
                context=1,
                batch_size=2,
                online=True,
            )
            clone.load_state_dict(state)

    threads = [
        threading.Thread(target=lambda: guarded(producer)),
        threading.Thread(target=lambda: guarded(consumer)),
        threading.Thread(target=lambda: guarded(snapshotter)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    replay.validate()


def test_stats_counters_ratio_persistence_and_reset() -> None:
    replay = _replay(sequence_length=2, context=0, online=True, batch_size=2)
    _add_rows(replay, 6)
    batch = replay.sample_raw("train", timeout=0.1)
    replay.update_context(
        batch.step_ids[:, :1],
        {
            "dyn/deter": np.zeros((2, 1, 2), np.float32),
            "dyn/stoch": np.zeros((2, 1, 1, 2), np.float32),
        },
    )
    stats = replay.stats()
    assert stats["inserted_rows"] == 6
    assert stats["inserted_items"] == 5
    assert stats["sampled_sequences"] == 2
    assert stats["updated_rows"] == 2
    assert stats["online_samples"] == 2
    assert stats["uniform_samples"] == 0
    assert stats["online_fraction"] == 1.0
    assert stats["replay_ratio"] == pytest.approx(4 / 6)
    restored = _replay(sequence_length=2, context=0, online=True, batch_size=2)
    restored.load_state_dict(replay.state_dict())
    assert restored.stats() == stats
    assert restored.stats(reset=True) == stats
    reset = restored.stats()
    assert reset["inserted_rows"] == 0
    assert reset["sampled_sequences"] == 0

    stale = _replay(capacity=3, sequence_length=3, context=1, online=True)
    _add_rows(stale, 10, natural_last=True)
    stale.sample_raw("train", timeout=0.1)
    assert stale.stats()["stale_online"] == 1
    stale_clone = _replay(capacity=3, sequence_length=3, context=1, online=True)
    stale_clone.load_state_dict(stale.state_dict())
    assert stale_clone.stats()["stale_online"] == 1
    assert stale_clone.stats(reset=True)["stale_online"] == 1
    assert stale_clone.stats()["stale_online"] == 0


def test_report_and_eval_sampling_leave_training_counters_unchanged() -> None:
    replay = _replay(sequence_length=2, context=0, online=False, batch_size=2)
    _add_rows(replay, 5)
    before = replay.stats()
    replay.sample_raw("report", timeout=0.1)
    replay.sample_raw("eval", timeout=0.1)
    after = replay.stats()
    for name in (
        "sample_calls",
        "sampled_sequences",
        "online_samples",
        "uniform_samples",
        "stale_online",
    ):
        assert after[name] == before[name] == 0
    assert after["replay_ratio"] == before["replay_ratio"] == 0.0
    replay.sample_raw("train", timeout=0.1)
    trained = replay.stats()
    assert trained["sample_calls"] == 1
    assert trained["sampled_sequences"] == 2
    assert trained["uniform_samples"] == 2
    assert trained["replay_ratio"] == pytest.approx(4 / 5)


def test_public_exports_and_jepa_replay_are_independent() -> None:
    required = {
        "ReplayKey",
        "ReplayBatch",
        "ReplayChunk",
        "ReplayWriter",
        "OnlineQueue",
        "UniformSelector",
        "ConsecutiveStream",
        "DreamerReplay",
        "REPLAY_SOURCE_SPEC",
    }
    assert required <= set(dreamer_package.__all__)
    for name in required:
        assert getattr(dreamer_package, name) is (
            REPLAY_SOURCE_SPEC
            if name == "REPLAY_SOURCE_SPEC"
            else getattr(replay_module, name)
        )
    assert jepa_replay.ReplayBatch.__module__ == "world_marl.jepa.replay"
    assert jepa_replay.ReplayBatch is not ReplayBatch
    assert jepa_replay.SequenceReplayBuffer.__module__ == "world_marl.jepa.replay"
    assert jepa_replay.SequenceReplayBuffer is not DreamerReplay
    legacy = jepa_replay.SequenceReplayBuffer(
        capacity=8,
        num_envs=1,
        observation_shape=(2,),
    )
    for index in range(4):
        legacy.add_step(
            observations=np.asarray([[index, index + 1]], np.float32),
            actions=np.asarray([index], np.int32),
            rewards=np.asarray([index], np.float32),
            dones=np.asarray([False]),
        )
    legacy_batch = legacy.sample(
        np.random.default_rng(3),
        batch_size=1,
        chunk_length=2,
        max_horizon=1,
    )
    assert legacy_batch.observations.shape == (1, 3, 2)
    assert legacy_batch.actions.shape == (1, 2)
