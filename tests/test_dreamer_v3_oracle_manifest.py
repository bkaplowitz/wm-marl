from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import world_marl.dreamer_v3_baseline.oracle as oracle_module
import world_marl.dreamer_v3_baseline.replay_oracle as replay_oracle_module
from world_marl.dreamer_v3_baseline.config import (
    DreamerProfile,
    DreamerV3Config,
    ObservationMode,
    resolve_dreamer_config,
)
from world_marl.dreamer_v3_baseline.oracle import (
    ORACLE_SCHEMA_VERSION,
    OracleHarness,
    OracleManifest,
    ParameterTranslator,
)


PAPER_OVERRIDES = {
    "agent.dec.simple.strided": True,
    "agent.enc.simple.strided": True,
    "agent.opt.beta2": 0.99,
    "run.steps": 1_000_000,
}

RSSM_SOURCE_HASHES = {
    "dreamerv3/agent.py": (
        "adce8e4274bc098c218bf9a20fd3327545f0ad7d850b5fe328597382e91b5269"
    ),
    "dreamerv3/rssm.py": (
        "d6d50166914e94fb8bd17a5d5dbda9d42cdd37b85819bb1e9fff3a64d4ad2eb6"
    ),
}

REPLAY_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "dreamer_v3"
REPLAY_MANIFEST = REPLAY_FIXTURE_DIR / "paper-proprio-replay.manifest.json"
REPLAY_FIXTURE = REPLAY_FIXTURE_DIR / "paper-proprio-replay.npz"
REPLAY_OFFICIAL_CHECKOUT = Path(
    os.environ.get(
        "DREAMERV3_ORACLE_CHECKOUT",
        "/private/tmp/danijar-dreamerv3-20260713",
    )
)
REPLAY_REQUEST_KEYS = {
    "case_name",
    "cases",
    "compute_dtype",
    "elements_dist_info",
    "elements_package_dir",
    "observation_mode",
    "official_checkout",
    "official_commit",
    "overrides",
    "profile",
    "python_executable",
    "row_schema",
    "runtime",
    "seed",
    "source_spec",
    "uuid_mode",
}


@pytest.fixture(scope="module")
def official_checkout() -> Path:
    path = Path(
        os.environ.get(
            "DREAMERV3_ORACLE_CHECKOUT",
            "/private/tmp/danijar-dreamerv3-20260713",
        )
    )
    if not (path / ".git").exists():
        pytest.skip("explicit DreamerV3 oracle checkout is unavailable")
    return path


def _write_tampered_manifest(
    source: Path,
    destination: Path,
    field: str,
    value,
) -> None:
    payload = json.loads(source.read_text())
    payload[field] = value
    destination.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )


def _write_tampered_replay_request(
    tmp_path: Path,
    *,
    request_mutation: Callable[[dict[str, Any]], None] | None = None,
    manifest_mutation: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    payload = json.loads(REPLAY_MANIFEST.read_text())
    request = json.loads(payload["generator_request"])
    request.setdefault("case_name", payload["case_name"])
    request.setdefault("seed", payload["seed"])
    if request_mutation is not None:
        request_mutation(request)
    payload["generator_request"] = json.dumps(
        request, sort_keys=True, separators=(",", ":")
    )
    if manifest_mutation is not None:
        manifest_mutation(payload)
    destination = tmp_path / "tampered-replay.manifest.json"
    destination.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return destination


def _load_tampered_replay(path: Path) -> OracleManifest:
    return OracleManifest.load(path, fixture_path=REPLAY_FIXTURE)


def test_replay_manifest_rejects_an_extra_generator_argument(tmp_path: Path) -> None:
    fixture_dir = Path(__file__).parent / "fixtures" / "dreamer_v3"
    source = fixture_dir / "paper-proprio-replay.manifest.json"
    payload = json.loads(source.read_text())
    payload["generator_command"].append("--untrusted")
    tampered = tmp_path / source.name
    tampered.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )

    with pytest.raises(ValueError, match="generator command"):
        OracleManifest.load(
            tampered,
            fixture_path=fixture_dir / "paper-proprio-replay.npz",
        )


@pytest.mark.parametrize("profile", ("paper", "upstream-current"))
def test_replay_manifest_binds_the_exact_generator_request(profile: str) -> None:
    stem = f"{profile}-proprio-replay"
    manifest = OracleManifest.load(
        REPLAY_FIXTURE_DIR / f"{stem}.manifest.json",
        fixture_path=REPLAY_FIXTURE_DIR / f"{stem}.npz",
    )
    request = json.loads(manifest.generator_request)

    assert set(request) == REPLAY_REQUEST_KEYS
    assert request["case_name"] == manifest.case_name
    assert request["seed"] == manifest.seed


def test_replay_manifest_rejects_an_extra_request_key(tmp_path: Path) -> None:
    def mutate(request: dict[str, Any]) -> None:
        request["untrusted"] = True

    path = _write_tampered_replay_request(tmp_path, request_mutation=mutate)

    with pytest.raises(ValueError, match="request keys"):
        _load_tampered_replay(path)


def test_replay_manifest_rejects_a_missing_request_key(tmp_path: Path) -> None:
    def mutate(request: dict[str, Any]) -> None:
        del request["uuid_mode"]

    path = _write_tampered_replay_request(tmp_path, request_mutation=mutate)

    with pytest.raises(ValueError, match="request keys"):
        _load_tampered_replay(path)


@pytest.mark.parametrize(
    ("command_index", "replacement"),
    (
        (1, str(Path(__file__).resolve())),
        (2, "--worker"),
    ),
)
def test_replay_manifest_rejects_a_forged_worker_command(
    tmp_path: Path,
    command_index: int,
    replacement: str,
) -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["generator_command"][command_index] = replacement

    path = _write_tampered_replay_request(tmp_path, manifest_mutation=mutate)

    with pytest.raises(ValueError, match="generator command"):
        _load_tampered_replay(path)


def test_replay_manifest_rejects_a_consistently_forged_interpreter(
    tmp_path: Path,
) -> None:
    def mutate_request(request: dict[str, Any]) -> None:
        request["python_executable"] = "/bin/false"

    def mutate_manifest(payload: dict[str, Any]) -> None:
        payload["generator_command"][0] = "/bin/false"

    path = _write_tampered_replay_request(
        tmp_path,
        request_mutation=mutate_request,
        manifest_mutation=mutate_manifest,
    )

    with pytest.raises(ValueError, match="generator command"):
        _load_tampered_replay(path)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("python_executable", 7),
        ("official_checkout", 7),
        ("official_checkout", "."),
    ),
)
def test_replay_manifest_rejects_noncanonical_path_coordinates(
    tmp_path: Path,
    key: str,
    value: Any,
) -> None:
    def mutate(request: dict[str, Any]) -> None:
        request[key] = value

    path = _write_tampered_replay_request(tmp_path, request_mutation=mutate)

    with pytest.raises(ValueError, match="coordinate"):
        _load_tampered_replay(path)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("case_name", "forged"),
        ("seed", 8),
        ("profile", "upstream-current"),
        ("observation_mode", "vision"),
        ("official_commit", "0" * 40),
        ("overrides", {}),
        ("source_spec", "config"),
    ),
)
def test_replay_manifest_rejects_request_manifest_coordinate_mismatch(
    tmp_path: Path,
    key: str,
    value: Any,
) -> None:
    def mutate(request: dict[str, Any]) -> None:
        request[key] = value

    path = _write_tampered_replay_request(tmp_path, request_mutation=mutate)

    with pytest.raises(ValueError, match="manifest coordinate"):
        _load_tampered_replay(path)


def test_replay_manifest_rejects_request_dtype_mismatch(tmp_path: Path) -> None:
    def mutate(request: dict[str, Any]) -> None:
        request["compute_dtype"] = "float64"

    path = _write_tampered_replay_request(tmp_path, request_mutation=mutate)

    with pytest.raises(ValueError, match="dtype"):
        _load_tampered_replay(path)


def test_replay_manifest_rejects_case_contract_tampering(tmp_path: Path) -> None:
    def mutate(request: dict[str, Any]) -> None:
        request["cases"]["primary"]["steps"] += 1

    path = _write_tampered_replay_request(tmp_path, request_mutation=mutate)

    with pytest.raises(ValueError, match="case contract"):
        _load_tampered_replay(path)


def test_replay_manifest_rejects_row_schema_tampering(tmp_path: Path) -> None:
    def mutate(request: dict[str, Any]) -> None:
        request["row_schema"]["reward"]["dtype"] = "float64"

    path = _write_tampered_replay_request(tmp_path, request_mutation=mutate)

    with pytest.raises(ValueError, match="row schema"):
        _load_tampered_replay(path)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("runtime", "worker_mode", "in-process"),
        ("shim_hashes", "UUID", "0" * 64),
        ("elements_helper_hashes", "elements/uuid.py", "0" * 64),
    ),
)
def test_replay_manifest_rejects_runtime_contract_tampering(
    tmp_path: Path,
    section: str,
    key: str,
    value: str,
) -> None:
    def mutate(request: dict[str, Any]) -> None:
        target = request["runtime"]
        if section != "runtime":
            target = target[section]
        target[key] = value

    path = _write_tampered_replay_request(tmp_path, request_mutation=mutate)

    with pytest.raises(ValueError, match="runtime contract"):
        _load_tampered_replay(path)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("elements_package_dir", "/tmp/untrusted-elements"),
        ("elements_dist_info", "/tmp/untrusted-elements.dist-info"),
        ("uuid_mode", "random"),
    ),
)
def test_replay_manifest_rejects_elements_and_uuid_coordinate_tampering(
    tmp_path: Path,
    key: str,
    value: str,
) -> None:
    def mutate(request: dict[str, Any]) -> None:
        request[key] = value

    path = _write_tampered_replay_request(tmp_path, request_mutation=mutate)

    with pytest.raises(ValueError, match="runtime coordinate"):
        _load_tampered_replay(path)


def test_replay_manifest_rejects_checkout_before_git_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate(request: dict[str, Any]) -> None:
        request["official_checkout"] = str(tmp_path / "forged-checkout")

    def reject_subprocess(*args, **kwargs):
        del args, kwargs
        raise AssertionError("checkout provenance reached a subprocess")

    path = _write_tampered_replay_request(tmp_path, request_mutation=mutate)
    monkeypatch.setattr(oracle_module.subprocess, "run", reject_subprocess)

    with pytest.raises(ValueError, match="official checkout"):
        OracleManifest.load(
            path,
            official_checkout=REPLAY_OFFICIAL_CHECKOUT,
            fixture_path=REPLAY_FIXTURE,
        )


def test_required_oracle_source_validator_cannot_be_omitted() -> None:
    with pytest.raises(ValueError, match="generator validator"):
        oracle_module.OracleSourceSpec(
            name="test_required_validator",
            revision_hashes=replay_oracle_module.REPLAY_SOURCE_SPEC.revision_hashes,
            execution_dtypes=("float32",),
            generator_validation_required=True,
        )


@pytest.mark.parametrize(
    ("mutation", "key", "value"),
    (
        ("extra", "untrusted", True),
        ("missing", "case_name", None),
        ("replace", "case_name", "forged"),
        ("replace", "seed", 8),
        ("replace", "official_checkout", "."),
    ),
)
def test_replay_worker_rejects_nonexact_source_request(
    mutation: str,
    key: str,
    value: Any,
) -> None:
    manifest = OracleManifest.load(REPLAY_MANIFEST, fixture_path=REPLAY_FIXTURE)
    request = json.loads(manifest.generator_request)
    if mutation == "missing":
        del request[key]
    else:
        request[key] = value

    completed = subprocess.run(
        manifest.generator_command,
        cwd=request["official_checkout"],
        input=json.dumps(request, sort_keys=True, separators=(",", ":")),
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "replay worker source request" in completed.stderr


def test_manifest_round_trip_validates_all_provenance_hashes(
    official_checkout: Path,
    tmp_path: Path,
) -> None:
    harness = OracleHarness(official_checkout, tmp_path)
    fixture_path, manifest_path = harness.write_fixture(
        case_name="manifest_round_trip",
        profile=DreamerProfile.PAPER,
        observation_mode=ObservationMode.VISION,
        arrays={
            "beta": np.asarray([2.0, 3.0], dtype=np.float32),
            "alpha": np.asarray([[1, 2]], dtype=np.int32),
        },
        seed=7,
        generator_command=("pytest", "manifest_round_trip"),
        source_spec="config",
    )

    manifest = OracleManifest.load(
        manifest_path,
        official_checkout=official_checkout,
        fixture_path=fixture_path,
    )

    assert manifest.schema_version == ORACLE_SCHEMA_VERSION
    assert manifest.case_name == "manifest_round_trip"
    assert manifest.profile is DreamerProfile.PAPER
    assert manifest.observation_mode is ObservationMode.VISION
    assert dict(manifest.overrides) == PAPER_OVERRIDES
    assert manifest.official_commit == ("bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01")
    assert manifest.source_spec == "config"
    assert tuple(manifest.official_file_hashes) == ("dreamerv3/configs.yaml",)
    assert len(manifest.profile_hash) == 64
    assert len(manifest.fixture_sha256) == 64
    assert manifest.tensor_schema["alpha"].shape == (1, 2)
    assert manifest.tensor_schema["alpha"].dtype == "int32"
    assert manifest.tensor_schema["beta"].shape == (2,)
    assert manifest.generator_command == ("pytest", "manifest_round_trip")
    assert len(manifest.canonical_hash()) == 64


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("schema_version", 999, "schema"),
        ("official_commit", "0" * 40, "commit"),
        (
            "official_file_hashes",
            {"dreamerv3/configs.yaml": "0" * 64},
            "source file",
        ),
        ("profile_hash", "0" * 64, "profile hash"),
        ("overrides", {}, "override"),
        ("jax_version", "0.0.0", "JAX"),
        ("dtype", "float32", "dtype"),
        ("device", "tpu", "device"),
        ("fixture_sha256", "0" * 64, "fixture"),
    ],
)
def test_manifest_rejects_wrong_schema_source_config_override_and_fixture_hashes(
    official_checkout: Path,
    tmp_path: Path,
    field: str,
    replacement,
    message: str,
) -> None:
    harness = OracleHarness(official_checkout, tmp_path)
    fixture_path, manifest_path = harness.write_fixture(
        case_name=f"tamper_{field}",
        profile=DreamerProfile.PAPER,
        observation_mode=ObservationMode.PROPRIO,
        arrays={"value": np.asarray([1.0], dtype=np.float32)},
        seed=3,
        generator_command=("pytest", "tamper"),
        source_spec="config",
    )
    tampered = tmp_path / f"tampered-{field}.json"
    _write_tampered_manifest(manifest_path, tampered, field, replacement)

    with pytest.raises(ValueError, match=message):
        OracleManifest.load(
            tampered,
            official_checkout=official_checkout,
            fixture_path=fixture_path,
        )


def test_fixture_writer_is_byte_deterministic_and_sorts_tensor_names(
    official_checkout: Path,
    tmp_path: Path,
) -> None:
    arrays = {
        "z_tensor": np.asarray([3.0, 4.0], dtype=np.float32),
        "a_tensor": np.asarray([[1, 2]], dtype=np.int16),
    }
    first = OracleHarness(official_checkout, tmp_path / "first")
    second = OracleHarness(official_checkout, tmp_path / "second")

    first_fixture, first_manifest = first.write_fixture(
        case_name="deterministic",
        profile=DreamerProfile.UPSTREAM_CURRENT,
        observation_mode=ObservationMode.VISION,
        arrays=arrays,
        seed=11,
        generator_command=("oracle", "deterministic"),
        source_spec="config",
    )
    second_fixture, second_manifest = second.write_fixture(
        case_name="deterministic",
        profile=DreamerProfile.UPSTREAM_CURRENT,
        observation_mode=ObservationMode.VISION,
        arrays=dict(reversed(tuple(arrays.items()))),
        seed=11,
        generator_command=("oracle", "deterministic"),
        source_spec="config",
    )

    assert first_fixture.read_bytes() == second_fixture.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    with np.load(first_fixture, allow_pickle=False) as fixture:
        assert fixture.files == ["a_tensor", "z_tensor"]
        np.testing.assert_array_equal(fixture["a_tensor"], arrays["a_tensor"])


def test_manifest_rejects_offline_wrong_source_hash_and_missing_fixture(
    official_checkout: Path,
    tmp_path: Path,
) -> None:
    harness = OracleHarness(official_checkout, tmp_path)
    fixture_path, manifest_path = harness.write_fixture(
        case_name="offline_validation",
        profile=DreamerProfile.PAPER,
        observation_mode=ObservationMode.VISION,
        arrays={"value": np.asarray([1.0], dtype=np.float32)},
        seed=5,
        generator_command=("pytest", "offline_validation"),
        source_spec="config",
    )
    tampered = tmp_path / "offline-wrong-source.manifest.json"
    _write_tampered_manifest(
        manifest_path,
        tampered,
        "official_file_hashes",
        {"dreamerv3/configs.yaml": "f" * 64},
    )

    with pytest.raises(ValueError, match="source file hash"):
        OracleManifest.load(tampered, fixture_path=fixture_path)

    fixture_path.rename(tmp_path / "detached-fixture.npz")
    with pytest.raises(ValueError, match="fixture.*missing"):
        OracleManifest.load(manifest_path)


def test_manifest_rejects_supplied_config_from_a_different_authority_coordinate(
    official_checkout: Path,
    tmp_path: Path,
) -> None:
    harness = OracleHarness(official_checkout, tmp_path)
    fixture_path, manifest_path = harness.write_fixture(
        case_name="config_coordinate",
        profile=DreamerProfile.PAPER,
        observation_mode=ObservationMode.VISION,
        arrays={"value": np.asarray([1.0], dtype=np.float32)},
        seed=17,
        generator_command=("pytest", "config_coordinate"),
        source_spec="config",
    )
    paper_proprio = resolve_dreamer_config(
        DreamerProfile.PAPER,
        ObservationMode.PROPRIO,
    )
    wrong_mode = tmp_path / "wrong-mode.manifest.json"
    _write_tampered_manifest(
        manifest_path,
        wrong_mode,
        "profile_hash",
        paper_proprio.canonical_hash(),
    )

    with pytest.raises(ValueError, match="observation mode"):
        OracleManifest.load(
            wrong_mode,
            fixture_path=fixture_path,
            config=paper_proprio,
        )

    current_vision = resolve_dreamer_config(
        DreamerProfile.UPSTREAM_CURRENT,
        ObservationMode.VISION,
    )
    wrong_profile = tmp_path / "wrong-profile.manifest.json"
    _write_tampered_manifest(
        manifest_path,
        wrong_profile,
        "profile_hash",
        current_vision.canonical_hash(),
    )
    with pytest.raises(ValueError, match="profile"):
        OracleManifest.load(
            wrong_profile,
            fixture_path=fixture_path,
            config=current_vision,
        )

    legacy = DreamerV3Config(action_dim=4, observation_shape=(8, 8, 3))
    with pytest.raises(ValueError, match="canonical supplied config"):
        OracleManifest.load(
            manifest_path,
            fixture_path=fixture_path,
            config=legacy,
        )


def test_generic_oracle_apis_require_explicit_source_authority(
    official_checkout: Path,
    tmp_path: Path,
) -> None:
    harness = OracleHarness(official_checkout, tmp_path)
    arrays = {"value": np.asarray([1.0], dtype=np.float32)}

    with pytest.raises(TypeError, match="source_spec"):
        harness.write_fixture(
            case_name="generic_without_source",
            profile=DreamerProfile.PAPER,
            observation_mode=ObservationMode.VISION,
            arrays=arrays,
            seed=19,
            generator_command=("pytest", "generic_without_source"),
        )

    fixture_path = tmp_path / "direct-create.npz"
    np.savez(fixture_path, **arrays)
    with pytest.raises(TypeError, match="source_spec"):
        OracleManifest.create(
            case_name="direct_without_source",
            profile=DreamerProfile.PAPER,
            observation_mode=ObservationMode.VISION,
            official_checkout=official_checkout,
            fixture_path=fixture_path,
            arrays=arrays,
            seed=23,
            generator_command=("pytest", "direct_without_source"),
        )


def test_case_specific_source_specs_pin_exact_files_for_future_oracles(
    official_checkout: Path,
    tmp_path: Path,
) -> None:
    source_spec_type = getattr(oracle_module, "OracleSourceSpec")
    register_source_spec = getattr(oracle_module, "register_oracle_source_spec")
    source_spec = source_spec_type(
        name="test_rssm",
        revision_hashes={
            oracle_module.PAPER_REVISION: RSSM_SOURCE_HASHES,
            oracle_module.UPSTREAM_CURRENT_REVISION: RSSM_SOURCE_HASHES,
        },
    )
    register_source_spec(source_spec)

    for profile in DreamerProfile:
        harness = OracleHarness(
            official_checkout,
            tmp_path / profile.value,
        )
        fixture_path, manifest_path = harness.write_fixture(
            case_name="rssm_initial",
            profile=profile,
            observation_mode=ObservationMode.PROPRIO,
            arrays={"state": np.zeros((1, 4), dtype=np.float32)},
            seed=13,
            generator_command=("pytest", "rssm_initial"),
            source_spec=source_spec.name,
        )

        manifest = OracleManifest.load(manifest_path)

        assert manifest.source_spec == source_spec.name
        assert dict(manifest.official_file_hashes) == RSSM_SOURCE_HASHES
        assert fixture_path.exists()


def test_repeated_config_cases_are_byte_identical_and_pid_is_out_of_band(
    official_checkout: Path,
    tmp_path: Path,
) -> None:
    first = OracleHarness(official_checkout, tmp_path / "first")
    second = OracleHarness(official_checkout, tmp_path / "second")

    first_fixture, first_manifest = first.run_config_case(
        DreamerProfile.PAPER,
        ObservationMode.PROPRIO,
        case_name="config_deterministic",
    )
    second_fixture, second_manifest = second.run_config_case(
        DreamerProfile.PAPER,
        ObservationMode.PROPRIO,
        case_name="config_deterministic",
    )

    assert first.last_worker_pid is not None
    assert second.last_worker_pid is not None
    assert first.last_worker_pid != os.getpid()
    assert second.last_worker_pid != os.getpid()
    assert first_fixture.read_bytes() == second_fixture.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    with np.load(first_fixture, allow_pickle=False) as fixture:
        assert "worker_pid" not in fixture.files


def test_config_generator_provenance_replays_the_exact_command_and_request(
    official_checkout: Path,
    tmp_path: Path,
) -> None:
    harness = OracleHarness(official_checkout, tmp_path)
    fixture_path, manifest_path = harness.run_config_case(
        DreamerProfile.UPSTREAM_CURRENT,
        ObservationMode.VISION,
        case_name="config_replayable",
    )
    manifest = OracleManifest.load(
        manifest_path,
        official_checkout=official_checkout,
        fixture_path=fixture_path,
    )

    assert manifest.generator_command[-1] == "_config_worker"
    assert manifest.generator_request is not None
    assert json.loads(manifest.generator_request) == {
        "official_checkout": str(official_checkout.resolve()),
        "official_commit": oracle_module.UPSTREAM_CURRENT_REVISION,
        "observation_mode": ObservationMode.VISION.value,
        "overrides": {},
        "profile": DreamerProfile.UPSTREAM_CURRENT.value,
        "source_spec": "config",
    }
    replayed = subprocess.run(
        manifest.generator_command,
        cwd=official_checkout,
        input=manifest.generator_request,
        check=True,
        capture_output=True,
        text=True,
    )
    replayed_payload = json.loads(replayed.stdout)

    assert int(replayed_payload["worker_pid"]) != os.getpid()
    with np.load(fixture_path, allow_pickle=False) as fixture:
        assert tuple(fixture.files) == tuple(sorted(replayed_payload["arrays"]))
        for name, spec in replayed_payload["arrays"].items():
            expected = np.asarray(spec["values"], dtype=spec["dtype"])
            np.testing.assert_array_equal(fixture[name], expected)


@pytest.mark.parametrize(
    ("profile", "mode", "expected_overrides", "expected_profile"),
    [
        (
            DreamerProfile.PAPER,
            ObservationMode.PROPRIO,
            PAPER_OVERRIDES,
            {
                "rssm": [512, 64, 32, 4],
                "encoder": [4, 64, 1],
                "dmc": [0, 1, 64, 64, 1, -1],
                "optimizer": [0.99],
                "run": [1_000_000.0, 1024.0],
            },
        ),
        (
            DreamerProfile.UPSTREAM_CURRENT,
            ObservationMode.VISION,
            {},
            {
                "rssm": [8192, 1024, 32, 64],
                "encoder": [64, 1024, 0],
                "dmc": [1, 0, 64, 64, 1, -1],
                "optimizer": [0.999],
                "run": [1_100_000.0, 256.0],
            },
        ),
    ],
)
def test_config_case_runs_in_a_process_and_profiles_apply_only_declared_overrides(
    official_checkout: Path,
    tmp_path: Path,
    profile: DreamerProfile,
    mode: ObservationMode,
    expected_overrides: dict[str, object],
    expected_profile: dict[str, list[float]],
) -> None:
    before_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=official_checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    before_bytes = (official_checkout / "dreamerv3/configs.yaml").read_bytes()
    harness = OracleHarness(official_checkout, tmp_path)

    fixture_path, manifest_path = harness.run_config_case(
        profile,
        mode,
        case_name=f"config_{profile.value}_{mode.value}",
    )

    after_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=official_checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert after_status == before_status
    assert (official_checkout / "dreamerv3/configs.yaml").read_bytes() == before_bytes
    manifest = OracleManifest.load(
        manifest_path,
        official_checkout=official_checkout,
        fixture_path=fixture_path,
    )
    assert dict(manifest.overrides) == expected_overrides
    assert harness.last_worker_pid is not None
    assert harness.last_worker_pid != os.getpid()
    with np.load(fixture_path, allow_pickle=False) as fixture:
        assert "worker_pid" not in fixture.files
        for name, expected in expected_profile.items():
            np.testing.assert_allclose(fixture[name], expected, rtol=0.0, atol=0.0)


def test_parameter_translator_requires_bijective_complete_consumption() -> None:
    translator = ParameterTranslator()
    translator.register("source/a", "destination/a")
    translator.register("source/b", "destination/b", transform="transpose")

    translated = translator.translate(
        {
            "source/a": np.asarray([1.0, 2.0], dtype=np.float32),
            "source/b": np.asarray([[1.0, 2.0]], dtype=np.float32),
        },
        {
            "destination/a": (2,),
            "destination/b": (2, 1),
        },
    )

    np.testing.assert_array_equal(
        translated["destination/a"],
        np.asarray([1.0, 2.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        translated["destination/b"],
        np.asarray([[1.0], [2.0]], dtype=np.float32),
    )
    translator.assert_fully_consumed()


def test_parameter_translator_rejects_duplicates_and_unconsumed_parameters() -> None:
    translator = ParameterTranslator()
    translator.register("source/a", "destination/a")
    with pytest.raises(ValueError, match="source.*already registered"):
        translator.register("source/a", "destination/b")
    with pytest.raises(ValueError, match="destination.*already registered"):
        translator.register("source/b", "destination/a")

    with pytest.raises(ValueError, match="unregistered source"):
        translator.translate(
            {
                "source/a": np.ones((1,), dtype=np.float32),
                "source/extra": np.ones((1,), dtype=np.float32),
            },
            {"destination/a": (1,)},
        )
    with pytest.raises(ValueError, match="unregistered destination"):
        translator.translate(
            {"source/a": np.ones((1,), dtype=np.float32)},
            {"destination/a": (1,), "destination/extra": (1,)},
        )
    with pytest.raises(ValueError, match="unconsumed"):
        translator.assert_fully_consumed()


def test_parameter_translator_rejects_shape_mismatch_and_unknown_transform() -> None:
    translator = ParameterTranslator()
    with pytest.raises(ValueError, match="unknown transform"):
        translator.register("source/a", "destination/a", transform="rotate")

    translator.register("source/a", "destination/a")
    with pytest.raises(ValueError, match="shape"):
        translator.translate(
            {"source/a": np.ones((2,), dtype=np.float32)},
            {"destination/a": (1,)},
        )
