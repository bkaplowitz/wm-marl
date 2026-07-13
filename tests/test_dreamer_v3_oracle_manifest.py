from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from world_marl.dreamer_v3_baseline.config import (
    DreamerProfile,
    ObservationMode,
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
        ("device", "", "device"),
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
    )
    second_fixture, second_manifest = second.write_fixture(
        case_name="deterministic",
        profile=DreamerProfile.UPSTREAM_CURRENT,
        observation_mode=ObservationMode.VISION,
        arrays=dict(reversed(tuple(arrays.items()))),
        seed=11,
        generator_command=("oracle", "deterministic"),
    )

    assert first_fixture.read_bytes() == second_fixture.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    with np.load(first_fixture, allow_pickle=False) as fixture:
        assert fixture.files == ["a_tensor", "z_tensor"]
        np.testing.assert_array_equal(fixture["a_tensor"], arrays["a_tensor"])


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
    assert profile.value in " ".join(manifest.generator_command)
    assert mode.value in " ".join(manifest.generator_command)
    with np.load(fixture_path, allow_pickle=False) as fixture:
        assert int(fixture["worker_pid"][0]) != os.getpid()
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
