from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
import pytest

from world_marl.dreamer_v3_baseline.config import DreamerProfile
from world_marl.dreamer_v3_baseline.network_oracle import NETWORKS_SOURCE_SPEC
from world_marl.dreamer_v3_baseline.oracle import (
    DISTRIBUTIONS_SOURCE_SPEC,
    ORACLE_SCHEMA_VERSION,
    PAPER_REVISION,
    UPSTREAM_CURRENT_REVISION,
    OracleManifest,
    ParameterTranslator,
    TensorSpec,
    _git_show,
    _parameter_path,
    _source_allows_dtype,
    _source_hashes_for,
)
from world_marl.dreamer_v3_baseline.replay_oracle import REPLAY_SOURCE_SPEC
from world_marl.dreamer_v3_baseline.rssm_oracle import RSSM_SOURCE_SPEC


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "dreamer_v3"
STEMS = (
    "paper-proprio-distributions",
    "upstream-current-proprio-distributions",
    "paper-proprio-replay",
    "upstream-current-proprio-replay",
    "paper-proprio-rssm",
    "paper-proprio-rssm-float32",
    "upstream-current-proprio-rssm",
    "upstream-current-proprio-rssm-float32",
    "paper-vision-networks",
    "paper-vision-networks-float32",
    "upstream-current-vision-networks",
    "upstream-current-vision-networks-float32",
)
PROFILE_REVISIONS = {
    DreamerProfile.PAPER: PAPER_REVISION,
    DreamerProfile.UPSTREAM_CURRENT: UPSTREAM_CURRENT_REVISION,
}
SOURCE_NAMES = tuple(
    spec.name
    for spec in (
        DISTRIBUTIONS_SOURCE_SPEC,
        NETWORKS_SOURCE_SPEC,
        REPLAY_SOURCE_SPEC,
        RSSM_SOURCE_SPEC,
    )
)


class _StringSubclass(str):
    pass


class _StringSequence(Sequence[str]):
    def __init__(self, *items: str) -> None:
        self._items = items

    def __getitem__(self, index: int) -> str:
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)


EXOTIC_STRING_TYPES = (np.str_, _StringSubclass)
NON_SEQUENCE_PATH_FACTORIES: tuple[Callable[[str], object], ...] = (
    lambda segment: {segment: None},
    lambda segment: {segment},
    lambda segment: (item for item in (segment,)),
)
EXPECTED_NPZ_DIGESTS = {
    "paper-proprio-distributions": (
        "04b28028fd05df124af6e51204827475c67b5bf87e139299c07d7576135fad3a"
    ),
    "upstream-current-proprio-distributions": (
        "04b28028fd05df124af6e51204827475c67b5bf87e139299c07d7576135fad3a"
    ),
    "paper-proprio-replay": (
        "c65d650b6359b335470d43abd3d4f2bd4352f1599510835fada493194910f08e"
    ),
    "upstream-current-proprio-replay": (
        "c65d650b6359b335470d43abd3d4f2bd4352f1599510835fada493194910f08e"
    ),
    "paper-proprio-rssm": (
        "6990bae306c641a059d33177cee2df6f62e9ad8f9c425133f4e5b9f4a362d72e"
    ),
    "paper-proprio-rssm-float32": (
        "daa4a5085781076aac279a2690f2629d085d57d8a2ae5f76eb6328a3092d95f0"
    ),
    "upstream-current-proprio-rssm": (
        "6990bae306c641a059d33177cee2df6f62e9ad8f9c425133f4e5b9f4a362d72e"
    ),
    "upstream-current-proprio-rssm-float32": (
        "daa4a5085781076aac279a2690f2629d085d57d8a2ae5f76eb6328a3092d95f0"
    ),
    "paper-vision-networks": (
        "be8f367281a41d5f54efe3918216e1e67ece72fdbd9720b18d250acf71511711"
    ),
    "paper-vision-networks-float32": (
        "e15ae7e34717833737b258559c8f7aa0550fbf2a6d1a6cc1d49864c51a151b69"
    ),
    "upstream-current-vision-networks": (
        "43473353baf491ec463c5d9e424babc7f60faa49263fe1c38c6e0674019e30d8"
    ),
    "upstream-current-vision-networks-float32": (
        "ff0268ea5e5d2df446afe6d82b66be9622b7b8f342f5ae8405c6cb79a81906d5"
    ),
}


def _payload(stem: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / f"{stem}.manifest.json").read_text())


def _copy_pair(tmp_path: Path, stem: str) -> tuple[Path, Path]:
    manifest = tmp_path / f"{stem}.manifest.json"
    fixture = tmp_path / f"{stem}.npz"
    manifest.write_bytes((FIXTURE_ROOT / manifest.name).read_bytes())
    fixture.write_bytes((FIXTURE_ROOT / fixture.name).read_bytes())
    return manifest, fixture


def _write_payload(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def _pair_inputs(
    stem: str = STEMS[0],
) -> tuple[OracleManifest, dict[str, np.ndarray]]:
    manifest = OracleManifest.load(FIXTURE_ROOT / f"{stem}.manifest.json")
    with np.load(FIXTURE_ROOT / f"{stem}.npz", allow_pickle=False) as fixture:
        arrays = {name: fixture[name] for name in fixture.files}
    return manifest, arrays


def _noncanonical_manifest_bytes(canonical: bytes, variant: str) -> bytes:
    payload = json.loads(canonical)
    if variant == "pretty":
        return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if variant == "reordered":
        reordered = dict(reversed(tuple(payload.items())))
        return (
            json.dumps(reordered, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode()
    if variant == "duplicate":
        marker = b'"case_name":"distributions",'
        assert canonical.count(marker) == 1
        return canonical.replace(marker, marker + marker, 1)
    if variant == "missing-newline":
        return canonical.removesuffix(b"\n")
    if variant == "extra-newline":
        return canonical + b"\n"
    raise AssertionError(f"unknown noncanonical manifest variant: {variant}")


def test_oracle_tooling_contains_no_private_runtime_or_source_execution() -> None:
    root = Path("src/world_marl/dreamer_v3_baseline")
    paths = tuple(
        root / name
        for name in (
            "oracle.py",
            "network_oracle.py",
            "rssm_oracle.py",
            "replay_oracle.py",
            "replay_oracle_contract.py",
        )
    )
    source = "\n".join(path.read_text() for path in paths)
    forbidden = (
        "generator_validator",
        "generator_resolver",
        "callback_implementation_fingerprint",
        "REPLAY_RUNTIME_CONTRACT",
        "isolated-ast-exec",
        "exec(",
    )
    assert not [token for token in forbidden if token in source]
    assert "import ast" not in source


@pytest.mark.parametrize("profile", tuple(DreamerProfile))
def test_profile_selects_exact_official_revision(profile: DreamerProfile) -> None:
    stem = (
        "paper-proprio-distributions"
        if profile is DreamerProfile.PAPER
        else "upstream-current-proprio-distributions"
    )
    assert (
        OracleManifest.load(FIXTURE_ROOT / f"{stem}.manifest.json").official_commit
        == PROFILE_REVISIONS[profile]
    )


def test_source_specs_pin_both_authority_revisions() -> None:
    assert set(SOURCE_NAMES) == {"distributions", "networks", "replay", "rssm"}
    for name in SOURCE_NAMES:
        for revision in (PAPER_REVISION, UPSTREAM_CURRENT_REVISION):
            hashes = _source_hashes_for(name, revision)
            assert hashes
            assert all(len(digest) == 64 for digest in hashes.values())
        assert _source_allows_dtype(
            name, "float32" if name in {"networks", "replay", "rssm"} else "bfloat16"
        )


@pytest.mark.parametrize("stem", STEMS)
def test_fixture_manifest_validates_without_live_checkout(stem: str) -> None:
    manifest = OracleManifest.load(FIXTURE_ROOT / f"{stem}.manifest.json")
    assert manifest.schema_version == ORACLE_SCHEMA_VERSION
    assert manifest.fixture_sha256 == EXPECTED_NPZ_DIGESTS[stem]
    assert manifest.official_commit == PROFILE_REVISIONS[manifest.profile]
    assert dict(manifest.official_file_hashes) == dict(
        _source_hashes_for(manifest.source_spec, manifest.official_commit)
    )


@pytest.mark.parametrize("stem", STEMS)
def test_fixture_manifest_schema_exactly_describes_npz(stem: str) -> None:
    manifest = OracleManifest.load(FIXTURE_ROOT / f"{stem}.manifest.json")
    with np.load(FIXTURE_ROOT / f"{stem}.npz", allow_pickle=False) as fixture:
        assert tuple(fixture.files) == tuple(manifest.tensor_schema)
        for name, spec in manifest.tensor_schema.items():
            assert fixture[name].shape == spec.shape
            assert fixture[name].dtype.name == spec.dtype


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("schema_version", 999, "schema version"),
        ("official_commit", "0" * 40, "profile authority"),
        ("fixture_sha256", "not-a-digest", "fixture hash"),
    ),
)
def test_manifest_rejects_invalid_metadata(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    manifest_path, _ = _copy_pair(tmp_path, STEMS[0])
    payload = json.loads(manifest_path.read_text())
    payload[field] = value
    _write_payload(manifest_path, payload)
    with pytest.raises(ValueError, match=match):
        OracleManifest.load(manifest_path)


def test_manifest_rejects_source_hash_drift(tmp_path: Path) -> None:
    manifest_path, _ = _copy_pair(tmp_path, STEMS[0])
    payload = json.loads(manifest_path.read_text())
    source_hashes = payload["official_file_hashes"]
    assert isinstance(source_hashes, dict)
    source_hashes[next(iter(source_hashes))] = "0" * 64
    _write_payload(manifest_path, payload)
    with pytest.raises(ValueError, match="pinned source spec"):
        OracleManifest.load(manifest_path)


def test_manifest_rejects_runtime_provenance_fields(tmp_path: Path) -> None:
    manifest_path, _ = _copy_pair(tmp_path, STEMS[0])
    payload = json.loads(manifest_path.read_text())
    payload["jax_version"] = "runtime provenance does not belong in fixtures"
    _write_payload(manifest_path, payload)
    with pytest.raises(ValueError, match="unexpected fields"):
        OracleManifest.load(manifest_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", True),
        ("case_name", np.str_("distributions")),
        ("profile", np.str_("paper")),
        ("observation_mode", np.str_("proprio")),
        ("official_commit", np.str_(PAPER_REVISION)),
        ("official_file_hashes", MappingProxyType({})),
        ("dtype", np.str_("bfloat16")),
        ("seed", np.int64(0)),
        ("tensor_schema", ()),
        ("generator_command", ("python",)),
        ("generator_request", {"profile": "paper"}),
        ("fixture_file", Path("fixture.npz")),
        ("fixture_sha256", np.str_("0" * 64)),
    ),
)
def test_manifest_decoder_rejects_nonexact_serialized_primitives(
    field: str, value: object
) -> None:
    payload = _payload(STEMS[0])
    payload[field] = value

    with pytest.raises((TypeError, ValueError)):
        OracleManifest.from_dict(payload)


@pytest.mark.parametrize(
    "variant",
    ("pretty", "reordered", "duplicate", "missing-newline", "extra-newline"),
)
def test_manifest_load_rejects_noncanonical_top_level_bytes(
    tmp_path: Path, variant: str
) -> None:
    manifest_path, fixture_path = _copy_pair(tmp_path, STEMS[0])
    manifest_path.write_bytes(
        _noncanonical_manifest_bytes(manifest_path.read_bytes(), variant)
    )

    with pytest.raises(ValueError, match="canonical"):
        OracleManifest.load(manifest_path, fixture_path=fixture_path)


def test_manifest_validation_uses_retained_direct_tables_after_registry_deletion() -> (
    None
):
    from world_marl.dreamer_v3_baseline import oracle

    for name in (
        "OracleSourceSpec",
        "_ORACLE_SOURCE_SPECS",
        "register_oracle_source_spec",
        "source_spec_for",
    ):
        assert not hasattr(oracle, name)
    assert isinstance(oracle._SOURCE_HASHES, MappingProxyType)
    assert isinstance(oracle._SOURCE_DTYPES, MappingProxyType)

    manifest = OracleManifest.load(FIXTURE_ROOT / f"{STEMS[0]}.manifest.json")
    assert manifest.source_spec == "distributions"
    assert dict(manifest.official_file_hashes) == dict(
        _source_hashes_for(manifest.source_spec, manifest.official_commit)
    )
    assert _source_allows_dtype(manifest.source_spec, manifest.dtype)


def test_task1c_deletes_registry_and_retains_direct_fixture_source_tables() -> None:
    root = Path("src/world_marl/dreamer_v3_baseline")
    transitional = {
        "OracleSourceSpec",
        "register_oracle_source_spec",
        "source_spec_for",
        "_ORACLE_SOURCE_SPECS",
    }
    callers = {name: set() for name in transitional}
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        referenced.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        )
        referenced.update(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        )
        for name in transitional:
            if name in referenced:
                callers[name].add(path.name)
    assert callers == {name: set() for name in transitional}

    oracle_tree = ast.parse((root / "oracle.py").read_text())
    oracle_assignments = {
        target.id
        for node in oracle_tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Name)
    }
    assert {
        "DISTRIBUTIONS_SOURCE_SPEC",
        "NETWORKS_SOURCE_SPEC",
        "REPLAY_SOURCE_SPEC",
        "RSSM_SOURCE_SPEC",
        "_SOURCE_DTYPES",
        "_SOURCE_HASHES",
    } <= oracle_assignments
    manifest = next(
        node
        for node in oracle_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OracleManifest"
    )
    manifest_references = {
        node.id for node in ast.walk(manifest) if isinstance(node, ast.Name)
    }
    assert not manifest_references & transitional
    assert not [
        node
        for node in ast.walk(oracle_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "OracleSourceSpec"
    ]


@pytest.mark.parametrize(
    ("field", "mutation"),
    (
        ("profile", lambda value: "upstream-current"),
        ("observation_mode", lambda value: "vision"),
        ("source_revision", lambda value: UPSTREAM_CURRENT_REVISION),
        ("fixture_stem", lambda value: "paper-proprio-replay"),
        ("case_name", lambda value: "replay"),
        ("source_spec", lambda value: "replay"),
        ("dtype", lambda value: "float32"),
        ("seed", lambda value: 1),
        ("fixture_file", lambda value: "other.npz"),
        ("schema_version", lambda value: 999),
        ("unexpected", lambda value: True),
    ),
)
def test_manifest_rejects_noncanonical_generator_request(
    tmp_path: Path, field: str, mutation
) -> None:
    manifest_path, _ = _copy_pair(tmp_path, STEMS[0])
    payload = json.loads(manifest_path.read_text())
    request = json.loads(payload["generator_request"])
    request[field] = mutation(request.get(field))
    payload["generator_request"] = json.dumps(request, sort_keys=True)
    _write_payload(manifest_path, payload)

    with pytest.raises(ValueError, match="generator request"):
        OracleManifest.load(manifest_path)


@pytest.mark.parametrize(
    "command",
    (
        ["wrong"],
        "python",
        ["python", "-m", "world_marl.dreamer_v3_baseline.fixture_generator"],
    ),
)
def test_manifest_rejects_noncanonical_generator_command(
    tmp_path: Path, command: object
) -> None:
    manifest_path, _ = _copy_pair(tmp_path, STEMS[0])
    payload = json.loads(manifest_path.read_text())
    payload["generator_command"] = command
    _write_payload(manifest_path, payload)

    with pytest.raises((TypeError, ValueError), match="generator command"):
        OracleManifest.load(manifest_path)


def test_manifest_rejects_fixture_and_tensor_schema_drift(tmp_path: Path) -> None:
    manifest_path, fixture_path = _copy_pair(tmp_path, STEMS[0])
    payload = json.loads(manifest_path.read_text())
    tensor_schema = payload["tensor_schema"]
    assert isinstance(tensor_schema, dict)
    tensor_schema[next(iter(tensor_schema))]["shape"] = [999]
    _write_payload(manifest_path, payload)
    with pytest.raises(ValueError, match="tensor schema mismatch"):
        OracleManifest.load(manifest_path)

    manifest_path.write_bytes((FIXTURE_ROOT / manifest_path.name).read_bytes())
    fixture_path.write_bytes(fixture_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="fixture hash mismatch"):
        OracleManifest.load(manifest_path)


def test_git_show_reads_revision_blob_without_executing_it(tmp_path: Path) -> None:
    checkout = tmp_path / "reference"
    subprocess.run(["git", "init", "-q", checkout], check=True)
    subprocess.run(
        ["git", "-C", checkout, "config", "user.email", "oracle@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", checkout, "config", "user.name", "Oracle Test"], check=True
    )
    source = checkout / "official.py"
    source.write_text("raise RuntimeError('must not execute')\n")
    subprocess.run(["git", "-C", checkout, "add", "official.py"], check=True)
    subprocess.run(["git", "-C", checkout, "commit", "-qm", "fixture"], check=True)
    revision = subprocess.run(
        ["git", "-C", checkout, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert _git_show(checkout, revision, "official.py") == source.read_bytes()


def test_parameter_translator_requires_a_bijection_and_exact_shapes() -> None:
    translator = ParameterTranslator()
    translator.register("dense.kernel", "dense.kernel", transform="transpose")
    translator.register("dense.bias", "dense.bias")
    translated = translator.translate(
        {
            "dense.kernel": np.arange(6, dtype=np.float32).reshape(2, 3),
            "dense.bias": np.zeros((2,), np.float32),
        },
        {"dense.kernel": (3, 2), "dense.bias": (2,)},
    )
    np.testing.assert_array_equal(
        translated["dense.kernel"], np.arange(6, dtype=np.float32).reshape(2, 3).T
    )
    translator.assert_fully_consumed()

    with pytest.raises(ValueError, match="source parameter already registered"):
        translator.register("dense.kernel", "other")
    with pytest.raises(ValueError, match="destination parameter already registered"):
        translator.register("other", "dense.bias")


def test_parameter_translator_accepts_exact_builtin_sequence_paths() -> None:
    translator = ParameterTranslator()
    translator.register(("dense", "kernel"), ["model", "kernel"])

    translated = translator.translate(
        {"dense.kernel": np.arange(4, dtype=np.float32).reshape(2, 2)},
        {"model.kernel": (2, 2)},
    )

    np.testing.assert_array_equal(
        translated["model.kernel"], np.arange(4, dtype=np.float32).reshape(2, 2)
    )


def _translator_consumption_snapshot(
    translator: ParameterTranslator,
) -> tuple[tuple[object, ...], frozenset[str], frozenset[str]]:
    state = vars(translator)
    consumed_sources = cast(set[str], state["_consumed_sources"])
    consumed_destinations = cast(set[str], state["_consumed_destinations"])
    return (
        translator.registry,
        frozenset(consumed_sources),
        frozenset(consumed_destinations),
    )


@pytest.mark.parametrize(
    ("case", "expected_error"),
    (
        ("extra_source", "unregistered source parameters"),
        ("extra_destination", "unregistered destination parameters"),
        ("missing_pair", "unconsumed registered parameters"),
        ("first_identity_shape", "parameter shape mismatch for x"),
        ("later_identity_shape", "parameter shape mismatch for y"),
        ("reshape_failure", "cannot reshape array"),
    ),
)
def test_parameter_translator_rejected_whole_translation_preserves_state(
    case: str, expected_error: str
) -> None:
    translator = ParameterTranslator()
    translator.register("a", "x")
    if case == "reshape_failure":
        translator.register("b", "y", transform="reshape", reshape=(2, 2))
        prior_value = np.arange(4, dtype=np.float32)
        later_value = np.arange(3, dtype=np.float32)
        later_shape = (2, 2)
    else:
        translator.register("b", "y")
        prior_value = np.ones((1,), dtype=np.float32)
        later_value = np.ones((1,), dtype=np.float32)
        later_shape = (1,)

    translator.consume("b", "y", prior_value, later_shape)
    before = _translator_consumption_snapshot(translator)
    assert before[1:] == (frozenset({"b"}), frozenset({"y"}))

    source_parameters = {
        "a": np.zeros((1,), dtype=np.float32),
        "b": later_value,
    }
    destination_shapes = {"x": (1,), "y": later_shape}
    if case == "extra_source":
        source_parameters["extra"] = np.zeros((1,), dtype=np.float32)
    elif case == "extra_destination":
        destination_shapes["extra"] = (1,)
    elif case == "missing_pair":
        source_parameters.pop("b")
        destination_shapes.pop("y")
    elif case == "first_identity_shape":
        destination_shapes["x"] = (2,)
    elif case == "later_identity_shape":
        destination_shapes["y"] = (2,)

    with pytest.raises(ValueError, match=expected_error):
        translator.translate(source_parameters, destination_shapes)

    assert _translator_consumption_snapshot(translator) == before


def test_parameter_translator_successful_whole_translation_commits_state() -> None:
    translator = ParameterTranslator()
    translator.register("a", "x")
    translator.register("b", "y")
    translator.consume("b", "y", np.ones((1,), dtype=np.float32), (1,))

    translated = translator.translate(
        {
            "a": np.array([1.0], dtype=np.float32),
            "b": np.array([2.0], dtype=np.float32),
        },
        {"x": (1,), "y": (1,)},
    )

    np.testing.assert_array_equal(translated["x"], np.array([1.0], np.float32))
    np.testing.assert_array_equal(translated["y"], np.array([2.0], np.float32))
    assert _translator_consumption_snapshot(translator)[1:] == (
        frozenset({"a", "b"}),
        frozenset({"x", "y"}),
    )


@pytest.mark.parametrize(
    "path",
    (
        "weight",
        ["weight"],
        ("weight",),
        _StringSequence("weight"),
    ),
)
def test_parameter_path_accepts_scalar_and_sequence_controls(
    path: str | Sequence[str],
) -> None:
    assert _parameter_path(path) == "weight"


@pytest.mark.parametrize(
    "path_factory",
    NON_SEQUENCE_PATH_FACTORIES,
    ids=("mapping", "set", "generator"),
)
def test_parameter_path_rejects_non_sequence_iterables(
    path_factory: Callable[[str], object],
) -> None:
    path = cast(str | Sequence[str], path_factory("weight"))
    with pytest.raises(TypeError, match="exact string or a sequence"):
        _parameter_path(path)


@pytest.mark.parametrize(
    "path_factory",
    NON_SEQUENCE_PATH_FACTORIES,
    ids=("mapping", "set", "generator"),
)
@pytest.mark.parametrize("coordinate", ("source", "destination"))
def test_parameter_translator_register_rejects_non_sequence_iterables(
    path_factory: Callable[[str], object],
    coordinate: str,
) -> None:
    translator = ParameterTranslator()
    source = cast(
        str | Sequence[str], path_factory("a") if coordinate == "source" else "a"
    )
    destination = cast(
        str | Sequence[str],
        path_factory("x") if coordinate == "destination" else "x",
    )

    with pytest.raises(TypeError, match="exact string or a sequence"):
        translator.register(source, destination)

    assert translator.registry == ()


@pytest.mark.parametrize(
    "path_factory",
    NON_SEQUENCE_PATH_FACTORIES,
    ids=("mapping", "set", "generator"),
)
@pytest.mark.parametrize("coordinate", ("source", "destination"))
def test_parameter_translator_consume_rejects_non_sequence_iterables(
    path_factory: Callable[[str], object],
    coordinate: str,
) -> None:
    translator = ParameterTranslator()
    translator.register("a", "x")
    source = cast(
        str | Sequence[str], path_factory("a") if coordinate == "source" else "a"
    )
    destination = cast(
        str | Sequence[str],
        path_factory("x") if coordinate == "destination" else "x",
    )
    value = np.zeros((1,), dtype=np.float32)

    with pytest.raises(TypeError, match="exact string or a sequence"):
        translator.consume(source, destination, value, (1,))

    with pytest.raises(ValueError, match="unconsumed parameters"):
        translator.assert_fully_consumed()
    np.testing.assert_array_equal(translator.consume("a", "x", value, (1,)), value)
    translator.assert_fully_consumed()


def test_parameter_translator_rejects_incomplete_and_shape_mismatched_sets() -> None:
    translator = ParameterTranslator()
    with pytest.raises(ValueError, match="unknown transform"):
        translator.register("bad", "bad", transform="rotate")
    with pytest.raises(ValueError, match="reshape target"):
        translator.register("bad", "bad", transform="reshape")
    translator.register("a", "x")
    translator.register("b", "y", transform="reshape", reshape=(2, 2))
    with pytest.raises(ValueError, match="unregistered source parameters"):
        translator.translate(
            {"a": np.zeros((1,)), "b": np.zeros((4,)), "c": np.zeros((1,))},
            {"x": (1,), "y": (2, 2)},
        )
    with pytest.raises(ValueError, match="unconsumed registered parameters"):
        translator.translate({"a": np.zeros((1,))}, {"x": (1,)})
    with pytest.raises(ValueError, match="shape mismatch"):
        translator.translate(
            {"a": np.zeros((2,)), "b": np.zeros((4,))},
            {"x": (1,), "y": (2, 2)},
        )


@pytest.mark.parametrize("string_type", EXOTIC_STRING_TYPES)
@pytest.mark.parametrize(
    ("coordinate", "as_sequence"),
    (
        ("source", False),
        ("destination", False),
        ("source", True),
        ("destination", True),
    ),
)
def test_parameter_translator_register_rejects_nonexact_string_paths(
    string_type: type[str],
    coordinate: str,
    as_sequence: bool,
) -> None:
    translator = ParameterTranslator()
    exotic = string_type("a")
    invalid = (exotic,) if as_sequence else exotic
    source = invalid if coordinate == "source" else "a"
    destination = invalid if coordinate == "destination" else "x"

    with pytest.raises(TypeError, match="exact string"):
        translator.register(source, destination)

    assert translator.registry == ()


@pytest.mark.parametrize("string_type", EXOTIC_STRING_TYPES)
@pytest.mark.parametrize("coordinate", ("source", "destination"))
def test_parameter_translator_consume_rejects_nonexact_string_paths(
    string_type: type[str], coordinate: str
) -> None:
    translator = ParameterTranslator()
    translator.register("a", "x")
    source = string_type("a") if coordinate == "source" else "a"
    destination = string_type("x") if coordinate == "destination" else "x"

    with pytest.raises(TypeError, match="exact string"):
        translator.consume(source, destination, np.zeros((1,)), (1,))

    with pytest.raises(ValueError, match="unconsumed parameters"):
        translator.assert_fully_consumed()


@pytest.mark.parametrize("string_type", EXOTIC_STRING_TYPES)
@pytest.mark.parametrize("coordinate", ("source", "destination"))
def test_parameter_translator_translate_rejects_nonexact_mapping_keys(
    string_type: type[str], coordinate: str
) -> None:
    translator = ParameterTranslator()
    translator.register("a", "x")
    source_key = string_type("a") if coordinate == "source" else "a"
    destination_key = string_type("x") if coordinate == "destination" else "x"

    with pytest.raises(TypeError, match="exact string"):
        translator.translate(
            {source_key: np.zeros((1,))},
            {destination_key: (1,)},
        )


@pytest.mark.parametrize("string_type", EXOTIC_STRING_TYPES)
def test_fixture_generator_parser_registration_rejects_nonexact_names(
    monkeypatch: pytest.MonkeyPatch, string_type: type[str]
) -> None:
    from world_marl.dreamer_v3_baseline import fixture_generator

    registry: dict[str, object] = {}
    monkeypatch.setattr(fixture_generator, "_PARSER_REGISTRY", registry)

    with pytest.raises(TypeError, match="exact string"):
        fixture_generator._register_parser(string_type("probe"))(lambda parsers: None)

    assert registry == {}


def test_fixture_generator_refresh_manifest_parser() -> None:
    from world_marl.dreamer_v3_baseline.fixture_generator import (
        _PARSER_REGISTRY,
        _parse_args,
    )

    assert tuple(_PARSER_REGISTRY) == ("refresh-manifest",)
    args = _parse_args(
        [
            "refresh-manifest",
            "--profile",
            "paper",
            "--observation-mode",
            "proprio",
            "--reference-checkout",
            "/tmp/reference",
            "--source-revision",
            PAPER_REVISION,
            "--output-dir",
            "/tmp/fixtures",
            "--fixture-stem",
            "paper-proprio-rssm",
        ]
    )
    assert args.command == "refresh-manifest"
    assert args.handler.__name__ == "refresh_manifest"
    assert args.profile == "paper"
    assert args.observation_mode == "proprio"

    for abbreviated in ("--prof", "--observation-m", "--source-rev"):
        invalid = [
            "refresh-manifest",
            "--profile",
            "paper",
            "--observation-mode",
            "proprio",
            "--reference-checkout",
            "/tmp/reference",
            "--source-revision",
            PAPER_REVISION,
            "--output-dir",
            "/tmp/fixtures",
            "--fixture-stem",
            "paper-proprio-rssm",
        ]
        invalid[
            invalid.index(
                {
                    "--prof": "--profile",
                    "--observation-m": "--observation-mode",
                    "--source-rev": "--source-revision",
                }[abbreviated]
            )
        ] = abbreviated
        with pytest.raises(SystemExit):
            _parse_args(invalid)

    with pytest.raises(SystemExit):
        _parse_args(["refresh-manifest", "--profile", "paper"])

    with pytest.raises(SystemExit):
        _parse_args(
            [
                "refresh-manifest",
                "--profile",
                "paper",
                "--observation-mode",
                "proprio",
                "--reference-checkout",
                "/tmp/reference",
                "--source-revision",
                PAPER_REVISION,
                "--output-dir",
                "/tmp/fixtures",
                "--fixture-stem",
                "paper-proprio-unknown",
            ]
        )


@pytest.mark.parametrize(
    "revision",
    (np.str_(PAPER_REVISION), type("RevisionString", (str,), {})(PAPER_REVISION)),
)
def test_fixture_generator_rejects_nonexact_revision_boundaries(
    tmp_path: Path, revision: object
) -> None:
    from world_marl.dreamer_v3_baseline import fixture_generator

    with pytest.raises(TypeError, match="revision"):
        fixture_generator._canonical_request(
            profile=DreamerProfile.PAPER,
            observation_mode="proprio",
            source_revision=revision,
            fixture_stem=STEMS[0],
        )

    args = fixture_generator._parse_args(
        [
            "refresh-manifest",
            "--profile",
            "paper",
            "--observation-mode",
            "proprio",
            "--reference-checkout",
            str(tmp_path),
            "--source-revision",
            PAPER_REVISION,
            "--output-dir",
            str(tmp_path),
            "--fixture-stem",
            STEMS[0],
        ]
    )
    args.source_revision = revision
    with pytest.raises(TypeError, match="revision"):
        fixture_generator.refresh_manifest(args)


def test_refresh_manifest_changes_only_manifest_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_marl.dreamer_v3_baseline import fixture_generator
    from world_marl.dreamer_v3_baseline import oracle

    stem = STEMS[0]
    manifest_path, fixture_path = _copy_pair(tmp_path, stem)
    fixture_before = fixture_path.read_bytes()
    payload = json.loads(manifest_path.read_text())
    source_hashes = payload["official_file_hashes"]
    assert isinstance(source_hashes, dict)

    def fake_git_show(checkout: Path, revision: str, path: str) -> bytes:
        del checkout, revision
        digest = source_hashes[path]
        return {digest: path.encode()}[digest]

    hashes = {path: hashlib.sha256(path.encode()).hexdigest() for path in source_hashes}
    payload["official_file_hashes"] = hashes
    _write_payload(manifest_path, payload)
    monkeypatch.setattr(fixture_generator, "_git_show", fake_git_show)
    for path, digest in hashes.items():
        source_hashes[path] = digest
    monkeypatch.setattr(
        fixture_generator, "_source_hashes_for", lambda name, revision: hashes
    )
    monkeypatch.setattr(oracle, "_source_hashes_for", lambda name, revision: hashes)
    args = fixture_generator._parse_args(
        [
            "refresh-manifest",
            "--profile",
            "paper",
            "--observation-mode",
            "proprio",
            "--reference-checkout",
            str(tmp_path),
            "--source-revision",
            PAPER_REVISION,
            "--output-dir",
            str(tmp_path),
            "--fixture-stem",
            stem,
        ]
    )
    args.handler(args)
    assert fixture_path.read_bytes() == fixture_before
    refreshed = json.loads(manifest_path.read_text())
    assert json.loads(refreshed["generator_request"]) == {
        "case_name": "distributions",
        "dtype": "bfloat16",
        "fixture_file": f"{stem}.npz",
        "fixture_stem": stem,
        "observation_mode": "proprio",
        "profile": "paper",
        "schema_version": ORACLE_SCHEMA_VERSION,
        "seed": 0,
        "source_revision": PAPER_REVISION,
        "source_spec": "distributions",
    }


@pytest.mark.parametrize(
    ("flag", "value", "match"),
    (
        ("--profile", "upstream-current", "fixture stem profile"),
        ("--observation-mode", "vision", "fixture stem observation mode"),
        ("--source-revision", UPSTREAM_CURRENT_REVISION, "profile authority"),
    ),
)
def test_refresh_manifest_rejects_mismatched_coordinates(
    tmp_path: Path, flag: str, value: str, match: str
) -> None:
    from world_marl.dreamer_v3_baseline import fixture_generator

    stem = STEMS[0]
    _copy_pair(tmp_path, stem)
    values = {
        "--profile": "paper",
        "--observation-mode": "proprio",
        "--source-revision": PAPER_REVISION,
        "--fixture-stem": stem,
    }
    values[flag] = value
    args = fixture_generator._parse_args(
        [
            "refresh-manifest",
            "--profile",
            values["--profile"],
            "--observation-mode",
            values["--observation-mode"],
            "--reference-checkout",
            str(tmp_path),
            "--source-revision",
            values["--source-revision"],
            "--output-dir",
            str(tmp_path),
            "--fixture-stem",
            values["--fixture-stem"],
        ]
    )
    with pytest.raises(ValueError, match=match):
        args.handler(args)


@pytest.mark.parametrize(
    "variant",
    ("pretty", "reordered", "duplicate", "missing-newline", "extra-newline"),
)
def test_refresh_rejects_noncanonical_manifest_before_source_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    from world_marl.dreamer_v3_baseline import fixture_generator

    stem = STEMS[0]
    manifest_path, _ = _copy_pair(tmp_path, stem)
    noncanonical = _noncanonical_manifest_bytes(manifest_path.read_bytes(), variant)
    manifest_path.write_bytes(noncanonical)
    source_reads: list[str] = []

    def forbidden_git_show(checkout: Path, revision: str, path: str) -> bytes:
        del checkout, revision
        source_reads.append(path)
        raise AssertionError("source read preceded canonical manifest validation")

    monkeypatch.setattr(fixture_generator, "_git_show", forbidden_git_show)
    args = fixture_generator._parse_args(
        [
            "refresh-manifest",
            "--profile",
            "paper",
            "--observation-mode",
            "proprio",
            "--reference-checkout",
            str(tmp_path),
            "--source-revision",
            PAPER_REVISION,
            "--output-dir",
            str(tmp_path),
            "--fixture-stem",
            stem,
        ]
    )

    with pytest.raises(ValueError, match="canonical"):
        args.handler(args)
    assert source_reads == []
    assert manifest_path.read_bytes() == noncanonical


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("case_name", "evil"),
        ("official_commit", "0" * 40),
        ("source_spec", "replay"),
        ("dtype", "float32"),
        ("seed", 1),
        ("fixture_file", "other.npz"),
        ("fixture_sha256", "0" * 64),
        ("schema_version", 999),
        ("generator_request", "{}"),
        ("generator_command", ["wrong"]),
        ("unexpected", True),
    ),
)
def test_refresh_rejects_prior_manifest_drift_before_source_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    from world_marl.dreamer_v3_baseline import fixture_generator

    stem = STEMS[0]
    manifest_path, _ = _copy_pair(tmp_path, stem)
    payload = json.loads(manifest_path.read_text())
    payload[field] = value
    _write_payload(manifest_path, payload)
    source_reads: list[str] = []

    def forbidden_git_show(checkout: Path, revision: str, path: str) -> bytes:
        del checkout, revision
        source_reads.append(path)
        raise AssertionError("source read preceded prior-pair validation")

    monkeypatch.setattr(fixture_generator, "_git_show", forbidden_git_show)
    args = fixture_generator._parse_args(
        [
            "refresh-manifest",
            "--profile",
            "paper",
            "--observation-mode",
            "proprio",
            "--reference-checkout",
            str(tmp_path),
            "--source-revision",
            PAPER_REVISION,
            "--output-dir",
            str(tmp_path),
            "--fixture-stem",
            stem,
        ]
    )

    with pytest.raises(ValueError):
        args.handler(args)
    assert source_reads == []


def test_refresh_rejects_traversal_and_symlink_escape_before_source_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_marl.dreamer_v3_baseline import fixture_generator

    def forbidden_git_show(checkout: Path, revision: str, path: str) -> bytes:
        del checkout, revision, path
        raise AssertionError("source read preceded path validation")

    monkeypatch.setattr(fixture_generator, "_git_show", forbidden_git_show)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    escaped_stem = "paper-proprio-x/../../escaped"
    escaped_manifest = tmp_path / "escaped.manifest.json"
    escaped_fixture = tmp_path / "escaped.npz"
    escaped_manifest.write_bytes(
        (FIXTURE_ROOT / f"{STEMS[0]}.manifest.json").read_bytes()
    )
    escaped_fixture.write_bytes((FIXTURE_ROOT / f"{STEMS[0]}.npz").read_bytes())
    (output_dir / "paper-proprio-x").mkdir()

    outside = tmp_path / "outside"
    outside.mkdir()
    outside_manifest, outside_fixture = _copy_pair(outside, STEMS[0])
    (output_dir / outside_manifest.name).symlink_to(outside_manifest)
    (output_dir / outside_fixture.name).symlink_to(outside_fixture)

    with pytest.raises(SystemExit):
        fixture_generator._parse_args(
            [
                "refresh-manifest",
                "--profile",
                "paper",
                "--observation-mode",
                "proprio",
                "--reference-checkout",
                str(tmp_path),
                "--source-revision",
                PAPER_REVISION,
                "--output-dir",
                str(output_dir),
                "--fixture-stem",
                escaped_stem,
            ]
        )

    args = fixture_generator._parse_args(
        [
            "refresh-manifest",
            "--profile",
            "paper",
            "--observation-mode",
            "proprio",
            "--reference-checkout",
            str(tmp_path),
            "--source-revision",
            PAPER_REVISION,
            "--output-dir",
            str(output_dir),
            "--fixture-stem",
            STEMS[0],
        ]
    )
    with pytest.raises(ValueError, match="symlink"):
        args.handler(args)


@pytest.mark.parametrize("invalid", ("filename", "hash", "schema"))
def test_write_pair_validation_failure_leaves_no_half_pair(
    tmp_path: Path, invalid: str
) -> None:
    from world_marl.dreamer_v3_baseline.fixture_generator import _write_pair

    stem = STEMS[0]
    manifest = OracleManifest.load(FIXTURE_ROOT / f"{stem}.manifest.json")
    with np.load(FIXTURE_ROOT / f"{stem}.npz", allow_pickle=False) as fixture:
        arrays = {name: fixture[name] for name in fixture.files}
    if invalid == "filename":
        manifest = replace(manifest, fixture_file="wrong.npz")
    elif invalid == "hash":
        manifest = replace(manifest, fixture_sha256="0" * 64)
    else:
        schema = dict(manifest.tensor_schema)
        first = next(iter(schema))
        schema[first] = TensorSpec((999,), schema[first].dtype)
        manifest = replace(manifest, tensor_schema=schema)

    with pytest.raises(ValueError):
        _write_pair(
            output_dir=tmp_path,
            fixture_stem=stem,
            arrays=arrays,
            manifest=manifest,
        )
    assert not (tmp_path / f"{stem}.npz").exists()
    assert not (tmp_path / f"{stem}.manifest.json").exists()


@pytest.mark.parametrize("string_type", EXOTIC_STRING_TYPES)
def test_write_pair_rejects_nonexact_array_keys_before_filesystem_work(
    tmp_path: Path, string_type: type[str]
) -> None:
    from world_marl.dreamer_v3_baseline.fixture_generator import _write_pair

    stem = STEMS[0]
    manifest, arrays = _pair_inputs(stem)
    first = next(iter(arrays))
    arrays = {
        string_type(name) if name == first else name: value
        for name, value in arrays.items()
    }
    output_dir = tmp_path / "uncreated"

    with pytest.raises(TypeError, match="array keys must be exact strings"):
        _write_pair(
            output_dir=output_dir,
            fixture_stem=stem,
            arrays=arrays,
            manifest=manifest,
        )

    assert not output_dir.exists()


def test_write_pair_refuses_existing_destinations_without_mutation(
    tmp_path: Path,
) -> None:
    from world_marl.dreamer_v3_baseline.fixture_generator import _write_pair

    stem = STEMS[0]
    fixture_path = tmp_path / f"{stem}.npz"
    manifest_path = tmp_path / f"{stem}.manifest.json"
    fixture_path.write_bytes(b"fixture sentinel")
    manifest_path.write_bytes(b"manifest sentinel")
    manifest = OracleManifest.load(FIXTURE_ROOT / f"{stem}.manifest.json")
    with np.load(FIXTURE_ROOT / f"{stem}.npz", allow_pickle=False) as fixture:
        arrays = {name: fixture[name] for name in fixture.files}

    with pytest.raises(ValueError, match="already exists"):
        _write_pair(
            output_dir=tmp_path,
            fixture_stem=stem,
            arrays=arrays,
            manifest=manifest,
        )
    assert fixture_path.read_bytes() == b"fixture sentinel"
    assert manifest_path.read_bytes() == b"manifest sentinel"


def test_write_pair_manifest_stage_failure_leaves_no_half_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_marl.dreamer_v3_baseline.fixture_generator import _write_pair

    stem = STEMS[0]
    manifest = OracleManifest.load(FIXTURE_ROOT / f"{stem}.manifest.json")
    with np.load(FIXTURE_ROOT / f"{stem}.npz", allow_pickle=False) as fixture:
        arrays = {name: fixture[name] for name in fixture.files}

    def fail_save(self: OracleManifest, path: str | Path) -> Path:
        del self
        Path(path).write_bytes(b"partial staged manifest")
        raise OSError("injected manifest stage failure")

    monkeypatch.setattr(OracleManifest, "save", fail_save)
    with pytest.raises(OSError, match="injected"):
        _write_pair(
            output_dir=tmp_path,
            fixture_stem=stem,
            arrays=arrays,
            manifest=manifest,
        )
    assert not (tmp_path / f"{stem}.npz").exists()
    assert not (tmp_path / f"{stem}.manifest.json").exists()
    assert not tuple(tmp_path.iterdir())


def test_manifest_save_does_not_follow_predictable_temp_symlink(
    tmp_path: Path,
) -> None:
    manifest, _ = _pair_inputs()
    destination = tmp_path / "saved.manifest.json"
    outside = tmp_path / "outside-manifest-sentinel"
    sentinel = b"outside manifest sentinel"
    outside.write_bytes(sentinel)
    predictable = tmp_path / f".{destination.name}.tmp"
    predictable.symlink_to(outside)

    manifest.save(destination)

    assert outside.read_bytes() == sentinel
    assert predictable.is_symlink()
    assert destination.is_file() and not destination.is_symlink()
    assert json.loads(destination.read_bytes()) == manifest.to_dict()


def test_deterministic_npz_does_not_follow_predictable_temp_symlink(
    tmp_path: Path,
) -> None:
    from world_marl.dreamer_v3_baseline.oracle import _write_deterministic_npz

    destination = tmp_path / "saved.npz"
    outside = tmp_path / "outside-npz-sentinel"
    sentinel = b"outside npz sentinel"
    outside.write_bytes(sentinel)
    predictable = tmp_path / f".{destination.name}.tmp"
    predictable.symlink_to(outside)
    arrays = {"value": np.arange(4, dtype=np.float32)}

    _write_deterministic_npz(destination, arrays)

    assert outside.read_bytes() == sentinel
    assert predictable.is_symlink()
    assert destination.is_file() and not destination.is_symlink()
    with np.load(destination, allow_pickle=False) as loaded:
        np.testing.assert_array_equal(loaded["value"], arrays["value"])


@pytest.mark.parametrize("string_type", EXOTIC_STRING_TYPES)
def test_deterministic_npz_rejects_nonexact_array_keys_before_filesystem_work(
    tmp_path: Path, string_type: type[str]
) -> None:
    from world_marl.dreamer_v3_baseline.oracle import _write_deterministic_npz

    output_dir = tmp_path / "uncreated"
    destination = output_dir / "fixture.npz"

    with pytest.raises(TypeError, match="array keys must be exact strings"):
        _write_deterministic_npz(
            destination,
            {string_type("value"): np.arange(4, dtype=np.float32)},
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    "legacy_name",
    (
        ".{stem}.npz.stage",
        ".{stem}.manifest.json.stage",
        "..{stem}.npz.stage.tmp",
        "..{stem}.manifest.json.stage.tmp",
    ),
)
def test_write_pair_does_not_follow_predictable_stage_symlinks(
    tmp_path: Path, legacy_name: str
) -> None:
    from world_marl.dreamer_v3_baseline.fixture_generator import _write_pair

    stem = STEMS[0]
    manifest, arrays = _pair_inputs(stem)
    outside = tmp_path / "outside-stage-sentinel"
    sentinel = b"outside stage sentinel"
    outside.write_bytes(sentinel)
    legacy = tmp_path / legacy_name.format(stem=stem)
    legacy.symlink_to(outside)

    fixture_path, manifest_path = _write_pair(
        output_dir=tmp_path,
        fixture_stem=stem,
        arrays=arrays,
        manifest=manifest,
    )

    assert outside.read_bytes() == sentinel
    assert legacy.is_symlink()
    assert fixture_path.is_file() and not fixture_path.is_symlink()
    assert manifest_path.is_file() and not manifest_path.is_symlink()


@pytest.mark.parametrize("failing_call", (1, 2))
def test_write_pair_link_failures_roll_back_both_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_call: int,
) -> None:
    from world_marl.dreamer_v3_baseline import fixture_generator

    stem = STEMS[0]
    manifest, arrays = _pair_inputs(stem)
    real_link = os.link
    calls = 0

    def fail_selected_link(source: str | Path, destination: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls == failing_call:
            raise OSError(f"injected link failure {failing_call}")
        real_link(source, destination)

    monkeypatch.setattr(fixture_generator.os, "link", fail_selected_link)
    with pytest.raises(OSError, match="injected link failure"):
        fixture_generator._write_pair(
            output_dir=tmp_path,
            fixture_stem=stem,
            arrays=arrays,
            manifest=manifest,
        )
    assert calls == failing_call
    assert not (tmp_path / f"{stem}.npz").exists()
    assert not (tmp_path / f"{stem}.manifest.json").exists()
    assert not tuple(tmp_path.iterdir())


def test_write_pair_npz_write_failure_cleans_random_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_marl.dreamer_v3_baseline import fixture_generator, oracle

    stem = STEMS[0]
    manifest, arrays = _pair_inputs(stem)

    def fail_write(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("injected NPZ write failure")

    monkeypatch.setattr(oracle.np.lib.format, "write_array", fail_write)
    with pytest.raises(OSError, match="injected NPZ write failure"):
        fixture_generator._write_pair(
            output_dir=tmp_path,
            fixture_stem=stem,
            arrays=arrays,
            manifest=manifest,
        )
    assert not (tmp_path / f"{stem}.npz").exists()
    assert not (tmp_path / f"{stem}.manifest.json").exists()
    assert not tuple(tmp_path.iterdir())


def test_architecture_caller_inventory_covers_fixture_generator_imports() -> None:
    root = Path("src/world_marl/dreamer_v3_baseline")
    architecture = (root / "ARCHITECTURE.md").read_text()
    section = architecture.split("### 3.1 Complete live-symbol migration inventory", 1)[
        1
    ].split("#### Legacy import-site inventory", 1)[0]
    rows = {
        cells[0].removeprefix("`").removesuffix("`"): cells[3]
        for line in section.splitlines()
        if line.startswith("| `")
        for cells in [[cell.strip() for cell in line.strip().strip("|").split("|")]]
    }
    package_modules = {path.stem for path in root.glob("*.py")}
    missing = set()
    for path in sorted(root.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not (
                node.module.startswith("world_marl.dreamer_v3_baseline.")
                or (node.level and node.module in package_modules)
            ):
                continue
            source_module = node.module.rsplit(".", 1)[-1] + ".py"
            token = (
                "E"
                if path.name == "__init__.py"
                else "O"
                if "oracle" in path.name
                else path.name
            )
            for alias in node.names:
                key = f"{source_module}::{alias.name}"
                if key in rows and token not in {
                    item.strip() for item in rows[key].split(",")
                }:
                    missing.add((key, path.name, token))
    assert not missing


def test_dead_generic_oracle_apis_are_absent_from_module_and_package() -> None:
    import ast

    import world_marl.dreamer_v3_baseline as package
    import world_marl.dreamer_v3_baseline.oracle as oracle

    source = Path("src/world_marl/dreamer_v3_baseline/oracle.py").read_text()
    names = {
        node.name
        for node in ast.parse(source).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert (
        not {
            "OracleInvocation",
            "OracleHarness",
            "profile_overrides",
            "official_revision",
        }
        & names
    )
    assert "CONFIG_SOURCE_SPEC" not in source
    assert "OracleHarness" not in package.__all__
    assert "OracleInvocation" not in package.__all__
    for transitional in (
        "OracleSourceSpec",
        "register_oracle_source_spec",
        "source_spec_for",
    ):
        assert transitional not in names
        assert transitional not in oracle.__all__
        assert not hasattr(oracle, transitional)
        assert transitional not in package.__all__
        assert not hasattr(package, transitional)
