from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import yaml

if __package__:
    from .config import (
        DreamerProfile,
        DreamerV3Config,
        ObservationMode,
        resolve_dreamer_config,
    )
else:  # pragma: no cover - exercised by the subprocess worker.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from world_marl.dreamer_v3_baseline.config import (
        DreamerProfile,
        DreamerV3Config,
        ObservationMode,
        resolve_dreamer_config,
    )


ORACLE_SCHEMA_VERSION = 2
PAPER_REVISION = "bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01"
UPSTREAM_CURRENT_REVISION = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"
_CASE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_CONFIG_SOURCE_SHA256 = (
    "9dff9c7062e3e33951cb54c6dd4b598aaf7e56e18e2cff39c812eaa797bcfcfc"
)
_DISTRIBUTIONS_SOURCE_HASHES = {
    "embodied/jax/heads.py": (
        "437641cde21e7f9e3f69b88ad8f6b7e7c22e54eec8c5b19eef6127afde1a9b3f"
    ),
    "embodied/jax/nets.py": (
        "9a1c0c71ad7d3596572a44416e78434f777d8f4dbcbe8ca0dd6b86bb8246392c"
    ),
    "embodied/jax/outs.py": (
        "7e80691f175c71be614f089023cce3a809e0d026c6d5ce89bf566d5f11eb3ed0"
    ),
}
_PAPER_OVERRIDES: dict[str, Any] = {
    "agent.dec.simple.strided": True,
    "agent.enc.simple.strided": True,
    "agent.opt.beta2": 0.99,
    "run.steps": 1_000_000,
}

GeneratorProvenanceValidator = Callable[
    ["OracleManifest", Mapping[str, Any], Path | None],
    None,
]


@dataclass(frozen=True)
class OracleSourceSpec:
    name: str
    revision_hashes: Mapping[str, Mapping[str, str]]
    execution_dtypes: tuple[str, ...] = ()
    generator_validation_required: bool = False
    generator_validator: GeneratorProvenanceValidator | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not _CASE_PATTERN.fullmatch(self.name):
            raise ValueError(f"invalid oracle source spec name: {self.name!r}")
        normalized: dict[str, Mapping[str, str]] = {}
        for revision, file_hashes in sorted(self.revision_hashes.items()):
            if not _COMMIT_PATTERN.fullmatch(revision):
                raise ValueError("oracle source revision must be a full Git object id")
            if not file_hashes:
                raise ValueError("oracle source spec must contain official files")
            files: dict[str, str] = {}
            for path, digest in sorted(file_hashes.items()):
                if (
                    not path
                    or path.startswith("/")
                    or ".." in Path(path).parts
                    or not _SHA256_PATTERN.fullmatch(digest)
                ):
                    raise ValueError(f"invalid oracle source entry: {path!r}")
                files[path] = digest
            normalized[revision] = MappingProxyType(files)
        required_revisions = {PAPER_REVISION, UPSTREAM_CURRENT_REVISION}
        if set(normalized) != required_revisions:
            raise ValueError("oracle source spec must pin both authority revisions")
        object.__setattr__(
            self,
            "revision_hashes",
            MappingProxyType(normalized),
        )
        dtypes = tuple(jnp.dtype(dtype).name for dtype in self.execution_dtypes)
        if len(set(dtypes)) != len(dtypes):
            raise ValueError("oracle source execution dtypes must be unique")
        object.__setattr__(self, "execution_dtypes", dtypes)
        if self.generator_validation_required and self.generator_validator is None:
            raise ValueError("oracle source requires a generator validator")
        if self.generator_validator is not None and not callable(
            self.generator_validator
        ):
            raise TypeError("oracle source generator validator must be callable")

    def hashes_for(self, revision: str) -> Mapping[str, str]:
        try:
            return self.revision_hashes[revision]
        except KeyError as error:
            raise ValueError(
                f"oracle source spec {self.name!r} does not pin revision {revision}"
            ) from error

    def allows_execution_dtype(self, dtype: str, canonical_dtype: str) -> bool:
        allowed = self.execution_dtypes or (jnp.dtype(canonical_dtype).name,)
        return jnp.dtype(dtype).name in allowed


_ORACLE_SOURCE_SPECS: dict[str, OracleSourceSpec] = {}


def register_oracle_source_spec(source_spec: OracleSourceSpec) -> None:
    if (
        source_spec.generator_validation_required
        and source_spec.generator_validator is None
    ):
        raise ValueError("oracle source requires a generator validator")
    existing = _ORACLE_SOURCE_SPECS.get(source_spec.name)
    if existing is not None and existing != source_spec:
        raise ValueError(f"oracle source spec already registered: {source_spec.name}")
    _ORACLE_SOURCE_SPECS[source_spec.name] = source_spec


def oracle_source_spec(name: str) -> OracleSourceSpec:
    try:
        return _ORACLE_SOURCE_SPECS[name]
    except KeyError as error:
        raise ValueError(f"unknown oracle source spec: {name}") from error


CONFIG_SOURCE_SPEC = OracleSourceSpec(
    name="config",
    revision_hashes={
        PAPER_REVISION: {"dreamerv3/configs.yaml": _CONFIG_SOURCE_SHA256},
        UPSTREAM_CURRENT_REVISION: {"dreamerv3/configs.yaml": _CONFIG_SOURCE_SHA256},
    },
)
register_oracle_source_spec(CONFIG_SOURCE_SPEC)

DISTRIBUTIONS_SOURCE_SPEC = OracleSourceSpec(
    name="distributions",
    revision_hashes={
        PAPER_REVISION: _DISTRIBUTIONS_SOURCE_HASHES,
        UPSTREAM_CURRENT_REVISION: _DISTRIBUTIONS_SOURCE_HASHES,
    },
)
register_oracle_source_spec(DISTRIBUTIONS_SOURCE_SPEC)


def official_revision(profile: DreamerProfile | str) -> str:
    resolved = DreamerProfile(profile)
    if resolved is DreamerProfile.PAPER:
        return PAPER_REVISION
    return UPSTREAM_CURRENT_REVISION


def profile_overrides(profile: DreamerProfile | str) -> Mapping[str, Any]:
    resolved = DreamerProfile(profile)
    values = _PAPER_OVERRIDES if resolved is DreamerProfile.PAPER else {}
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("tensor dimensions must be nonnegative")
        try:
            np.dtype(self.dtype)
        except TypeError as error:
            raise ValueError(f"invalid tensor dtype: {self.dtype}") from error

    def to_dict(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class OracleManifest:
    schema_version: int
    case_name: str
    profile: DreamerProfile | str
    observation_mode: ObservationMode | str
    official_commit: str
    source_spec: str
    official_file_hashes: Mapping[str, str]
    profile_hash: str
    overrides: Mapping[str, Any]
    jax_version: str
    dtype: str
    device: str
    seed: int
    tensor_schema: Mapping[str, TensorSpec]
    generator_command: tuple[str, ...]
    generator_request: str | None
    fixture_file: str
    fixture_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", DreamerProfile(self.profile))
        object.__setattr__(
            self,
            "observation_mode",
            ObservationMode(self.observation_mode),
        )
        object.__setattr__(
            self,
            "official_file_hashes",
            MappingProxyType(dict(sorted(self.official_file_hashes.items()))),
        )
        object.__setattr__(
            self,
            "overrides",
            MappingProxyType(dict(sorted(self.overrides.items()))),
        )
        object.__setattr__(
            self,
            "tensor_schema",
            MappingProxyType(dict(sorted(self.tensor_schema.items()))),
        )
        object.__setattr__(self, "generator_command", tuple(self.generator_command))
        if self.generator_request is not None:
            canonical_request = _canonical_generator_request(self.generator_request)
            object.__setattr__(self, "generator_request", canonical_request)

    @classmethod
    def create(
        cls,
        *,
        case_name: str,
        profile: DreamerProfile | str,
        observation_mode: ObservationMode | str,
        official_checkout: str | Path,
        fixture_path: str | Path,
        arrays: Mapping[str, np.ndarray],
        seed: int,
        generator_command: Sequence[str],
        source_spec: str | OracleSourceSpec,
        generator_request: Mapping[str, Any] | None = None,
        dtype: str | None = None,
        device: str | None = None,
    ) -> OracleManifest:
        resolved_profile = DreamerProfile(profile)
        resolved_mode = ObservationMode(observation_mode)
        config = resolve_dreamer_config(resolved_profile, resolved_mode)
        checkout = Path(official_checkout).resolve()
        revision = official_revision(resolved_profile)
        resolved_source_spec = (
            source_spec
            if isinstance(source_spec, OracleSourceSpec)
            else oracle_source_spec(source_spec)
        )
        expected_source_hashes = resolved_source_spec.hashes_for(revision)
        source_hashes = {
            path: _sha256_bytes(_git_show(checkout, revision, path))
            for path in expected_source_hashes
        }
        if source_hashes != dict(expected_source_hashes):
            raise ValueError(
                f"official checkout does not match source spec "
                f"{resolved_source_spec.name!r}"
            )
        fixture = Path(fixture_path)
        assert config.run is not None
        execution_dtype = jnp.dtype(dtype or config.run.compute_dtype).name
        if not resolved_source_spec.allows_execution_dtype(
            execution_dtype,
            config.run.compute_dtype,
        ):
            raise ValueError(
                f"oracle source spec {resolved_source_spec.name!r} does not allow "
                f"execution dtype {execution_dtype!r}"
            )
        return cls(
            schema_version=ORACLE_SCHEMA_VERSION,
            case_name=case_name,
            profile=resolved_profile,
            observation_mode=resolved_mode,
            official_commit=revision,
            source_spec=resolved_source_spec.name,
            official_file_hashes=source_hashes,
            profile_hash=config.canonical_hash(),
            overrides=profile_overrides(resolved_profile),
            jax_version=jax.__version__,
            dtype=execution_dtype,
            device=device or jax.default_backend(),
            seed=seed,
            tensor_schema={
                name: TensorSpec(tuple(array.shape), array.dtype.name)
                for name, array in sorted(arrays.items())
            },
            generator_command=tuple(generator_command),
            generator_request=(
                _canonical_generator_request(generator_request)
                if generator_request is not None
                else None
            ),
            fixture_file=fixture.name,
            fixture_sha256=_sha256_path(fixture),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "device": self.device,
            "dtype": self.dtype,
            "fixture_file": self.fixture_file,
            "fixture_sha256": self.fixture_sha256,
            "generator_command": list(self.generator_command),
            "generator_request": self.generator_request,
            "jax_version": self.jax_version,
            "observation_mode": self.observation_mode.value,
            "official_commit": self.official_commit,
            "official_file_hashes": dict(self.official_file_hashes),
            "overrides": dict(self.overrides),
            "profile": self.profile.value,
            "profile_hash": self.profile_hash,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "source_spec": self.source_spec,
            "tensor_schema": {
                name: spec.to_dict() for name, spec in self.tensor_schema.items()
            },
        }

    def canonical_hash(self) -> str:
        return _sha256_bytes(_canonical_json(self.to_dict()))

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_bytes(_canonical_json(self.to_dict()) + b"\n")
        temporary.replace(destination)
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        official_checkout: str | Path | None = None,
        fixture_path: str | Path | None = None,
        config: DreamerV3Config | None = None,
    ) -> OracleManifest:
        source = Path(path)
        payload = json.loads(source.read_text())
        manifest = cls(
            schema_version=payload["schema_version"],
            case_name=payload["case_name"],
            profile=payload["profile"],
            observation_mode=payload["observation_mode"],
            official_commit=payload["official_commit"],
            source_spec=payload["source_spec"],
            official_file_hashes=payload["official_file_hashes"],
            profile_hash=payload["profile_hash"],
            overrides=payload["overrides"],
            jax_version=payload["jax_version"],
            dtype=payload["dtype"],
            device=payload["device"],
            seed=payload["seed"],
            tensor_schema={
                name: TensorSpec(tuple(spec["shape"]), spec["dtype"])
                for name, spec in payload["tensor_schema"].items()
            },
            generator_command=tuple(payload["generator_command"]),
            generator_request=payload["generator_request"],
            fixture_file=payload["fixture_file"],
            fixture_sha256=payload["fixture_sha256"],
        )
        fixture = (
            Path(fixture_path)
            if fixture_path is not None
            else source.parent / manifest.fixture_file
        )
        manifest.validate(
            official_checkout=official_checkout,
            fixture_path=fixture,
            config=config,
        )
        return manifest

    def validate(
        self,
        *,
        official_checkout: str | Path | None = None,
        fixture_path: str | Path | None = None,
        config: DreamerV3Config | None = None,
    ) -> None:
        if self.schema_version != ORACLE_SCHEMA_VERSION:
            raise ValueError(
                f"oracle schema version {self.schema_version} is unsupported"
            )
        if not _CASE_PATTERN.fullmatch(self.case_name):
            raise ValueError(f"invalid oracle case name: {self.case_name!r}")
        if not _COMMIT_PATTERN.fullmatch(self.official_commit):
            raise ValueError("official commit is not a full Git object id")
        expected_commit = official_revision(self.profile)
        if self.official_commit != expected_commit:
            raise ValueError("official commit does not match profile authority")
        source_spec = oracle_source_spec(self.source_spec)
        expected_source_hashes = dict(source_spec.hashes_for(expected_commit))
        if dict(self.official_file_hashes) != expected_source_hashes:
            raise ValueError(
                "oracle source file hashes do not match the pinned source spec"
            )
        expected_overrides = dict(profile_overrides(self.profile))
        if dict(self.overrides) != expected_overrides:
            raise ValueError("oracle override map does not match profile")
        if config is not None:
            config.validate()
            if config._legacy:
                raise ValueError("oracle requires a canonical supplied config")
            if config.profile is not self.profile:
                raise ValueError("supplied config profile does not match manifest")
            if config.observation_mode is not self.observation_mode:
                raise ValueError(
                    "supplied config observation mode does not match manifest"
                )
            expected_config = config
        else:
            expected_config = resolve_dreamer_config(
                self.profile,
                self.observation_mode,
            )
        if self.profile_hash != expected_config.canonical_hash():
            raise ValueError("oracle profile hash does not match resolved config")
        if not _SHA256_PATTERN.fullmatch(self.profile_hash):
            raise ValueError("oracle profile hash is malformed")
        if self.jax_version != jax.__version__:
            raise ValueError("oracle JAX version does not match the runtime")
        assert expected_config.run is not None
        if not source_spec.allows_execution_dtype(
            self.dtype,
            expected_config.run.compute_dtype,
        ):
            raise ValueError(
                "oracle dtype is not allowed by the pinned source specification"
            )
        if source_spec.execution_dtypes:
            if self.generator_request is None:
                raise ValueError(
                    "multi-dtype oracle source requires an explicit generator request"
                )
        request: dict[str, Any] | None = None
        if self.generator_request is not None:
            request = json.loads(self.generator_request)
            if not isinstance(request, dict):
                raise ValueError("oracle generator request must be a JSON object")
            if source_spec.execution_dtypes and "compute_dtype" not in request:
                raise ValueError(
                    "multi-dtype oracle source requires an explicit generator "
                    "compute dtype"
                )
            if "compute_dtype" in request and (
                jnp.dtype(request["compute_dtype"]).name != jnp.dtype(self.dtype).name
            ):
                raise ValueError(
                    "oracle dtype does not match the executed generator request"
                )
        expected_device = jax.default_backend()
        if self.device != expected_device:
            raise ValueError(
                f"oracle device {self.device!r} does not match runtime "
                f"{expected_device!r}"
            )
        if not self.generator_command:
            raise ValueError("oracle generator command must be recorded")
        if self.seed < 0:
            raise ValueError("oracle seed must be nonnegative")
        validator = source_spec.generator_validator
        if source_spec.generator_validation_required and validator is None:
            raise ValueError("oracle source generator validator is unavailable")
        if validator is not None:
            if request is None:
                raise ValueError("oracle source generator validator requires a request")
            validator(
                self,
                request,
                Path(official_checkout).resolve()
                if official_checkout is not None
                else None,
            )
        if official_checkout is not None:
            checkout = Path(official_checkout).resolve()
            for path, recorded_hash in self.official_file_hashes.items():
                expected_hash = _sha256_bytes(
                    _git_show(checkout, self.official_commit, path)
                )
                if recorded_hash != expected_hash:
                    raise ValueError(f"oracle source file hash mismatch: {path}")
        if not _SHA256_PATTERN.fullmatch(self.fixture_sha256):
            raise ValueError("oracle fixture hash is malformed")
        if fixture_path is not None:
            fixture = Path(fixture_path)
            if not fixture.is_file():
                raise ValueError(f"oracle fixture is missing: {fixture}")
            if fixture.name != self.fixture_file:
                raise ValueError("oracle fixture filename does not match manifest")
            if _sha256_path(fixture) != self.fixture_sha256:
                raise ValueError("oracle fixture hash mismatch")
            self._validate_tensor_schema(fixture)

    def _validate_tensor_schema(self, fixture_path: Path) -> None:
        with np.load(fixture_path, allow_pickle=False) as fixture:
            if tuple(fixture.files) != tuple(self.tensor_schema):
                raise ValueError("oracle tensor schema names do not match fixture")
            for name, spec in self.tensor_schema.items():
                array = fixture[name]
                if tuple(array.shape) != spec.shape or array.dtype.name != spec.dtype:
                    raise ValueError(f"oracle tensor schema mismatch: {name}")


@dataclass(frozen=True)
class ParameterMapping:
    source: str
    destination: str
    transform: str = "identity"
    reshape: tuple[int, ...] | None = None


class ParameterTranslator:
    _TRANSFORMS = frozenset({"identity", "transpose", "reshape"})

    def __init__(self) -> None:
        self._by_source: dict[str, ParameterMapping] = {}
        self._by_destination: dict[str, ParameterMapping] = {}
        self._consumed_sources: set[str] = set()
        self._consumed_destinations: set[str] = set()

    @property
    def registry(self) -> tuple[ParameterMapping, ...]:
        return tuple(self._by_source[name] for name in sorted(self._by_source))

    def register(
        self,
        source: str | Sequence[str],
        destination: str | Sequence[str],
        *,
        transform: str = "identity",
        reshape: Sequence[int] | None = None,
    ) -> None:
        source_path = _parameter_path(source)
        destination_path = _parameter_path(destination)
        if transform not in self._TRANSFORMS:
            raise ValueError(f"unknown transform: {transform}")
        if transform == "reshape" and reshape is None:
            raise ValueError("reshape transform requires a target shape")
        if transform != "reshape" and reshape is not None:
            raise ValueError("reshape target is only valid for reshape transform")
        if source_path in self._by_source:
            raise ValueError(f"source parameter already registered: {source_path}")
        if destination_path in self._by_destination:
            raise ValueError(
                f"destination parameter already registered: {destination_path}"
            )
        mapping = ParameterMapping(
            source_path,
            destination_path,
            transform,
            tuple(reshape) if reshape is not None else None,
        )
        self._by_source[source_path] = mapping
        self._by_destination[destination_path] = mapping

    def reset_consumption(self) -> None:
        self._consumed_sources.clear()
        self._consumed_destinations.clear()

    def consume(
        self,
        source: str | Sequence[str],
        destination: str | Sequence[str],
        value: np.ndarray,
        destination_shape: Sequence[int],
    ) -> np.ndarray:
        source_path = _parameter_path(source)
        destination_path = _parameter_path(destination)
        mapping = self._by_source.get(source_path)
        if mapping is None or mapping.destination != destination_path:
            raise ValueError(
                f"parameter mapping is not registered: {source_path} -> "
                f"{destination_path}"
            )
        if source_path in self._consumed_sources:
            raise ValueError(f"source parameter consumed more than once: {source_path}")
        if destination_path in self._consumed_destinations:
            raise ValueError(
                f"destination parameter consumed more than once: {destination_path}"
            )
        transformed = _transform_parameter(np.asarray(value), mapping)
        expected_shape = tuple(destination_shape)
        if transformed.shape != expected_shape:
            raise ValueError(
                f"parameter shape mismatch for {destination_path}: "
                f"{transformed.shape} != {expected_shape}"
            )
        self._consumed_sources.add(source_path)
        self._consumed_destinations.add(destination_path)
        return transformed

    def translate(
        self,
        source_parameters: Mapping[str, np.ndarray],
        destination_shapes: Mapping[str, Sequence[int] | np.ndarray],
    ) -> dict[str, np.ndarray]:
        self.reset_consumption()
        source_paths = set(source_parameters)
        destination_paths = set(destination_shapes)
        registered_sources = set(self._by_source)
        registered_destinations = set(self._by_destination)
        extra_sources = source_paths - registered_sources
        if extra_sources:
            raise ValueError(
                "unregistered source parameters: " + ", ".join(sorted(extra_sources))
            )
        extra_destinations = destination_paths - registered_destinations
        if extra_destinations:
            raise ValueError(
                "unregistered destination parameters: "
                + ", ".join(sorted(extra_destinations))
            )
        missing_sources = registered_sources - source_paths
        missing_destinations = registered_destinations - destination_paths
        if missing_sources or missing_destinations:
            missing = sorted(missing_sources | missing_destinations)
            raise ValueError("unconsumed registered parameters: " + ", ".join(missing))
        translated: dict[str, np.ndarray] = {}
        for mapping in self.registry:
            destination = destination_shapes[mapping.destination]
            shape = (
                destination.shape
                if isinstance(destination, np.ndarray)
                else destination
            )
            translated[mapping.destination] = self.consume(
                mapping.source,
                mapping.destination,
                source_parameters[mapping.source],
                shape,
            )
        self.assert_fully_consumed()
        return translated

    def assert_fully_consumed(self) -> None:
        missing_sources = set(self._by_source) - self._consumed_sources
        missing_destinations = set(self._by_destination) - self._consumed_destinations
        if missing_sources or missing_destinations:
            missing = sorted(missing_sources | missing_destinations)
            raise ValueError("unconsumed parameters: " + ", ".join(missing))


class OracleHarness:
    def __init__(
        self,
        official_checkout: str | Path,
        fixture_dir: str | Path,
        *,
        python_executable: str | Path | None = None,
    ) -> None:
        self.official_checkout = Path(official_checkout).resolve()
        self.fixture_dir = Path(fixture_dir)
        self.python_executable = str(python_executable or sys.executable)
        self._last_worker_pid: int | None = None
        if not self.official_checkout.is_dir():
            raise ValueError("official checkout does not exist")
        for revision in (PAPER_REVISION, UPSTREAM_CURRENT_REVISION):
            _git_object_exists(self.official_checkout, revision)

    @property
    def last_worker_pid(self) -> int | None:
        return self._last_worker_pid

    def write_fixture(
        self,
        *,
        case_name: str,
        profile: DreamerProfile | str,
        observation_mode: ObservationMode | str,
        arrays: Mapping[str, np.ndarray],
        seed: int,
        generator_command: Sequence[str],
        source_spec: str | OracleSourceSpec,
        generator_request: Mapping[str, Any] | None = None,
        dtype: str | None = None,
    ) -> tuple[Path, Path]:
        if not _CASE_PATTERN.fullmatch(case_name):
            raise ValueError(f"invalid oracle case name: {case_name!r}")
        if not arrays:
            raise ValueError("oracle fixture must contain at least one tensor")
        normalized: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            if not _CASE_PATTERN.fullmatch(name):
                raise ValueError(f"invalid oracle tensor name: {name!r}")
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError("oracle tensors cannot use object dtype")
            normalized[name] = array
        resolved_profile = DreamerProfile(profile)
        resolved_mode = ObservationMode(observation_mode)
        self.fixture_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{resolved_profile.value}-{resolved_mode.value}-{case_name}"
        fixture_path = self.fixture_dir / f"{stem}.npz"
        manifest_path = self.fixture_dir / f"{stem}.manifest.json"
        _write_deterministic_npz(fixture_path, normalized)
        manifest = OracleManifest.create(
            case_name=case_name,
            profile=resolved_profile,
            observation_mode=resolved_mode,
            official_checkout=self.official_checkout,
            fixture_path=fixture_path,
            arrays=normalized,
            seed=seed,
            generator_command=generator_command,
            generator_request=generator_request,
            source_spec=source_spec,
            dtype=dtype,
        )
        manifest.validate(
            official_checkout=self.official_checkout,
            fixture_path=fixture_path,
        )
        manifest.save(manifest_path)
        return fixture_path, manifest_path

    def run_config_case(
        self,
        profile: DreamerProfile | str,
        observation_mode: ObservationMode | str,
        *,
        case_name: str = "config",
        seed: int = 0,
    ) -> tuple[Path, Path]:
        resolved_profile = DreamerProfile(profile)
        resolved_mode = ObservationMode(observation_mode)
        request = {
            "official_checkout": str(self.official_checkout),
            "official_commit": official_revision(resolved_profile),
            "observation_mode": resolved_mode.value,
            "overrides": dict(profile_overrides(resolved_profile)),
            "profile": resolved_profile.value,
            "source_spec": CONFIG_SOURCE_SPEC.name,
        }
        command = (
            self.python_executable,
            str(Path(__file__).resolve()),
            "_config_worker",
        )
        completed = subprocess.run(
            command,
            cwd=self.official_checkout,
            input=_canonical_json(request).decode(),
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        worker_pid = int(payload["worker_pid"])
        if worker_pid <= 0 or worker_pid == os.getpid():
            raise ValueError("oracle config worker did not cross a process boundary")
        self._last_worker_pid = worker_pid
        arrays = {
            name: np.asarray(spec["values"], dtype=spec["dtype"])
            for name, spec in payload["arrays"].items()
        }
        _validate_config_case_arrays(
            arrays,
            resolve_dreamer_config(resolved_profile, resolved_mode),
        )
        return self.write_fixture(
            case_name=case_name,
            profile=resolved_profile,
            observation_mode=resolved_mode,
            arrays=arrays,
            seed=seed,
            generator_command=command,
            generator_request=request,
            source_spec=CONFIG_SOURCE_SPEC.name,
        )

    def run_distributions_case(
        self,
        profile: DreamerProfile | str,
        observation_mode: ObservationMode | str,
        *,
        case_name: str = "distributions",
        seed: int = 0,
    ) -> tuple[Path, Path]:
        resolved_profile = DreamerProfile(profile)
        resolved_mode = ObservationMode(observation_mode)
        request = {
            "official_checkout": str(self.official_checkout),
            "official_commit": official_revision(resolved_profile),
            "observation_mode": resolved_mode.value,
            "overrides": dict(profile_overrides(resolved_profile)),
            "profile": resolved_profile.value,
            "seed": seed,
            "source_spec": DISTRIBUTIONS_SOURCE_SPEC.name,
        }
        command = (
            self.python_executable,
            str(Path(__file__).resolve()),
            "_distributions_worker",
        )
        completed = subprocess.run(
            command,
            cwd=self.official_checkout,
            input=_canonical_json(request).decode(),
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        worker_pid = int(payload["worker_pid"])
        if worker_pid <= 0 or worker_pid == os.getpid():
            raise ValueError(
                "oracle distributions worker did not cross a process boundary"
            )
        self._last_worker_pid = worker_pid
        arrays = {
            name: np.asarray(spec["values"], dtype=spec["dtype"])
            for name, spec in payload["arrays"].items()
        }
        return self.write_fixture(
            case_name=case_name,
            profile=resolved_profile,
            observation_mode=resolved_mode,
            arrays=arrays,
            seed=seed,
            generator_command=command,
            generator_request=request,
            source_spec=DISTRIBUTIONS_SOURCE_SPEC.name,
        )

    def run_networks_case(
        self,
        profile: DreamerProfile | str,
        observation_mode: ObservationMode | str,
        *,
        case_name: str | None = None,
        seed: int = 0,
        compute_dtype: str = "bfloat16",
    ) -> tuple[Path, Path]:
        from world_marl.dreamer_v3_baseline.network_oracle import run_networks_case

        return run_networks_case(
            self,
            profile,
            observation_mode,
            case_name=case_name,
            seed=seed,
            compute_dtype=compute_dtype,
        )


def _validate_config_case_arrays(
    arrays: Mapping[str, np.ndarray],
    config: DreamerV3Config,
) -> None:
    assert config.rssm is not None
    assert config.encoder is not None
    assert config.decoder is not None
    assert config.optimizer is not None
    assert config.run is not None
    expected = {
        "rssm": np.asarray(
            [
                config.rssm.deter,
                config.rssm.hidden,
                config.rssm.stoch,
                config.rssm.classes,
            ],
            dtype=np.int64,
        ),
        "encoder": np.asarray(
            [config.encoder.depth, config.encoder.units, config.encoder.strided],
            dtype=np.int64,
        ),
        "decoder": np.asarray(
            [config.decoder.depth, config.decoder.units, config.decoder.strided],
            dtype=np.int64,
        ),
        "dmc": np.asarray(
            [
                config.observation_mode is ObservationMode.VISION,
                config.observation_mode is ObservationMode.PROPRIO,
                *config.run.image_size,
                config.run.action_repeat,
                config.run.camera,
            ],
            dtype=np.int64,
        ),
        "optimizer": np.asarray([config.optimizer.beta2], dtype=np.float64),
        "run": np.asarray(
            [config.run.steps, config.run.replay_ratio],
            dtype=np.float64,
        ),
    }
    for name, value in expected.items():
        if name not in arrays or not np.array_equal(arrays[name], value):
            raise ValueError(f"official config case disagrees with native {name}")


def _config_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    checkout = Path(request["official_checkout"]).resolve()
    revision = str(request["official_commit"])
    profile = DreamerProfile(request["profile"])
    mode = ObservationMode(request["observation_mode"])
    if revision != official_revision(profile):
        raise ValueError("worker revision does not match requested profile")
    overrides = dict(request["overrides"])
    if overrides != dict(profile_overrides(profile)):
        raise ValueError("worker override map does not match requested profile")
    source_spec = oracle_source_spec(str(request["source_spec"]))
    if source_spec != CONFIG_SOURCE_SPEC:
        raise ValueError("config worker requires the registered config source spec")
    source_files = tuple(source_spec.hashes_for(revision))
    source = yaml.safe_load(_git_show(checkout, revision, source_files[0]))
    defaults = source["defaults"]
    mode_config = source[f"dmc_{mode.value}"]
    size_config = source["size200m" if mode is ObservationMode.VISION else "size1m"]
    rssm_size = size_config[r".*\.rssm"]
    depth = int(size_config[r".*\.depth"])
    units = int(size_config[r".*\.units"])
    rssm_defaults = defaults["agent"]["dyn"]["rssm"]
    encoder_defaults = defaults["agent"]["enc"]["simple"]
    decoder_defaults = defaults["agent"]["dec"]["simple"]
    optimizer_defaults = defaults["agent"]["opt"]
    dmc_defaults = defaults["env"]["dmc"]
    encoder_strided = overrides.get(
        "agent.enc.simple.strided",
        encoder_defaults["strided"],
    )
    decoder_strided = overrides.get(
        "agent.dec.simple.strided",
        decoder_defaults["strided"],
    )
    beta2 = overrides.get("agent.opt.beta2", optimizer_defaults["beta2"])
    steps = overrides.get("run.steps", mode_config["run"]["steps"])
    image = mode_config.get("env.dmc.image", dmc_defaults["image"])
    proprio = mode_config.get("env.dmc.proprio", dmc_defaults["proprio"])
    arrays = {
        "decoder": {
            "dtype": "int64",
            "values": [depth, units, int(decoder_strided)],
        },
        "encoder": {
            "dtype": "int64",
            "values": [depth, units, int(encoder_strided)],
        },
        "dmc": {
            "dtype": "int64",
            "values": [
                int(image),
                int(proprio),
                *dmc_defaults["size"],
                dmc_defaults["repeat"],
                dmc_defaults["camera"],
            ],
        },
        "optimizer": {"dtype": "float64", "values": [beta2]},
        "rssm": {
            "dtype": "int64",
            "values": [
                rssm_size["deter"],
                rssm_size["hidden"],
                rssm_defaults["stoch"],
                rssm_size["classes"],
            ],
        },
        "run": {
            "dtype": "float64",
            "values": [steps, mode_config["run"]["train_ratio"]],
        },
    }
    return {"arrays": arrays, "worker_pid": os.getpid()}


def _distributions_worker(request: Mapping[str, Any]) -> dict[str, Any]:
    checkout = Path(request["official_checkout"]).resolve()
    revision = str(request["official_commit"])
    profile = DreamerProfile(request["profile"])
    ObservationMode(request["observation_mode"])
    if revision != official_revision(profile):
        raise ValueError("worker revision does not match requested profile")
    if dict(request["overrides"]) != dict(profile_overrides(profile)):
        raise ValueError("worker override map does not match requested profile")
    source_spec = oracle_source_spec(str(request["source_spec"]))
    if source_spec != DISTRIBUTIONS_SOURCE_SPEC:
        raise ValueError(
            "distributions worker requires the registered distributions source spec"
        )
    sources: dict[str, bytes] = {}
    for path, digest in source_spec.hashes_for(revision).items():
        source = _git_show(checkout, revision, path)
        if _sha256_bytes(source) != digest:
            raise ValueError(f"official distribution source hash mismatch: {path}")
        sources[path] = source
    official_outs = _load_official_outs(sources["embodied/jax/outs.py"], revision)
    official_symexp = _load_official_function(
        sources["embodied/jax/nets.py"],
        "symexp",
        {"jnp": jax.numpy},
        revision,
    )
    official_symlog = _load_official_function(
        sources["embodied/jax/nets.py"],
        "symlog",
        {"jnp": jax.numpy},
        revision,
    )
    bounded_normal = _load_official_method(
        sources["embodied/jax/heads.py"],
        "Head",
        "bounded_normal",
        {
            "f32": jax.numpy.float32,
            "jax": jax,
            "jnp": jax.numpy,
            "nets": SimpleNamespace(Linear=object, symexp=official_symexp),
            "outs": official_outs,
        },
        revision,
    )
    symexp_twohot = _load_official_method(
        sources["embodied/jax/heads.py"],
        "Head",
        "symexp_twohot",
        {
            "f32": jax.numpy.float32,
            "jax": jax,
            "jnp": jax.numpy,
            "nets": SimpleNamespace(Linear=object, symexp=official_symexp),
            "outs": official_outs,
        },
        revision,
    )
    arrays = _official_distribution_arrays(
        official_outs,
        official_symlog,
        official_symexp,
        bounded_normal,
        symexp_twohot,
        int(request["seed"]),
    )
    return {
        "arrays": {
            name: {
                "dtype": array.dtype.name,
                "values": array.tolist(),
            }
            for name, array in sorted(arrays.items())
        },
        "worker_pid": os.getpid(),
    }


def _load_official_outs(source: bytes, revision: str) -> ModuleType:
    module = ModuleType("dreamerv3_official_outs")
    code = compile(
        source,
        f"{revision}:embodied/jax/outs.py",
        "exec",
    )
    exec(code, module.__dict__)
    runtime_jax = module.jax

    module.jax = SimpleNamespace(
        nn=runtime_jax.nn,
        random=_OfficialRandomFacade(runtime_jax.random),
        scipy=runtime_jax.scipy,
    )
    return module


class _OfficialRandomFacade:
    def __init__(self, runtime_random: Any) -> None:
        self._runtime_random = runtime_random

    def bernoulli(
        self,
        seed: jax.Array,
        probability: jax.Array,
        axis: int,
        shape: tuple[int, ...],
    ) -> jax.Array:
        if axis != -1:
            raise ValueError("official Binary sample axis must be -1")
        return self._runtime_random.bernoulli(seed, probability, shape)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime_random, name)


class _SuppliedCategoricalNoise:
    def __init__(
        self,
        *,
        expected_logits: np.ndarray | jax.Array,
        expected_output_shape: Sequence[int],
        noise: np.ndarray | jax.Array,
    ) -> None:
        self.expected_logits = np.asarray(jax.device_get(expected_logits))
        self.expected_output_shape = tuple(expected_output_shape)
        supplied_noise = np.asarray(jax.device_get(noise))
        expected_noise_shape = self.expected_output_shape + (
            self.expected_logits.shape[-1],
        )
        if tuple(supplied_noise.shape) != expected_noise_shape:
            raise ValueError(
                "supplied categorical noise shape does not match the expected "
                f"primitive shape: {supplied_noise.shape} != {expected_noise_shape}"
            )
        if supplied_noise.dtype != self.expected_logits.dtype:
            raise ValueError(
                "supplied categorical noise dtype does not match logits: "
                f"{supplied_noise.dtype} != {self.expected_logits.dtype}"
            )
        self.noise = jax.numpy.asarray(supplied_noise)
        self.calls = 0

    def __call__(
        self,
        seed: jax.Array,
        logits: jax.Array,
        axis: int = -1,
        shape: Sequence[int] | None = None,
    ) -> jax.Array:
        del seed
        if axis != -1:
            raise ValueError("official Categorical sample axis must be -1")
        if shape is None or tuple(shape) != self.expected_output_shape:
            raise ValueError(
                "categorical requested output shape does not match supplied case: "
                f"{shape} != {self.expected_output_shape}"
            )
        if tuple(logits.shape) != self.expected_logits.shape:
            raise ValueError(
                "categorical logits shape does not match supplied case: "
                f"{logits.shape} != {self.expected_logits.shape}"
            )
        batch_shape = tuple(logits.shape[:-1])
        if (
            batch_shape
            and self.expected_output_shape[-len(batch_shape) :] != batch_shape
        ):
            raise ValueError(
                "categorical output shape does not end in the logits batch shape"
            )
        expected_noise_shape = self.expected_output_shape + (logits.shape[-1],)
        if tuple(self.noise.shape) != expected_noise_shape:
            raise ValueError(
                "supplied categorical noise shape changed after construction"
            )
        self._assert_logits(logits)
        shape_prefix = len(self.expected_output_shape) - len(batch_shape)
        expanded_logits = jax.lax.expand_dims(logits, tuple(range(shape_prefix)))
        self.calls += 1
        return jax.numpy.argmax(self.noise + expanded_logits, axis=axis)

    def _assert_logits(self, logits: jax.Array) -> None:
        expected = self.expected_logits

        def assert_equal(value: np.ndarray) -> None:
            if not np.array_equal(value, expected):
                raise ValueError(
                    "categorical logits do not match the authoritative supplied-noise "
                    "case"
                )

        if isinstance(logits, jax.core.Tracer):
            jax.debug.callback(assert_equal, logits)
        else:
            assert_equal(np.asarray(jax.device_get(logits)))


@contextmanager
def _supplied_categorical_noise_scope(
    random_namespace: Any,
    *,
    expected_logits: np.ndarray | jax.Array,
    expected_output_shape: Sequence[int],
    noise: np.ndarray | jax.Array,
) -> Iterator[None]:
    injected = _SuppliedCategoricalNoise(
        expected_logits=expected_logits,
        expected_output_shape=expected_output_shape,
        noise=noise,
    )
    missing = object()
    original = vars(random_namespace).get("categorical", missing)
    setattr(random_namespace, "categorical", injected)
    try:
        yield
        if injected.calls != 1:
            raise ValueError(
                "supplied categorical noise case must invoke the primitive exactly once"
            )
    finally:
        if original is missing:
            delattr(random_namespace, "categorical")
        else:
            setattr(random_namespace, "categorical", original)


def _load_official_function(
    source: bytes,
    function_name: str,
    namespace: Mapping[str, Any],
    revision: str,
) -> Any:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1:
        raise ValueError(f"official function is not unique: {function_name}")
    module = ast.Module(body=matches, type_ignores=[])
    ast.fix_missing_locations(module)
    globals_dict = dict(namespace)
    exec(
        compile(module, f"{revision}:official:{function_name}", "exec"),
        globals_dict,
    )
    return globals_dict[function_name]


def _load_official_method(
    source: bytes,
    class_name: str,
    method_name: str,
    namespace: Mapping[str, Any],
    revision: str,
) -> Any:
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise ValueError(f"official class is not unique: {class_name}")
    matches = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    ]
    if len(matches) != 1:
        raise ValueError(f"official method is not unique: {class_name}.{method_name}")
    module = ast.Module(body=matches, type_ignores=[])
    ast.fix_missing_locations(module)
    globals_dict = dict(namespace)
    exec(
        compile(
            module,
            f"{revision}:official:{class_name}.{method_name}",
            "exec",
        ),
        globals_dict,
    )
    return globals_dict[method_name]


class _OfficialHeadStub:
    def __init__(
        self,
        values: Mapping[str, jax.Array],
        *,
        shape: tuple[int, ...],
        bins: int = 255,
        minstd: float = 0.1,
        maxstd: float = 1.0,
    ) -> None:
        self.values = values
        self.space = SimpleNamespace(discrete=False, shape=shape)
        self.bins = bins
        self.minstd = minstd
        self.maxstd = maxstd
        self.kw: dict[str, Any] = {}

    def sub(self, name: str, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        value = self.values[name]
        return lambda inputs: value


def _official_distribution_arrays(
    outs: ModuleType,
    official_symlog: Any,
    official_symexp: Any,
    bounded_normal: Any,
    symexp_twohot: Any,
    seed: int,
) -> dict[str, np.ndarray]:
    jnp = jax.numpy
    keys = jax.random.split(jax.random.PRNGKey(seed), 8)
    arrays: dict[str, jax.Array] = {}

    scalar_input = jnp.asarray(
        [-1e20, -100.0, -1.0, -0.0, 0.0, 1.0, 100.0, 1e20],
        jnp.float32,
    )
    scalar_symlog = official_symlog(scalar_input)
    scalar_roundtrip = official_symexp(scalar_symlog)
    arrays.update(
        {
            "scalar.input": scalar_input,
            "scalar.roundtrip": scalar_roundtrip,
            "scalar.symlog": scalar_symlog,
        }
    )

    mse_mean = jnp.asarray([[-2.0, -0.5, 0.0], [0.5, 2.0, 5.0]], jnp.float32)
    mse_target = jnp.asarray([[-1.0, 0.5, 2.0], [0.0, -2.0, 3.0]], jnp.float32)
    mse = outs.MSE(mse_mean)
    arrays.update(
        {
            "mse.grad_mean": jax.grad(
                lambda value: outs.MSE(value).loss(mse_target).sum()
            )(mse_mean),
            "mse.grad_target": jax.grad(lambda value: mse.loss(value).sum())(
                mse_target
            ),
            "mse.loss": mse.loss(mse_target),
            "mse.mean": mse_mean,
            "mse.pred": mse.pred(),
            "mse.seed": keys[0],
            "mse.target": mse_target,
        }
    )

    normal_mean = jnp.asarray([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]], jnp.float32)
    normal_stddev = jnp.asarray([[0.2, 0.5, 1.0], [1.5, 2.0, 0.1]], jnp.float32)
    normal_event = jnp.asarray([[-3.0, 0.25, 1.5], [0.25, 3.0, -1.0]], jnp.float32)
    normal_other_mean = jnp.asarray([[0.0, -1.0, 2.0], [1.0, 0.0, -0.5]], jnp.float32)
    normal_other_stddev = jnp.asarray([[1.0, 0.3, 2.0], [0.5, 1.5, 0.25]], jnp.float32)
    normal = outs.Normal(normal_mean, normal_stddev)
    normal_other = outs.Normal(normal_other_mean, normal_other_stddev)
    arrays.update(
        {
            "normal.entropy": normal.entropy(),
            "normal.event": normal_event,
            "normal.grad_target": jax.grad(lambda value: normal.loss(value).sum())(
                normal_event
            ),
            "normal.kl": normal.kl(normal_other),
            "normal.logp": normal.logp(normal_event),
            "normal.loss": normal.loss(normal_event),
            "normal.mean": normal_mean,
            "normal.other_mean": normal_other_mean,
            "normal.other_stddev": normal_other_stddev,
            "normal.pred": normal.pred(),
            "normal.prob": normal.prob(normal_event),
            "normal.sample": normal.sample(keys[1], shape=(2,)),
            "normal.seed": keys[1],
            "normal.stddev": normal_stddev,
        }
    )

    bounded_raw_mean = jnp.asarray([[2.0, -2.0, 0.1], [3.0, -3.0, 0.0]], jnp.float32)
    bounded_raw_stddev = jnp.asarray(
        [[-5.0, -2.0, 0.0], [2.0, 5.0, -10.0]], jnp.float32
    )
    bounded_event = jnp.asarray([[1.2, -1.4, 0.0], [2.0, -2.0, 0.5]], jnp.float32)

    def make_bounded(raw_mean: jax.Array, raw_stddev: jax.Array) -> Any:
        stub = _OfficialHeadStub(
            {"mean": raw_mean, "stddev": raw_stddev},
            shape=(3,),
        )
        return bounded_normal(stub, jnp.zeros((2, 1), jnp.float32))

    bounded = make_bounded(bounded_raw_mean, bounded_raw_stddev)
    arrays.update(
        {
            "bounded.entropy": bounded.entropy(),
            "bounded.event": bounded_event,
            "bounded.grad_raw_mean": jax.grad(
                lambda value: (
                    make_bounded(value, bounded_raw_stddev).loss(bounded_event).sum()
                )
            )(bounded_raw_mean),
            "bounded.grad_raw_stddev": jax.grad(
                lambda value: (
                    make_bounded(bounded_raw_mean, value).loss(bounded_event).sum()
                )
            )(bounded_raw_stddev),
            "bounded.logp": bounded.logp(bounded_event),
            "bounded.loss": bounded.loss(bounded_event),
            "bounded.mean": bounded.mean,
            "bounded.pred": bounded.pred(),
            "bounded.prob": bounded.prob(bounded_event),
            "bounded.raw_mean": bounded_raw_mean,
            "bounded.raw_stddev": bounded_raw_stddev,
            "bounded.sample": bounded.sample(keys[2]),
            "bounded.seed": keys[2],
            "bounded.stddev": bounded.stddev,
        }
    )

    binary_logit = jnp.asarray([-100.0, -10.0, -0.0, 0.0, 10.0, 100.0], jnp.float32)
    binary_event = jnp.asarray([1.0, 0.0, 0.0, 1.0, 1.0, 0.0], jnp.float32)
    binary = outs.Binary(binary_logit)
    arrays.update(
        {
            "binary.event": binary_event,
            "binary.logit": binary_logit,
            "binary.logp": binary.logp(binary_event),
            "binary.loss": binary.loss(binary_event),
            "binary.pred": binary.pred(),
            "binary.prob": binary.prob(binary_event),
            "binary.sample": binary.sample(keys[3], shape=(3,)),
            "binary.seed": keys[3],
        }
    )

    categorical_logits = jnp.asarray(
        [[-100.0, 0.0, 1.0, 2.0], [10.0, -10.0, 0.0, -5.0]],
        jnp.float32,
    )
    categorical_other_logits = jnp.asarray(
        [[2.0, 1.0, 0.0, -2.0], [-3.0, 4.0, 0.5, 1.0]],
        jnp.float32,
    )
    categorical_event = jnp.asarray([3, 1], jnp.int32)
    categorical = outs.Categorical(categorical_logits, 0.01)
    categorical_other = outs.Categorical(categorical_other_logits, 0.01)
    categorical_supplied_noise = jnp.asarray(
        [
            [[8.0, 0.0, 0.0, 0.0], [0.0, 7.5, 0.0, 0.0]],
            [[0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 7.0, 0.0]],
            [[0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 7.0]],
        ],
        jnp.float32,
    )
    with _supplied_categorical_noise_scope(
        outs.jax.random,
        expected_logits=categorical.logits,
        expected_output_shape=(3, 2),
        noise=categorical_supplied_noise,
    ):
        categorical_supplied_sample = categorical.sample(keys[4], shape=(3,))
    arrays.update(
        {
            "categorical.effective_logits": categorical.logits,
            "categorical.entropy": categorical.entropy(),
            "categorical.event": categorical_event,
            "categorical.kl": categorical.kl(categorical_other),
            "categorical.logits": categorical_logits,
            "categorical.logp": categorical.logp(categorical_event),
            "categorical.loss": categorical.loss(categorical_event),
            "categorical.other_logits": categorical_other_logits,
            "categorical.pred": categorical.pred(),
            "categorical.prob": categorical.prob(categorical_event),
            "categorical.probs": jax.nn.softmax(categorical.logits),
            "categorical.sample": categorical.sample(keys[4], shape=(3,)),
            "categorical.seed": keys[4],
            "categorical.supplied_noise": categorical_supplied_noise,
            "categorical.supplied_sample": categorical_supplied_sample,
        }
    )

    onehot_logits = jnp.asarray(
        [[-3.0, 0.0, 2.0, 1.0], [5.0, -2.0, 0.5, -1.0]],
        jnp.float32,
    )
    onehot_other_logits = jnp.asarray(
        [[1.0, 2.0, -1.0, 0.0], [-2.0, 0.0, 1.0, 3.0]],
        jnp.float32,
    )
    onehot_event = jax.nn.one_hot(jnp.asarray([2, 0]), 4, dtype=jnp.float32)
    onehot_weights = jnp.asarray(
        [[-1.0, 0.5, 2.0, -0.25], [0.1, -0.4, 1.5, 2.0]],
        jnp.float32,
    )
    onehot = outs.OneHot(onehot_logits, 0.01)
    onehot_other = outs.OneHot(onehot_other_logits, 0.01)
    onehot_supplied_noise = jnp.asarray(
        [
            [[5.0, 0.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0]],
            [[0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0]],
        ],
        jnp.float32,
    )
    with _supplied_categorical_noise_scope(
        outs.jax.random,
        expected_logits=onehot.dist.logits,
        expected_output_shape=(2, 2),
        noise=onehot_supplied_noise,
    ):
        onehot_supplied_sample = onehot.sample(keys[5], shape=(2,))

    def supplied_onehot_objective(value: jax.Array) -> jax.Array:
        candidate = outs.OneHot(value, 0.01)
        with _supplied_categorical_noise_scope(
            outs.jax.random,
            expected_logits=onehot.dist.logits,
            expected_output_shape=(2, 2),
            noise=onehot_supplied_noise,
        ):
            sample = candidate.sample(keys[5], shape=(2,))
        return (sample * onehot_weights).sum()

    arrays.update(
        {
            "onehot.effective_logits": onehot.dist.logits,
            "onehot.entropy": onehot.entropy(),
            "onehot.event": onehot_event,
            "onehot.grad_target": jax.grad(lambda value: onehot.loss(value).sum())(
                onehot_event
            ),
            "onehot.kl": onehot.kl(onehot_other),
            "onehot.logits": onehot_logits,
            "onehot.logp": onehot.logp(onehot_event),
            "onehot.loss": onehot.loss(onehot_event),
            "onehot.other_logits": onehot_other_logits,
            "onehot.pred": onehot.pred(),
            "onehot.prob": onehot.prob(onehot_event),
            "onehot.sample": onehot.sample(keys[5]),
            "onehot.sample_grad": jax.grad(
                lambda value: (
                    outs.OneHot(value, 0.01).sample(keys[5]) * onehot_weights
                ).sum()
            )(onehot_logits),
            "onehot.sample_weights": onehot_weights,
            "onehot.seed": keys[5],
            "onehot.supplied_noise": onehot_supplied_noise,
            "onehot.supplied_sample": onehot_supplied_sample,
            "onehot.supplied_sample_grad": jax.grad(supplied_onehot_objective)(
                onehot_logits
            ),
        }
    )

    odd_logits = jnp.stack(
        [
            jnp.zeros((255,), jnp.float32),
            jnp.linspace(-2.0, 2.0, 255, dtype=jnp.float32),
            jnp.linspace(2.0, -2.0, 255, dtype=jnp.float32),
            jnp.sin(jnp.linspace(-3.0, 3.0, 255, dtype=jnp.float32)),
            jnp.zeros((255,), jnp.float32).at[-1].set(5.0),
        ]
    )
    odd_target = jnp.asarray([-1e20, -1.0, 0.25, 10.0, 1e20], jnp.float32)

    def make_twohot(logits: jax.Array, bins: int) -> Any:
        stub = _OfficialHeadStub({"logits": logits}, shape=(), bins=bins)
        return symexp_twohot(stub, jnp.zeros((*logits.shape[:-1], 1), jnp.float32))

    odd = make_twohot(odd_logits, 255)
    arrays.update(
        {
            "twohot_odd.bins": odd.bins,
            "twohot_odd.grad_logits": jax.grad(
                lambda value: make_twohot(value, 255).loss(odd_target).sum()
            )(odd_logits),
            "twohot_odd.grad_target": jax.grad(lambda value: odd.loss(value).sum())(
                odd_target
            ),
            "twohot_odd.logits": odd_logits,
            "twohot_odd.loss": odd.loss(odd_target),
            "twohot_odd.pred": odd.pred(),
            "twohot_odd.target": odd_target,
        }
    )

    even_logits = jnp.stack(
        [
            jnp.zeros((8,), jnp.float32),
            jnp.linspace(-2.0, 2.0, 8, dtype=jnp.float32),
            jnp.linspace(2.0, -2.0, 8, dtype=jnp.float32),
        ]
    )
    even_target = jnp.asarray([-1e20, 0.2, 1e20], jnp.float32)
    even = make_twohot(even_logits, 8)
    arrays.update(
        {
            "twohot_even.bins": even.bins,
            "twohot_even.logits": even_logits,
            "twohot_even.loss": even.loss(even_target),
            "twohot_even.pred": even.pred(),
            "twohot_even.seed": keys[6],
            "twohot_even.target": even_target,
        }
    )

    aggregate_mean = jnp.linspace(-1.0, 1.0, 12, dtype=jnp.float32).reshape(2, 2, 3)
    aggregate_stddev = jnp.linspace(0.2, 1.3, 12, dtype=jnp.float32).reshape(2, 2, 3)
    aggregate_event = jnp.linspace(-2.0, 2.0, 12, dtype=jnp.float32).reshape(2, 2, 3)
    aggregate_other_mean = jnp.linspace(1.0, -1.0, 12, dtype=jnp.float32).reshape(
        2, 2, 3
    )
    aggregate_other_stddev = jnp.linspace(1.5, 0.4, 12, dtype=jnp.float32).reshape(
        2, 2, 3
    )
    aggregate = outs.Agg(outs.Normal(aggregate_mean, aggregate_stddev), 2, jnp.mean)
    aggregate_other = outs.Agg(
        outs.Normal(aggregate_other_mean, aggregate_other_stddev),
        2,
        jnp.mean,
    )
    arrays.update(
        {
            "aggregate.entropy": aggregate.entropy(),
            "aggregate.event": aggregate_event,
            "aggregate.kl": aggregate.kl(aggregate_other),
            "aggregate.logp": aggregate.logp(aggregate_event),
            "aggregate.loss": aggregate.loss(aggregate_event),
            "aggregate.mean": aggregate_mean,
            "aggregate.other_mean": aggregate_other_mean,
            "aggregate.other_stddev": aggregate_other_stddev,
            "aggregate.pred": aggregate.pred(),
            "aggregate.prob": aggregate.prob(aggregate_event),
            "aggregate.sample": aggregate.sample(keys[7]),
            "aggregate.seed": keys[7],
            "aggregate.stddev": aggregate_stddev,
        }
    )

    aggregate_mse_mean = jnp.linspace(-2.0, 2.0, 12, dtype=jnp.float32).reshape(2, 2, 3)
    aggregate_mse_target = jnp.linspace(1.0, -1.0, 12, dtype=jnp.float32).reshape(
        2, 2, 3
    )
    aggregate_mse = outs.Agg(outs.MSE(aggregate_mse_mean), 2)
    arrays.update(
        {
            "aggregate_mse.grad_target": jax.grad(
                lambda value: aggregate_mse.loss(value).sum()
            )(aggregate_mse_target),
            "aggregate_mse.loss": aggregate_mse.loss(aggregate_mse_target),
            "aggregate_mse.mean": aggregate_mse_mean,
            "aggregate_mse.target": aggregate_mse_target,
        }
    )
    return {
        name: np.asarray(jax.device_get(value))
        for name, value in sorted(arrays.items())
    }


def _parameter_path(path: str | Sequence[str]) -> str:
    if isinstance(path, str):
        normalized = path
    else:
        normalized = "/".join(path)
    if not normalized or normalized.startswith("/") or normalized.endswith("/"):
        raise ValueError(f"invalid parameter path: {normalized!r}")
    return normalized


def _transform_parameter(array: np.ndarray, mapping: ParameterMapping) -> np.ndarray:
    if mapping.transform == "identity":
        return array
    if mapping.transform == "transpose":
        return array.T
    assert mapping.reshape is not None
    return array.reshape(mapping.reshape)


def _canonical_generator_request(request: Mapping[str, Any] | str) -> str:
    payload = json.loads(request) if isinstance(request, str) else dict(request)
    if not isinstance(payload, dict):
        raise ValueError("oracle generator request must be a JSON object")
    if any(not isinstance(key, str) for key in payload):
        raise ValueError("oracle generator request keys must be strings")
    return _canonical_json(payload).decode()


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git_show(checkout: Path, revision: str, path: str) -> bytes:
    _git_object_exists(checkout, revision)
    completed = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _git_object_exists(checkout: Path, revision: str) -> None:
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            cwd=checkout,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(f"official checkout is missing commit {revision}") from error


def _write_deterministic_npz(
    destination: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_STORED
    ) as archive:
        for name, array in sorted(arrays.items()):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    temporary.replace(destination)


def _main(argv: Sequence[str]) -> int:
    workers = {
        "_config_worker": _config_worker,
        "_distributions_worker": _distributions_worker,
    }
    if len(argv) != 1 or argv[0] not in workers:
        raise SystemExit("oracle.py is an internal fixture worker")
    request = json.loads(sys.stdin.read())
    sys.stdout.write(json.dumps(workers[argv[0]](request), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess boundary.
    raise SystemExit(_main(sys.argv[1:]))


__all__ = [
    "CONFIG_SOURCE_SPEC",
    "DISTRIBUTIONS_SOURCE_SPEC",
    "ORACLE_SCHEMA_VERSION",
    "OracleHarness",
    "OracleManifest",
    "OracleSourceSpec",
    "PAPER_REVISION",
    "ParameterMapping",
    "ParameterTranslator",
    "TensorSpec",
    "UPSTREAM_CURRENT_REVISION",
    "official_revision",
    "oracle_source_spec",
    "profile_overrides",
    "register_oracle_source_spec",
]
