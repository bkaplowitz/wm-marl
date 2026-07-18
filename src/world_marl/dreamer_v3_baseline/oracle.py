from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import tempfile
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import core as jax_core

from .config import DreamerProfile, ObservationMode


Array = npt.NDArray[np.generic]
ORACLE_SCHEMA_VERSION = 3
PAPER_REVISION = "bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01"
UPSTREAM_CURRENT_REVISION = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"
_PROFILE_REVISIONS = MappingProxyType(
    {
        DreamerProfile.PAPER: PAPER_REVISION,
        DreamerProfile.UPSTREAM_CURRENT: UPSTREAM_CURRENT_REVISION,
    }
)
_CASE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _canonical_generator_request(value: str | Mapping[str, object]) -> str:
    if type(value) is str:
        payload = json.loads(value)
    elif type(value) is dict:
        payload = value
    else:
        raise TypeError("oracle generator request must be an exact string or dict")
    if type(payload) is not dict or any(type(key) is not str for key in payload):
        raise ValueError("oracle generator request must be a JSON object")
    return _canonical_json(payload).decode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _git_show(checkout: str | Path, revision: str, path: str) -> bytes:
    root = Path(checkout)
    if not root.is_dir():
        raise ValueError(f"official checkout does not exist: {root}")
    if type(revision) is not str or not _COMMIT_PATTERN.fullmatch(revision):
        raise ValueError("official revision must be a full Git object id")
    if type(path) is not str or not path or path.startswith("/"):
        raise ValueError(f"invalid official source path: {path!r}")
    if ".." in Path(path).parts:
        raise ValueError(f"invalid official source path: {path!r}")
    try:
        return subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{path}"],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors="replace").strip()
        raise ValueError(
            f"cannot read official source {revision}:{path}: {detail}"
        ) from error


_DISTRIBUTION_HASHES = MappingProxyType(
    {
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
)

_NETWORK_HASHES = MappingProxyType(
    {
        "dreamerv3/rssm.py": (
            "d6d50166914e94fb8bd17a5d5dbda9d42cdd37b85819bb1e9fff3a64d4ad2eb6"
        ),
        "embodied/jax/heads.py": _DISTRIBUTION_HASHES["embodied/jax/heads.py"],
        "embodied/jax/nets.py": _DISTRIBUTION_HASHES["embodied/jax/nets.py"],
    }
)

_RSSM_HASHES = MappingProxyType(
    {
        "dreamerv3/agent.py": (
            "adce8e4274bc098c218bf9a20fd3327545f0ad7d850b5fe328597382e91b5269"
        ),
        "dreamerv3/configs.yaml": (
            "9dff9c7062e3e33951cb54c6dd4b598aaf7e56e18e2cff39c812eaa797bcfcfc"
        ),
        "dreamerv3/rssm.py": _NETWORK_HASHES["dreamerv3/rssm.py"],
        "embodied/jax/heads.py": _DISTRIBUTION_HASHES["embodied/jax/heads.py"],
        "embodied/jax/nets.py": _DISTRIBUTION_HASHES["embodied/jax/nets.py"],
        "embodied/jax/outs.py": _DISTRIBUTION_HASHES["embodied/jax/outs.py"],
    }
)

REPLAY_SOURCE_HASHES = MappingProxyType(
    {
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
)


_SOURCE_HASHES = MappingProxyType(
    {
        name: MappingProxyType(
            {
                revision: hashes
                for revision in (PAPER_REVISION, UPSTREAM_CURRENT_REVISION)
            }
        )
        for name, hashes in {
            "distributions": _DISTRIBUTION_HASHES,
            "networks": _NETWORK_HASHES,
            "replay": REPLAY_SOURCE_HASHES,
            "rssm": _RSSM_HASHES,
        }.items()
    }
)
_SOURCE_DTYPES = MappingProxyType(
    {
        "distributions": ("bfloat16",),
        "networks": ("bfloat16", "float32"),
        "replay": ("float32",),
        "rssm": ("bfloat16", "float32"),
    }
)


def _source_hashes_for(name: str, revision: str) -> Mapping[str, str]:
    if type(name) is not str:
        raise TypeError("fixture source name must be an exact string")
    if type(revision) is not str:
        raise TypeError("fixture source revision must be an exact string")
    try:
        return _SOURCE_HASHES[name][revision]
    except KeyError as error:
        raise ValueError(
            f"unsupported fixture source coordinate: {name}@{revision}"
        ) from error


def _source_allows_dtype(name: str, dtype: str) -> bool:
    if type(name) is not str or type(dtype) is not str:
        raise TypeError("fixture source name and dtype must be exact strings")
    try:
        return dtype in _SOURCE_DTYPES[name]
    except KeyError as error:
        raise ValueError(f"unsupported fixture source: {name}") from error


@dataclass(frozen=True)
class _FixtureSourceName:
    name: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in _SOURCE_HASHES:
            raise ValueError(f"unsupported fixture source: {self.name!r}")


DISTRIBUTIONS_SOURCE_SPEC = _FixtureSourceName("distributions")
NETWORKS_SOURCE_SPEC = _FixtureSourceName("networks")
REPLAY_SOURCE_SPEC = _FixtureSourceName("replay")
RSSM_SOURCE_SPEC = _FixtureSourceName("rssm")


_FIXTURE_CASES = MappingProxyType(
    {
        "paper-proprio-distributions": (
            DreamerProfile.PAPER,
            ObservationMode.PROPRIO,
            "distributions",
            "distributions",
            "bfloat16",
            0,
        ),
        "upstream-current-proprio-distributions": (
            DreamerProfile.UPSTREAM_CURRENT,
            ObservationMode.PROPRIO,
            "distributions",
            "distributions",
            "bfloat16",
            0,
        ),
        "paper-proprio-replay": (
            DreamerProfile.PAPER,
            ObservationMode.PROPRIO,
            "replay",
            "replay",
            "float32",
            7,
        ),
        "upstream-current-proprio-replay": (
            DreamerProfile.UPSTREAM_CURRENT,
            ObservationMode.PROPRIO,
            "replay",
            "replay",
            "float32",
            7,
        ),
        "paper-proprio-rssm": (
            DreamerProfile.PAPER,
            ObservationMode.PROPRIO,
            "rssm",
            "rssm",
            "bfloat16",
            0,
        ),
        "paper-proprio-rssm-float32": (
            DreamerProfile.PAPER,
            ObservationMode.PROPRIO,
            "rssm-float32",
            "rssm",
            "float32",
            0,
        ),
        "upstream-current-proprio-rssm": (
            DreamerProfile.UPSTREAM_CURRENT,
            ObservationMode.PROPRIO,
            "rssm",
            "rssm",
            "bfloat16",
            0,
        ),
        "upstream-current-proprio-rssm-float32": (
            DreamerProfile.UPSTREAM_CURRENT,
            ObservationMode.PROPRIO,
            "rssm-float32",
            "rssm",
            "float32",
            0,
        ),
        "paper-vision-networks": (
            DreamerProfile.PAPER,
            ObservationMode.VISION,
            "networks",
            "networks",
            "bfloat16",
            0,
        ),
        "paper-vision-networks-float32": (
            DreamerProfile.PAPER,
            ObservationMode.VISION,
            "networks-float32",
            "networks",
            "float32",
            0,
        ),
        "upstream-current-vision-networks": (
            DreamerProfile.UPSTREAM_CURRENT,
            ObservationMode.VISION,
            "networks",
            "networks",
            "bfloat16",
            0,
        ),
        "upstream-current-vision-networks-float32": (
            DreamerProfile.UPSTREAM_CURRENT,
            ObservationMode.VISION,
            "networks-float32",
            "networks",
            "float32",
            0,
        ),
    }
)


def _fixture_coordinates(
    stem: str,
) -> tuple[DreamerProfile, ObservationMode, str, str, str, int]:
    if type(stem) is not str or not _CASE_PATTERN.fullmatch(stem):
        raise ValueError(f"unsupported fixture stem: {stem!r}")
    try:
        return _FIXTURE_CASES[stem]
    except KeyError as error:
        raise ValueError(f"unsupported fixture stem: {stem!r}") from error


def _canonical_fixture_request(stem: str) -> str:
    profile, mode, case, source, dtype, seed = _fixture_coordinates(stem)
    return _canonical_generator_request(
        {
            "case_name": case,
            "dtype": dtype,
            "fixture_file": f"{stem}.npz",
            "fixture_stem": stem,
            "observation_mode": mode.value,
            "profile": profile.value,
            "schema_version": ORACLE_SCHEMA_VERSION,
            "seed": seed,
            "source_revision": _PROFILE_REVISIONS[profile],
            "source_spec": source,
        }
    )


def _canonical_fixture_command(stem: str) -> tuple[str, ...]:
    profile, mode, _case, _source, _dtype, _seed = _fixture_coordinates(stem)
    return (
        "python",
        "-m",
        "world_marl.dreamer_v3_baseline.fixture_generator",
        "refresh-manifest",
        "--profile",
        profile.value,
        "--observation-mode",
        mode.value,
        "--reference-checkout",
        "<reference-checkout>",
        "--source-revision",
        _PROFILE_REVISIONS[profile],
        "--output-dir",
        "<fixture-dir>",
        "--fixture-stem",
        stem,
    )


def _require_dict(value: object, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact dict")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{name} keys must be exact strings")
    return value


def _require_list(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an exact list")
    return value


def _require_str(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    return value


def _require_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    return value


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if type(self.shape) is not tuple:
            raise TypeError("oracle tensor shape must be an exact tuple")
        if any(type(size) is not int or size < 0 for size in self.shape):
            raise ValueError("oracle tensor shape must contain nonnegative integers")
        if type(self.dtype) is not str or not self.dtype:
            raise TypeError("oracle tensor dtype must be an exact nonempty string")
        try:
            np.dtype(self.dtype)
        except TypeError as error:
            raise ValueError(f"invalid oracle tensor dtype: {self.dtype}") from error

    @classmethod
    def from_dict(cls, value: object) -> TensorSpec:
        record = _require_dict(value, "oracle tensor spec")
        if set(record) != {"dtype", "shape"}:
            raise ValueError("oracle tensor spec has incorrect fields")
        shape = _require_list(record["shape"], "oracle tensor shape")
        sizes = tuple(_require_int(size, "oracle tensor shape item") for size in shape)
        return cls(sizes, _require_str(record["dtype"], "oracle tensor dtype"))

    def to_dict(self) -> dict[str, object]:
        return {"dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True)
class OracleManifest:
    schema_version: int
    case_name: str
    profile: DreamerProfile
    observation_mode: ObservationMode
    official_commit: str
    source_spec: str
    official_file_hashes: Mapping[str, str]
    dtype: str
    seed: int
    tensor_schema: Mapping[str, TensorSpec]
    generator_command: tuple[str, ...]
    generator_request: str
    fixture_file: str
    fixture_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise TypeError("oracle schema version must be an exact integer")
        for name, value in (
            ("case name", self.case_name),
            ("official commit", self.official_commit),
            ("source spec", self.source_spec),
            ("dtype", self.dtype),
            ("generator request", self.generator_request),
            ("fixture file", self.fixture_file),
            ("fixture hash", self.fixture_sha256),
        ):
            if type(value) is not str:
                raise TypeError(f"oracle {name} must be an exact string")
        if type(self.profile) is not DreamerProfile:
            raise TypeError("oracle profile must be DreamerProfile")
        if type(self.observation_mode) is not ObservationMode:
            raise TypeError("oracle observation mode must be ObservationMode")
        if type(self.seed) is not int:
            raise TypeError("oracle seed must be an exact integer")
        hashes = dict(self.official_file_hashes)
        if any(
            type(key) is not str or type(value) is not str
            for key, value in hashes.items()
        ):
            raise TypeError("oracle source hashes must map exact strings")
        schema = dict(self.tensor_schema)
        if any(
            type(key) is not str or type(value) is not TensorSpec
            for key, value in schema.items()
        ):
            raise TypeError("oracle tensor schema must map strings to TensorSpec")
        if type(self.generator_command) is not tuple or any(
            type(part) is not str for part in self.generator_command
        ):
            raise TypeError("oracle generator command must be an exact string tuple")
        if (
            _canonical_generator_request(self.generator_request)
            != self.generator_request
        ):
            raise ValueError("oracle generator request must be canonical JSON")
        object.__setattr__(
            self, "official_file_hashes", MappingProxyType(dict(sorted(hashes.items())))
        )
        object.__setattr__(
            self, "tensor_schema", MappingProxyType(dict(sorted(schema.items())))
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OracleManifest:
        record = _require_dict(payload, "oracle manifest")
        required = {
            "case_name",
            "dtype",
            "fixture_file",
            "fixture_sha256",
            "generator_command",
            "generator_request",
            "observation_mode",
            "official_commit",
            "official_file_hashes",
            "profile",
            "schema_version",
            "seed",
            "source_spec",
            "tensor_schema",
        }
        if set(record) != required:
            missing = required - set(record)
            extra = set(record) - required
            if missing:
                raise ValueError(
                    "oracle manifest missing fields: " + ", ".join(sorted(missing))
                )
            raise ValueError(
                "oracle manifest has unexpected fields: " + ", ".join(sorted(extra))
            )
        hashes_record = _require_dict(
            record["official_file_hashes"], "oracle source hashes"
        )
        hashes = {
            key: _require_str(value, f"oracle source digest {key}")
            for key, value in hashes_record.items()
        }
        schema_record = _require_dict(record["tensor_schema"], "oracle tensor schema")
        schema = {
            key: TensorSpec.from_dict(value) for key, value in schema_record.items()
        }
        command_values = _require_list(
            record["generator_command"], "oracle generator command"
        )
        command = tuple(
            _require_str(value, "oracle generator command item")
            for value in command_values
        )
        return cls(
            schema_version=_require_int(
                record["schema_version"], "oracle schema version"
            ),
            case_name=_require_str(record["case_name"], "oracle case name"),
            profile=DreamerProfile(_require_str(record["profile"], "oracle profile")),
            observation_mode=ObservationMode(
                _require_str(record["observation_mode"], "oracle observation mode")
            ),
            official_commit=_require_str(record["official_commit"], "official commit"),
            source_spec=_require_str(record["source_spec"], "oracle source spec"),
            official_file_hashes=hashes,
            dtype=_require_str(record["dtype"], "oracle dtype"),
            seed=_require_int(record["seed"], "oracle seed"),
            tensor_schema=schema,
            generator_command=command,
            generator_request=_require_str(
                record["generator_request"], "oracle generator request"
            ),
            fixture_file=_require_str(record["fixture_file"], "oracle fixture file"),
            fixture_sha256=_require_str(
                record["fixture_sha256"], "oracle fixture hash"
            ),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        official_checkout: str | Path | None = None,
        fixture_path: str | Path | None = None,
    ) -> OracleManifest:
        source = Path(path)
        source_bytes = source.read_bytes()
        payload: object = json.loads(source_bytes)
        manifest = cls.from_dict(_require_dict(payload, "oracle manifest"))
        if source_bytes != _canonical_json(manifest.to_dict()) + b"\n":
            raise ValueError("oracle manifest must use canonical JSON bytes")
        fixture = (
            Path(fixture_path)
            if fixture_path is not None
            else source.parent / manifest.fixture_file
        )
        manifest.validate(official_checkout=official_checkout, fixture_path=fixture)
        return manifest

    def validate(
        self,
        *,
        official_checkout: str | Path | None = None,
        fixture_path: str | Path | None = None,
    ) -> None:
        if self.schema_version != ORACLE_SCHEMA_VERSION:
            raise ValueError(
                f"oracle schema version {self.schema_version} is unsupported"
            )
        if not _CASE_PATTERN.fullmatch(self.case_name):
            raise ValueError(f"invalid oracle case name: {self.case_name!r}")
        if not _COMMIT_PATTERN.fullmatch(self.official_commit):
            raise ValueError("official commit is not a full Git object id")
        if (
            not self.fixture_file.endswith(".npz")
            or Path(self.fixture_file).name != self.fixture_file
        ):
            raise ValueError("oracle fixture filename is not a basename NPZ")
        stem = self.fixture_file.removesuffix(".npz")
        profile, mode, case, source, dtype, seed = _fixture_coordinates(stem)
        if (
            self.profile,
            self.observation_mode,
            self.case_name,
            self.source_spec,
            self.dtype,
            self.seed,
        ) != (profile, mode, case, source, dtype, seed):
            raise ValueError(
                "oracle manifest coordinates do not match supported fixture stem"
            )
        revision = _PROFILE_REVISIONS[profile]
        if self.official_commit != revision:
            raise ValueError("official commit does not match profile authority")
        if self.generator_request != _canonical_fixture_request(stem):
            raise ValueError(
                "oracle generator request does not match fixture coordinates"
            )
        if self.generator_command != _canonical_fixture_command(stem):
            raise ValueError(
                "oracle generator command does not match fixture coordinates"
            )
        expected_hashes = dict(_source_hashes_for(source, revision))
        if dict(self.official_file_hashes) != expected_hashes:
            raise ValueError("oracle source hashes do not match pinned source spec")
        if not _source_allows_dtype(source, dtype):
            raise ValueError("oracle dtype is not allowed by source spec")
        if not _SHA256_PATTERN.fullmatch(self.fixture_sha256):
            raise ValueError("oracle fixture hash is malformed")
        if official_checkout is not None:
            for path, digest in self.official_file_hashes.items():
                if (
                    _sha256_bytes(_git_show(official_checkout, revision, path))
                    != digest
                ):
                    raise ValueError(f"oracle source file hash mismatch: {path}")
        if fixture_path is not None:
            fixture = Path(fixture_path)
            if fixture.is_symlink():
                raise ValueError("oracle fixture must not be a symlink")
            if not fixture.is_file():
                raise ValueError(f"oracle fixture is missing: {fixture}")
            if fixture.name != self.fixture_file:
                raise ValueError("oracle fixture filename does not match manifest")
            if _sha256_path(fixture) != self.fixture_sha256:
                raise ValueError("oracle fixture hash mismatch")
            with np.load(fixture, allow_pickle=False) as arrays:
                if tuple(arrays.files) != tuple(self.tensor_schema):
                    raise ValueError("oracle tensor schema names do not match fixture")
                for name, tensor_spec in self.tensor_schema.items():
                    value = arrays[name]
                    if (
                        value.shape != tensor_spec.shape
                        or value.dtype.name != tensor_spec.dtype
                    ):
                        raise ValueError(f"oracle tensor schema mismatch: {name}")

    def to_dict(self) -> dict[str, object]:
        return {
            "case_name": self.case_name,
            "dtype": self.dtype,
            "fixture_file": self.fixture_file,
            "fixture_sha256": self.fixture_sha256,
            "generator_command": list(self.generator_command),
            "generator_request": self.generator_request,
            "observation_mode": self.observation_mode.value,
            "official_commit": self.official_commit,
            "official_file_hashes": dict(self.official_file_hashes),
            "profile": self.profile.value,
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
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(_canonical_json(self.to_dict()) + b"\n")
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination


@dataclass(frozen=True)
class ParameterMapping:
    source: str
    destination: str
    transform: str = "identity"
    reshape: tuple[int, ...] | None = None


def _parameter_path(path: str | Sequence[str]) -> str:
    if type(path) is str:
        result = path
    else:
        if isinstance(path, str):
            raise TypeError("parameter path must be an exact string")
        runtime_path = cast(object, path)
        if not isinstance(runtime_path, Sequence):
            raise TypeError(
                "parameter path must be an exact string or a sequence of exact strings"
            )
        segments = tuple(path)
        if any(type(segment) is not str for segment in segments):
            raise TypeError("parameter path segments must be exact strings")
        result = ".".join(segments)
    if not result or result.startswith(".") or result.endswith(".") or ".." in result:
        raise ValueError(f"invalid parameter path: {result!r}")
    return result


def _transform_parameter(value: Array, mapping: ParameterMapping) -> Array:
    if mapping.transform == "identity":
        return value
    if mapping.transform == "transpose":
        return value.T
    assert mapping.reshape is not None
    return value.reshape(mapping.reshape)


def _prepare_parameter(
    value: Array,
    mapping: ParameterMapping,
    destination_shape: Sequence[int],
) -> Array:
    transformed = _transform_parameter(np.asarray(value), mapping)
    expected_shape = tuple(destination_shape)
    if transformed.shape != expected_shape:
        raise ValueError(
            f"parameter shape mismatch for {mapping.destination}: {transformed.shape} != {expected_shape}"
        )
    return transformed


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
        if (transform == "reshape") != (reshape is not None):
            raise ValueError("reshape target and transform must be used together")
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
        value: Array,
        destination_shape: Sequence[int],
    ) -> Array:
        source_path = _parameter_path(source)
        destination_path = _parameter_path(destination)
        mapping = self._by_source.get(source_path)
        if mapping is None or mapping.destination != destination_path:
            raise ValueError(
                f"parameter mapping is not registered: {source_path} -> {destination_path}"
            )
        if source_path in self._consumed_sources:
            raise ValueError(f"source parameter consumed more than once: {source_path}")
        if destination_path in self._consumed_destinations:
            raise ValueError(
                f"destination parameter consumed more than once: {destination_path}"
            )
        transformed = _prepare_parameter(value, mapping, destination_shape)
        self._consumed_sources.add(source_path)
        self._consumed_destinations.add(destination_path)
        return transformed

    def translate(
        self,
        source_parameters: Mapping[str, Array],
        destination_shapes: Mapping[str, Sequence[int] | Array],
    ) -> dict[str, Array]:
        if any(type(path) is not str for path in source_parameters):
            raise TypeError("source parameter mapping keys must be exact strings")
        if any(type(path) is not str for path in destination_shapes):
            raise TypeError("destination parameter mapping keys must be exact strings")
        source_paths = set(source_parameters)
        destination_paths = set(destination_shapes)
        extra_sources = source_paths - set(self._by_source)
        if extra_sources:
            raise ValueError(
                "unregistered source parameters: " + ", ".join(sorted(extra_sources))
            )
        extra_destinations = destination_paths - set(self._by_destination)
        if extra_destinations:
            raise ValueError(
                "unregistered destination parameters: "
                + ", ".join(sorted(extra_destinations))
            )
        missing = (set(self._by_source) - source_paths) | (
            set(self._by_destination) - destination_paths
        )
        if missing:
            raise ValueError(
                "unconsumed registered parameters: " + ", ".join(sorted(missing))
            )
        translated: dict[str, Array] = {}
        for mapping in self.registry:
            destination = destination_shapes[mapping.destination]
            shape = (
                tuple(np.shape(destination))
                if isinstance(destination, np.ndarray)
                else destination
            )
            translated[mapping.destination] = _prepare_parameter(
                source_parameters[mapping.source],
                mapping,
                shape,
            )
        consumed_sources = set(self._by_source)
        consumed_destinations = set(self._by_destination)
        self._consumed_sources.clear()
        self._consumed_sources.update(consumed_sources)
        self._consumed_destinations.clear()
        self._consumed_destinations.update(consumed_destinations)
        return translated

    def assert_fully_consumed(self) -> None:
        missing = (set(self._by_source) - self._consumed_sources) | (
            set(self._by_destination) - self._consumed_destinations
        )
        if missing:
            raise ValueError("unconsumed parameters: " + ", ".join(sorted(missing)))


def _write_deterministic_npz(path: str | Path, arrays: Mapping[str, Array]) -> Path:
    if any(type(name) is not str for name in arrays):
        raise TypeError("array keys must be exact strings")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            with zipfile.ZipFile(
                stream, "w", compression=zipfile.ZIP_STORED
            ) as archive:
                for name, value in sorted(arrays.items()):
                    buffer = io.BytesIO()
                    np.lib.format.write_array(
                        buffer, np.asarray(value), allow_pickle=False
                    )
                    info = zipfile.ZipInfo(
                        f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                    )
                    info.compress_type = zipfile.ZIP_STORED
                    info.external_attr = 0o600 << 16
                    archive.writestr(info, buffer.getvalue())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


class _SuppliedCategoricalNoise:
    def __init__(
        self,
        *,
        expected_logits: Array | jax.Array,
        expected_output_shape: Sequence[int],
        noise: Array | jax.Array,
    ) -> None:
        self.expected_logits = np.asarray(jax.device_get(expected_logits))
        self.expected_output_shape = tuple(expected_output_shape)
        supplied_noise = np.asarray(jax.device_get(noise))
        expected_shape = self.expected_output_shape + (self.expected_logits.shape[-1],)
        if supplied_noise.shape != expected_shape:
            raise ValueError(
                f"supplied categorical noise shape mismatch: {supplied_noise.shape} != {expected_shape}"
            )
        if supplied_noise.dtype != self.expected_logits.dtype:
            raise ValueError(
                "supplied categorical noise dtype does not match logits: "
                f"{supplied_noise.dtype} != {self.expected_logits.dtype}"
            )
        self.noise = jnp.asarray(supplied_noise)
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
            raise ValueError("categorical sample axis must be -1")
        if shape is None or tuple(shape) != self.expected_output_shape:
            raise ValueError("categorical output shape does not match supplied case")
        if tuple(logits.shape) != self.expected_logits.shape:
            raise ValueError("categorical logits shape does not match supplied case")
        batch_shape = tuple(logits.shape[:-1])
        if (
            batch_shape
            and self.expected_output_shape[-len(batch_shape) :] != batch_shape
        ):
            raise ValueError(
                "categorical output shape does not end in logits batch shape"
            )
        self._assert_logits(logits)
        shape_prefix = len(self.expected_output_shape) - len(batch_shape)
        expanded_logits = jax.lax.expand_dims(logits, tuple(range(shape_prefix)))
        self.calls += 1
        return jnp.argmax(self.noise + expanded_logits, axis=axis)

    def _assert_logits(self, logits: jax.Array) -> None:
        expected = self.expected_logits

        def assert_equal(value: Array) -> None:
            if not np.array_equal(value, expected):
                raise ValueError("categorical logits do not match supplied-noise case")

        if isinstance(logits, jax_core.Tracer):
            jax.debug.callback(assert_equal, logits)
        else:
            assert_equal(np.asarray(jax.device_get(logits)))


@contextmanager
def _supplied_categorical_noise_scope(
    random_namespace: Any,
    *,
    expected_logits: Array | jax.Array,
    expected_output_shape: Sequence[int],
    noise: Array | jax.Array,
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
            raise ValueError("supplied categorical noise case must invoke exactly once")
    finally:
        if original is missing:
            delattr(random_namespace, "categorical")
        else:
            setattr(random_namespace, "categorical", original)


__all__ = [
    "DISTRIBUTIONS_SOURCE_SPEC",
    "NETWORKS_SOURCE_SPEC",
    "ORACLE_SCHEMA_VERSION",
    "PAPER_REVISION",
    "REPLAY_SOURCE_HASHES",
    "REPLAY_SOURCE_SPEC",
    "RSSM_SOURCE_SPEC",
    "UPSTREAM_CURRENT_REVISION",
    "OracleManifest",
    "ParameterMapping",
    "ParameterTranslator",
    "TensorSpec",
]
