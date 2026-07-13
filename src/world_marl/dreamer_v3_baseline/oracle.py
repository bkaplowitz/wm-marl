from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import jax
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


ORACLE_SCHEMA_VERSION = 1
PAPER_REVISION = "bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01"
UPSTREAM_CURRENT_REVISION = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"
OFFICIAL_FILES = ("dreamerv3/configs.yaml",)
_CASE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_PAPER_OVERRIDES: dict[str, Any] = {
    "agent.dec.simple.strided": True,
    "agent.enc.simple.strided": True,
    "agent.opt.beta2": 0.99,
    "run.steps": 1_000_000,
}


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
    official_file_hashes: Mapping[str, str]
    profile_hash: str
    overrides: Mapping[str, Any]
    jax_version: str
    dtype: str
    device: str
    seed: int
    tensor_schema: Mapping[str, TensorSpec]
    generator_command: tuple[str, ...]
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
        dtype: str | None = None,
        device: str | None = None,
    ) -> OracleManifest:
        resolved_profile = DreamerProfile(profile)
        resolved_mode = ObservationMode(observation_mode)
        config = resolve_dreamer_config(resolved_profile, resolved_mode)
        checkout = Path(official_checkout).resolve()
        revision = official_revision(resolved_profile)
        source_hashes = {
            path: _sha256_bytes(_git_show(checkout, revision, path))
            for path in OFFICIAL_FILES
        }
        fixture = Path(fixture_path)
        return cls(
            schema_version=ORACLE_SCHEMA_VERSION,
            case_name=case_name,
            profile=resolved_profile,
            observation_mode=resolved_mode,
            official_commit=revision,
            official_file_hashes=source_hashes,
            profile_hash=config.canonical_hash(),
            overrides=profile_overrides(resolved_profile),
            jax_version=jax.__version__,
            dtype=dtype or config.run.compute_dtype,  # type: ignore[union-attr]
            device=device or jax.default_backend(),
            seed=seed,
            tensor_schema={
                name: TensorSpec(tuple(array.shape), array.dtype.name)
                for name, array in sorted(arrays.items())
            },
            generator_command=tuple(generator_command),
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
            "jax_version": self.jax_version,
            "observation_mode": self.observation_mode.value,
            "official_commit": self.official_commit,
            "official_file_hashes": dict(self.official_file_hashes),
            "overrides": dict(self.overrides),
            "profile": self.profile.value,
            "profile_hash": self.profile_hash,
            "schema_version": self.schema_version,
            "seed": self.seed,
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
            fixture_file=payload["fixture_file"],
            fixture_sha256=payload["fixture_sha256"],
        )
        fixture = Path(fixture_path) if fixture_path is not None else None
        if fixture is None:
            candidate = source.parent / manifest.fixture_file
            fixture = candidate if candidate.exists() else None
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
        expected_overrides = dict(profile_overrides(self.profile))
        if dict(self.overrides) != expected_overrides:
            raise ValueError("oracle override map does not match profile")
        expected_config = config or resolve_dreamer_config(
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
        if self.dtype != expected_config.run.compute_dtype:
            raise ValueError("oracle dtype does not match the resolved config")
        if not self.device:
            raise ValueError("oracle device must be recorded")
        if tuple(self.official_file_hashes) != OFFICIAL_FILES:
            raise ValueError("oracle source file set is incomplete")
        if official_checkout is not None:
            checkout = Path(official_checkout).resolve()
            for path, recorded_hash in self.official_file_hashes.items():
                expected_hash = _sha256_bytes(
                    _git_show(checkout, self.official_commit, path)
                )
                if recorded_hash != expected_hash:
                    raise ValueError(f"oracle source file hash mismatch: {path}")
        elif any(
            not _SHA256_PATTERN.fullmatch(value)
            for value in self.official_file_hashes.values()
        ):
            raise ValueError("oracle source file hash is malformed")
        if not self.generator_command:
            raise ValueError("oracle generator command must be recorded")
        if self.seed < 0:
            raise ValueError("oracle seed must be nonnegative")
        if not _SHA256_PATTERN.fullmatch(self.fixture_sha256):
            raise ValueError("oracle fixture hash is malformed")
        if fixture_path is not None:
            fixture = Path(fixture_path)
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
        if not self.official_checkout.is_dir():
            raise ValueError("official checkout does not exist")
        for revision in (PAPER_REVISION, UPSTREAM_CURRENT_REVISION):
            _git_object_exists(self.official_checkout, revision)

    def write_fixture(
        self,
        *,
        case_name: str,
        profile: DreamerProfile | str,
        observation_mode: ObservationMode | str,
        arrays: Mapping[str, np.ndarray],
        seed: int,
        generator_command: Sequence[str],
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
        }
        command = (
            self.python_executable,
            str(Path(__file__).resolve()),
            "_config_worker",
        )
        completed = subprocess.run(
            command,
            cwd=self.official_checkout,
            input=json.dumps(request, sort_keys=True),
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
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
            generator_command=command
            + ("--request", json.dumps(request, sort_keys=True)),
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
    source = yaml.safe_load(_git_show(checkout, revision, OFFICIAL_FILES[0]))
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
        "worker_pid": {"dtype": "int64", "values": [os.getpid()]},
    }
    return {"arrays": arrays}


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
    if tuple(argv) != ("_config_worker",):
        raise SystemExit("oracle.py is an internal fixture worker")
    request = json.loads(sys.stdin.read())
    sys.stdout.write(json.dumps(_config_worker(request), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess boundary.
    raise SystemExit(_main(sys.argv[1:]))


__all__ = [
    "OFFICIAL_FILES",
    "ORACLE_SCHEMA_VERSION",
    "OracleHarness",
    "OracleManifest",
    "PAPER_REVISION",
    "ParameterMapping",
    "ParameterTranslator",
    "TensorSpec",
    "UPSTREAM_CURRENT_REVISION",
    "official_revision",
    "profile_overrides",
]
