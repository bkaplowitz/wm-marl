from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import os
import platform
import re
import subprocess
import sys
import textwrap
import threading
import traceback
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial as bind
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

if __package__:
    from .config import DreamerProfile, ObservationMode
    from .oracle import (
        PAPER_REVISION,
        UPSTREAM_CURRENT_REVISION,
        OracleHarness,
        OracleInvocation,
        OracleSourceSpec,
        _canonical_json,
        _git_show,
        _sha256_bytes,
        official_revision,
        profile_overrides,
        register_oracle_source_spec,
    )
    from .replay_oracle_contract import (
        REPLAY_COMMAND_DESCRIPTOR,
        REPLAY_GENERATOR_FILE_HASHES,
        REPLAY_RUNTIME_CONTRACT,
    )
else:  # pragma: no cover - exercised by the isolated worker.
    package_dir = Path(__file__).resolve().parent
    world_marl_dir = package_dir.parent
    world_marl = sys.modules.setdefault("world_marl", ModuleType("world_marl"))
    world_marl.__path__ = [str(world_marl_dir)]
    dreamer_package = sys.modules.setdefault(
        "world_marl.dreamer_v3_baseline",
        ModuleType("world_marl.dreamer_v3_baseline"),
    )
    dreamer_package.__path__ = [str(package_dir)]
    world_marl.dreamer_v3_baseline = dreamer_package
    from world_marl.dreamer_v3_baseline.config import (
        DreamerProfile,
        ObservationMode,
    )
    from world_marl.dreamer_v3_baseline.oracle import (
        PAPER_REVISION,
        UPSTREAM_CURRENT_REVISION,
        OracleHarness,
        OracleInvocation,
        OracleSourceSpec,
        _canonical_json,
        _git_show,
        _sha256_bytes,
        official_revision,
        profile_overrides,
        register_oracle_source_spec,
    )
    from world_marl.dreamer_v3_baseline.replay_oracle_contract import (
        REPLAY_COMMAND_DESCRIPTOR,
        REPLAY_GENERATOR_FILE_HASHES,
        REPLAY_RUNTIME_CONTRACT,
    )


_SOURCE_HASHES = {
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

_ELEMENTS_VERSION = str(REPLAY_RUNTIME_CONTRACT["elements_version"])
_ELEMENTS_MODE = str(REPLAY_RUNTIME_CONTRACT["elements_mode"])
_NUMPY_VERSION = str(REPLAY_RUNTIME_CONTRACT["numpy_version"])
_WORKER_MODE = str(REPLAY_RUNTIME_CONTRACT["worker_mode"])
_ELEMENTS_HELPER_HASHES = dict(REPLAY_RUNTIME_CONTRACT["elements_helper_hashes"])
_NATIVE_MODULE_NAMES = frozenset(
    {
        "world_marl.dreamer_v3_baseline.replay",
        "world_marl.dreamer_v3_baseline.replay_oracle",
    }
)
_REQUEST_KEYS = frozenset(
    {
        "case_name",
        "cases",
        "compute_dtype",
        "generator_files",
        "observation_mode",
        "official_commit",
        "overrides",
        "profile",
        "row_schema",
        "runtime",
        "seed",
        "source_spec",
        "uuid_mode",
    }
)
_EXECUTION_KEYS = frozenset(
    {
        "elements_dist_info",
        "elements_package_dir",
        "official_checkout",
        "python_executable",
    }
)
_GENERATOR_PACKAGE_PREFIX = "world_marl/dreamer_v3_baseline/"
_GENERATOR_CONTRACT_PATH = "world_marl/dreamer_v3_baseline/replay_oracle_contract.py"
_CONTRACT_SELF_DIGEST_PATTERN = re.compile(
    rb'(?m)^(REPLAY_CONTRACT_SELF_SHA256 = \(\n    ")[0-9a-f]{64}("\n\)$)'
)


def _native_module_violations() -> list[str]:
    replay_path = Path(__file__).resolve().with_name("replay.py")
    oracle_path = Path(__file__).resolve()
    violations = set()
    for name, module in tuple(sys.modules.items()):
        if name in _NATIVE_MODULE_NAMES:
            violations.add(name)
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        try:
            module_path = Path(filename).resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if module_path == replay_path or (
            module_path == oracle_path and name != "__main__"
        ):
            violations.add(f"{name}:{module_path.name}")
    return sorted(violations)


def _require_isolated_worker_modules() -> list[str]:
    violations = _native_module_violations()
    if violations:
        raise ValueError(f"forbidden native replay modules loaded: {violations}")
    return violations


def _installed_elements_provenance(
    package_dir: Path,
    dist_info: Path,
) -> tuple[str, dict[str, str]]:
    metadata = (dist_info / "METADATA").read_text()
    versions = [
        line.split(":", 1)[1].strip()
        for line in metadata.splitlines()
        if line.startswith("Version:")
    ]
    if len(versions) != 1:
        raise ValueError("Elements distribution metadata has no unique version")
    hashes = {
        path: hashlib.sha256((package_dir / Path(path).name).read_bytes()).hexdigest()
        for path in sorted(_ELEMENTS_HELPER_HASHES)
    }
    return versions[0], hashes


class _Section:
    def __call__(self, function):
        return function

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _Timer:
    @staticmethod
    def section(name):
        del name
        return _Section()


class _UUID:
    debug_id: int | None = None

    @classmethod
    def reset(cls, *, debug: bool) -> None:
        cls.debug_id = 0 if debug else None

    def __init__(self, value=None):
        if value is None:
            if type(self).debug_id is None:
                raise ValueError("replay oracle UUIDs require debug mode")
            type(self).debug_id += 1
            self.value = type(self).debug_id.to_bytes(16, "big")
        elif isinstance(value, _UUID):
            self.value = value.value
        elif isinstance(value, int):
            self.value = value.to_bytes(16, "big")
        elif isinstance(value, bytes):
            self.value = value
        elif isinstance(value, np.ndarray):
            self.value = value.tobytes()
        elif isinstance(value, str):
            self.value = int(value).to_bytes(16, "big")
        else:
            raise ValueError(value)
        if len(self.value) != 16:
            raise ValueError("UUID must contain exactly 16 bytes")

    def __bytes__(self):
        return self.value

    def __int__(self):
        return int.from_bytes(self.value, "big")

    def __str__(self):
        return str(int(self))

    def __array__(self):
        return np.frombuffer(self.value, np.uint8)

    def __getitem__(self, index):
        return self.__array__()[index]

    def __eq__(self, other):
        return isinstance(other, _UUID) and self.value == other.value

    def __hash__(self):
        return hash(self.value)


class _RWLock:
    @property
    @contextmanager
    def reading(self):
        yield

    @property
    @contextmanager
    def writing(self):
        yield


class _Limiters:
    @staticmethod
    def wait(predicate, message):
        if not predicate():
            raise RuntimeError(message)


def _timestamp(*, millis=False):
    del millis
    return "oracle"


def _live_shim_hashes() -> dict[str, str]:
    helpers = {
        "Limiters": _Limiters,
        "RWLock": _RWLock,
        "Section": _Section,
        "Timer": _Timer,
        "UUID": _UUID,
        "timestamp": _timestamp,
    }
    return {
        name: hashlib.sha256(
            textwrap.dedent(inspect.getsource(helper)).encode()
        ).hexdigest()
        for name, helper in sorted(helpers.items())
    }


def _runtime_contract() -> dict[str, Any]:
    return json.loads(json.dumps(REPLAY_RUNTIME_CONTRACT, sort_keys=True))


def _live_generator_file_hashes() -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    result = {}
    for relative_path in sorted(REPLAY_GENERATOR_FILE_HASHES):
        source = (
            package_dir / relative_path.removeprefix(_GENERATOR_PACKAGE_PREFIX)
        ).read_bytes()
        if relative_path == _GENERATOR_CONTRACT_PATH:
            source = _normalized_contract_source(source)
        result[relative_path] = hashlib.sha256(source).hexdigest()
    return result


def _normalized_contract_source(source: bytes) -> bytes:
    matches = tuple(_CONTRACT_SELF_DIGEST_PATTERN.finditer(source))
    if len(matches) != 1:
        raise ValueError(
            "replay generator contract must contain exactly one self digest"
        )
    match = matches[0]
    start, stop = match.span()
    normalized = match.group(1) + b"0" * 64 + match.group(2)
    return source[:start] + normalized + source[stop:]


def _validate_live_generator_contract(request: Mapping[str, Any]) -> None:
    expected_runtime = _runtime_contract()
    live_files = _live_generator_file_hashes()
    if live_files != dict(REPLAY_GENERATOR_FILE_HASHES) or not _same_contract(
        request["generator_files"], live_files
    ):
        raise ValueError("replay generator content does not match the frozen contract")
    if platform.python_implementation() != expected_runtime["python_implementation"]:
        raise ValueError("replay generator Python implementation changed")
    if platform.python_version() != expected_runtime["python_version"]:
        raise ValueError("replay generator Python version changed")
    if np.__version__ != expected_runtime["numpy_version"]:
        raise ValueError("replay generator NumPy version changed")
    if _live_shim_hashes() != expected_runtime["shim_hashes"]:
        raise ValueError("replay generator shim content changed")


def _case_contract(seed: int) -> dict[str, Any]:
    return {
        "capacity": {
            "batch": 1,
            "capacity": 3,
            "chunk_size": 3,
            "collection_latent_bases": {"dyn/deter": 1000, "dyn/stoch": 2000},
            "first_steps": [0, 4, 8],
            "last_steps": [3, 7],
            "online": True,
            "raw_length": 4,
            "seed": seed,
            "selector_checkpoints": [6, 8],
            "selector_draws": 12,
            "steps": 10,
            "terminal_steps": [],
        },
        "primary": {
            "batch": 1,
            "capacity": 20,
            "chunk_size": 3,
            "collection_latent_bases": {"dyn/deter": 1000, "dyn/stoch": 2000},
            "consecutive": 2,
            "context": 1,
            "first_steps": [0, 4, 8],
            "last_steps": [3],
            "latent_update_bases": {"dyn/deter": 100, "dyn/stoch": 200},
            "online": True,
            "raw_length": 5,
            "seed": seed,
            "sequence_length": 2,
            "steps": 11,
            "terminal_steps": [],
        },
    }


def _row_schema_contract() -> dict[str, dict[str, Any]]:
    return {
        "action": {"dtype": "float32", "shape": []},
        "dyn/deter": {"dtype": "float32", "shape": [2]},
        "dyn/stoch": {"dtype": "float32", "shape": [1, 2]},
        "is_first": {"dtype": "bool", "shape": []},
        "is_last": {"dtype": "bool", "shape": []},
        "is_terminal": {"dtype": "bool", "shape": []},
        "reward": {"dtype": "float32", "shape": []},
        "value": {"dtype": "int32", "shape": []},
    }


def _same_contract(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        return False
    return _canonical_json(dict(actual)) == _canonical_json(dict(expected))


def _request_path(
    request: Mapping[str, Any],
    key: str,
    *,
    boundary: str,
    resolve: bool = True,
) -> Path:
    value = request[key]
    if type(value) is not str:
        raise ValueError(f"{boundary} path coordinate changed: {key}")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{boundary} path coordinate changed: {key}")
    try:
        return path.resolve() if resolve else path.absolute()
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"{boundary} path coordinate changed: {key}") from error


def _execution_path(
    coordinates: Mapping[str, str | Path | None],
    key: str,
    *,
    default: str | Path | None = None,
    resolve: bool = True,
) -> Path:
    value = coordinates.get(key)
    if value is None:
        value = default
    if value is None:
        raise ValueError(f"replay generator execution coordinate is required: {key}")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(
            f"replay generator execution coordinate must be absolute: {key}"
        )
    try:
        return path.resolve() if resolve else path.absolute()
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            f"replay generator execution coordinate changed: {key}"
        ) from error


def _validate_replay_generator_provenance(
    manifest: Any,
    request: Mapping[str, Any],
    official_checkout: Path | None,
) -> None:
    del official_checkout
    if set(request) != _REQUEST_KEYS:
        raise ValueError("replay generator request keys do not match the contract")
    if manifest.case_name != "replay" or type(manifest.seed) is not int:
        raise ValueError("replay generator manifest coordinate is not authorized")
    if manifest.seed != 7:
        raise ValueError("replay generator manifest coordinate is not authorized")
    expected_coordinates = {
        "case_name": manifest.case_name,
        "compute_dtype": manifest.dtype,
        "observation_mode": manifest.observation_mode.value,
        "official_commit": manifest.official_commit,
        "overrides": dict(manifest.overrides),
        "profile": manifest.profile.value,
        "seed": manifest.seed,
        "source_spec": manifest.source_spec,
    }
    actual_coordinates = {key: request[key] for key in expected_coordinates}
    if not _same_contract(actual_coordinates, expected_coordinates):
        raise ValueError("replay generator request manifest coordinate changed")
    if manifest.observation_mode is not ObservationMode.PROPRIO:
        raise ValueError("replay generator manifest coordinate is not authorized")
    if manifest.dtype != "float32" or manifest.source_spec != "replay":
        raise ValueError("replay generator manifest coordinate is not authorized")
    if not _same_contract(request["cases"], _case_contract(manifest.seed)):
        raise ValueError("replay generator case contract changed")
    if not _same_contract(request["row_schema"], _row_schema_contract()):
        raise ValueError("replay generator row schema changed")
    if not _same_contract(request["runtime"], _runtime_contract()):
        raise ValueError("replay generator runtime contract changed")
    if not _same_contract(request["generator_files"], REPLAY_GENERATOR_FILE_HASHES):
        raise ValueError("replay generator file contract changed")
    if request["uuid_mode"] != "debug-counter":
        raise ValueError("replay generator runtime coordinate changed")
    if tuple(manifest.generator_command) != REPLAY_COMMAND_DESCRIPTOR:
        raise ValueError("replay generator command does not match the contract")


def _stable_replay_request(
    profile: DreamerProfile,
    mode: ObservationMode,
    case_name: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "cases": _case_contract(seed),
        "compute_dtype": "float32",
        "generator_files": dict(REPLAY_GENERATOR_FILE_HASHES),
        "observation_mode": mode.value,
        "official_commit": official_revision(profile),
        "overrides": dict(profile_overrides(profile)),
        "profile": profile.value,
        "row_schema": _row_schema_contract(),
        "runtime": _runtime_contract(),
        "seed": seed,
        "source_spec": "replay",
        "uuid_mode": "debug-counter",
    }


def _build_replay_invocation(
    request: Mapping[str, Any],
    coordinates: Mapping[str, str | Path | None],
) -> OracleInvocation:
    _validate_live_generator_contract(request)
    checkout = _execution_path(coordinates, "official_checkout")
    python = _execution_path(
        coordinates,
        "python_executable",
        default=Path(sys.executable),
        resolve=False,
    )
    if python.resolve() != Path(sys.executable).resolve():
        raise ValueError("replay generator interpreter provenance changed")
    if python != Path(sys.executable).absolute():
        raise ValueError("replay generator interpreter provenance changed")
    package_dir = _execution_path(
        coordinates,
        "elements_package_dir",
    )
    dist_info = _execution_path(
        coordinates,
        "elements_dist_info",
    )
    elements_version, elements_hashes = _installed_elements_provenance(
        package_dir, dist_info
    )
    if elements_version != _ELEMENTS_VERSION:
        raise ValueError("replay generator Elements version changed")
    if elements_hashes != _ELEMENTS_HELPER_HASHES:
        raise ValueError("replay generator Elements helper files changed")
    execution = {
        "elements_dist_info": str(dist_info),
        "elements_package_dir": str(package_dir),
        "official_checkout": str(checkout),
        "python_executable": str(python),
    }
    return OracleInvocation(
        command=(str(python), str(Path(__file__).resolve()), "_worker"),
        cwd=checkout,
        generator_request=_canonical_json(
            {"execution": execution, "request": dict(request)}
        ).decode(),
    )


def _resolve_replay_generator_invocation(
    manifest: Any,
    coordinates: Mapping[str, str | Path | None],
) -> OracleInvocation:
    if manifest.generator_request is None:
        raise ValueError("replay generator invocation requires a request")
    request = json.loads(manifest.generator_request)
    _validate_replay_generator_provenance(manifest, request, None)
    return _build_replay_invocation(request, coordinates)


REPLAY_SOURCE_SPEC = OracleSourceSpec(
    name="replay",
    revision_hashes={
        PAPER_REVISION: _SOURCE_HASHES,
        UPSTREAM_CURRENT_REVISION: _SOURCE_HASHES,
    },
    execution_dtypes=("float32",),
    generator_validation_required=True,
    generator_validator_id="replay-generator-validator-v3",
    generator_validator=_validate_replay_generator_provenance,
    generator_resolution_required=True,
    generator_resolver_id="replay-generator-resolver-v2",
    generator_resolver=_resolve_replay_generator_invocation,
)
register_oracle_source_spec(REPLAY_SOURCE_SPEC)


def _extract_classes(
    source: bytes,
    filename: str,
    class_names: Sequence[str],
    globals_: Mapping[str, Any],
) -> ModuleType:
    tree = ast.parse(source, filename=filename)
    wanted = set(class_names)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in wanted
    ]
    if {node.name for node in nodes} != wanted:
        raise ValueError(f"missing exact source classes in {filename}")
    module = ModuleType(filename)
    module.__dict__.update(globals_)
    exact = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(exact)
    exec(compile(exact, filename, "exec"), module.__dict__)
    return module


def _load_source_classes(sources: Mapping[str, bytes], revision: str):
    elements = SimpleNamespace(
        Path=Path,
        RWLock=_RWLock,
        UUID=_UUID,
        timer=_Timer(),
        timestamp=_timestamp,
    )
    selectors = _extract_classes(
        sources["embodied/core/selectors.py"],
        f"{revision}:embodied/core/selectors.py",
        ("Uniform",),
        {"collections": __import__("collections"), "np": np, "threading": threading},
    )
    chunk = _extract_classes(
        sources["embodied/core/chunk.py"],
        f"{revision}:embodied/core/chunk.py",
        ("Chunk",),
        {
            "elements": elements,
            "io": io,
            "np": np,
            "sys": sys,
            "traceback": traceback,
        },
    )
    replay = _extract_classes(
        sources["embodied/core/replay.py"],
        f"{revision}:embodied/core/replay.py",
        ("Replay",),
        {
            "ThreadPoolExecutor": ThreadPoolExecutor,
            "bind": bind,
            "chunklib": chunk,
            "defaultdict": defaultdict,
            "deque": deque,
            "elements": elements,
            "limiters": _Limiters,
            "np": np,
            "selectors": selectors,
            "threading": threading,
        },
    )
    streams = _extract_classes(
        sources["embodied/core/streams.py"],
        f"{revision}:embodied/core/streams.py",
        ("Consec",),
        {"base": SimpleNamespace(Stream=object), "np": np},
    )
    return elements, chunk.Chunk, selectors.Uniform, replay.Replay, streams.Consec


def _source_attestation(revision, elements, Chunk, Uniform, Replay, Consec):
    classes = (Chunk, Consec, Replay, Uniform)
    origins = {
        "Chunk.append": Chunk.append,
        "Chunk.update": Chunk.update,
        "Consec.__next__": Consec.__next__,
        "Replay._sample": Replay._sample,
        "Replay.add": Replay.add,
        "Replay.sample": Replay.sample,
        "Replay.update": Replay.update,
        "Uniform.__call__": Uniform.__call__,
        "Uniform.__delitem__": Uniform.__delitem__,
        "Uniform.__setitem__": Uniform.__setitem__,
    }
    expected_origins = {
        "Chunk": f"{revision}:embodied/core/chunk.py",
        "Consec": f"{revision}:embodied/core/streams.py",
        "Replay": f"{revision}:embodied/core/replay.py",
        "Uniform": f"{revision}:embodied/core/selectors.py",
    }
    method_origins = {
        name: method.__code__.co_filename for name, method in sorted(origins.items())
    }
    for name, filename in method_origins.items():
        if filename != expected_origins[name.split(".", 1)[0]]:
            raise ValueError(f"official method origin mismatch: {name}")

    replay_globals = Replay.add.__globals__
    chunk_globals = Chunk.append.__globals__
    uniform_globals = Uniform.__call__.__globals__
    consec_globals = Consec.__next__.__globals__
    bindings = {
        "chunk.elements": chunk_globals.get("elements") is elements,
        "chunk.numpy": chunk_globals.get("np") is np,
        "consec.numpy": consec_globals.get("np") is np,
        "replay.chunk": replay_globals.get("chunklib").Chunk is Chunk,
        "replay.elements": replay_globals.get("elements") is elements,
        "replay.numpy": replay_globals.get("np") is np,
        "replay.uniform": replay_globals.get("selectors").Uniform is Uniform,
        "uniform.numpy": uniform_globals.get("np") is np,
    }
    if not all(bindings.values()):
        raise ValueError("official source bindings did not survive AST loading")
    return {
        "bindings": sorted(name for name, valid in bindings.items() if valid),
        "classes": sorted(cls.__name__ for cls in classes),
        "method_origins": method_origins,
    }


def _row(
    index: int,
    case: Mapping[str, Any],
    row_schema: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    flags = {
        "is_first": index in case["first_steps"],
        "is_last": index in case["last_steps"],
        "is_terminal": index in case["terminal_steps"],
    }
    result = {}
    for name, spec in sorted(row_schema.items()):
        shape = tuple(spec["shape"])
        dtype = np.dtype(spec["dtype"])
        if name in flags:
            value = flags[name]
        elif name.startswith(("dyn/", "enc/", "dec/")):
            base = case["collection_latent_bases"][name]
            value = base + index + np.arange(np.prod(shape)).reshape(shape)
        else:
            value = index
        result[name] = np.asarray(value, dtype=dtype).reshape(shape)
    return result


def _key_arrays(keys):
    return (
        np.asarray([int(chunk_id) for chunk_id, _ in keys], np.int64),
        np.asarray([offset for _, offset in keys], np.int32),
    )


def _rng_bytes(rng: np.random.Generator) -> np.ndarray:
    payload = json.dumps(rng.bit_generator.state, sort_keys=True).encode()
    return np.frombuffer(payload, np.uint8).copy()


def _official_arrays(
    Replay,
    Consec,
    elements,
    primary: Mapping[str, Any],
    capacity_case: Mapping[str, Any],
    row_schema: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    elements.UUID.reset(debug=True)
    replay = Replay(
        length=primary["raw_length"],
        capacity=primary["capacity"],
        chunksize=primary["chunk_size"],
        online=primary["online"],
        seed=primary["seed"],
    )
    for index in range(primary["steps"]):
        replay.add(_row(index, primary, row_schema))

    collection_chunks = sorted(
        replay.chunks.values(), key=lambda chunk: int(chunk.uuid)
    )
    collection_deter = np.concatenate(
        [chunk.data["dyn/deter"][: chunk.length] for chunk in collection_chunks]
    )
    collection_stoch = np.concatenate(
        [chunk.data["dyn/stoch"][: chunk.length] for chunk in collection_chunks]
    )

    item_ids = np.asarray(list(replay.items), np.int64)
    start_chunks, start_offsets = _key_arrays(list(replay.items.values()))
    online_chunks, online_offsets = _key_arrays(list(replay.queue))
    train = replay.sample(primary["batch"], "train")
    report = replay.sample(primary["batch"], "report")
    stream = iter(
        Consec(
            [report],
            length=primary["sequence_length"],
            consec=primary["consecutive"],
            prefix=primary["context"],
            strict=True,
            contiguous=True,
        )
    )
    consecutive0 = next(stream)
    consecutive1 = next(stream)

    update_length = primary["raw_length"] - primary["context"]
    update = {"stepid": report["stepid"][:, primary["context"] :].copy()}
    for name, base in sorted(primary["latent_update_bases"].items()):
        shape = (primary["batch"], update_length, *row_schema[name]["shape"])
        update[name] = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        update[name] += base
    replay.update(update)
    chunks = sorted(replay.chunks.values(), key=lambda chunk: int(chunk.uuid))
    written_values = np.concatenate(
        [chunk.data["value"][: chunk.length] for chunk in chunks]
    )
    written_deter = np.concatenate(
        [chunk.data["dyn/deter"][: chunk.length] for chunk in chunks]
    )
    written_stoch = np.concatenate(
        [chunk.data["dyn/stoch"][: chunk.length] for chunk in chunks]
    )

    elements.UUID.reset(debug=True)
    capacity = Replay(
        length=capacity_case["raw_length"],
        capacity=capacity_case["capacity"],
        chunksize=capacity_case["chunk_size"],
        online=capacity_case["online"],
        seed=capacity_case["seed"],
    )
    checkpoints = {}
    for index in range(capacity_case["steps"]):
        capacity.add(_row(index, capacity_case, row_schema))
        if index in capacity_case["selector_checkpoints"]:
            checkpoints[index] = np.asarray(capacity.sampler.keys, np.int64)
    if set(checkpoints) != set(capacity_case["selector_checkpoints"]):
        raise ValueError("capacity selector checkpoints were not reached")

    capacity_item_ids = np.asarray(list(capacity.items), np.int64)
    capacity_start_chunks, capacity_start_offsets = _key_arrays(
        list(capacity.items.values())
    )
    capacity_queue_chunks, capacity_queue_offsets = _key_arrays(list(capacity.queue))
    capacity_chunk_ids = np.asarray(
        sorted(int(chunk_id) for chunk_id in capacity.chunks), np.int64
    )
    capacity_refs = np.asarray(
        [
            capacity.refs[next(cid for cid in capacity.refs if int(cid) == chunk_id)]
            for chunk_id in capacity_chunk_ids
        ],
        np.int64,
    )
    rng_before = _rng_bytes(capacity.sampler.rng)
    capacity_train = capacity.sample(capacity_case["batch"], "train")
    rng_after = _rng_bytes(capacity.sampler.rng)
    selector_draws = np.asarray(
        [capacity.sampler() for _ in range(capacity_case["selector_draws"])],
        np.int64,
    )

    arrays = {
        "capacity.chunk_ids": capacity_chunk_ids,
        "capacity.fifo": np.asarray(capacity.fifo, np.int64),
        "capacity.item_ids": capacity_item_ids,
        "capacity.queue_chunks": capacity_queue_chunks,
        "capacity.queue_offsets": capacity_queue_offsets,
        "capacity.refs": capacity_refs,
        "capacity.rng_after_online": rng_after,
        "capacity.rng_before_online": rng_before,
        "capacity.selector_draws": selector_draws,
        "capacity.selector_keys": np.asarray(capacity.sampler.keys, np.int64),
        "capacity.start_chunks": capacity_start_chunks,
        "capacity.start_offsets": capacity_start_offsets,
        "capacity.train_values": capacity_train["value"],
        "collection.deter": collection_deter,
        "collection.stoch": collection_stoch,
        "consecutive0.consec": consecutive0["consec"],
        "consecutive0.first": consecutive0["is_first"],
        "consecutive0.last": consecutive0["is_last"],
        "consecutive0.stepid": consecutive0["stepid"],
        "consecutive0.values": consecutive0["value"],
        "consecutive1.consec": consecutive1["consec"],
        "consecutive1.first": consecutive1["is_first"],
        "consecutive1.last": consecutive1["is_last"],
        "consecutive1.stepid": consecutive1["stepid"],
        "consecutive1.values": consecutive1["value"],
        "raw.item_ids": item_ids,
        "raw.online_chunks": online_chunks,
        "raw.online_offsets": online_offsets,
        "raw.report_first": report["is_first"],
        "raw.report_deter": report["dyn/deter"],
        "raw.report_last": report["is_last"],
        "raw.report_stepid": report["stepid"],
        "raw.report_stoch": report["dyn/stoch"],
        "raw.report_terminal": report["is_terminal"],
        "raw.report_values": report["value"],
        "raw.start_chunks": start_chunks,
        "raw.start_offsets": start_offsets,
        "raw.train_values": train["value"],
        "writeback.deter": written_deter,
        "writeback.logical_values": written_values,
        "writeback.stoch": written_stoch,
    }
    arrays.update(
        {
            f"capacity.intermediate{index}": value
            for index, value in sorted(checkpoints.items())
        }
    )
    return {name: np.asarray(value) for name, value in sorted(arrays.items())}


def _worker(envelope: Mapping[str, Any]) -> dict[str, Any]:
    _require_isolated_worker_modules()
    if set(envelope) != {"execution", "request"}:
        raise ValueError("replay worker invocation envelope keys changed")
    request = envelope["request"]
    execution = envelope["execution"]
    if not isinstance(request, Mapping) or not isinstance(execution, Mapping):
        raise ValueError("replay worker invocation envelope changed")
    if set(request) != _REQUEST_KEYS:
        raise ValueError("replay worker source request keys changed")
    if set(execution) != _EXECUTION_KEYS:
        raise ValueError("replay worker execution coordinate keys changed")
    if (
        request["case_name"] != "replay"
        or type(request["seed"]) is not int
        or request["seed"] != 7
    ):
        raise ValueError("replay worker source request coordinate changed")
    checkout = _request_path(
        execution,
        "official_checkout",
        boundary="replay worker execution",
    )
    revision = str(request["official_commit"])
    profile = DreamerProfile(request["profile"])
    mode = ObservationMode(request["observation_mode"])
    if mode is not ObservationMode.PROPRIO:
        raise ValueError("replay oracle only authorizes proprio fixtures")
    if revision != official_revision(profile):
        raise ValueError("replay worker revision does not match profile")
    if not _same_contract(request["overrides"], profile_overrides(profile)):
        raise ValueError("replay worker override map does not match profile")
    if request["source_spec"] != REPLAY_SOURCE_SPEC.name:
        raise ValueError("replay worker source spec mismatch")
    if request["compute_dtype"] != "float32":
        raise ValueError("replay worker requires float32 latent entries")
    if request["uuid_mode"] != "debug-counter":
        raise ValueError("replay worker UUID mode is not authorized")
    expected_runtime = _runtime_contract()
    if not _same_contract(request["runtime"], expected_runtime):
        raise ValueError("replay worker runtime provenance is not authorized")
    if not _same_contract(request["generator_files"], REPLAY_GENERATOR_FILE_HASHES):
        raise ValueError("replay worker generator file contract changed")
    _validate_live_generator_contract(request)
    package_dir = _request_path(
        execution,
        "elements_package_dir",
        boundary="replay worker execution",
    )
    dist_info = _request_path(
        execution,
        "elements_dist_info",
        boundary="replay worker execution",
    )
    elements_version, elements_hashes = _installed_elements_provenance(
        package_dir, dist_info
    )
    if elements_version != _ELEMENTS_VERSION:
        raise ValueError("replay worker Elements version changed")
    if elements_hashes != _ELEMENTS_HELPER_HASHES:
        raise ValueError("replay worker Elements helper files changed")
    request_python = _request_path(
        execution,
        "python_executable",
        boundary="replay worker execution",
        resolve=False,
    )
    if request_python != Path(sys.executable).absolute():
        raise ValueError("replay worker interpreter provenance changed")
    if not _same_contract(request["cases"], _case_contract(request["seed"])):
        raise ValueError("replay worker source cases are not authorized")
    if not _same_contract(request["row_schema"], _row_schema_contract()):
        raise ValueError("replay worker row schema is not authorized")

    sources = {}
    for path, digest in REPLAY_SOURCE_SPEC.hashes_for(revision).items():
        source = _git_show(checkout, revision, path)
        if _sha256_bytes(source) != digest:
            raise ValueError(f"official replay source hash mismatch: {path}")
        sources[path] = source
    defaults = yaml.safe_load(sources["dreamerv3/configs.yaml"])["defaults"]
    if defaults["replay"]["fracs"] != {
        "uniform": 1.0,
        "priority": 0.0,
        "recency": 0.0,
    }:
        raise ValueError("official replay selector defaults changed")
    if defaults["replay_context"] != 1 or defaults["replay"]["online"] is not True:
        raise ValueError("official replay context/online defaults changed")

    elements, Chunk, Uniform, Replay, Consec = _load_source_classes(sources, revision)
    attestation = _source_attestation(
        revision, elements, Chunk, Uniform, Replay, Consec
    )
    attestation["native_module_violations"] = _require_isolated_worker_modules()
    arrays = _official_arrays(
        Replay,
        Consec,
        elements,
        request["cases"]["primary"],
        request["cases"]["capacity"],
        request["row_schema"],
    )
    arrays.update(
        {
            "source_config.batch_length": np.asarray(
                defaults["batch_length"], np.int32
            ),
            "source_config.chunk_size": np.asarray(
                defaults["replay"]["chunksize"], np.int32
            ),
            "source_config.context": np.asarray(defaults["replay_context"], np.int32),
            "source_config.online": np.asarray(defaults["replay"]["online"], bool),
            "source_config.uniform": np.asarray(
                defaults["replay"]["fracs"]["uniform"], np.float32
            ),
        }
    )
    return {
        "arrays": {
            name: {"dtype": value.dtype.name, "values": value.tolist()}
            for name, value in sorted(arrays.items())
        },
        "compute_dtype": "float32",
        "runtime": expected_runtime,
        "source_attestation": attestation,
        "worker_pid": os.getpid(),
    }


def run_replay_case(
    harness: OracleHarness,
    profile: DreamerProfile | str,
    observation_mode: ObservationMode | str = ObservationMode.PROPRIO,
    *,
    elements_package_dir: str | Path,
    elements_dist_info: str | Path,
    case_name: str = "replay",
    seed: int = 7,
) -> tuple[Path, Path]:
    profile = DreamerProfile(profile)
    mode = ObservationMode(observation_mode)
    if mode is not ObservationMode.PROPRIO:
        raise ValueError("replay oracle only authorizes proprio fixtures")
    if case_name != "replay" or seed != 7:
        raise ValueError("replay oracle case name and seed are fixed")
    request = _stable_replay_request(profile, mode, case_name, seed)
    invocation = _build_replay_invocation(
        request,
        {
            "elements_dist_info": elements_dist_info,
            "elements_package_dir": elements_package_dir,
            "official_checkout": harness.official_checkout,
            "python_executable": harness.python_executable,
        },
    )
    completed = subprocess.run(
        invocation.command,
        cwd=invocation.cwd,
        input=invocation.generator_request,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if payload["compute_dtype"] != "float32":
        raise ValueError("replay worker reported the wrong dtype")
    if payload["runtime"] != request["runtime"]:
        raise ValueError("replay worker runtime attestation changed")
    attestation = payload["source_attestation"]
    if attestation.get("native_module_violations") != []:
        raise ValueError("replay worker native module attestation changed")
    if attestation["classes"] != ["Chunk", "Consec", "Replay", "Uniform"]:
        raise ValueError("replay worker source class attestation changed")
    expected_bindings = {
        "chunk.elements",
        "chunk.numpy",
        "consec.numpy",
        "replay.chunk",
        "replay.elements",
        "replay.numpy",
        "replay.uniform",
        "uniform.numpy",
    }
    if set(attestation["bindings"]) != expected_bindings:
        raise ValueError("replay worker source binding attestation changed")
    for name, filename in attestation["method_origins"].items():
        class_name = name.split(".", 1)[0]
        source_path = {
            "Chunk": "embodied/core/chunk.py",
            "Consec": "embodied/core/streams.py",
            "Replay": "embodied/core/replay.py",
            "Uniform": "embodied/core/selectors.py",
        }[class_name]
        if filename != f"{official_revision(profile)}:{source_path}":
            raise ValueError("replay worker method origin attestation changed")
    worker_pid = int(payload["worker_pid"])
    if worker_pid <= 0 or worker_pid == os.getpid():
        raise ValueError("replay oracle worker did not cross a process boundary")
    harness._last_worker_pid = worker_pid
    arrays = {
        name: np.asarray(spec["values"], dtype=spec["dtype"])
        for name, spec in payload["arrays"].items()
    }
    return harness.write_fixture(
        case_name=case_name,
        profile=profile,
        observation_mode=mode,
        arrays=arrays,
        seed=seed,
        generator_command=REPLAY_COMMAND_DESCRIPTOR,
        generator_request=request,
        source_spec=REPLAY_SOURCE_SPEC,
        dtype="float32",
    )


def _main(argv: Sequence[str]) -> int:
    if tuple(argv) != ("_worker",):
        raise SystemExit("replay_oracle.py is an internal fixture worker")
    request = json.loads(sys.stdin.read())
    sys.stdout.write(json.dumps(_worker(request), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess boundary.
    raise SystemExit(_main(sys.argv[1:]))


__all__ = ["REPLAY_SOURCE_SPEC", "run_replay_case"]
