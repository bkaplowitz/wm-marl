"""Independent RNG streams and stable fingerprints for JEPA experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np


JAX_STREAM_IDS = {
    "initialization": 1,
    "world_model": 2,
    "policy": 3,
    "evaluation": 4,
}

NUMPY_STREAM_IDS = {
    "initial_collection": 11,
    "online_collection": 12,
    "online_validation_collection": 13,
    "world_model_replay": 14,
    "policy_replay": 15,
    "validation_replay": 16,
}


@dataclass
class JaxRngStreams:
    """Own JAX keys by subsystem, with an opt-in legacy shared mode."""

    seed: int
    isolated: bool
    _keys: dict[str, jax.Array]

    @classmethod
    def create(cls, seed: int, *, isolated: bool) -> "JaxRngStreams":
        root = jax.random.PRNGKey(seed)
        if isolated:
            keys = {
                name: jax.random.fold_in(root, stream_id)
                for name, stream_id in JAX_STREAM_IDS.items()
            }
        else:
            keys = {"shared": root}
        return cls(seed=int(seed), isolated=bool(isolated), _keys=keys)

    def _slot(self, name: str) -> str:
        if name not in JAX_STREAM_IDS:
            raise KeyError(f"unknown JAX RNG stream: {name}")
        return name if self.isolated else "shared"

    def take(self, name: str) -> jax.Array:
        """Advance a stream and return one subkey."""

        slot = self._slot(name)
        self._keys[slot], subkey = jax.random.split(self._keys[slot])
        return subkey

    def current(self, name: str) -> jax.Array:
        """Return the current key for a stateful training routine."""

        return self._keys[self._slot(name)]

    def update(self, name: str, key: jax.Array) -> None:
        """Store the key returned by a stateful training routine."""

        self._keys[self._slot(name)] = key

    def manifest(self) -> dict[str, Any]:
        return {
            "mode": "isolated" if self.isolated else "legacy-shared",
            "base_seed": self.seed,
            "jax_stream_ids": dict(JAX_STREAM_IDS) if self.isolated else {},
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible snapshot of every JAX RNG stream."""

        return {
            "seed": self.seed,
            "isolated": self.isolated,
            "keys": {
                name: np.asarray(jax.device_get(key), dtype=np.uint32).tolist()
                for name, key in self._keys.items()
            },
        }

    def restore_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore stream keys from :meth:`state_dict`."""

        if int(state["seed"]) != self.seed or bool(state["isolated"]) != self.isolated:
            raise ValueError("JAX RNG snapshot does not match the configured streams")
        expected_slots = set(self._keys)
        restored_slots = set(state["keys"])
        if restored_slots != expected_slots:
            raise ValueError(
                "JAX RNG snapshot slots do not match: "
                f"expected {sorted(expected_slots)}, got {sorted(restored_slots)}"
            )
        self._keys = {
            name: jnp.asarray(value, dtype=jnp.uint32)
            for name, value in state["keys"].items()
        }


@dataclass
class NumpyRngStreams:
    """Own deterministic NumPy generators by data-producing subsystem."""

    seed: int
    isolated: bool
    _generators: dict[str, np.random.Generator]

    @classmethod
    def create(cls, seed: int, *, isolated: bool) -> "NumpyRngStreams":
        if isolated:
            generators = {
                name: np.random.default_rng(
                    np.random.SeedSequence([int(seed), stream_id])
                )
                for name, stream_id in NUMPY_STREAM_IDS.items()
            }
        else:
            generators = {"shared": np.random.default_rng(seed)}
        return cls(seed=int(seed), isolated=bool(isolated), _generators=generators)

    def get(self, name: str) -> np.random.Generator:
        if name not in NUMPY_STREAM_IDS:
            raise KeyError(f"unknown NumPy RNG stream: {name}")
        return self._generators[name if self.isolated else "shared"]

    def manifest(self) -> dict[str, Any]:
        derived_seeds = {}
        if self.isolated:
            derived_seeds = {
                name: np.random.SeedSequence([self.seed, stream_id])
                .generate_state(2)
                .tolist()
                for name, stream_id in NUMPY_STREAM_IDS.items()
            }
        return {
            "mode": "isolated" if self.isolated else "legacy-shared",
            "base_seed": self.seed,
            "numpy_stream_ids": dict(NUMPY_STREAM_IDS) if self.isolated else {},
            "numpy_derived_seeds": derived_seeds,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible snapshot of every NumPy RNG stream."""

        return {
            "seed": self.seed,
            "isolated": self.isolated,
            "bit_generators": {
                name: _json_compatible(generator.bit_generator.state)
                for name, generator in self._generators.items()
            },
        }

    def restore_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore generator states from :meth:`state_dict`."""

        if int(state["seed"]) != self.seed or bool(state["isolated"]) != self.isolated:
            raise ValueError("NumPy RNG snapshot does not match the configured streams")
        expected_slots = set(self._generators)
        restored_slots = set(state["bit_generators"])
        if restored_slots != expected_slots:
            raise ValueError(
                "NumPy RNG snapshot slots do not match: "
                f"expected {sorted(expected_slots)}, got {sorted(restored_slots)}"
            )
        for name, bit_generator_state in state["bit_generators"].items():
            self._generators[name].bit_generator.state = bit_generator_state


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def fingerprint_arrays(arrays: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for named array-like values."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.asarray(jax.device_get(arrays[name]))
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def fingerprint_pytree(tree: Any) -> str:
    """Return a stable SHA-256 digest for a JAX parameter pytree."""

    leaves, treedef = jax.tree_util.tree_flatten(tree)
    digest = hashlib.sha256(str(treedef).encode("utf-8"))
    for index, leaf in enumerate(leaves):
        value = np.asarray(jax.device_get(leaf))
        digest.update(np.asarray(index, dtype=np.int64).tobytes())
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()
