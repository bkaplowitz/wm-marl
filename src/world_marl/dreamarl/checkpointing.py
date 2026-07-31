"""Atomic exact-resume checkpoints for DreaMARL training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.checkpointing import (
    load_train_state,
    save_train_state,
)
from world_marl.dreamarl.learner import DreaMARLLearnerState
from world_marl.dreamarl.replay import JointSequenceReplay
from world_marl.dreamarl.runtime import DriverState, PolicyContext
from world_marl.logging import to_jsonable


def save_dreamarl_checkpoint(
    directory: str | Path,
    *,
    learner_state: DreaMARLLearnerState,
    driver: DriverState,
    policy_context: PolicyContext,
    replay: JointSequenceReplay,
    metadata: dict[str, Any],
) -> Path:
    """Persist learner, runtime, replay, and accounting under one directory."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    save_train_state(root / "learner.msgpack", learner_state)
    _save_array_tree(root / "runtime.npz", (driver, policy_context))
    replay.save(root)
    temporary = root / ".metadata.json.tmp"
    temporary.write_text(
        json.dumps(to_jsonable(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(root / "metadata.json")
    return root


def load_dreamarl_checkpoint(
    directory: str | Path,
    *,
    learner_template: DreaMARLLearnerState,
    driver_template: DriverState,
    policy_context_template: PolicyContext,
    replay: JointSequenceReplay,
) -> tuple[
    DreaMARLLearnerState,
    DriverState,
    PolicyContext,
    dict[str, Any],
]:
    """Restore a checkpoint into matching initialized templates."""

    root = Path(directory)
    replay.load(root)
    learner = jax.tree.map(
        _restore_device_leaf,
        load_train_state(root / "learner.msgpack", learner_template),
    )
    driver, context = _load_array_tree(
        root / "runtime.npz", (driver_template, policy_context_template)
    )
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    return learner, driver, context, metadata


def _restore_device_leaf(value: Any) -> Any:
    """Move serialized learner arrays back to JAX without touching metadata."""

    if isinstance(value, (np.ndarray, np.generic)):
        return jnp.asarray(value)
    return value


def _save_array_tree(path: Path, tree: Any) -> None:
    leaves = jax.tree.leaves(tree)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{
                f"leaf_{index:05d}": np.asarray(value)
                for index, value in enumerate(leaves)
            },
        )
    temporary.replace(path)


def _load_array_tree(path: Path, template: Any) -> Any:
    template_leaves, structure = jax.tree.flatten(template)
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(template_leaves):
            raise ValueError("runtime checkpoint does not match template")
        leaves = [
            jnp.asarray(
                archive[f"leaf_{index:05d}"],
                dtype=getattr(target, "dtype", None),
            )
            for index, target in enumerate(template_leaves)
        ]
    return jax.tree.unflatten(structure, leaves)
