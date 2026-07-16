"""Checkpoint save/load helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flax import serialization
from flax.training.train_state import TrainState


def save_checkpoint(
    checkpoint_dir: str | Path,
    train_state: TrainState,
    *,
    metadata: dict[str, Any],
) -> Path:
    """Save train-state parameters and metadata."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    params_path = checkpoint_path / "checkpoint.msgpack"
    metadata_path = checkpoint_path / "metadata.json"
    params_tmp = checkpoint_path / ".checkpoint.msgpack.tmp"
    metadata_tmp = checkpoint_path / ".metadata.json.tmp"
    params_tmp.write_bytes(serialization.to_bytes(train_state.params))
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    params_tmp.replace(params_path)
    metadata_tmp.replace(metadata_path)
    return params_path


def load_params(checkpoint_file: str | Path, target_params):
    """Load parameters into a target pytree with the same structure."""
    return serialization.from_bytes(target_params, Path(checkpoint_file).read_bytes())


def save_train_state(checkpoint_file: str | Path, train_state: Any) -> Path:
    """Persist a complete learner state, including optimizer and target state."""

    checkpoint_path = Path(checkpoint_file)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    temporary_path.write_bytes(serialization.to_bytes(train_state))
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def load_train_state(checkpoint_file: str | Path, target_train_state: Any):
    """Restore a complete learner state into a matching initialized template."""

    return serialization.from_bytes(
        target_train_state,
        Path(checkpoint_file).read_bytes(),
    )


def save_pytree(checkpoint_file: str | Path, tree: Any) -> Path:
    """Persist an auxiliary pytree such as an EMA policy bundle."""

    checkpoint_path = Path(checkpoint_file)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    temporary_path.write_bytes(serialization.to_bytes(tree))
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def load_pytree(checkpoint_file: str | Path, target_tree: Any):
    """Restore an auxiliary pytree into a matching template."""

    return serialization.from_bytes(target_tree, Path(checkpoint_file).read_bytes())


def load_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Load checkpoint metadata."""
    metadata_path = Path(checkpoint_dir) / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


# TODO: add checkpointing for world model
