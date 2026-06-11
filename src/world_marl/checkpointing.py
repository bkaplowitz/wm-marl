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
  params_path.write_bytes(serialization.to_bytes(train_state.params))
  metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
  return params_path


def load_params(checkpoint_file: str | Path, target_params):
  """Load parameters into a target pytree with the same structure."""
  return serialization.from_bytes(target_params, Path(checkpoint_file).read_bytes())


def load_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
  """Load checkpoint metadata."""
  metadata_path = Path(checkpoint_dir) / "metadata.json"
  return json.loads(metadata_path.read_text(encoding="utf-8"))
