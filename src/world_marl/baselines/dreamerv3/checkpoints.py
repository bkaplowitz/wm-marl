"""Read-only helpers for upstream Elements checkpoint collections."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UpstreamCheckpoint:
  path: Path
  env_steps: int | None


def latest_checkpoint(checkpoint_root: str | Path) -> UpstreamCheckpoint:
  root = Path(checkpoint_root)
  pointer = root / "latest"
  if not pointer.is_file():
    raise FileNotFoundError(f"checkpoint pointer is missing: {pointer}")
  path = root / pointer.read_text(encoding="utf-8").strip()
  if not (path / "done").is_file():
    raise FileNotFoundError(f"latest checkpoint is incomplete: {path}")
  step_path = path / "step.pkl"
  env_steps = None
  if step_path.is_file():
    with step_path.open("rb") as handle:
      env_steps = int(pickle.load(handle))
  return UpstreamCheckpoint(path=path, env_steps=env_steps)
