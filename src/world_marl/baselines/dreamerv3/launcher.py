"""Subprocess launchers for the pinned upstream DreamerV3 implementation."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from world_marl.baselines.dreamerv3.artifacts import normalize_training_artifacts
from world_marl.baselines.dreamerv3.config import (
  OFFICIAL_DREAMERV3_COMMIT,
  DreamerV3RunSpec,
)


def timestamp() -> str:
  return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def upstream_revision(upstream_root: str | Path) -> str:
  result = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=Path(upstream_root),
    check=True,
    capture_output=True,
    text=True,
  )
  return result.stdout.strip()


def verify_upstream(upstream_root: str | Path) -> str:
  upstream_root = Path(upstream_root)
  main = upstream_root / "dreamerv3" / "main.py"
  if not main.is_file():
    raise FileNotFoundError(
      f"DreamerV3 upstream checkout is missing at {upstream_root}. "
      "Run 'git submodule update --init --recursive'."
    )
  revision = upstream_revision(upstream_root)
  if revision != OFFICIAL_DREAMERV3_COMMIT:
    raise RuntimeError(
      f"DreamerV3 revision mismatch: expected {OFFICIAL_DREAMERV3_COMMIT}, "
      f"found {revision}"
    )
  status = subprocess.run(
    ["git", "status", "--porcelain"],
    cwd=upstream_root,
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()
  if status:
    raise RuntimeError(
      "DreamerV3 checkout has local modifications; refusing to run a "
      f"non-canonical baseline:\n{status}"
    )
  return revision


def _write_json(path: Path, payload: Any) -> None:
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _process_environment(spec: DreamerV3RunSpec) -> dict[str, str]:
  env = os.environ.copy()
  if "MUJOCO_GL" not in env:
    env["MUJOCO_GL"] = "glfw" if platform.system() == "Darwin" else "egl"
  env["PYTHONUNBUFFERED"] = "1"
  if spec.wandb_project:
    env["WANDB_PROJECT"] = spec.wandb_project
  if spec.wandb_entity:
    env["WANDB_ENTITY"] = spec.wandb_entity
  return env


def run_training(
  spec: DreamerV3RunSpec,
  *,
  resume: bool = False,
  dry_run: bool = False,
) -> int:
  """Launch official DreamerV3 and normalize its online score artifacts."""
  revision = verify_upstream(spec.upstream_root)
  experiment_dir = spec.experiment_dir.resolve()
  upstream_logdir = spec.upstream_logdir.resolve()
  if upstream_logdir.exists() and not resume:
    raise FileExistsError(
      f"upstream logdir already exists: {upstream_logdir}; pass resume=True "
      "only when intentionally resuming the exact same configuration"
    )
  experiment_dir.mkdir(parents=True, exist_ok=True)
  metadata = {
    **spec.to_dict(),
    "created_at": timestamp(),
    "verified_upstream_commit": revision,
    "host_platform": platform.platform(),
    "resume": resume,
    "dry_run": dry_run,
  }
  _write_json(experiment_dir / "launch.json", metadata)
  if dry_run:
    print(" ".join(spec.command))
    return 0

  if not spec.python.exists():
    raise FileNotFoundError(
      f"DreamerV3 Python executable not found: {spec.python}. "
      "Run world-marl-setup-dreamerv3 first or pass --python."
    )

  log_path = experiment_dir / "process.log"
  with log_path.open("a", encoding="utf-8") as log:
    process = subprocess.Popen(
      spec.command,
      cwd=spec.upstream_root,
      env=_process_environment(spec),
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1,
    )
    assert process.stdout is not None
    try:
      for line in process.stdout:
        sys.stdout.write(line)
        log.write(line)
        log.flush()
      returncode = process.wait()
    except KeyboardInterrupt:
      process.terminate()
      returncode = process.wait()

  summary = normalize_training_artifacts(
    experiment_dir,
    upstream_root=spec.upstream_root,
    task=spec.task,
    seed=spec.seed,
    train_steps_budget=spec.train_steps,
  )
  _write_json(
    experiment_dir / "outcome.json",
    {
      "returncode": returncode,
      "completed": returncode == 0,
      "training_summary": summary,
    },
  )
  return returncode
