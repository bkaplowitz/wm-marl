"""Fixed-checkpoint evaluation through the official DreamerV3 evaluator."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from world_marl.baselines.dreamerv3.artifacts import normalize_evaluation_artifacts
from world_marl.baselines.dreamerv3.checkpoints import latest_checkpoint
from world_marl.baselines.dreamerv3.config import absolute_path, default_dreamerv3_python
from world_marl.baselines.dreamerv3.launcher import timestamp, verify_upstream


@dataclass(frozen=True)
class DreamerV3EvaluationSpec:
  experiment_dir: Path
  episodes: int = 20
  envs: int = 4
  episode_length: int = 1_000
  eval_seed: int = 10_000
  success_threshold: float | None = None
  python: Path | None = None

  def __post_init__(self) -> None:
    object.__setattr__(
      self, "experiment_dir", Path(self.experiment_dir).expanduser().resolve()
    )
    if self.python is not None:
      object.__setattr__(self, "python", absolute_path(self.python))
    if self.episodes < 1 or self.envs < 1 or self.episode_length < 1:
      raise ValueError("episodes, envs, and episode_length must be >= 1")

  @property
  def eval_steps(self) -> int:
    rounds = (self.episodes + self.envs - 1) // self.envs
    return rounds * self.envs * (self.episode_length + 1)


def _read_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def evaluation_command(
  spec: DreamerV3EvaluationSpec,
  *,
  eval_dir: Path,
) -> tuple[list[str], dict[str, Any]]:
  launch = _read_json(spec.experiment_dir / "launch.json")
  upstream_root = Path(launch["upstream_root"])
  python = spec.python or Path(launch.get("python", default_dreamerv3_python()))
  checkpoint = latest_checkpoint(spec.experiment_dir / "upstream" / "ckpt")
  command = [
    str(python),
    str(upstream_root / "dreamerv3" / "main.py"),
    "--logdir",
    str(eval_dir / "upstream"),
    "--configs",
    *launch["configs"],
    "--task",
    launch["task"],
    "--seed",
    str(spec.eval_seed),
    "--script",
    "eval_only",
    "--run.from_checkpoint",
    str(checkpoint.path),
    "--run.steps",
    str(spec.eval_steps),
    "--run.envs",
    str(spec.envs),
    "--jax.platform",
    launch["platform"],
  ]
  launch = {
    **launch,
    "evaluated_checkpoint": str(checkpoint.path),
    "checkpoint_train_env_steps": checkpoint.env_steps,
  }
  return command, launch


def run_evaluation(
  spec: DreamerV3EvaluationSpec,
  *,
  dry_run: bool = False,
) -> tuple[int, Path]:
  eval_dir = spec.experiment_dir / "evaluation" / (
    f"latest_{spec.episodes}eps_seed{spec.eval_seed}"
  )
  if eval_dir.exists():
    raise FileExistsError(f"evaluation directory already exists: {eval_dir}")
  command, launch = evaluation_command(spec, eval_dir=eval_dir)
  verify_upstream(launch["upstream_root"])
  eval_dir.mkdir(parents=True)
  metadata = {
    "created_at": timestamp(),
    "checkpoint_policy": "latest periodic upstream checkpoint",
    "checkpoint": launch["evaluated_checkpoint"],
    "checkpoint_train_env_steps": launch["checkpoint_train_env_steps"],
    "training_experiment": str(spec.experiment_dir.resolve()),
    "episodes": spec.episodes,
    "envs": spec.envs,
    "episode_length": spec.episode_length,
    "eval_steps_budget": spec.eval_steps,
    "eval_seed": spec.eval_seed,
    "success_threshold": spec.success_threshold,
    "command": command,
    "dry_run": dry_run,
  }
  _write_json(eval_dir / "evaluation_launch.json", metadata)
  if dry_run:
    print(" ".join(command))
    return 0, eval_dir

  env = os.environ.copy()
  if "MUJOCO_GL" not in env:
    env["MUJOCO_GL"] = "glfw" if platform.system() == "Darwin" else "egl"
  with (eval_dir / "process.log").open("w", encoding="utf-8") as log:
    process = subprocess.Popen(
      command,
      cwd=launch["upstream_root"],
      env=env,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
      sys.stdout.write(line)
      log.write(line)
      log.flush()
    returncode = process.wait()
  summary = normalize_evaluation_artifacts(
    eval_dir,
    requested_episodes=spec.episodes,
    train_env_steps=int(
      launch["checkpoint_train_env_steps"]
      if launch["checkpoint_train_env_steps"] is not None
      else launch["train_env_steps_budget"]
    ),
    success_threshold=spec.success_threshold,
  )
  _write_json(
    eval_dir / "outcome.json",
    {
      "returncode": returncode,
      "completed": (
        returncode == 0 and summary["completed_episodes"] == spec.episodes
      ),
      "summary": summary,
    },
  )
  if returncode == 0 and summary["completed_episodes"] < spec.episodes:
    returncode = 2
  return returncode, eval_dir
