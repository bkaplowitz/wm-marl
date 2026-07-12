from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pytest

from world_marl.baselines.dreamerv3.config import (
  OFFICIAL_DREAMERV3_COMMIT,
  DreamerV3RunSpec,
  default_upstream_root,
)
from world_marl.baselines.dreamerv3.evaluation import (
  DreamerV3EvaluationSpec,
  evaluation_command,
  run_evaluation,
)
from world_marl.baselines.dreamerv3.launcher import (
  run_training,
  verify_upstream,
)


def test_pinned_upstream_checkout_is_present():
  assert verify_upstream(default_upstream_root()) == OFFICIAL_DREAMERV3_COMMIT


def test_canonical_command_uses_only_official_config_and_explicit_budget(tmp_path):
  spec = DreamerV3RunSpec(
    experiment_dir=tmp_path / "run",
    python=Path(sys.executable),
    platform="cpu",
  )
  command = spec.command
  assert Path(command[0]).is_absolute()
  assert command[1].endswith("external/dreamerv3/dreamerv3/main.py")
  assert command[command.index("--configs") + 1] == "dmc_proprio"
  assert command[command.index("--task") + 1] == "dmc_reacher_easy"
  assert command[command.index("--run.steps") + 1] == "500000"
  assert "--agent.imag_length" not in command
  assert "--batch_size" not in command


def test_wandb_is_an_upstream_logger_override(tmp_path):
  spec = DreamerV3RunSpec(
    experiment_dir=tmp_path / "run",
    python=Path(sys.executable),
    platform="cpu",
    wandb_project="world-marl",
  )
  index = spec.command.index("--logger.outputs")
  assert spec.command[index + 1:index + 4] == ["jsonl", "scope", "wandb"]


def test_python_path_is_not_dereferenced_out_of_virtualenv(tmp_path, monkeypatch):
  target = tmp_path / "base-python"
  target.touch()
  virtualenv_python = tmp_path / "venv" / "bin" / "python"
  virtualenv_python.parent.mkdir(parents=True)
  virtualenv_python.symlink_to(target)
  monkeypatch.chdir(tmp_path)
  spec = DreamerV3RunSpec(
    experiment_dir=tmp_path / "run",
    python=Path("venv/bin/python"),
    platform="cpu",
  )
  assert spec.command[0] == str(virtualenv_python)


def test_dry_run_writes_complete_launch_metadata(tmp_path):
  spec = DreamerV3RunSpec(
    experiment_dir=tmp_path / "run",
    python=Path(sys.executable),
    platform="cpu",
  )
  assert run_training(spec, dry_run=True) == 0
  launch = json.loads((spec.experiment_dir / "launch.json").read_text())
  assert launch["verified_upstream_commit"] == OFFICIAL_DREAMERV3_COMMIT
  assert launch["train_env_steps_budget"] == 500_000
  assert launch["eval_env_steps_budget"] == 0
  assert launch["total_real_env_steps_budget"] == 500_000


def test_evaluation_command_uses_latest_checkpoint_and_separate_budget(tmp_path):
  train_spec = DreamerV3RunSpec(
    experiment_dir=tmp_path / "run",
    python=Path(sys.executable),
    platform="cpu",
  )
  run_training(train_spec, dry_run=True)
  checkpoint_root = train_spec.upstream_logdir / "ckpt"
  checkpoint = checkpoint_root / "checkpoint-123"
  checkpoint.mkdir(parents=True)
  (checkpoint / "done").touch()
  with (checkpoint / "step.pkl").open("wb") as handle:
    pickle.dump(123, handle)
  (checkpoint_root / "latest").write_text(checkpoint.name)
  eval_spec = DreamerV3EvaluationSpec(
    experiment_dir=train_spec.experiment_dir,
    episodes=20,
    envs=4,
  )
  eval_dir = train_spec.experiment_dir / "evaluation" / "test"
  command, launch = evaluation_command(eval_spec, eval_dir=eval_dir)
  assert command[command.index("--script") + 1] == "eval_only"
  assert command[command.index("--run.from_checkpoint") + 1] == str(checkpoint)
  assert command[command.index("--run.steps") + 1] == "20020"
  assert launch["train_env_steps_budget"] == 500_000
  assert launch["checkpoint_train_env_steps"] == 123


def test_evaluation_refuses_missing_checkpoint(tmp_path):
  train_spec = DreamerV3RunSpec(
    experiment_dir=tmp_path / "run",
    python=Path(sys.executable),
    platform="cpu",
  )
  run_training(train_spec, dry_run=True)
  with pytest.raises(FileNotFoundError, match="checkpoint pointer is missing"):
    run_evaluation(
      DreamerV3EvaluationSpec(experiment_dir=train_spec.experiment_dir),
      dry_run=True,
    )
