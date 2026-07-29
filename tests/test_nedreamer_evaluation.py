from __future__ import annotations

import sys
from pathlib import Path

from world_marl.baselines.nedreamer.config import NEDreamerRunSpec
from world_marl.baselines.nedreamer.evaluation import (
    NEDreamerEvaluationSpec,
    evaluation_command,
)
from world_marl.baselines.nedreamer.launcher import run_training


def test_evaluation_uses_final_checkpoint_and_fixed_protocol(tmp_path):
    train = NEDreamerRunSpec(
        experiment_dir=tmp_path / "run",
        python=Path(sys.executable),
        device="cpu",
    )
    run_training(train, dry_run=True)
    train.upstream_logdir.mkdir()
    (train.upstream_logdir / "latest.pt").touch()
    spec = NEDreamerEvaluationSpec(
        experiment_dir=train.experiment_dir,
        episodes=20,
        eval_seed=10_000,
        device="cuda:0",
    )
    command, launch = evaluation_command(spec, eval_dir=tmp_path / "eval")
    assert command[command.index("--checkpoint") + 1].endswith("latest.pt")
    assert command[command.index("--episodes") + 1] == "20"
    assert command[command.index("--eval-seed") + 1] == "10000"
    assert launch["train_real_transition_budget"] == 1_100_000
