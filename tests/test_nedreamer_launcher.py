from __future__ import annotations

import json
import sys
from pathlib import Path

from dreamarl.baselines.nedreamer.artifacts import normalize_training_artifacts
from dreamarl.baselines.nedreamer.config import (
    OFFICIAL_NEDREAMER_COMMIT,
    NEDreamerRunSpec,
    default_upstream_root,
)
from dreamarl.baselines.nedreamer.environment import resolved_requirements
from dreamarl.baselines.nedreamer.launcher import run_training, verify_upstream


def test_pinned_nedreamer_checkout_is_clean():
    assert verify_upstream(default_upstream_root()) == OFFICIAL_NEDREAMER_COMMIT


def test_command_preserves_official_method_and_records_transition_accounting(tmp_path):
    spec = NEDreamerRunSpec(
        experiment_dir=tmp_path / "run",
        python=Path(sys.executable),
        train_steps=1_100_000,
        device="cuda:1",
    )
    assert "model.rep_loss=ne_dreamer" in spec.command
    assert "device=cuda:1" in spec.command
    assert "env.steps=1100000" in spec.command
    assert "trainer.steps=1100000" in spec.command
    metadata = spec.to_dict()
    assert metadata["train_real_transition_budget"] == 1_100_000
    assert metadata["native_action_repeat"] == 2
    assert metadata["environment_decision_budget"] == 550_000


def test_dry_run_writes_source_pin_and_full_command(tmp_path):
    spec = NEDreamerRunSpec(
        experiment_dir=tmp_path / "run",
        python=Path(sys.executable),
        device="cpu",
    )
    assert run_training(spec, dry_run=True) == 0
    launch = json.loads((spec.experiment_dir / "launch.json").read_text())
    assert launch["verified_upstream_commit"] == OFFICIAL_NEDREAMER_COMMIT
    assert launch["observation_mode"] == "vision"
    assert launch["evaluation_real_transition_budget"] == 0


def test_normalizer_separates_training_and_periodic_evaluation(tmp_path):
    upstream = tmp_path / "run" / "official"
    upstream.mkdir(parents=True)
    rows = [
        {"step": 100, "episode/score": 5.0},
        {"step": 100, "episode/eval_score": 7.0},
        {"step": 200, "episode/score": 9.0},
    ]
    (upstream / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    summary = normalize_training_artifacts(
        tmp_path / "run",
        upstream_logdir=upstream,
        task="dmc_walker_walk",
        seed=0,
        train_steps_budget=200,
    )
    assert summary["online_training_episodes"]["mean"] == 7.0
    assert summary["periodic_deterministic_evaluations"]["mean"] == 7.0


def test_environment_requirements_match_pinned_upstream_file():
    requirements = resolved_requirements(default_upstream_root())
    assert "torch==2.8.0" in requirements
    assert "dm_control==1.0.9" in requirements
