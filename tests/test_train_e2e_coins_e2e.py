"""End-to-end smoke test for the CoinGame + prefit-world-model training loop.

Runs the *whole* ``run_training`` pipeline at tiny scale: random/policy
transition collection, world-model fit, the imagined model-rollout PPO loop,
checkpoint save, and the coin-specific reload-evaluation subprocess. It asserts
the loop **completes** and writes its artifacts -- not that it beats the
``min_improvement`` gate. With a world model prefit on a single random + single
policy rollout and soft states fed back over the horizon, imagined returns are
expected to drift, so the run may legitimately *run yet fail* its success gate.
Completion is the contract under test here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from world_marl.scripts import train_e2e


def _tiny_coins_args(out_dir: Path, monkeypatch) -> argparse.Namespace:
    argv = [
        "train_e2e",
        "--substrate", "coins",
        "--prefit-world-model",
        "--algorithm", "ippo",
        "--num-envs", "1",
        "--rollout-steps", "4",
        "--total-env-steps", "8",
        "--num-runs", "1",
        "--max-cycles", "4",
        "--negative-control", "none",
        "--eval-episodes", "1",
        "--eval-max-steps", "4",
        "--num-minibatches", "1",
        "--update-epochs", "1",
        "--wm-random-rollouts", "1",
        "--wm-initial-rollouts", "1",
        "--wm-fit-steps", "2",
        "--wm-integration-steps", "1",
        "--wm-hidden-dim", "8",
        "--out-dir", str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return train_e2e.parse_args()


def test_coins_prefit_run_training_completes_and_writes_artifacts(tmp_path, monkeypatch):
    args = _tiny_coins_args(tmp_path, monkeypatch)
    run_dir = tmp_path / "run_000"

    outcome = train_e2e.run_training(
        args,
        run_dir=run_dir,
        name="run_000",
        run_index=0,
        control=None,
    )

    # Completed without raising and produced a finite trained-return estimate
    # (the reload subprocess parsed and returned a number).
    assert outcome.checkpoint_dir == str(run_dir / "checkpoint")
    assert (run_dir / "checkpoint").is_dir()
    assert (run_dir / "checkpoint" / "checkpoint.msgpack").exists()
    # The coin-specific reload-eval subprocess ran and its JSON was persisted.
    assert (run_dir / "reload_evaluation.json").exists()
    assert (run_dir / "outcome.json").exists()
    assert (run_dir / "world_model_prefit.json").exists()
