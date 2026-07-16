"""End-to-end CLI smoke test for CEM-MPC (brax:reacher, minimal budgets).

Skipped automatically when brax is not importable (pytest.importorskip).
Calls ``main(argv)`` in-process and asserts on the written artifacts.
"""

from __future__ import annotations

import json

import numpy as np
import pytest


def test_cem_end_to_end_smoke(tmp_path):
    pytest.importorskip("brax")
    from world_marl.scripts.train_single_genwm import main

    exit_code = main(
        [
            "--env",
            "brax:reacher",
            "--arm",
            "continuous-transformer",
            "--policy-optimizer",
            "cem",
            "--num-envs",
            "2",
            "--max-cycles",
            "20",
            "--collect-steps",
            "8",
            "--train-steps",
            "2",
            "--policy-train-steps",
            "1",
            "--online-iterations",
            "0",
            "--eval-episodes",
            "1",
            "--batch-size",
            "4",
            "--model-dim",
            "16",
            "--num-layers",
            "1",
            "--num-heads",
            "2",
            "--obs-bins",
            "5",
            "--integration-steps",
            "2",
            "--cem-samples",
            "6",
            "--cem-topk",
            "2",
            "--cem-iters",
            "2",
            "--cem-horizon",
            "2",
            "--allow-fail",
            "--quiet",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    run_dirs = sorted(tmp_path.glob("genwm_*/run_00"))
    assert run_dirs, "run directory not created"
    outcome = json.loads((run_dirs[0] / "outcome.json").read_text())
    assert outcome["policy_optimizer"] == "cem"
    assert outcome["planner_metrics"]["num_solves"] > 0
    assert np.isfinite(outcome["policy_trained_mean"])
    assert (run_dirs[0] / "planner.json").exists()
    assert outcome["ppo_final_metrics"] == {}
