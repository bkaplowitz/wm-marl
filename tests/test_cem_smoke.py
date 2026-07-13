"""End-to-end CLI smoke test for CEM-MPC (brax:reacher, minimal budgets).

Skipped automatically when brax is not importable (pytest.importorskip).
Verifies that the full train_single_genwm pipeline:
  1. Exits 0 with --allow-fail
  2. Writes planner.json with the expected structure
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_cem_smoke_e2e():
    pytest.importorskip("brax")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        cmd = [
            sys.executable,
            "-m",
            "world_marl.scripts.train_single_genwm",
            "--env",
            "brax:reacher",
            "--arm",
            "discrete-transformer",
            "--policy-optimizer",
            "cem",
            "--num-envs",
            "2",
            "--collect-steps",
            "4",
            "--train-steps",
            "2",
            "--online-iterations",
            "0",
            "--eval-episodes",
            "1",
            "--max-cycles",
            "50",
            "--cem-samples",
            "6",
            "--cem-topk",
            "2",
            "--cem-iters",
            "2",
            "--cem-horizon",
            "2",
            "--allow-fail",
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(result.stdout)
        print(result.stderr)

        assert result.returncode == 0, (
            f"Train exited {result.returncode}; stderr: {result.stderr[-2000:]}"
        )

        # Locate experiment dir from stdout ("experiment dir: <path>")
        experiment_dir: Path | None = None
        for line in result.stdout.splitlines():
            if line.startswith("experiment dir:"):
                experiment_dir = Path(line.split("experiment dir:", 1)[1].strip())
                break

        assert experiment_dir is not None, (
            f"'experiment dir:' not found in stdout:\n{result.stdout}"
        )

        run_dir = experiment_dir / "run_00"
        planner_json = run_dir / "planner.json"
        assert planner_json.exists(), f"planner.json not found at {planner_json}"

        with open(planner_json) as f:
            planner_stats = json.load(f)

        expected_keys = {
            "topk_costs_mean",
            "solve_seconds_total",
            "horizon",
            "receding_horizon",
            "num_samples",
            "topk",
        }
        missing = expected_keys - set(planner_stats.keys())
        assert not missing, f"planner.json missing keys: {missing}"

        assert isinstance(planner_stats["topk_costs_mean"], float), (
            "topk_costs_mean must be a float"
        )
        assert planner_stats["solve_seconds_total"] >= 0, (
            "solve time should be non-negative"
        )
        assert planner_stats["horizon"] == 2
        assert planner_stats["topk"] == 2
