from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from dreamarl.baselines.marie.artifacts import (
    normalize_training_artifacts,
    read_official_result,
)
from dreamarl.baselines.marie.config import (
    OFFICIAL_MARIE_COMMIT,
    MARIERunSpec,
    default_upstream_root,
)
from dreamarl.baselines.marie.launcher import run_training, verify_upstream


def test_pinned_official_marie_checkout_is_present():
    assert verify_upstream(default_upstream_root()) == OFFICIAL_MARIE_COMMIT


@pytest.mark.parametrize(
    ("map_name", "steps", "temperature"),
    [("3m", "100000", "1.0"), ("3s_vs_4z", "100000", "1.0")],
)
def test_paper_gate_command_is_the_official_cli(tmp_path, map_name, steps, temperature):
    spec = MARIERunSpec(
        experiment_dir=tmp_path / "run",
        map_name=map_name,
        python=Path(sys.executable),
        mode="disabled",
    )
    command = spec.command
    assert command[1].endswith("external/marie/train.py")
    assert command[command.index("--env") + 1] == "starcraft"
    assert command[command.index("--env_name") + 1] == map_name
    assert command[command.index("--steps") + 1] == steps
    assert command[command.index("--temperature") + 1] == temperature
    assert command[-1] == "--ce_for_av"


def test_dry_run_records_source_and_effective_seed(tmp_path):
    spec = MARIERunSpec(
        experiment_dir=tmp_path / "run",
        python=Path(sys.executable),
        mode="disabled",
    )
    assert run_training(spec, dry_run=True) == 0
    launch = json.loads((spec.experiment_dir / "launch.json").read_text())
    assert launch["verified_upstream_commit"] == OFFICIAL_MARIE_COMMIT
    assert launch["source_policy"] == "unmodified official model and training code"
    assert launch["effective_internal_seed"] == 123


def test_official_result_is_preserved_and_normalized(tmp_path):
    result_path = tmp_path / "official.pkl"
    with result_path.open("wb") as handle:
        pickle.dump(
            {
                "steps": np.asarray([500, 1000]),
                "eval_win_rates": np.asarray([0.2, 0.6]),
                "eval_returns": np.asarray([4.0, 8.0]),
            },
            handle,
        )
    checkpoint = tmp_path / "model_final.pth"
    checkpoint.write_bytes(b"checkpoint")

    summary = normalize_training_artifacts(
        tmp_path / "experiment",
        result_path=result_path,
        checkpoint_path=checkpoint,
        map_name="3m",
        cli_seed=1,
        steps_budget=1000,
    )

    assert summary["latest_evaluation"]["eval_win_rate"] == pytest.approx(0.6)
    assert summary["best_evaluation"]["env_steps"] == 1000
    assert summary["normalized_win_rate_auc"] == pytest.approx(0.25)
    assert read_official_result(
        tmp_path / "experiment" / "upstream" / "marie_results.pkl"
    )[-1]["eval_return"] == pytest.approx(8.0)
    assert (
        tmp_path / "experiment" / "upstream" / "model_final.pth"
    ).read_bytes() == b"checkpoint"
