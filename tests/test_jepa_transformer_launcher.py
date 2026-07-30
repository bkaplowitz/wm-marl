from __future__ import annotations

import json
import sys
from pathlib import Path

from world_marl.jepa_transformer.config import JEPATransformerRunSpec
from world_marl.jepa_transformer.launcher import run_training
from world_marl.jepa_transformer.runtime import runtime_fingerprint
from world_marl.scripts.train_dmc_jepa_transformer import main as train_main


def test_m3_command_keeps_visual_cdp_control_settings(tmp_path):
    spec = JEPATransformerRunSpec(
        experiment_dir=tmp_path / "run",
        runtime_root=tmp_path / "runtime",
        python=Path(sys.executable),
        platform="cpu",
    )
    command = spec.command
    start = command.index("--configs") + 1
    assert command[start : start + 2] == ["dmc_vision", "jepa_transformer"]
    assert command[command.index("--run.steps") + 1] == "25000"
    assert "--agent.imag_length" not in command
    assert "--run.train_ratio" not in command
    assert spec.to_dict()["overlay_fingerprint"] == runtime_fingerprint()


def test_dry_run_records_the_official_source_and_only_registered_delta(tmp_path):
    runtime = tmp_path / "runtime"
    experiment = tmp_path / "experiment"
    spec = JEPATransformerRunSpec(
        experiment_dir=experiment,
        runtime_root=runtime,
        python=Path(sys.executable),
        platform="cpu",
    )
    assert run_training(spec, dry_run=True) == 0
    launch = json.loads((experiment / "launch.json").read_text())
    assert len(launch["verified_official_commit"]) == 40
    assert launch["verified_overlay_fingerprint"] == runtime_fingerprint()
    assert "causal Transformer" in launch["causal_delta"]


def test_cli_override_is_explicit_and_manifested(tmp_path):
    experiment = tmp_path / "experiment"
    assert train_main(
        [
            "--experiment-dir",
            str(experiment),
            "--runtime-root",
            str(tmp_path / "runtime"),
            "--python",
            sys.executable,
            "--platform",
            "cpu",
            "--override",
            "run.log_every=60",
            "--dry-run",
        ]
    ) == 0
    launch = json.loads((experiment / "launch.json").read_text())
    index = launch["command"].index("--run.log_every")
    assert launch["command"][index + 1] == "60"
