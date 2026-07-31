from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

import pytest

from world_marl.baselines.dreamerv3.evaluation import (
    DreamerV3EvaluationSpec,
    run_evaluation,
)
from world_marl.jepa_transformer.config import JEPATransformerRunSpec
from world_marl.jepa_transformer.launcher import require_free_disk, run_training
from world_marl.jepa_transformer.runtime import runtime_fingerprint
from world_marl.scripts.eval_dmc_jepa_transformer import main as eval_main
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
    assert "--jax.profiler" not in command
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


def test_latest_checkpoint_evaluation_uses_the_registered_runtime(tmp_path):
    runtime = tmp_path / "runtime"
    experiment = tmp_path / "experiment"
    spec = JEPATransformerRunSpec(
        experiment_dir=experiment,
        runtime_root=runtime,
        python=Path(sys.executable),
        platform="cpu",
    )
    assert run_training(spec, dry_run=True) == 0
    launch_path = experiment / "launch.json"
    launch = json.loads(launch_path.read_text())
    launch.pop("upstream_root")
    launch_path.write_text(json.dumps(launch))
    checkpoint_root = spec.upstream_logdir / "ckpt"
    checkpoint = checkpoint_root / "checkpoint-123"
    checkpoint.mkdir(parents=True)
    (checkpoint / "done").touch()
    with (checkpoint / "step.pkl").open("wb") as handle:
        pickle.dump(123, handle)
    (checkpoint_root / "latest").write_text(checkpoint.name)

    assert eval_main([str(experiment), "--python", sys.executable, "--dry-run"]) == 0
    metadata = json.loads(
        (
            experiment
            / "evaluation"
            / "latest_20eps_seed10000"
            / "evaluation_launch.json"
        ).read_text()
    )
    command = metadata["command"]
    assert command[1] == str(runtime / "dreamerv3" / "main.py")
    start = command.index("--configs") + 1
    assert command[start : start + 2] == ["dmc_vision", "jepa_transformer"]
    assert command[command.index("--run.from_checkpoint") + 1] == str(checkpoint)


def test_training_refuses_an_exhausted_output_volume(monkeypatch, tmp_path):
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        "world_marl.jepa_transformer.launcher.shutil.disk_usage",
        lambda path: usage._replace(free=1024),
    )
    with pytest.raises(RuntimeError, match="insufficient free disk"):
        require_free_disk(tmp_path)


def test_legacy_manifest_runtime_is_used_as_evaluation_cwd(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    experiment = tmp_path / "experiment"
    spec = JEPATransformerRunSpec(
        experiment_dir=experiment,
        runtime_root=runtime,
        python=Path(sys.executable),
        platform="cpu",
    )
    assert run_training(spec, dry_run=True) == 0
    launch_path = experiment / "launch.json"
    launch = json.loads(launch_path.read_text())
    launch.pop("upstream_root")
    launch_path.write_text(json.dumps(launch))
    checkpoint = spec.upstream_logdir / "ckpt" / "checkpoint-123"
    checkpoint.mkdir(parents=True)
    (checkpoint / "done").touch()
    with (checkpoint / "step.pkl").open("wb") as handle:
        pickle.dump(123, handle)
    (checkpoint.parent / "latest").write_text(checkpoint.name)

    observed = {}

    class FakeProcess:
        stdout = iter(())

        def wait(self):
            return 0

    def fake_popen(command, *, cwd, **kwargs):
        observed["cwd"] = cwd
        return FakeProcess()

    monkeypatch.setattr(
        "world_marl.baselines.dreamerv3.evaluation.subprocess.Popen", fake_popen
    )
    monkeypatch.setattr(
        "world_marl.baselines.dreamerv3.evaluation.normalize_evaluation_artifacts",
        lambda *args, **kwargs: {"completed_episodes": 20},
    )
    returncode, _ = run_evaluation(
        DreamerV3EvaluationSpec(experiment_dir=experiment),
        verify_fn=lambda path: "verified",
    )
    assert returncode == 0
    assert observed["cwd"] == str(runtime)
