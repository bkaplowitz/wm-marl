"""Gates and preservation for an assigned frozen-calibration GPU job."""

import importlib.util
import json
from pathlib import Path
import pickle

import pytest


SPEC = importlib.util.spec_from_file_location(
    "frozen_calibration_launch",
    Path(__file__).parents[1] / "analysis/ppo_audit/run_frozen_value_calibration.py",
)
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


def completed_run(root, step=50000):
    run = root / "runs/original_base-2s3z-seed1"
    checkpoint = run / "train/run/ckpt/final"
    checkpoint.mkdir(parents=True)
    (checkpoint / "done").touch()
    (checkpoint / "step.pkl").write_bytes(pickle.dumps(step))
    (checkpoint / "agent.pkl").write_bytes(b"opaque trusted checkpoint")
    (run / "train/run/config.yaml").write_text("seed: 1\n")
    (run / "manifest.json").write_text(json.dumps({"slot": 5, "gpu": 5}))
    (run / "status.json").write_text(json.dumps({"status": "complete"}))
    outcome = {
        "completed": True,
        "checkpoint": str(checkpoint),
        "summary": {"evaluation_protocol": {"episodes": 128}},
    }
    (run / "outcome.json").write_text(json.dumps(outcome))
    (root / "slot-5-supervisor.pid").write_text("12345")
    return run, checkpoint, outcome


@pytest.mark.parametrize(
    "failure",
    (
        "incomplete",
        "127_episodes",
        "supervisor_alive",
        "checkpoint_incomplete",
        "wrong_step",
    ),
)
def test_final_run_blocks_before_completed_final128_and_supervisor_exit(
    tmp_path, failure
):
    run, checkpoint, outcome = completed_run(tmp_path)
    if failure == "incomplete":
        outcome["completed"] = False
    elif failure == "127_episodes":
        outcome["summary"]["evaluation_protocol"]["episodes"] = 127
    elif failure == "checkpoint_incomplete":
        (checkpoint / "done").unlink()
    elif failure == "wrong_step":
        (checkpoint / "step.pkl").write_bytes(pickle.dumps(49999))
    (run / "outcome.json").write_text(json.dumps(outcome))
    with pytest.raises(ValueError):
        launch.final_run(
            tmp_path, run.name, alive=lambda _: failure == "supervisor_alive"
        )


def test_frozen_preservation_keeps_exact_checkpoint_and_configuration(tmp_path):
    run, checkpoint, _ = completed_run(tmp_path)
    training = launch.final_run(tmp_path, run.name, alive=lambda _: False)
    destination = tmp_path / "calibration"
    destination.mkdir()
    frozen, preserved = launch.preserve(training, destination, {"verified": True})
    assert (preserved / "agent.pkl").stat().st_ino == (
        checkpoint / "agent.pkl"
    ).stat().st_ino
    assert (frozen / "config.yaml").read_bytes() == (
        run / "train/run/config.yaml"
    ).read_bytes()
    (checkpoint / "agent.pkl").unlink()
    assert (preserved / "agent.pkl").read_bytes() == b"opaque trusted checkpoint"
    manifest = json.loads((destination / "snapshot_manifest.json").read_text())
    assert manifest["training"]["step"] == 50000
    assert all(len(row["sha256"]) == 64 for row in manifest["files"])


def test_gpu_occupied_prevents_any_destination_or_launch(monkeypatch, tmp_path):
    fake = {"manifest": {"gpu": 5}}
    monkeypatch.setattr(launch, "final_run", lambda *_: fake)
    monkeypatch.setattr(launch, "verify_stage", lambda *_: {})
    monkeypatch.setattr(
        launch, "gpu_state", lambda _: {"idle": False, "compute_pids": [42]}
    )
    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="not idle"):
        launch.main(
            [
                "--run",
                "original_base-2s3z-seed1",
                "--release-run",
                "original_base-2s3z-seed1",
                "--gpu",
                "5",
                "--stage",
                str(tmp_path),
                "--destination",
                str(destination),
                "--execute",
            ]
        )
    assert not destination.exists()
