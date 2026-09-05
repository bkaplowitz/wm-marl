"""A completed run is independently archivable without waiting for diagnostics."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).parents[1] / "analysis/ppo_audit/backup_final_artifacts.py"
SPEC = importlib.util.spec_from_file_location("final_backup", SCRIPT)
backup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backup)


def make_run(root, name, completed):
    run = root / "runs" / name
    checkpoint = run / "train/run/ckpt/50000"
    checkpoint.mkdir(parents=True)
    for name in ["agent.pkl", "step.pkl", "done"]:
        (checkpoint / name).write_bytes(b"test checkpoint data")
    (checkpoint.parent / "latest").write_text("50000")
    (run / "outcome.json").write_text(
        json.dumps({"completed": completed, "checkpoint": str(checkpoint)})
    )
    (run / "manifest.json").write_text('{"source_commit":"tested"}')
    for phase in ["train", "final128"]:
        config = run / phase / "run/config.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("task: smac_2s3z\n")
    (run / "final128/run/evaluation_summary.json").write_text('{"win_rate":0.5}')
    replay = run / "train/run/replay/chunk.npz"
    replay.parent.mkdir()
    replay.write_bytes(b"test replay data")
    return run


def plan(root, run, **kwargs):
    request = {"root": str(root), "run": run, "exports": [], **kwargs}
    result = subprocess.run(
        [sys.executable, "-c", backup.REMOTE_PLAN],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_successful_run_is_ready_while_peer_failed_and_marker_missing(tmp_path):
    make_run(tmp_path, "success", True)
    make_run(tmp_path, "failed_peer", False)
    result = plan(tmp_path, "success", ready_marker=str(tmp_path / "not_ready.json"))
    assert result["ready"]
    assert result["raw_replay_files"] == 1
    paths = {item["relative"] for item in result["files"]}
    assert {
        "checkpoint/50000/agent.pkl",
        "checkpoint/50000/step.pkl",
        "checkpoint/50000/done",
        "train/config.yaml",
        "final128/config.yaml",
        "raw_replay/chunk.npz",
    } <= paths
    assert not plan(tmp_path, "failed_peer")["ready"]


def test_completed_run_with_missing_replay_does_not_claim_complete_backup(tmp_path):
    run = make_run(tmp_path, "example", True)
    (run / "train/run/replay/chunk.npz").unlink()
    result = plan(tmp_path, "example")
    assert not result["ready"]
    assert any("no raw replay" in reason for reason in result["reasons"])


def test_later_diagnostic_plan_does_not_copy_weights_or_raw_replay_again(tmp_path):
    make_run(tmp_path, "example", True)
    result = plan(tmp_path, "example", exports_only=True)
    assert result["ready"]
    assert not any(
        item["relative"].startswith(("checkpoint/", "raw_replay/"))
        for item in result["files"]
    )


def test_watcher_selects_completed_peer_even_after_other_backup_attempts_fail(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "backup_watcher", SCRIPT.with_name("watch_final_backups.py")
    )
    watcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watcher)
    for name, completed in [
        ("success", True),
        ("training_failed", False),
        ("backup_failed", True),
    ]:
        make_run(tmp_path, name, completed)
    outcomes, ready = watcher.candidates(
        tmp_path, tmp_path / "archives", {"backup_failed": 3}, 3
    )
    assert ready == ["success"]
    assert not outcomes["training_failed"]["successful"]
