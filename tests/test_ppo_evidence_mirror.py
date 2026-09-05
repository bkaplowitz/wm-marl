"""Verify incremental evidence recovery without reading weights or credentials."""

import importlib.util
import base64
import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "analysis/ppo_audit/mirror_remote_results.py"
SPEC = importlib.util.spec_from_file_location("evidence_mirror", SCRIPT)
mirror = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mirror)


def remote_snapshot(root, offsets):
    proc = subprocess.run(
        [sys.executable, "-c", mirror.REMOTE],
        input=json.dumps({"root": str(root), "offsets": offsets}),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)


def test_mirror_preserves_complete_lines_without_duplicate_or_partial_records(tmp_path):
    remote = tmp_path / "remote"
    metrics = remote / "runs/example/train/run/metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    metrics.write_bytes(b'{"step":1}\n{"step":')
    local = tmp_path / "local"
    local.mkdir()
    first = remote_snapshot(remote, {})
    assert len(first["files"]) == 1
    count, complete = mirror.apply_file(local, first["files"][0])
    assert count == len(b'{"step":1}\n')
    assert not complete
    with metrics.open("ab") as stream:
        stream.write(b"2}\n")
    relative = first["files"][0]["path"]
    second = remote_snapshot(remote, {relative: count})
    _, complete = mirror.apply_file(local, second["files"][0])
    assert complete
    assert (local / relative).read_bytes() == b'{"step":1}\n{"step":2}\n'


def test_mirror_selects_outcomes_but_not_credentials_or_weight_contents(tmp_path):
    run = tmp_path / "runs/example"
    checkpoint = run / "train/run/ckpt/final"
    checkpoint.mkdir(parents=True)
    (run / "outcome.json").write_text('{"completed":true}')
    (run / "train/run/config.yaml").write_text("task: smac_3m\n")
    (tmp_path / ".netrc").write_text("test-only credential placeholder")
    (checkpoint / "agent.pkl").write_bytes(b"weight placeholder")
    (checkpoint / "done").touch()
    (checkpoint.parent / "latest").write_text("final")
    result = remote_snapshot(tmp_path, {})
    assert [r["path"] for r in result["files"]] == [
        "runs/example/outcome.json",
        "runs/example/train/run/config.yaml",
    ]
    assert result["outcomes"] == [{"completed": True}]
    assert result["checkpoint_inventory"][0]["complete"]


def test_remote_log_reset_preserves_the_earlier_local_evidence(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step":100}\n')
    content = b'{"step":1}\n'
    mirror.apply_file(
        tmp_path,
        {
            "path": "metrics.jsonl",
            "jsonl": True,
            "offset": 0,
            "size": len(content),
            "data": base64.b64encode(gzip.compress(content)).decode(),
        },
    )
    previous = list(tmp_path.glob("metrics.before-reset-*.jsonl"))
    assert len(previous) == 1
    assert previous[0].read_text() == '{"step":100}\n'
    assert path.read_bytes() == content


@pytest.mark.parametrize("path", ["../../outside.json", "/absolute/outcome.json"])
def test_mirror_rejects_paths_outside_local_root(tmp_path, path):
    with pytest.raises(ValueError):
        mirror.safe_target(tmp_path, path)


def test_local_summary_retains_metric_measurement_steps(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "mirror_summary", SCRIPT.with_name("summarize_mirror.py")
    )
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    run = tmp_path / "runs/example"
    metrics = run / "train/run/metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    (run / "manifest.json").write_text('{"slot":1}')
    metrics.write_text(
        '{"step":100,"train/opt/local_world/updates":1}\n'
        '{"step":5000,"eval/win_rate":0.25,"eval/return_mean":8.0}\n'
        '{"step":5010,"counters/ppo_update_calls":2}\n'
    )
    result = summary.summarize(tmp_path)["runs"][0]
    assert result["environment_step"] == 5010
    assert result["first_learner_metric_step"] == 100
    assert result["latest"]["train/opt/local_world/updates"] == {
        "step": 100,
        "value": 1,
    }
    assert result["curve"] == [
        {"step": 5000, "eval/win_rate": 0.25, "eval/return_mean": 8.0}
    ]


def test_mirror_failure_retains_connection_cause_without_command_or_arbitrary_stderr():
    result = mirror.failure_detail(
        subprocess.CalledProcessError(
            255,
            ["secret-command-placeholder"],
            stderr="ssh: Connection timed out\nprivate-placeholder",
        )
    )
    assert result["returncode"] == 255
    assert result["connection_errors"] == ["Connection timed out"]
    assert "placeholder" not in json.dumps(result)
