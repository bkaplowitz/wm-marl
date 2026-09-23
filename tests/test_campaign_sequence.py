import json
from types import SimpleNamespace

import pytest

from majepa import campaign, campaign_sequence as sequence


def client():
    return SimpleNamespace(flush=lambda: None)


def test_sequence_waits_verifies_tears_down_then_launches(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    manifest = {
        "jobs": [{"pod_id": "oldpod", "created_at": 100, "actual_rate": 4}],
        "runs": [{"name": "run"}],
        "allocations": [{"gpu_count": 4}],
        "rate": 1.6,
        "max_hours": 8,
    }
    campaign.write_json(old / "manifest.json", manifest)
    campaign.write_json(new / "manifest.json", {**manifest, "jobs": []})
    campaign.write_json(
        tmp_path / "sequence.json",
        {
            "budget": 160,
            "spent_before": 0,
            "predecessors": [str(old)],
            "stages": [[str(new)]],
        },
    )
    pods = [
        {"id": "oldpod", "desiredStatus": "RUNNING"},
        {"id": "unrelated", "desiredStatus": "RUNNING"},
    ]
    calls = []
    monkeypatch.setattr(campaign, "list_pods", lambda: pods)
    monkeypatch.setattr(campaign, "api", lambda path, **kw: calls.append((path, kw)))
    monkeypatch.setattr(sequence.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(sequence.time, "time", lambda: 3700)
    ready = False

    def results(*args):
        if not ready:
            raise ValueError("training run is not finished")
        return {"runs": [{"episodes": 100}]}

    monkeypatch.setattr(campaign, "collect_results", results)
    assert sequence.step(tmp_path, client()) == "waiting"
    assert calls == []
    ready = True
    assert sequence.step(tmp_path, client()) == "waiting"
    assert calls == [("pods/oldpod/stop", {"method": "POST"})]
    pods[0]["desiredStatus"] = "EXITED"
    assert sequence.step(tmp_path, client()) == "waiting"
    assert calls[-1] == ["runpodctl", "pod", "delete", "oldpod"]
    assert not any(isinstance(c, list) and "launch" in c for c in calls)
    pods.pop(0)
    assert sequence.step(tmp_path, client()) == "running"
    assert calls[-1][-3:] == ["--directory", str(new), "--once"]
    state = json.loads((tmp_path / "sequence-state.json").read_text())
    assert state["completed"][str(old)]["gpu_cost"] == 4
    assert all("unrelated" not in str(c) for c in calls)
    calls.clear()
    campaign.write_json(
        tmp_path / "sequence.json",
        {
            "budget": 20,
            "spent_before": 0,
            "predecessors": [str(old)],
            "stages": [[str(new)]],
        },
    )
    assert sequence.step(tmp_path, client()) == "budget_blocked"
    assert calls == []


def test_sequence_does_not_delete_failed_stopped_pod(tmp_path, monkeypatch):
    old = tmp_path / "old"
    old.mkdir()
    campaign.write_json(
        old / "manifest.json",
        {
            "jobs": [{"pod_id": "failedpod", "created_at": 100, "actual_rate": 4}],
            "runs": [{"name": "run"}],
        },
    )
    campaign.write_json(
        tmp_path / "sequence.json",
        {
            "budget": 160,
            "spent_before": 0,
            "predecessors": [str(old)],
            "stages": [],
        },
    )
    monkeypatch.setattr(
        campaign, "list_pods", lambda: [{"id": "failedpod", "desiredStatus": "EXITED"}]
    )
    monkeypatch.setattr(
        campaign,
        "collect_results",
        lambda *a: (_ for _ in ()).throw(ValueError("checkpoint is not verified")),
    )
    with pytest.raises(ValueError, match="checkpoint is not verified"):
        sequence.step(tmp_path, client())


def test_current_batch_teardown_does_not_relaunch_deleted_pod(tmp_path, monkeypatch):
    batch = tmp_path / "batch"
    batch.mkdir()
    campaign.write_json(
        batch / "manifest.json",
        {
            "jobs": [{"pod_id": "donepod", "created_at": 100, "actual_rate": 4}],
            "runs": [{"name": "run"}],
            "allocations": [{"gpu_count": 4}],
            "rate": 1.6,
            "max_hours": 8,
        },
    )
    campaign.write_json(
        tmp_path / "sequence.json",
        {
            "budget": 160,
            "predecessors": [],
            "stages": [[str(batch)]],
        },
    )
    pods = [{"id": "donepod", "desiredStatus": "EXITED"}]
    calls, flushes = [], []
    cached_client = SimpleNamespace(flush=lambda: flushes.append(True))
    monkeypatch.setattr(campaign, "list_pods", lambda: pods)
    monkeypatch.setattr(campaign, "collect_results", lambda *a: {"records": []})
    monkeypatch.setattr(sequence.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(sequence.time, "time", lambda: 3700)
    sequence.step(tmp_path, cached_client)
    assert calls == [["runpodctl", "pod", "delete", "donepod"]]
    assert (
        json.loads((batch / "manifest.json").read_text())["jobs"][0]["stopped_at"]
        == 3700
    )
    pods.clear()
    sequence.step(tmp_path, cached_client)
    assert sequence.step(tmp_path, cached_client) == "complete"
    assert len(flushes) == 3
    assert len(calls) == 1
