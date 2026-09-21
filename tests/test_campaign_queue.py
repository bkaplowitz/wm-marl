from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import copy
import hashlib
import json
import os
import pickle
import subprocess
import tarfile
import time
from unittest.mock import Mock

import pytest

from majepa import campaign_queue as queue


def manifest(directory, count=4):
    return {
        "campaign": "queue-test",
        "source_sha256": "source",
        "base_commit": "base",
        "wandb": {"entity": "test", "project": "test"},
        "storage": {"workspace_root": str(directory), "quota_gb": 10, "min_free_gb": 1},
        "runs": [
            {
                "name": f"run{index}",
                "expected_updates": 3,
                "config": {
                    "logdir": f"/RUN/run{index}/train",
                    "run.steps": 20,
                    "jax.train_devices": [0],
                    "jax.policy_devices": [0],
                    "script": "train",
                    "task": "smac_3m",
                    "seed": index,
                },
            }
            for index in range(count)
        ],
        "job": {
            "gpu_count": 2,
            "pod_id": "testpod",
            "deadline": time.time() + 100,
            "run_names": [f"run{index}" for index in range(count)],
        },
    }


def setup_queue(directory, count=4):
    spec = manifest(directory, count)
    (directory / "job.json").write_text(json.dumps(spec))
    return spec, queue.initialize(directory, spec)


def checkpoint(logdir, name="final", updates=3, steps=20):
    root = logdir / "ckpt"
    path = root / name
    path.mkdir(parents=True)
    (path / "done").touch()
    (path / "learner_update_calls.pkl").write_bytes(pickle.dumps(updates))
    (path / "step.pkl").write_bytes(pickle.dumps(steps))
    (root / "latest").write_text(name)
    return path


def test_initialization_is_durable_and_claims_are_exclusive(tmp_path):
    spec, original = setup_queue(tmp_path)
    assert queue.initialize(tmp_path, spec) == original
    ids = [item["wandb_id"] for item in original["jobs"]]
    assert len(set(ids)) == 4
    with ThreadPoolExecutor(max_workers=4) as pool:
        claimed = list(pool.map(lambda gpu: queue.claim(tmp_path, gpu), range(4)))
    assert len({item["name"] for item in claimed}) == 4
    assert queue.claim(tmp_path, 0) is None
    assert [
        item["wandb_id"] for item in queue.initialize(tmp_path, spec)["jobs"]
    ] == ids
    changed = copy.deepcopy(spec)
    changed["runs"][0]["config"]["seed"] = 20
    with pytest.raises(ValueError, match="changed"):
        queue.initialize(tmp_path, changed)


def test_initialization_preserves_controller_run_ids(tmp_path):
    spec = manifest(tmp_path, 1)
    spec["runs"][0].update(
        wandb_id="controller-train", evaluation_wandb_id="controller-eval"
    )
    entry = queue.initialize(tmp_path, spec)["jobs"][0]
    assert entry["wandb_id"] == "controller-train"
    assert entry["evaluation_wandb_id"] == "controller-eval"


def test_predecessor_and_dependencies_fail_closed(tmp_path):
    path = tmp_path / "predecessor.json"
    assert "not present" in queue.predecessor_wait(path)
    path.write_text(json.dumps({"jobs": [{"status": "running"}]}))
    assert queue.predecessor_wait(path)
    path.write_text(json.dumps({"jobs": [{"status": "failed"}]}))
    with pytest.raises(RuntimeError, match="predecessor failed"):
        queue.predecessor_wait(path)
    path.write_text(json.dumps({"jobs": [{"status": "complete"}]}))
    assert queue.predecessor_wait(path) is None
    spec = manifest(tmp_path, 2)
    spec["runs"][0]["depends_on"] = ["run1"]
    queue.initialize(tmp_path, spec)
    assert queue.claim(tmp_path, 0)["name"] == "run1"
    assert queue.claim(tmp_path, 1) is None
    queue.update_job(tmp_path, "run1", status="failed")
    assert queue.claim(tmp_path, 1) is None
    assert all(
        item["status"] == "failed"
        for item in json.loads((tmp_path / "queue.json").read_text())["jobs"]
    )


def test_dry_run_and_restart_never_spawn_or_retry(tmp_path, monkeypatch):
    spec, state = setup_queue(tmp_path)
    spawn = Mock(side_effect=AssertionError("unexpected spawn"))
    monkeypatch.setattr(queue.subprocess, "Popen", spawn)
    assert queue.run(tmp_path, dry_run=True)["spawned"] is False
    queue.claim(tmp_path, 0)
    with pytest.raises(RuntimeError, match="reconcile"):
        queue.run(tmp_path)
    queue.update_job(tmp_path, "run0", status="failed")
    assert queue.initialize(tmp_path, spec)["jobs"][0]["status"] == "failed"
    assert queue.claim(tmp_path, 1)["name"] == "run1"
    spawn.assert_not_called()


def test_storage_uses_allocated_bytes_and_quota(tmp_path, monkeypatch):
    spec, _ = setup_queue(tmp_path)
    check = Mock(return_value="9000000\t/workspace\n")
    monkeypatch.setattr(queue.subprocess, "check_output", check)
    monkeypatch.setattr(
        queue.shutil, "disk_usage", lambda root: Mock(free=100_000_000_000)
    )
    with pytest.raises(RuntimeError, match="storage guard"):
        queue.storage_check(tmp_path, spec["storage"])
    assert check.call_args.args[0][:3] == ["du", "-sk", "--exclude=*.tmp"]
    record = json.loads((tmp_path / "storage.json").read_text())
    assert record["used_bytes"] == 9000000 * 1024
    assert record["headroom_bytes"] == 10_000_000_000 - 9000000 * 1024


def test_checkpoint_budget_and_pruning_boundaries(tmp_path):
    _, state = setup_queue(tmp_path, 2)
    logdir = tmp_path / "jobs/run0/train"
    old = checkpoint(logdir, "old")
    final = checkpoint(logdir)
    sibling = checkpoint(tmp_path / "jobs/run1/train")
    outside = tmp_path / "outside"
    outside.mkdir()
    (logdir / "ckpt/link").symlink_to(outside, target_is_directory=True)
    assert queue.final_checkpoint(logdir, 3, 20) == final
    with pytest.raises(ValueError, match="expected 4"):
        queue.final_checkpoint(logdir, 4, 20)
    with pytest.raises(ValueError, match="pruning requires"):
        queue.prune_checkpoints(tmp_path, state["jobs"][0], final, {"verified": True})
    entry = dict(state["jobs"][0], status="complete")
    with pytest.raises(ValueError, match="pruning requires"):
        queue.prune_checkpoints(tmp_path, entry, final, {"verified": False})
    queue.prune_checkpoints(tmp_path, entry, final, {"verified": True})
    assert final.is_dir() and sibling.is_dir() and outside.is_dir()
    assert not old.exists()
    (logdir / "ckpt/latest").write_text("../../outside")
    with pytest.raises(ValueError, match="invalid final"):
        queue.final_checkpoint(logdir, 3, 20)


def test_frozen_configuration_roundtrip(tmp_path):
    from majepa.main import _load_configs, _resolve_config_profiles

    config = _resolve_config_profiles(_load_configs(), ["baseline"]).flat
    config = json.loads(json.dumps(config))
    config["logdir"] = "/RUN/example/train"
    entry = {"name": "example", "config": config}
    train = queue.phase_config(tmp_path, entry)
    evaluation = queue.phase_config(tmp_path, entry, tmp_path / "checkpoint")
    assert {key for key in train if train[key] != config[key]} == {"logdir"}
    assert {key for key in evaluation if evaluation[key] != config[key]} <= {
        "logdir",
        "script",
        "run.from_checkpoint",
        "run.eval_eps",
        "run.eval_policy_mode",
    }
    command = queue.phase_command(train, tmp_path / "train.yaml")
    assert command[-1] == str(tmp_path / "train.yaml")
    assert config["logdir"] == "/RUN/example/train"


def test_source_verification_checks_archive_and_extracted_code(tmp_path, monkeypatch):
    source = tmp_path / "repo/src/majepa/campaign_queue.py"
    source.parent.mkdir(parents=True)
    source.write_text("frozen source")
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname="src/majepa/campaign_queue.py")
    spec = {"source_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    monkeypatch.setattr(queue, "__file__", str(source))
    assert queue.verify_source(tmp_path, spec) == spec["source_sha256"]
    source.write_text("edited source")
    with pytest.raises(ValueError, match="extracted source mismatch"):
        queue.verify_source(tmp_path, spec)
    archive.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        queue.verify_source(tmp_path, spec)


def test_two_gpu_workers_reuse_devices_and_keep_sibling_running(tmp_path, monkeypatch):
    from majepa import campaign, evaluation, tracking

    spec, _ = setup_queue(tmp_path, 6)
    monkeypatch.setattr(queue, "interruptible", nullcontext)
    monkeypatch.setattr(queue, "verify_source", lambda *args: "source")
    monkeypatch.setattr(queue, "storage_check", lambda *args: {})
    monkeypatch.setattr(
        queue, "port_server", lambda directory, gpu: nullcontext(f"@gpu{gpu}")
    )
    monkeypatch.setattr(queue, "gpu_processes", lambda gpu: [])
    monkeypatch.setattr(campaign, "discover_sc2", lambda **kwargs: tmp_path)
    monkeypatch.setattr(queue.tempfile, "gettempdir", lambda: str(tmp_path))
    original_environment = os.environ.copy()
    phases = []

    def execute(directory, manifest, entry, gpu, address, phase, checkpoint_path=None):
        assert address == f"@gpu{gpu}"
        phases.append((entry["name"], gpu, phase))
        time.sleep(0.02)
        path = directory / "jobs" / entry["name"] / phase
        path.mkdir(parents=True)
        if phase == "train":
            checkpoint(path, "intermediate")
            checkpoint(path)
        else:
            summary = {key: [0] * 100 for key in evaluation._RAW_EVALUATION_KEYS}
            summary.update(episodes=100, policy_mode="eval")
            (path / "evaluation_summary.json").write_text(json.dumps(summary))
            (path / "evaluation_episodes.jsonl").write_text(
                "\n".join(json.dumps({"episode": index}) for index in range(100))
            )
        return path

    def publish(entity, project, run_id, kind, paths, metadata):
        if "run0" in str(paths[0]):
            raise RuntimeError("mock artifact upload failed")
        return {"verified": True, "artifact": run_id + kind}

    monkeypatch.setattr(queue, "execute_phase", execute)
    monkeypatch.setattr(tracking, "publish_run_artifact", publish)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda gpu: queue.worker(tmp_path, gpu), (0, 1)))
    finally:
        os.environ.clear()
        os.environ.update(original_environment)
    state = json.loads((tmp_path / "queue.json").read_text())
    assert state["jobs"][0]["status"] == "failed"
    assert all(item["status"] == "complete" for item in state["jobs"][1:])
    assert {gpu for _, gpu, phase in phases if phase == "train"} == {0, 1}
    assert all(
        sum(1 for _, device, phase in phases if device == gpu and phase == "train") >= 2
        for gpu in (0, 1)
    )
    assert (tmp_path / "jobs/run0/train/ckpt/intermediate").exists()
    assert not (tmp_path / "jobs/run1/train/ckpt/intermediate").exists()
    assert not (tmp_path / "outcome.json").exists()


def test_termination_targets_only_owned_child_group(monkeypatch):
    process = Mock(pid=12345)
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("owned", 10), 0]
    kill = Mock()
    monkeypatch.setattr(queue.os, "killpg", kill)
    queue.stop_child(process)
    assert [call.args[0] for call in kill.call_args_list] == [12345, 12345]
    process.poll.return_value = 0
    kill.reset_mock()
    queue.stop_child(process)
    kill.assert_not_called()


def test_supervisor_deadline_records_failure_and_stops_only_workers(
    tmp_path, monkeypatch
):
    spec, _ = setup_queue(tmp_path)
    spec["job"]["deadline"] = time.time() - 1
    (tmp_path / "job.json").write_text(json.dumps(spec))
    (tmp_path / "queue.json").unlink()
    monkeypatch.setattr(queue, "verify_source", lambda *args: None)
    monkeypatch.setattr(queue, "storage_check", lambda *args: None)
    children = [Mock(pid=100), Mock(pid=101)]
    for child in children:
        child.poll.return_value = None
    monkeypatch.setattr(queue.subprocess, "Popen", Mock(side_effect=children))
    stopped = Mock()
    monkeypatch.setattr(queue, "stop_child", stopped)
    with pytest.raises(TimeoutError, match="runtime deadline"):
        queue.run(tmp_path)
    assert [call.args[0] for call in stopped.call_args_list] == children
    assert json.loads((tmp_path / "outcome.json").read_text())["completed"] is False


def test_execute_phase_isolates_gpu_and_checks_actual_saved_config(
    tmp_path, monkeypatch
):
    import elements

    spec, state = setup_queue(tmp_path, 1)
    entry = queue.claim(tmp_path, 1)
    config = queue.phase_config(tmp_path, entry)
    path = tmp_path / "jobs/run0/train"
    path.mkdir(parents=True)
    elements.Config(config).save(path / "config.yaml")
    (path / "artifact_verification.json").write_text(
        json.dumps(
            {
                "source": {"verified": True},
                "config": {"verified": True},
            }
        )
    )
    process = Mock(pid=555, returncode=0)
    process.poll.side_effect = [None, 0, 0]
    spawn = Mock(return_value=process)
    monkeypatch.setattr(queue.subprocess, "Popen", spawn)
    monkeypatch.setattr(queue, "phase_command", lambda *args: ["fake-train"])
    monkeypatch.setattr(queue, "storage_check", lambda *args: {})
    monkeypatch.setattr(queue.time, "sleep", lambda *args: None)
    assert queue.execute_phase(tmp_path, spec, entry, 1, "@gpu1", "train") == path
    env = spawn.call_args.kwargs["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["PORTSERVER_ADDRESS"] == "@gpu1"
    assert env["WANDB_RUN_ID"] == entry["wandb_id"]
    assert spawn.call_args.kwargs["start_new_session"] is True
    state = json.loads((tmp_path / "queue.json").read_text())
    assert state["jobs"][0]["child_pid"] is None
    assert state["workers"]["1"]["heartbeat"] > 0
    elements.Config(dict(config, seed=999)).save(path / "config.yaml")
    process.poll.side_effect = [0, 0]
    with pytest.raises(ValueError, match="saved config differs"):
        queue.execute_phase(tmp_path, spec, entry, 1, "@gpu1", "train")


def test_busy_gpu_waits_without_claiming_or_signalling(tmp_path, monkeypatch):
    from majepa import campaign

    _, state = setup_queue(tmp_path, 1)
    monkeypatch.setattr(queue, "interruptible", nullcontext)
    monkeypatch.setattr(queue, "verify_source", lambda *args: "source")
    monkeypatch.setattr(queue, "port_server", lambda *args: nullcontext("@gpu0"))
    monkeypatch.setattr(queue, "gpu_processes", lambda gpu: [987654])
    monkeypatch.setattr(campaign, "discover_sc2", lambda **kwargs: tmp_path)
    monkeypatch.setattr(queue.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("SC2PATH", str(tmp_path))
    kill = Mock()
    monkeypatch.setattr(queue.os, "killpg", kill)

    def interrupt(seconds):
        assert (
            json.loads((tmp_path / "queue.json").read_text())["jobs"][0]["status"]
            == "pending"
        )
        raise InterruptedError("test SIGTERM")

    monkeypatch.setattr(queue.time, "sleep", interrupt)
    with pytest.raises(InterruptedError):
        queue.worker(tmp_path, 0)
    state = json.loads((tmp_path / "queue.json").read_text())
    assert state["workers"]["0"]["status"] == "failed"
    assert state["jobs"][0]["status"] == "pending"
    kill.assert_not_called()


def test_portserver_uses_the_installed_wheel_script(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    binary = tmp_path / "bin"
    binary.mkdir()
    executable = binary / "python"
    script = binary / "portserver.py"
    script.write_text("# wheel .data/scripts entry\n")
    monkeypatch.setattr(queue.sys, "executable", str(executable))
    process = MagicMock()
    process.poll.return_value = None
    launched = []
    monkeypatch.setattr(
        queue.subprocess,
        "Popen",
        lambda command, **kwargs: launched.append(command) or process,
    )
    connection = MagicMock()
    monkeypatch.setattr(queue.socket, "socket", lambda *a: connection)
    stopped = []
    monkeypatch.setattr(queue, "stop_child", stopped.append)
    with queue.port_server(tmp_path, 1) as address:
        assert address.startswith("@majepa-")
        assert launched[0][:2] == [str(executable), str(script)]
        assert launched[0][-1] == "59300-59599"
    assert stopped == [process]
