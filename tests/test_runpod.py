from __future__ import annotations

import argparse
import json
import sys

import pytest

from world_marl import runpod


def _compare_args(tmp_path, **overrides) -> argparse.Namespace:
    values = {
        "job": "compare-world-models",
        "job_args": [],
        "remote_out_root": "/workspace/outputs/wm_marl",
        "local_out_root": str(tmp_path),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_default_job_is_compare_world_models(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["world-marl-runpod"])

    args = runpod.parse_args()

    assert args.job == "compare-world-models"


def test_default_gpu_is_rtx_5090(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["world-marl-runpod"])

    args = runpod.parse_args()

    assert args.gpu_id == "NVIDIA GeForce RTX 5090"


def test_command_uses_wandb_detection():
    assert runpod.command_uses_wandb(
        ["uv", "run", "world-marl-optuna-single-genwm", "--wandb-project", "wm"]
    )
    assert runpod.command_uses_wandb(["--wandb-project=wm"])
    assert not runpod.command_uses_wandb(
        ["uv", "run", "world-marl-train-e2e", "--substrate", "coins"]
    )


def test_read_wandb_netrc_entry_fails_fast_without_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(runpod.Path, "home", classmethod(lambda cls: tmp_path))

    with pytest.raises(SystemExit, match="wandb login"):
        runpod.read_wandb_netrc_entry()

    (tmp_path / ".netrc").write_text(
        "machine example.com\n  login a\n  password b\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="wandb login"):
        runpod.read_wandb_netrc_entry()


def test_read_wandb_netrc_entry_returns_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(runpod.Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / ".netrc").write_text(
        "machine api.wandb.ai\n  login user\n  password secret\n", encoding="utf-8"
    )

    entry = runpod.read_wandb_netrc_entry()

    assert entry == "machine api.wandb.ai\n  login user\n  password secret\n"


def test_compare_defaults_match_wide_transformer_run(tmp_path):
    job = runpod.build_job_spec(_compare_args(tmp_path), "20260624T120000Z")

    assert job.command[:6] == [
        "env",
        "XLA_FLAGS=--xla_gpu_enable_triton_gemm=false",
        "uv",
        "run",
        "python",
        "-m",
    ]
    assert job.command[6:8] == [
        "world_marl.scripts.compare_world_models",
        "--flow-types",
    ]
    assert job.command[job.command.index("--flow-types") + 1] == "transformer"
    assert job.command[job.command.index("--fit-steps") + 1] == "100000"
    assert job.command[job.command.index("--chunk-steps") + 1] == "5000"
    assert job.command[job.command.index("--heldout-seeds") + 1] == "1"
    assert job.command[job.command.index("--rollout-envs") + 1] == "6000"
    assert job.command[job.command.index("--out-dir") + 1] == (
        "/workspace/outputs/wm_marl/compare-world-models/20260624T120000Z"
    )


def _benchmark_args(tmp_path, **overrides) -> argparse.Namespace:
    values = {
        "job": "benchmark-policy",
        "job_args": [],
        "remote_out_root": "/workspace/outputs/wm_marl",
        "local_out_root": str(tmp_path),
        "no_policy_warmstart": False,
        "policy_warmstart_updates": 1,
        "prefit_train_steps": 5000,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_benchmark_policy_job_is_accepted(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["world-marl-runpod", "--job", "benchmark-policy"])

    args = runpod.parse_args()

    assert args.job == "benchmark-policy"


def test_benchmark_policy_puts_own_flags_before_train_args(tmp_path):
    job = runpod.build_job_spec(_benchmark_args(tmp_path), "20260630T120000Z")

    assert job.command[:3] == ["uv", "run", "world-marl-benchmark-policy"]
    separator = job.command.index("--")
    out_dir_idx = job.command.index("--out-dir")
    flow_idx = job.command.index("--model-flow-types")
    # benchmark_policy reads its own flags only before the REMAINDER separator
    assert out_dir_idx < separator
    assert flow_idx < separator
    assert job.command[flow_idx + 1] == "transformer"
    assert job.command[out_dir_idx + 1] == (
        "/workspace/outputs/wm_marl/benchmark-policy/20260630T120000Z"
    )
    # train-e2e args are forwarded after the separator
    substrate_idx = job.command.index("--substrate")
    assert substrate_idx > separator
    assert job.command[substrate_idx + 1] == "coins"
    # the wrapper-managed out-dir must not leak into the train-e2e REMAINDER
    assert "--out-dir" not in job.command[separator + 1 :]


def test_frontier_quality_job_uses_managed_output_directory(tmp_path):
    args = _compare_args(
        tmp_path,
        job="frontier-world-model-quality",
        job_args=["--seed", "3"],
    )

    job = runpod.build_job_spec(args, "20260710T120000Z")

    assert job.command[:4] == [
        "env",
        "XLA_FLAGS=--xla_gpu_enable_triton_gemm=false",
        "MUJOCO_GL=egl",
        "uv",
    ]
    assert job.command[4:6] == ["run", "world-marl-frontier-wm-quality"]
    assert job.command[-4:] == [
        "--seed",
        "3",
        "--out-dir",
        "/workspace/outputs/wm_marl/frontier-world-model-quality/20260710T120000Z",
    ]


def test_frontier_quality_job_installs_dmc_extra_without_duplicates() -> None:
    args = argparse.Namespace(
        job="frontier-world-model-quality",
        sync_extra=["brax", "dmc"],
    )

    assert runpod.required_sync_extras(args) == ["brax", "dmc"]

    args.sync_extra = []
    assert runpod.required_sync_extras(args) == ["dmc"]


def test_remote_job_script_skips_unused_menagerie_download_for_dmc_extra() -> None:
    script = runpod.remote_job_script(
        "/root/wm-marl",
        ["uv", "run", "world-marl-frontier-wm-quality"],
        skip_uv_sync=False,
        sync_extras=["dmc"],
    )

    assert "sysconfig.get_path('purelib')" in script
    assert "mujoco_playground/external_deps/mujoco_menagerie" in script
    assert 'mkdir -p "$MENAGERIE_DIR"' in script
    assert "git clone" not in script
    assert script.index('mkdir -p "$MENAGERIE_DIR"') < script.index(
        "uv run world-marl-verify-install"
    )


def test_loads_runpod_api_key_from_dotenv_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "RUNPOD_API_KEY='from-dotenv'\nOTHER_VALUE=ignored\n",
        encoding="utf-8",
    )

    runpod.ensure_runpod_api_key(tmp_path)

    assert runpod.os.environ["RUNPOD_API_KEY"] == "from-dotenv"


def test_dotenv_does_not_override_existing_runpod_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "from-env")
    (tmp_path / ".env").write_text("RUNPOD_API_KEY='from-dotenv'\n", encoding="utf-8")

    runpod.ensure_runpod_api_key(tmp_path)

    assert runpod.os.environ["RUNPOD_API_KEY"] == "from-env"


def test_ssh_info_not_ready_reports_pod_status(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runpod,
        "fetch_pod_rest",
        lambda _: {
            "id": "pod-id",
            "publicIp": None,
            "portMappings": None,
            "desiredStatus": "INITIALIZING",
        },
    )

    with pytest.raises(RuntimeError, match="not ready.*INITIALIZING"):
        runpod.get_ssh_info("pod-id", tmp_path / "key")


def _create_args(**overrides) -> argparse.Namespace:
    values = {
        "template_id": "runpod-torch-v240",
        "gpu_id": "NVIDIA L40S",
        "gpu_count": 1,
        "cloud_type": "SECURE",
        "volume_gb": 100,
        "container_disk_gb": 50,
        "volume_mount_path": "/workspace",
        "ports": "22/tcp",
        "stop_after": None,
        "terminate_after": None,
        "terminate_after_hours": 0.0,
        "auto_stop_hours": 0.0,
        "ssh_key": "~/.ssh/runpod_key",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_create_pod_cmd_injects_public_key_env(tmp_path):
    key = tmp_path / "runpod_key"
    key.write_text("PRIVATE", encoding="utf-8")
    (tmp_path / "runpod_key.pub").write_text(
        "ssh-ed25519 AAAATEST me@host\n", encoding="utf-8"
    )

    cmd = runpod.build_create_pod_cmd(_create_args(ssh_key=str(key)), "pod-name")

    assert "--env" in cmd
    env_json = cmd[cmd.index("--env") + 1]
    assert json.loads(env_json) == {"PUBLIC_KEY": "ssh-ed25519 AAAATEST me@host"}


def test_create_pod_cmd_requires_public_key_file(tmp_path):
    key = tmp_path / "runpod_key"  # no .pub sibling
    key.write_text("PRIVATE", encoding="utf-8")

    with pytest.raises(SystemExit, match="public key"):
        runpod.build_create_pod_cmd(_create_args(ssh_key=str(key)), "pod-name")


def test_parse_direct_ssh_info_uses_public_ip_and_port22(tmp_path):
    pod = {
        "id": "p",
        "publicIp": "1.2.3.4",
        "portMappings": {"22": 18151},
        "desiredStatus": "RUNNING",
    }

    info = runpod.parse_direct_ssh_info(pod, tmp_path / "key")

    assert info.user == "root"
    assert info.host == "1.2.3.4"
    assert info.port == 18151
    assert info.key_path == tmp_path / "key"


def test_parse_direct_ssh_info_not_ready_without_port_mapping(tmp_path):
    pod = {
        "id": "p",
        "publicIp": None,
        "portMappings": None,
        "desiredStatus": "INITIALIZING",
    }

    with pytest.raises(RuntimeError, match="not ready.*INITIALIZING"):
        runpod.parse_direct_ssh_info(pod, tmp_path / "key")


def test_get_ssh_info_returns_direct_tcp_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runpod,
        "fetch_pod_rest",
        lambda _: {
            "id": "pod-id",
            "publicIp": "5.6.7.8",
            "portMappings": {"22": 12345},
            "desiredStatus": "RUNNING",
        },
    )

    info = runpod.get_ssh_info("pod-id", tmp_path / "key")

    assert (info.user, info.host, info.port) == ("root", "5.6.7.8", 12345)


def test_ensure_remote_rsync_installs_via_ssh(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(runpod, "run", lambda cmd, **kwargs: calls.append(cmd))
    info = runpod.SshInfo(
        user="root", host="1.2.3.4", port=22, key_path=tmp_path / "key"
    )

    runpod.ensure_remote_rsync(info)

    assert calls, "expected an ssh call"
    remote_script = calls[0][-1]
    assert "rsync" in remote_script
    assert "libegl1" in remote_script
    assert "apt-get install" in remote_script


def test_start_remote_job_detached_nohups_and_asserts_gpu(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(runpod, "run", lambda cmd, **kwargs: calls.append(cmd))
    info = runpod.SshInfo(
        user="root", host="1.2.3.4", port=22, key_path=tmp_path / "key"
    )

    runpod.start_remote_job_detached(
        "/root/wm-marl",
        ["uv", "run", "world-marl-benchmark-policy"],
        info,
        skip_uv_sync=True,
        sync_extras=[],
        remote_out_dir="/workspace/outputs/wm_marl/benchmark-policy/20260706T000000Z",
    )

    assert calls, "expected an ssh call"
    script = calls[0][-1]
    assert "nohup" in script, "job must not depend on the launcher's ssh session"
    assert "job.log" in script
    assert "JOB_DONE" in script and "JOB_FAILED" in script
    gpu_idx = script.index("jax.devices()")
    job_idx = script.index("world-marl-benchmark-policy")
    assert gpu_idx < job_idx, "GPU assertion must run before the benchmark job"


def test_sync_repo_excludes_dotenv(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(runpod, "run", lambda cmd, **kwargs: calls.append(cmd))
    info = runpod.SshInfo(
        user="root", host="1.2.3.4", port=22, key_path=tmp_path / "key"
    )

    runpod.sync_repo(tmp_path, "/root/wm-marl", info)

    rsync_calls = [c for c in calls if c and c[0] == "rsync"]
    assert rsync_calls, "expected an rsync invocation"
    argv = rsync_calls[0]
    excludes = [argv[i + 1] for i, a in enumerate(argv) if a == "--exclude"]
    assert ".env" in excludes


def test_write_manifest_round_trips_and_creates_dir(tmp_path):
    out_dir = tmp_path / "compare-world-models" / "20260706T000000Z"
    manifest = {"pod_id": "abc123", "status": "running"}

    path = runpod.write_manifest(out_dir, manifest)

    assert path == out_dir / "manifest.json"
    assert json.loads(path.read_text(encoding="utf-8")) == manifest


def _manifest_main_argv(tmp_path) -> list[str]:
    key = tmp_path / "runpod_key"
    key.write_text("private", encoding="utf-8")
    (tmp_path / "runpod_key.pub").write_text("ssh-ed25519 AAAA test", encoding="utf-8")
    return [
        "world-marl-runpod",
        "--local-out-root",
        str(tmp_path / "runs"),
        "--ssh-key",
        str(key),
    ]


def _forbid_subprocess(cmd, **kwargs):
    raise AssertionError(f"unexpected subprocess call: {cmd}")


def _patch_main_collaborators(monkeypatch, tmp_path) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(sys, "argv", _manifest_main_argv(tmp_path))
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setattr(runpod, "require_local_tools", lambda names: None)
    monkeypatch.setattr(runpod, "require_repo", lambda root: None)
    monkeypatch.setattr(runpod, "run_json", lambda cmd: {"id": "pod123"})
    monkeypatch.setattr(runpod, "run", _forbid_subprocess)
    info = runpod.SshInfo(
        user="root", host="1.2.3.4", port=22, key_path=tmp_path / "runpod_key"
    )
    monkeypatch.setattr(runpod, "wait_for_ssh", lambda pod_id, key, args: info)
    monkeypatch.setattr(runpod, "get_ssh_info", lambda pod_id, key: info)
    monkeypatch.setattr(runpod, "ensure_remote_rsync", lambda info: None)
    monkeypatch.setattr(runpod, "sync_repo", lambda root, remote, info: None)
    monkeypatch.setattr(
        runpod,
        "start_remote_job_detached",
        lambda *a, **k: calls.append("start_remote_job_detached"),
    )
    monkeypatch.setattr(
        runpod,
        "wait_for_detached_job",
        lambda *a, **k: (calls.append("wait_for_detached_job"), "done")[1],
    )
    monkeypatch.setattr(
        runpod,
        "download_outputs",
        lambda remote, local, info: calls.append("download_outputs"),
    )
    monkeypatch.setattr(
        runpod, "delete_pod", lambda pod_id: (calls.append("delete_pod"), True)[1]
    )
    monkeypatch.setattr(
        runpod,
        "stop_for_inspection",
        lambda pod_id, remote: calls.append("stop_for_inspection"),
    )
    return calls


def _read_manifest(tmp_path) -> dict:
    manifests = list((tmp_path / "runs").rglob("manifest.json"))
    assert len(manifests) == 1
    return json.loads(manifests[0].read_text(encoding="utf-8"))


def test_main_runs_job_detached_and_deletes_pod_on_success(tmp_path, monkeypatch):
    calls = _patch_main_collaborators(monkeypatch, tmp_path)

    assert runpod.main() == 0

    manifest = _read_manifest(tmp_path)
    assert manifest["pod_id"] == "pod123"
    assert manifest["status"] == "completed-pod-deleted"
    assert manifest["finished_at"]
    assert manifest["job"] == "compare-world-models"
    started = calls.index("start_remote_job_detached")
    waited = calls.index("wait_for_detached_job")
    downloaded = calls.index("download_outputs")
    assert started < waited < downloaded
    assert "stop_for_inspection" not in calls


def test_main_leaves_pod_running_when_monitoring_lost(tmp_path, monkeypatch):
    calls = _patch_main_collaborators(monkeypatch, tmp_path)

    def lose_monitoring(*a, **k):
        raise RuntimeError("Connection reset by peer")

    monkeypatch.setattr(runpod, "wait_for_detached_job", lose_monitoring)

    assert runpod.main() == 1

    manifest = _read_manifest(tmp_path)
    assert manifest["status"] == "detached-monitoring-lost"
    assert "stop_for_inspection" not in calls, "job may still be running"
    assert "delete_pod" not in calls


def test_main_abandons_monitoring_on_interrupt_without_stopping_job(
    tmp_path, monkeypatch
):
    calls = _patch_main_collaborators(monkeypatch, tmp_path)

    def interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(runpod, "wait_for_detached_job", interrupt)

    assert runpod.main() == 130

    manifest = _read_manifest(tmp_path)
    assert manifest["status"] == "detached-monitoring-abandoned"
    assert "stop_for_inspection" not in calls, "^C must never kill the remote job"
    assert "delete_pod" not in calls


def test_main_downloads_outputs_then_stops_pod_when_job_fails(tmp_path, monkeypatch):
    calls = _patch_main_collaborators(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runpod,
        "wait_for_detached_job",
        lambda *a, **k: (calls.append("wait_for_detached_job"), "failed")[1],
    )

    assert runpod.main() == 1

    manifest = _read_manifest(tmp_path)
    assert manifest["status"] == "job-failed-pod-stopped"
    assert manifest["finished_at"]
    assert calls.index("download_outputs") < calls.index("stop_for_inspection")
    assert "delete_pod" not in calls


def test_main_stops_pod_when_launch_fails_before_job_starts(tmp_path, monkeypatch):
    calls = _patch_main_collaborators(monkeypatch, tmp_path)

    def boom(root, remote, info):
        raise RuntimeError("rsync failed")

    monkeypatch.setattr(runpod, "sync_repo", boom)

    assert runpod.main() == 1

    manifest = _read_manifest(tmp_path)
    assert manifest["pod_id"] == "pod123"
    assert manifest["status"] == "stopped-for-inspection"
    assert manifest["finished_at"]
    assert "stop_for_inspection" in calls
