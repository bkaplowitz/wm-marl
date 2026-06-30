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
    monkeypatch.setattr(
        sys, "argv", ["world-marl-runpod", "--job", "benchmark-policy"]
    )

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
    assert "apt-get install" in remote_script


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
