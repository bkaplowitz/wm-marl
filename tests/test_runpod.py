from __future__ import annotations

import argparse
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
        "run_json",
        lambda _: {"error": "pod not ready", "status": "INITIALIZING"},
    )

    with pytest.raises(RuntimeError, match="ssh info not ready.*INITIALIZING"):
        runpod.get_ssh_info("pod-id", tmp_path / "key")
