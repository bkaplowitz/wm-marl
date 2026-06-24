from __future__ import annotations

import argparse

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
