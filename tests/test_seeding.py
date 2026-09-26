import os
from pathlib import Path

from majepa import main
from majepa.envs import smac
from majepa.launcher import runtime_environment


def test_smac_workers_use_distinct_seed_offsets(monkeypatch):
    config = main._resolve_config_profiles(main._load_configs(), ["baseline"])
    config = config.update({"seed": 7, "task": "smac_2s3z"})
    monkeypatch.setattr(smac, "SMACEnv", lambda task, **kwargs: kwargs["seed"])
    monkeypatch.setattr(main, "wrap_env", lambda env, config: env)
    assert [main.make_env(config, index) for index in (0, 1, 100_000)] == [
        7,
        8,
        100_007,
    ]


def test_launcher_preserves_host_environment_and_selects_local_code(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SC2PATH", str(tmp_path / "StarCraftII"))
    monkeypatch.setenv("PYTHONHASHSEED", "31")
    environment = runtime_environment(
        task="smac_3m",
        infrastructure_root=tmp_path,
        artifact_dir=tmp_path,
    )
    assert environment["PYTHONHASHSEED"] == "31"
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment["PYTHONPATH"].split(os.pathsep)[:2] == [
        str(tmp_path),
        str(Path(__file__).parents[1] / "src"),
    ]
    assert environment is not os.environ


def test_maintained_profile_uses_synchronous_snapshot_sampling():
    config = main._resolve_config_profiles(main._load_configs(), ["baseline"])
    assert config.run.replay_stream_mode == "snapshot_staggered"
    assert config.run.replay_startup_behavior_min_starts == 4
    assert config.run.replay_trace_batches == 16
    assert config.run.isolate_report_rng is True
