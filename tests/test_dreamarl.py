from __future__ import annotations

import json
from pathlib import Path

import pytest
import ruamel.yaml as yaml

from dreamarl.config import DreaMARLRunSpec
from dreamarl.contracts import verify_run_contract
from dreamarl.launcher import run_training
from dreamarl.main import _validate_script
from dreamarl.replay import ExponentialRecency, RecentReplay
from dreamarl.runtime import algorithm_root
from dreamarl.scripts.eval_dreamarl import main as eval_main


def _spec(tmp_path: Path, **updates) -> DreaMARLRunSpec:
    values = {
        "experiment_dir": tmp_path / "run",
        "task": "meltingpot_externality_mushrooms__dense",
        "num_agents": 5,
        "seed": 7,
        "train_steps": 50_000,
        "platform": "cpu",
        "python": Path("/usr/bin/python3"),
        "wandb_project": "world-marl",
        "wandb_entity": "osaze-obahor",
    }
    values.update(updates)
    return DreaMARLRunSpec(**values)


def test_run_spec_exposes_only_the_locked_algorithm(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    manifest = spec.to_dict()
    assert manifest["world_model"] == "parallel_transformer"
    assert manifest["world_model_objective"] == "embedding"
    assert manifest["visual_encoder"] == "simple"
    assert manifest["spatial_mask_topology"] == "fixed_count"
    assert manifest["spatial_mask_ratio"] == 0.5
    assert manifest["embedding_target"] == "ema"
    assert manifest["embedding_loss"] == "cosine"
    assert manifest["sigreg_scale"] == 0.05
    assert manifest["sigreg_aggregation"] == "per_agent"
    assert manifest["replay_context"] == 128
    assert manifest["environment_seed_mode"].startswith(
        "construction-time Lab2D seed stream"
    )
    assert manifest["environment_reproducibility"] == (
        "construction_seed_controlled_not_trajectory_deterministic"
    )
    assert manifest["train_agent_steps_budget"] == 250_000
    assert not any(
        flag in spec.command
        for flag in (
            "--agent.dyn.typ",
            "--agent.enc.typ",
            "--agent.objective",
            "--agent.spatial_jepa.topology",
        )
    )
    contract = verify_run_contract(spec)
    assert contract["marl_stage"] == "b0"
    assert contract["policy_peer_access"] is False
    assert contract["imagination_atomicity"] == (
        "team starts remain grouped while every transition uses only its own action"
    )
    assert contract["environment_reproducibility"] == (
        "construction_seed_controlled_not_trajectory_deterministic"
    )


def test_single_and_multi_agent_runs_share_one_contract(tmp_path: Path) -> None:
    singleton = verify_run_contract(_spec(tmp_path, num_agents=1))
    multi = verify_run_contract(_spec(tmp_path, num_agents=7))
    differing = {"num_agents"}
    assert {key: value for key, value in singleton.items() if key not in differing} == {
        key: value for key, value in multi.items() if key not in differing
    }
    assert singleton["policy_information"] == multi["policy_information"]
    assert singleton["marl_architecture"] == "shared independent local JEPA"
    assert singleton["agent_axis_adapter"] == "[B,T,A,...] <-> [B*A,T,...]"


def test_canonical_yaml_matches_the_locked_manifest(tmp_path: Path) -> None:
    defaults = yaml.YAML(typ="safe").load(
        (algorithm_root() / "configs.yaml").read_text(encoding="utf-8")
    )["defaults"]
    config = defaults["agent"]
    manifest = _spec(tmp_path).to_dict()
    assert config["dyn"]["typ"] == manifest["world_model"]
    assert config["enc"]["typ"] == manifest["visual_encoder"]
    assert config["objective"] == manifest["world_model_objective"]
    assert config["embedding_target"] == manifest["embedding_target"]
    assert config["embedding_loss"] == manifest["embedding_loss"]
    assert config["spatial_jepa"]["topology"] == manifest["spatial_mask_topology"]
    assert config["sigreg"]["aggregation"] == "per_agent"
    assert config["marl"] == {
        "stage": "b0",
        "execution": "strict_decentralized",
        "agent_jepa": {
            "slots": 8,
            "width": 256,
            "heads": 4,
            "layers": 2,
            "predictor_layers": 2,
            "ffup": 4,
            "predictor_hidden": 512,
            "local_grad_scale": 0.0,
            "k0_scale": 0.1,
            "future_scale": 1.0,
            "future_set_scale": 1.0,
            "teacher_rate": 0.01,
            "teacher_every": 1,
            "mask_min": 0.25,
            "mask_max": 0.5,
            "matching_temperature": 0.02,
            "sinkhorn_iterations": 10,
            "predicted_set_scale": 1.0,
            "source_set_scale": 1.0,
            "hidden_coverage_scale": 1.0,
            "variance_scale": 0.1,
            "covariance_scale": 0.1,
            "slot_target_std": 0.1,
            "utility_probe": False,
            "act": "silu",
            "norm": "rms",
            "winit": "trunc_normal_in",
        },
    }
    temporal = config["dyn"]["parallel_transformer"]
    assert defaults["replay_context"] == temporal["context"] * temporal["layers"]


def test_curve_evaluation_is_explicit(tmp_path: Path) -> None:
    baseline = _spec(tmp_path)
    measured = _spec(
        tmp_path,
        curve_eval_interval=50_000,
        curve_eval_episodes=20,
        curve_eval_seed_offset=50_000,
    )
    assert "--run.curve_eval_interval" not in baseline.command
    index = measured.command.index("--run.curve_eval_interval")
    assert measured.command[index + 1] == "50000"


def test_training_cadence_is_explicit_and_recorded(tmp_path: Path) -> None:
    spec = _spec(tmp_path, train_ratio=1024.0)
    index = spec.command.index("--run.train_ratio")
    assert spec.command[index + 1] == "1024.0"
    assert spec.to_dict()["optimizer_updates_per_environment_step"] == 1.0


def test_recent_replay_keeps_the_exponential_selector_when_empty() -> None:
    replay = RecentReplay(length=4, capacity=32, recency_decay=0.9998, seed=7)
    assert isinstance(replay.sampler, ExponentialRecency)


def test_b1_launch_selects_only_training_time_agent_jepa(tmp_path: Path) -> None:
    spec = _spec(tmp_path, marl_stage="b1")
    stage = spec.command.index("--agent.marl.stage")
    contract = verify_run_contract(spec)
    assert spec.command[stage + 1] == "b1"
    assert contract["agent_axis_jepa"].startswith("whole-agent-masked prediction")
    assert contract["team_teacher"].startswith("training-only EMA set encoder")
    assert contract["team_teacher_execution_access"] is False
    assert contract["policy_peer_access"] is False


def test_bounded_imagination_starts_are_recorded_and_forwarded(tmp_path: Path) -> None:
    spec = _spec(tmp_path, imagination_starts=16)
    index = spec.command.index("--agent.imag_last")
    assert spec.command[index + 1] == "16"
    assert spec.to_dict()["imagination_starts"] == 16


def test_generic_reporting_modes_are_rejected_for_marl() -> None:
    for script in ("train_eval", "parallel", "parallel_env", "parallel_replay"):
        with pytest.raises(ValueError, match="single-agent reporting"):
            _validate_script(script, 5)
    _validate_script("train", 5)
    _validate_script("parallel", 1)


def test_dry_run_records_the_locked_contract(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    assert run_training(spec, dry_run=True) == 0
    manifest = json.loads(
        (spec.experiment_dir / "launch.json").read_text(encoding="utf-8")
    )
    assert manifest["implementation"] == "first-party decoder-free DreaMARL"
    assert manifest["spatial_mask_topology"] == "fixed_count"
    assert manifest["configs"] == ["meltingpot_vision"]


def test_visual_dmc_uses_the_same_singleton_algorithm(tmp_path: Path) -> None:
    spec = _spec(tmp_path, task="dmc_reacher_easy", num_agents=1)
    assert spec.configs == ["dmc_vision"]


def test_fixed_evaluation_rebuilds_the_training_architecture(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    assert run_training(spec, dry_run=True) == 0
    checkpoint = spec.logdir / "ckpt" / "checkpoint-123"
    checkpoint.mkdir(parents=True)
    (checkpoint / "done").touch()
    (checkpoint.parent / "latest").write_text(checkpoint.name, encoding="utf-8")

    assert eval_main([str(spec.experiment_dir), "--dry-run"]) == 0
    launch_file = next((spec.experiment_dir / "evaluation").glob("*.launch.json"))
    command = json.loads(launch_file.read_text(encoding="utf-8"))["command"]
    assert command[command.index("--agent.num_agents") + 1] == "5"
    assert "--agent.marl.mechanism" not in command
    assert "--agent.behavior.objective" not in command
    assert "--replay_context" not in command
