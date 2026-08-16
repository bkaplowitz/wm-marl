from __future__ import annotations

import json
from pathlib import Path

import ruamel.yaml as yaml

from dreamarl.config import DreaMARLRunSpec
from dreamarl.contracts import verify_run_contract
from dreamarl.launcher import run_training
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
    assert contract["policy_peer_access"] is False
    assert contract["imagination_atomicity"] == (
        "all decentralized actions are sampled before one joint-conditioned step"
    )


def test_single_and_multi_agent_runs_share_one_contract(tmp_path: Path) -> None:
    singleton = verify_run_contract(_spec(tmp_path, num_agents=1))
    multi = verify_run_contract(_spec(tmp_path, num_agents=7))
    assert {key: value for key, value in singleton.items() if key != "num_agents"} == {
        key: value for key, value in multi.items() if key != "num_agents"
    }
    assert singleton["marl_architecture"] == "joint-action-conditioned local JEPA"
    assert singleton["agent_axis_adapter"] == "[B,T,A,...] <-> [B*A,T,...]"


def test_canonical_yaml_matches_the_locked_manifest(tmp_path: Path) -> None:
    config = yaml.YAML(typ="safe").load(
        (algorithm_root() / "configs.yaml").read_text(encoding="utf-8")
    )["defaults"]["agent"]
    manifest = _spec(tmp_path).to_dict()
    assert config["dyn"]["typ"] == manifest["world_model"]
    assert config["enc"]["typ"] == manifest["visual_encoder"]
    assert config["objective"] == manifest["world_model_objective"]
    assert config["embedding_target"] == manifest["embedding_target"]
    assert config["embedding_loss"] == manifest["embedding_loss"]
    assert config["spatial_jepa"]["topology"] == manifest["spatial_mask_topology"]


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
