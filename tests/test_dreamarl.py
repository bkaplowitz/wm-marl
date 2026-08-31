from __future__ import annotations

import json
from pathlib import Path

import pytest

from dreamarl.config import DreaMARLRunSpec, PUBLIC_ALGORITHMS
from dreamarl.contracts import verify_run_contract
from dreamarl.launcher import run_training
from dreamarl.main import _load_configs, _resolve_config_profiles, _validate_script
from dreamarl.runtime import algorithm_root
from dreamarl.scripts.eval_dreamarl import main as eval_main


def _spec(tmp_path: Path, **updates) -> DreaMARLRunSpec:
    values = {
        "experiment_dir": tmp_path / "final",
        "task": "smac_3m",
        "num_agents": 3,
        "seed": 234,
        "train_steps": 50_000,
        "platform": "cpu",
        "python": Path("/usr/bin/python3"),
    }
    values.update(updates)
    return DreaMARLRunSpec(**values)


def test_repository_exposes_only_final_dreamarl() -> None:
    configs = _load_configs()

    assert PUBLIC_ALGORITHMS == ("final-dreamarl",)
    assert set(configs) == {
        "defaults",
        "dreamarl_final",
        "smac_vector",
        "debug",
    }


def test_final_profile_resolves_to_the_locked_architecture(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    resolved = _resolve_config_profiles(_load_configs(), spec.configs)

    assert spec.configs == ["smac_vector", "dreamarl_final"]
    assert resolved.replay_context == 192
    assert resolved.replay.sampling == "recent_world_uniform_behavior"
    assert resolved.replay.size == 250_000
    assert not resolved.replay.online
    assert resolved.run.train_ratio == 1024
    assert resolved.run.actor_critic_start_step == 3000
    assert resolved.agent.action_mask_reduction == "balanced"
    assert resolved.agent.policy.units == 512
    assert resolved.agent.imag_loss.actent == pytest.approx(6e-4)
    assert resolved.agent.marl.ctde.actor_lr == pytest.approx(1e-5)
    assert resolved.agent.marl.ctde.teammate_belief.enabled
    multistep = resolved.agent.marl.ctde.multistep_jepa
    assert multistep.enabled and multistep.belief_context
    assert tuple(multistep.horizons) == (1, 2, 4, 8)
    assert multistep.action_counterfactual_mode == "cyclic"
    assert multistep.plan_aggregation == "identity_attention"
    assert multistep.plan_attention_heads == 4
    assert resolved.agent.loss_scales.ctde_multistep_jepa_action == 0.0


def test_manifest_and_command_have_no_architecture_overrides(tmp_path: Path) -> None:
    spec = _spec(tmp_path, task="smac_8m", num_agents=8)
    manifest = spec.to_dict()

    assert "--agent.marl.stage" not in spec.command
    assert "--run.train_ratio" not in spec.command
    assert "--replay.sampling" not in spec.command
    assert manifest["algorithm"] == "final-dreamarl"
    assert manifest["train_ratio"] == 1024.0
    assert manifest["replay_sampling"] == "recent_world_uniform_behavior"
    assert manifest["replay_context"] == 192
    assert manifest["ctde"]["temporal_transformer"]["layers"] == 12
    assert manifest["ctde"]["actor_units"] == 512
    assert manifest["ctde"]["actor_learning_rate"] == pytest.approx(1e-5)
    assert manifest["policy_modules"] == [
        "enc",
        "dyn",
        "pol",
        "ctde_teammate_belief",
        "ctde_teammate_actor",
    ]


def test_contract_preserves_strict_decentralized_execution(tmp_path: Path) -> None:
    contract = verify_run_contract(_spec(tmp_path))

    assert contract["execution"]["mode"] == "strict_decentralized"
    assert contract["execution"]["policy_peer_access"] is False
    assert contract["execution"]["runtime_communication"] is False
    assert contract["training"]["actor_objective"] == "score_function_reinforce"
    assert contract["ctde"]["training_only"] is True
    assert contract["ctde"]["role_aware_peer_plan"] is True


def test_non_final_inputs_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported algorithm"):
        _spec(tmp_path, algorithm="ctde-one-step")
    with pytest.raises(ValueError, match="requires at least two agents"):
        _spec(tmp_path, num_agents=1)
    with pytest.raises(ValueError, match="supports SMAC"):
        _spec(tmp_path, task="dmc_reacher_easy")


def test_only_final_train_and_eval_modes_are_supported() -> None:
    for script in ("train_eval", "parallel", "parallel_env", "parallel_replay"):
        with pytest.raises(ValueError, match="only train and eval_only"):
            _validate_script(script, 3)
    _validate_script("train", 3)


def test_dry_run_and_fixed_eval_preserve_the_final_profile(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    assert run_training(spec, dry_run=True) == 0
    manifest = json.loads(
        (spec.experiment_dir / "launch.json").read_text(encoding="utf-8")
    )
    assert manifest["algorithm"] == "final-dreamarl"
    assert manifest["configs"] == ["smac_vector", "dreamarl_final"]

    checkpoint = spec.logdir / "ckpt" / "checkpoint-123"
    checkpoint.mkdir(parents=True)
    (checkpoint / "done").touch()
    (checkpoint.parent / "latest").write_text(checkpoint.name, encoding="utf-8")
    assert eval_main([str(spec.experiment_dir), "--dry-run"]) == 0
    launch_file = next((spec.experiment_dir / "evaluation").glob("*.launch.json"))
    launch = json.loads(launch_file.read_text(encoding="utf-8"))
    assert launch["algorithm"] == "final-dreamarl"
    assert launch["configs"] == ["smac_vector", "dreamarl_final"]


def test_config_file_is_packaged() -> None:
    assert (algorithm_root() / "configs.yaml").is_file()
