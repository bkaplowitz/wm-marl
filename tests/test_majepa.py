from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import elements

import embodied.jax.outs as outs

from majepa.config import MAJEPARunSpec, PUBLIC_ALGORITHMS
from majepa.contracts import verify_run_contract
from majepa.launcher import run_training
from majepa.main import _load_configs, _resolve_config_profiles, _validate_script
from majepa.models.heads import apply_action_mask, apply_legal_unimix
from majepa.runtime import algorithm_root
from majepa.scripts.evaluate import main as evaluate
from majepa.scripts.train import main as train
from majepa.train import train as train_loop


def _spec(tmp_path: Path, **updates) -> MAJEPARunSpec:
    values = {
        "experiment_dir": tmp_path / "run",
        "task": "smac_3m",
        "num_agents": 3,
        "seed": 1,
        "train_steps": 50_000,
        "platform": "cpu",
        "python": Path("/usr/bin/python3"),
    }
    values.update(updates)
    return MAJEPARunSpec(**values)


def test_only_the_locked_algorithm_is_public() -> None:
    configs = _load_configs()
    assert PUBLIC_ALGORITHMS == ("ma-jepa",)
    assert {"defaults", "ma_jepa", "smac_vector", "debug"} == set(configs)


def test_locked_profile_values(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    resolved = _resolve_config_profiles(_load_configs(), spec.configs)

    assert spec.configs == ["smac_vector", "ma_jepa"]
    assert spec.train_envs == resolved.run.envs == 16
    assert resolved.replay_context == 192
    assert resolved.replay.sampling == "recent_world_uniform_behavior"
    assert resolved.run.train_ratio == 128
    assert resolved.run.world_model_start_step == 0
    assert resolved.run.ppo_start_step == 5000
    assert resolved.agent.imag_length == 5
    assert resolved.agent.action_mask_reduction == "balanced"
    assert resolved.agent.policy.units == 1024
    assert resolved.agent.collection_unimix == pytest.approx(0.05)
    assert resolved.agent.ppo.epochs == 5
    assert resolved.agent.ppo.clip_epsilon == pytest.approx(0.2)
    assert resolved.agent.ppo.entropy_coefficient == pytest.approx(1e-2)
    assert not resolved.agent.ppo.entropy_schedule.enabled
    assert resolved.agent.ppo.lam == pytest.approx(0.95)
    assert resolved.agent.ppo.actor_lr == pytest.approx(3e-5)
    assert resolved.agent.ppo.critic_lr == pytest.approx(3e-5)
    assert resolved.agent.slowvalue.rate == pytest.approx(1.0)
    ctde = resolved.agent.marl.ctde
    assert ctde.teammate_belief.enabled
    assert ctde.teammate_belief.actor_residual
    assert tuple(ctde.multistep_jepa.horizons) == (1, 2, 4, 8)
    assert ctde.multistep_jepa.action_counterfactual_mode == "all_legal_mean"
    assert ctde.multistep_jepa.plan_aggregation == "mean"
    assert resolved.agent.loss_scales.ctde_multistep_jepa_action == pytest.approx(0.25)


def test_manifest_describes_the_locked_architecture(tmp_path: Path) -> None:
    manifest = _spec(tmp_path, task="smac_8m", num_agents=8).to_dict()
    assert manifest["implementation"] == "MA-JEPA"
    assert manifest["algorithm"] == "ma-jepa"
    assert manifest["train_ratio"] == 128.0
    assert manifest["replay_sampling"] == "recent_world_uniform_behavior"
    assert manifest["ctde"]["actor_units"] == 1024
    assert manifest["ctde"]["actor_learning_rate"] == pytest.approx(3e-5)
    assert manifest["ctde"]["critic_learning_rate"] == pytest.approx(3e-5)
    assert manifest["ctde"]["ppo_start_step"] == 5000
    assert manifest["ctde"]["imagination_horizon"] == 5
    assert manifest["ctde"]["ppo"]["epochs"] == 5
    assert manifest["ctde"]["ppo"]["clip_epsilon"] == pytest.approx(0.2)
    assert manifest["actor_objective"] == "clipped_imagined_ppo"
    assert manifest["learner_batches_per_environment_step"] == pytest.approx(0.125)
    assert manifest["actor_updates_per_environment_step"] == pytest.approx(0.625)
    assert manifest["ctde"]["action_counterfactuals"] == "all_legal_mean"
    assert manifest["ctde"]["action_counterfactual_scale"] == pytest.approx(0.25)


def test_execution_contract_is_decentralized(tmp_path: Path) -> None:
    contract = verify_run_contract(_spec(tmp_path))
    assert contract["execution"]["mode"] == "strict_decentralized"
    assert contract["execution"]["policy_peer_access"] is False
    assert contract["execution"]["runtime_communication"] is False
    assert contract["ctde"]["training_only"] is True


def test_invalid_public_inputs_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported algorithm"):
        _spec(tmp_path, algorithm="ablation")
    with pytest.raises(ValueError, match="requires at least two agents"):
        _spec(tmp_path, num_agents=1)
    with pytest.raises(ValueError, match="supports SMAC"):
        _spec(tmp_path, task="dmc_reacher_easy")
    for envs in (0, -1):
        with pytest.raises(ValueError, match="train_envs must be positive"):
            _spec(tmp_path, train_envs=envs)
    for script in ("train_eval", "parallel", "parallel_env", "parallel_replay"):
        with pytest.raises(ValueError, match="only train and eval_only"):
            _validate_script(script, 3)


@pytest.mark.parametrize("steps", [0, -1, 43.5])
def test_runtime_rejects_invalid_budgets_before_initialization(steps):
    with pytest.raises(ValueError, match="positive integer"):
        train_loop(None, None, None, None, None, SimpleNamespace(steps=steps))


@pytest.mark.parametrize("envs", [None, 1, 4, 16])
def test_parallel_training_cli_preserves_the_total_budget(
    tmp_path: Path, envs: int | None
):
    experiment = tmp_path / "parallel"
    assert (
        train(
            [
                "--task",
                "smac_3m",
                "--num-agents",
                "3",
                *(["--train-envs", str(envs)] if envs is not None else []),
                "--total-env-steps",
                "50000",
                "--eval-envs",
                "2",
                "--experiment-dir",
                str(experiment),
                "--platform",
                "cpu",
                "--dry-run",
            ]
        )
        == 0
    )
    manifest = json.loads((experiment / "launch.json").read_text())
    flags, other = elements.Flags(configs=["defaults"]).parse_known(
        manifest["command"][3:]
    )
    resolved = elements.Flags(
        _resolve_config_profiles(_load_configs(), flags.configs)
    ).parse(other)
    expected_envs = 16 if envs is None else envs
    assert resolved.run.envs == expected_envs
    assert resolved.run.steps == 50000
    assert resolved.run.train_ratio == 128
    assert manifest["train_envs"] == expected_envs
    assert manifest["train_env_steps_budget"] == 50000
    assert manifest["replay_prefill_steps"] == 5000
    assert manifest["train_agent_steps_budget"] == 150000
    assert manifest["evaluation_protocol"]["envs"] == 2


def test_dry_run_and_evaluation_keep_the_locked_profile(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    assert run_training(spec, dry_run=True) == 0
    manifest = json.loads((spec.experiment_dir / "launch.json").read_text())
    assert manifest["configs"] == ["smac_vector", "ma_jepa"]

    checkpoint = spec.logdir / "ckpt" / "checkpoint-123"
    checkpoint.mkdir(parents=True)
    (checkpoint / "done").touch()
    (checkpoint.parent / "latest").write_text(checkpoint.name)
    assert evaluate([str(spec.experiment_dir), "--dry-run"]) == 0
    launch_file = next((spec.experiment_dir / "evaluation").glob("*.launch.json"))
    launch = json.loads(launch_file.read_text())
    assert launch["algorithm"] == "ma-jepa"
    assert launch["configs"] == ["smac_vector", "ma_jepa"]


def test_config_file_is_packaged() -> None:
    assert (algorithm_root() / "configs.yaml").is_file()


def test_collection_unimix_only_mixes_legal_actions() -> None:
    distribution = outs.Categorical(jnp.array([[4.0, 0.0, -2.0, 3.0]]))
    distribution.raw_logits = jnp.array([[4.0, 0.0, -2.0, 3.0]])
    policy = apply_action_mask(
        {"action": distribution},
        jnp.array([[True, False, True, False]]),
        "action",
    )
    mixed = apply_legal_unimix(policy, "action", 0.2)["action"]
    probabilities = jax.nn.softmax(mixed.logits, axis=-1)
    expected = 0.8 * jax.nn.softmax(jnp.array([4.0, -2.0])) + 0.1

    np.testing.assert_allclose(np.asarray(probabilities)[0, [0, 2]], expected)
    np.testing.assert_array_equal(np.asarray(probabilities)[0, [1, 3]], 0.0)
