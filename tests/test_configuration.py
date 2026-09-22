import json
from types import SimpleNamespace
from unittest.mock import Mock

import elements
import pytest

from majepa.config import MAJEPARunSpec
from majepa.main import _load_configs, _resolve_config_profiles
from majepa.scripts.evaluate import _evaluation_protocol, _latest_checkpoint
from majepa.scripts.evaluate import main as evaluate_main
from majepa.scripts.train import main as train_main


def test_default_config_uses_reference_map_and_wandb(tmp_path):
    configs = _load_configs()
    assert configs["baseline"] == {}

    defaults = _resolve_config_profiles(configs, ["defaults"])
    assert defaults.task == "smac_2s3z"
    assert defaults.agent.num_agents == 5
    assert defaults.logger.outputs == ("jsonl", "wandb")
    assert defaults.logdir == "~/logdir/{timestamp}"
    assert _resolve_config_profiles(configs, ["baseline"]).flat == defaults.flat
    spec = MAJEPARunSpec(tmp_path, "smac_2s3z", 5)
    _, flags = elements.Flags(configs=["baseline"]).parse_known(spec.command[3:])
    resolved = elements.Flags(defaults).parse(flags)
    assert resolved.flat == defaults.update(logdir=str(spec.logdir)).flat


def test_training_setup_records_experiment_controls(tmp_path):
    arguments = [
        "--task",
        "smac_2s3z",
        "--num-agents",
        "5",
        "--experiment-dir",
        str(tmp_path),
        "--dry-run",
        "--wm-critic-value-scale",
        "0.2",
        "--wm-joint-objective-scale",
        "0.3",
        "--wm-joint-prediction-gradient",
        "--wm-joint-prediction-scale",
        "0.4",
        "--no-local-prior",
        "--action-margin-loss-scale",
        "0.0",
        "--categorical-stoch",
        "8",
        "--categorical-classes",
        "16",
    ]
    assert train_main(arguments) == 0
    manifest = json.loads((tmp_path / "launch.json").read_text())
    command = manifest["command"]
    config = _resolve_config_profiles(_load_configs(), manifest["configs"])
    for key, expected in (
        ("critic_value_scale", 0.2),
        ("joint_objective_scale", 0.3),
        ("joint_prediction", True),
        ("joint_prediction_scale", 0.4),
    ):
        index = command.index("--agent.world_model_gradients." + key)
        resolved = elements.Flags(config).parse(command[index : index + 2])
        assert resolved.agent.world_model_gradients[key] == expected
        assert manifest["world_model_gradients"][key] == expected
    for option, config_key, expected in (
        ("--agent.dyn.parallel_transformer.local_prior", "local_prior", False),
        (
            "--agent.loss_scales.ctde_multistep_jepa_action",
            "action_margin_loss_scale",
            0.0,
        ),
        ("--agent.dyn.parallel_transformer.stoch", "categorical_stoch", 8),
        ("--agent.dyn.parallel_transformer.classes", "categorical_classes", 16),
    ):
        assert command[command.index(option) + 1] == str(expected)
        assert manifest["model_controls"][config_key] == expected


@pytest.mark.parametrize("scale", [-1.0, float("nan"), float("inf")])
def test_run_spec_rejects_invalid_world_gradient_scale(tmp_path, scale):
    with pytest.raises(ValueError, match="wm_critic_value_scale"):
        MAJEPARunSpec(tmp_path, "smac_2s3z", 5, wm_critic_value_scale=scale)


@pytest.mark.parametrize("scale", [-1.0, float("nan"), float("inf")])
def test_run_spec_rejects_invalid_joint_prediction_scale(tmp_path, scale):
    with pytest.raises(ValueError, match="wm_joint_prediction_scale"):
        MAJEPARunSpec(tmp_path, "smac_2s3z", 5, wm_joint_prediction_scale=scale)


@pytest.mark.parametrize(
    "field", ["wm_joint_objective_scale", "action_margin_loss_scale"]
)
@pytest.mark.parametrize("scale", [-1.0, float("nan"), float("inf")])
def test_run_spec_rejects_invalid_nonnegative_scale(tmp_path, field, scale):
    with pytest.raises(ValueError, match=field):
        MAJEPARunSpec(tmp_path, "smac_2s3z", 5, **{field: scale})


@pytest.mark.parametrize("field", ["categorical_stoch", "categorical_classes"])
@pytest.mark.parametrize("size", [0, -1])
def test_run_spec_rejects_invalid_categorical_size(tmp_path, field, size):
    with pytest.raises(ValueError, match=field):
        MAJEPARunSpec(tmp_path, "smac_2s3z", 5, **{field: size})


def test_training_setup_records_joint_prediction_scale(tmp_path):
    assert (
        train_main(
            [
                "--task",
                "smac_2s3z",
                "--num-agents",
                "5",
                "--experiment-dir",
                str(tmp_path),
                "--dry-run",
                "--wm-joint-prediction-gradient",
                "--wm-joint-prediction-scale",
                "0.1",
            ]
        )
        == 0
    )
    manifest = json.loads((tmp_path / "launch.json").read_text())
    command = manifest["command"]
    key = "--agent.world_model_gradients.joint_prediction_scale"
    assert command[command.index(key) + 1] == "0.1"
    assert manifest["world_model_gradients"]["joint_prediction_scale"] == 0.1


def test_evaluation_uses_shell_protocol_and_complete_checkpoint(tmp_path):
    manifest = MAJEPARunSpec(
        tmp_path,
        "smac_2s3z",
        5,
        seed=7,
        wm_critic_value_scale=0.1,
        wm_joint_prediction_gradient=True,
    ).to_dict()
    (tmp_path / "launch.json").write_text(json.dumps(manifest))
    checkpoint = tmp_path / "run" / "ckpt" / "complete"
    checkpoint.mkdir(parents=True)
    (checkpoint.parent / "latest").write_text(checkpoint.name)
    (checkpoint / "done").touch()
    assert evaluate_main([str(tmp_path), "--dry-run"]) == 0
    (launch,) = (tmp_path / "evaluation").glob("*.launch.json")
    result = json.loads(launch.read_text())
    assert result["episodes"] == 100
    assert result["envs"] == 4
    assert result["eval_seed"] == 7
    assert result["policy_mode"] == "deterministic"
    assert result["configs"] == [*manifest["configs"], "eval_only"]
    command = result["command"]
    assert command[command.index("--run.from_checkpoint") + 1] == str(checkpoint)
    assert command[command.index("--run.eval_policy_mode") + 1] == "eval"
    _, flags = elements.Flags(configs=["baseline"]).parse_known(command[3:])
    config = _resolve_config_profiles(_load_configs(), result["configs"])
    resolved = elements.Flags(config).parse(flags)
    for key, value in {
        "script": "eval_only",
        "run.eval_worker_offset": 100000,
        "run.world_model_start_step": 0,
        "run.curve_eval_interval": 0,
        "run.eval_envs": 1,
        "jax.precompile": False,
    }.items():
        assert resolved.flat[key] == value
    assert (
        command[command.index("--agent.world_model_gradients.critic_value_scale") + 1]
        == "0.1"
    )
    assert (
        command[command.index("--agent.world_model_gradients.joint_prediction") + 1]
        == "True"
    )
    (checkpoint / "done").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete checkpoint"):
        _latest_checkpoint(tmp_path)


def test_evaluation_protocol_respects_explicit_overrides(tmp_path):
    manifest = MAJEPARunSpec(tmp_path, "smac_2s3z", 5).to_dict()
    manifest["evaluation_protocol"] = {
        "episodes": 32,
        "envs": 1,
        "seed_offset": 50_000,
    }
    assert _evaluation_protocol(manifest, episodes=None, envs=None, eval_seed=None) == (
        100,
        4,
        0,
    )
    assert _evaluation_protocol(manifest, episodes=100, envs=4, eval_seed=123) == (
        100,
        4,
        123,
    )


@pytest.mark.parametrize("overrides", [False, True])
def test_evaluation_defaults_follow_profile(tmp_path, monkeypatch, overrides):
    from majepa import config as specification, main

    configs = _load_configs()
    configs["eval_only"]["run"].update(
        eval_eps=7, envs=3, eval_worker_offset=123456, eval_policy_mode="eval_sample"
    )
    elements.Config(configs).save(tmp_path / "configs.yaml")
    monkeypatch.setattr(specification, "__file__", str(tmp_path / "config.py"))
    monkeypatch.setattr(main, "folder", tmp_path)
    manifest = MAJEPARunSpec(tmp_path, "smac_2s3z", 5, seed=7).to_dict()
    assert manifest["evaluation_protocol"] == {
        "episodes": 7,
        "envs": 3,
        "worker_offset": 123456,
        "seed_offset": 0,
        "policy_mode": "stochastic",
    }
    (tmp_path / "launch.json").write_text(json.dumps(manifest))
    checkpoint = tmp_path / "run/ckpt/complete"
    checkpoint.mkdir(parents=True)
    (checkpoint / "done").touch()
    (checkpoint.parent / "latest").write_text(checkpoint.name)
    arguments = [str(tmp_path), "--dry-run"]
    if overrides:
        arguments += [
            "--episodes",
            "5",
            "--envs",
            "2",
            "--eval-seed",
            "123",
            "--policy-mode",
            "deterministic",
        ]
    assert evaluate_main(arguments) == 0
    (launch,) = (tmp_path / "evaluation").glob("*.launch.json")
    result = json.loads(launch.read_text())
    assert result["episodes"] == (5 if overrides else 7)
    assert result["envs"] == (2 if overrides else 3)
    assert result["eval_seed"] == (123 if overrides else 7)
    assert result["worker_offset"] == 123456
    assert result["policy_mode"] == ("deterministic" if overrides else "stochastic")
    parsed, flags = elements.Flags(configs=["baseline"]).parse_known(
        result["command"][3:]
    )
    resolved = elements.Flags(_resolve_config_profiles(configs, parsed.configs)).parse(
        flags
    )
    assert resolved.run.eval_eps == result["episodes"]
    assert resolved.run.envs == result["envs"]
    assert resolved.run.eval_worker_offset == result["worker_offset"]
    assert resolved.run.eval_policy_mode == ("eval" if overrides else "eval_sample")


def test_wrapper_wandb_runs_have_separate_ids_and_never_resume(tmp_path, monkeypatch):
    from majepa import launcher
    from majepa.scripts import evaluate

    monkeypatch.setenv("SC2PATH", str(tmp_path))
    monkeypatch.setenv("WANDB_RUN_ID", "inherited-run")
    monkeypatch.setenv("WANDB_JOB_TYPE", "inherited-type")
    monkeypatch.setenv("WANDB_RESUME", "allow")
    train = Mock(return_value=SimpleNamespace(stdout=[], wait=lambda: 0))
    evaluation = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(
        launcher,
        "subprocess",
        SimpleNamespace(
            Popen=train,
            PIPE=launcher.subprocess.PIPE,
            STDOUT=launcher.subprocess.STDOUT,
        ),
    )
    monkeypatch.setattr(evaluate, "subprocess", SimpleNamespace(run=evaluation))
    assert launcher.run_training(MAJEPARunSpec(tmp_path, "smac_2s3z", 5)) == 0
    checkpoint = tmp_path / "run/ckpt/complete"
    checkpoint.mkdir(parents=True)
    (checkpoint / "done").touch()
    (checkpoint.parent / "latest").write_text(checkpoint.name)
    assert evaluate_main([str(tmp_path)]) == 0
    assert evaluate_main([str(tmp_path)]) == 0
    calls = [train.call_args, *evaluation.call_args_list]
    environments = [call.kwargs["env"] for call in calls]
    assert len({env["WANDB_RUN_ID"] for env in environments}) == 3
    for env, phase in zip(environments, ("train", "final100", "final100")):
        assert env["WANDB_RUN_ID"] != "inherited-run"
        assert env["WANDB_RUN_ID"].endswith("-" + phase)
        assert env["WANDB_JOB_TYPE"] == phase
        assert env["WANDB_RESUME"] == "never"
