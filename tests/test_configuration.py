import json

import elements
import pytest

from majepa.config import MAJEPARunSpec
from majepa.main import _load_configs, _resolve_config_profiles
from majepa.scripts.evaluate import _evaluation_protocol, _latest_checkpoint
from majepa.scripts.evaluate import main as evaluate_main
from majepa.scripts.train import main as train_main


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


def test_evaluation_uses_manifest_protocol_and_complete_checkpoint(tmp_path):
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
    assert result["episodes"] == manifest["evaluation_protocol"]["episodes"]
    assert result["envs"] == manifest["evaluation_protocol"]["envs"]
    assert result["eval_seed"] == 7 + manifest["evaluation_protocol"]["seed_offset"]
    assert result["policy_mode"] == "deterministic"
    assert result["configs"] == manifest["configs"]
    command = result["command"]
    assert command[command.index("--run.from_checkpoint") + 1] == str(checkpoint)
    assert command[command.index("--run.eval_policy_mode") + 1] == "eval"
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
    assert _evaluation_protocol(manifest, episodes=100, envs=4, eval_seed=123) == (
        100,
        4,
        123,
    )
