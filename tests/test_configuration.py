import json

import elements
import pytest

from majepa.config import MAJEPARunSpec
from majepa.main import _load_configs, _resolve_config_profiles
from majepa.scripts.evaluate import _evaluation_protocol, _latest_checkpoint
from majepa.scripts.evaluate import main as evaluate_main
from majepa.scripts.train import main as train_main


@pytest.mark.parametrize("samples", [None, 1, 2])
def test_training_setup_accepts_imagined_action_sample_count(tmp_path, samples):
    arguments = [
        "--task",
        "smac_2s3z",
        "--num-agents",
        "5",
        "--experiment-dir",
        str(tmp_path),
        "--dry-run",
    ]
    if samples is not None:
        arguments.extend(["--imag-action-samples", str(samples)])
    assert train_main(arguments) == 0
    expected = 2 if samples is None else samples
    manifest = json.loads((tmp_path / "launch.json").read_text())
    command = manifest["command"]
    index = command.index("--agent.imag_action_samples")
    config = _resolve_config_profiles(_load_configs(), manifest["configs"])
    resolved = elements.Flags(config).parse(command[index : index + 2])
    assert resolved.agent.imag_action_samples == expected
    assert manifest["ctde"]["imag_action_samples"] == expected


@pytest.mark.parametrize("samples", [0, 3, True, 1.0])
def test_run_spec_rejects_invalid_imagination_sample_count(tmp_path, samples):
    with pytest.raises(ValueError, match="imag_action_samples must be 1 or 2"):
        MAJEPARunSpec(tmp_path, "smac_2s3z", 5, imag_action_samples=samples)


def test_evaluation_uses_manifest_protocol_and_complete_checkpoint(tmp_path):
    manifest = MAJEPARunSpec(tmp_path, "smac_2s3z", 5, seed=7).to_dict()
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
