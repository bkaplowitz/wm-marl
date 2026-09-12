from pathlib import Path

import pytest

from majepa.main import _load_configs, _resolve_config_profiles
from majepa.scripts.evaluate import evaluation_config


def _config(profile="reference"):
    return _resolve_config_profiles(_load_configs(), [profile])


@pytest.mark.parametrize(
    ("profile", "deter", "lr"),
    [
        ("reference", 4096, 1e-4),
        ("wm_lr15e5", 4096, 1.5e-4),
        ("wm2048", 2048, 1e-4),
        ("wm2048_lr15e5", 2048, 1.5e-4),
    ],
)
def test_sweep_profiles_change_only_size_and_world_learning_rate(profile, deter, lr):
    config = _config(profile)
    expected = _config().update(
        {
            "agent.dyn.parallel_transformer.deter": deter,
            "agent.opt.lr": lr,
            "agent.marl.ctde.opt.lr": lr,
        }
    )
    assert config.flat == expected.flat


def test_evaluation_uses_saved_architecture_and_heldout_protocol(tmp_path):
    saved = _config("wm2048_lr15e5").update(seed=7, task="smac_3s_vs_4z")
    saved.save(tmp_path / "config.yaml")
    checkpoint = tmp_path / "ckpt" / "20260912T000000"
    checkpoint.mkdir(parents=True)
    (checkpoint.parent / "latest").write_text(checkpoint.name)
    (checkpoint / "done").touch()
    result = evaluation_config(tmp_path)
    assert result.agent.flat == saved.agent.flat
    assert result.seed == 7
    assert result.task == "smac_3s_vs_4z"
    assert result.script == "eval_only"
    assert result.run.eval_eps == 100
    assert result.run.envs == 4
    assert result.run.eval_worker_offset == 100000
    assert result.run.from_checkpoint == str(checkpoint)
    assert result.run.curve_eval_interval == 0
    assert Path(result.logdir) == tmp_path / "final100"
    (checkpoint / "done").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete checkpoint"):
        evaluation_config(tmp_path)


def test_evaluation_does_not_overwrite_previous_results(tmp_path):
    _config().save(tmp_path / "config.yaml")
    checkpoint = tmp_path / "ckpt" / "complete"
    checkpoint.mkdir(parents=True)
    (checkpoint.parent / "latest").write_text(checkpoint.name)
    (checkpoint / "done").touch()
    output = tmp_path / "final100"
    output.mkdir()
    (output / "evaluation_summary.json").write_text("{}")
    with pytest.raises(FileExistsError, match="already exists"):
        evaluation_config(tmp_path)
