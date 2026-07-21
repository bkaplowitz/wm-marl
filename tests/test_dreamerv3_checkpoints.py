from __future__ import annotations

import pickle

import pytest

from world_marl.baselines.dreamerv3.checkpoints import latest_checkpoint


def test_latest_checkpoint_resolves_pointer_and_step(tmp_path):
    checkpoint = tmp_path / "ckpt" / "timestamp"
    checkpoint.mkdir(parents=True)
    (checkpoint / "done").touch()
    with (checkpoint / "step.pkl").open("wb") as handle:
        pickle.dump(42_000, handle)
    (checkpoint.parent / "latest").write_text("timestamp")
    result = latest_checkpoint(checkpoint.parent)
    assert result.path == checkpoint
    assert result.env_steps == 42_000


def test_latest_checkpoint_rejects_incomplete_save(tmp_path):
    checkpoint = tmp_path / "ckpt" / "timestamp"
    checkpoint.mkdir(parents=True)
    (checkpoint.parent / "latest").write_text("timestamp")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        latest_checkpoint(checkpoint.parent)
