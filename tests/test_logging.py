from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from world_marl import logging as run_logging
from world_marl.logging import RunLogger, WandbConfig


class _FakeConfig(dict):
    def update(self, payload, *, allow_val_change=False):
        assert allow_val_change
        super().update(payload)


class _FakeRun:
    def __init__(self):
        self.logged = []
        self.defined_metrics = []
        self.summary = {}
        self.config = _FakeConfig()
        self.exit_code = None

    def define_metric(self, name, **kwargs):
        self.defined_metrics.append((name, kwargs))

    def log(self, payload):
        self.logged.append(payload)

    def finish(self, *, exit_code):
        self.exit_code = exit_code


def _fake_wandb(run: _FakeRun):
    return SimpleNamespace(
        init=lambda **kwargs: run,
        Image=lambda path, caption=None: ("image", path, caption),
        Video=lambda path, format, caption=None: ("video", path, format, caption),
    )


def test_run_logger_mirrors_scalars_and_keeps_local_metrics(tmp_path, monkeypatch):
    run = _FakeRun()
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(run))
    logger = RunLogger(
        tmp_path,
        wandb_config=WandbConfig(
            project="test-project",
            config={"seed": 3},
        ),
    )

    logger.append_metrics(
        {
            "phase": "fit",
            "env_steps": 128,
            "model": {"loss": 0.25},
            "ignored": [1, 2, 3],
        }
    )
    logger.update_config({"resolved": {"latent_dim": 64}})
    logger.update_summary({"eval": {"return_mean": 950.0}})
    logger.close(exit_code=0)

    local_row = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert local_row["model"]["loss"] == 0.25
    assert run.logged[0]["model/loss"] == 0.25
    assert run.logged[0]["budget/train_env_steps"] == 128
    assert (
        "report/*",
        {"step_metric": "budget/train_env_steps"},
    ) in run.defined_metrics
    assert "ignored" not in run.logged[0]
    assert run.config["resolved"]["latent_dim"] == 64
    assert run.summary["eval/return_mean"] == 950.0
    assert run.exit_code == 0


def test_run_logger_wandb_failure_does_not_break_local_logging(tmp_path, monkeypatch):
    def fail_init(**kwargs):
        raise RuntimeError("tracking unavailable")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fail_init))
    with pytest.warns(RuntimeWarning, match="tracking unavailable"):
        logger = RunLogger(
            tmp_path,
            wandb_config=WandbConfig(project="test-project"),
        )

    logger.append_metrics({"phase": "fit", "model/loss": 1.0})

    assert (tmp_path / "metrics.jsonl").is_file()
    assert not logger.wandb_enabled


def test_run_logger_encodes_video_before_wandb_upload(tmp_path, monkeypatch):
    run = _FakeRun()
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(run))

    def fake_write_mp4(path, frames, fps):
        assert frames.shape == (3, 8, 8, 3)
        assert fps == 10
        path.write_bytes(b"mp4")

    monkeypatch.setattr(run_logging, "_write_mp4", fake_write_mp4)
    logger = RunLogger(
        tmp_path,
        wandb_config=WandbConfig(project="test-project"),
    )
    path = logger.write_video(
        "videos/eval.mp4",
        [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)],
        fps=10,
        key="videos/eval",
        caption="evaluation",
    )

    assert path == tmp_path / "videos/eval.mp4"
    assert path.read_bytes() == b"mp4"
    assert run.logged[-1]["videos/eval"][0] == "video"


def test_run_logger_plot_failure_does_not_stop_training(tmp_path, monkeypatch):
    import matplotlib.figure

    def fail_savefig(self, *args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fail_savefig)
    logger = RunLogger(tmp_path)

    with pytest.warns(RuntimeWarning, match="Plot saving failed"):
        path = logger.plot_world_model_loss([1.0, 0.5])

    assert path == tmp_path / "world_model_loss.png"
    assert not path.exists()
