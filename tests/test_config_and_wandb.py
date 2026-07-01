from __future__ import annotations

import sys

import pytest

from world_marl.logging import RunLogger
from world_marl.scripts import train_e2e


def _write_config(tmp_path, body: str):
    path = tmp_path / "run.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_config_file_sets_defaults(tmp_path, monkeypatch):
    config = _write_config(tmp_path, "num_runs: 7\nseed: 123\n")
    monkeypatch.setattr(sys, "argv", ["world-marl-train-e2e", "--config", str(config)])

    args = train_e2e.parse_args()

    assert args.num_runs == 7
    assert args.seed == 123


def test_cli_flag_overrides_config(tmp_path, monkeypatch):
    config = _write_config(tmp_path, "num_runs: 7\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["world-marl-train-e2e", "--config", str(config), "--num-runs", "2"],
    )

    args = train_e2e.parse_args()

    assert args.num_runs == 2


def test_unknown_config_key_errors(tmp_path, monkeypatch):
    config = _write_config(tmp_path, "not_a_real_key: 1\n")
    monkeypatch.setattr(sys, "argv", ["world-marl-train-e2e", "--config", str(config)])

    with pytest.raises(SystemExit):
        train_e2e.parse_args()


class _FakeWandbRun:
    def __init__(self) -> None:
        self.logged: list[dict] = []

    def log(self, row: dict) -> None:
        self.logged.append(row)


def test_run_logger_mirrors_metrics_to_wandb(tmp_path):
    fake = _FakeWandbRun()
    logger = RunLogger(tmp_path, wandb_run=fake)

    logger.append_metrics({"update": 1, "reward": 0.5})

    assert fake.logged == [{"update": 1, "reward": 0.5}]


def test_run_logger_without_wandb_is_unaffected(tmp_path):
    logger = RunLogger(tmp_path)

    logger.append_metrics({"update": 1, "reward": 0.5})

    assert (tmp_path / "metrics.jsonl").exists()
