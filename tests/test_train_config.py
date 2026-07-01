from __future__ import annotations

import dataclasses
import sys

from world_marl.config import TrainConfig
from world_marl.scripts import train_e2e


def test_trainconfig_defaults_match_argparse(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["world-marl-train-e2e"])
    namespace = train_e2e.parse_args()
    assert dataclasses.asdict(TrainConfig()) == vars(namespace)


def test_from_namespace_round_trips(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["world-marl-train-e2e", "--num-runs", "7", "--wm-flow-type", "discrete"],
    )
    namespace = train_e2e.parse_args()
    cfg = TrainConfig.from_namespace(namespace)
    assert dataclasses.asdict(cfg) == vars(namespace)
    assert cfg.num_runs == 7
    assert cfg.wm_flow_type == "discrete"
