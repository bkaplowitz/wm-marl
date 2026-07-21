from __future__ import annotations

import sys
from types import SimpleNamespace

from world_marl.scripts import train_dmc_jepa_bootstrap


def test_bootstrap_enables_determinism_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["world-marl-train-dmc-jepa"])
    monkeypatch.setattr(
        train_dmc_jepa_bootstrap,
        "configure_deterministic_environment",
        lambda: calls.append("determinism"),
    )
    monkeypatch.setitem(
        sys.modules,
        "world_marl.scripts.train_dmc_jepa",
        SimpleNamespace(main=lambda: calls.append("train")),
    )

    train_dmc_jepa_bootstrap.main()

    assert calls == ["determinism", "train"]


def test_bootstrap_allows_explicit_nondeterministic_compute(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["world-marl-train-dmc-jepa", "--no-deterministic-compute"],
    )
    monkeypatch.setattr(
        train_dmc_jepa_bootstrap,
        "configure_deterministic_environment",
        lambda: calls.append("determinism"),
    )
    monkeypatch.setitem(
        sys.modules,
        "world_marl.scripts.train_dmc_jepa",
        SimpleNamespace(main=lambda: calls.append("train")),
    )

    train_dmc_jepa_bootstrap.main()

    assert calls == ["train"]
