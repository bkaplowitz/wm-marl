from __future__ import annotations

import sys

from world_marl.scripts import train_dmc_jepa


def test_single_agent_jepa_cli_accepts_brax_env(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "world-marl-validate-single-agent-world-model",
            "--env",
            "brax:reacher",
            "--collect-steps",
            "8",
            "--validation-steps",
            "8",
            "--chunk-length",
            "4",
            "--open-loop-horizon",
            "2",
        ],
    )

    args = train_dmc_jepa.parse_args()

    assert args.env == "brax:reacher"
    assert args.env_workers == 1
