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
    assert args.policy_objective == "direct"
    assert args.regularizer == "sigreg"
    assert args.online_reset_replay_env


def test_single_agent_jepa_cli_can_disable_online_replay_reset(monkeypatch):
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
            "--no-online-reset-replay-env",
        ],
    )

    args = train_dmc_jepa.parse_args()

    assert not args.online_reset_replay_env


def test_single_agent_jepa_cli_uses_regularizer_weight_alias(monkeypatch):
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
            "--regularizer-weight",
            "0.25",
        ],
    )

    args = train_dmc_jepa.parse_args()

    assert args.regularizer_weight == 0.25


def test_single_agent_jepa_cli_allows_model_only_history_context(monkeypatch):
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
            "--context-window",
            "2",
            "--policy-train-steps",
            "0",
        ],
    )

    args = train_dmc_jepa.parse_args()

    assert args.context_window == 2


def test_single_agent_jepa_cli_rejects_policy_history_context(monkeypatch):
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
            "--context-window",
            "2",
            "--policy-train-steps",
            "1",
        ],
    )

    try:
        train_dmc_jepa.parse_args()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected context-window policy guard to fail")
