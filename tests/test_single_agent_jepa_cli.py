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
    assert args.online_freeze_encoder


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


def test_single_agent_jepa_cli_can_disable_online_encoder_freeze(monkeypatch):
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
            "--no-online-freeze-encoder",
        ],
    )

    args = train_dmc_jepa.parse_args()

    assert not args.online_freeze_encoder


def test_single_agent_jepa_cli_accepts_online_interface_drift_flags(monkeypatch):
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
            "--online-freeze-encoder",
            "--online-interface-eval-episodes",
            "7",
            "--online-interface-eval-num-envs",
            "3",
            "--online-behavior-distill-weight",
            "0.25",
            "--online-candidate-refit",
            "--online-validation-steps",
            "9",
            "--online-candidate-gate-metric",
            "model/control_prediction_loss",
            "--online-candidate-min-recent-improvement",
            "0.01",
            "--online-candidate-max-anchor-degradation",
            "0.02",
            "--control-alignment",
            "procrustes",
            "--online-latent-anchor-weight",
            "0.1",
            "--online-control-prediction-weight",
            "0.2",
        ],
    )

    args = train_dmc_jepa.parse_args()

    assert args.online_freeze_encoder
    assert args.online_interface_eval_episodes == 7
    assert args.online_interface_eval_num_envs == 3
    assert args.online_behavior_distill_weight == 0.25
    assert args.online_candidate_refit
    assert args.online_validation_steps == 9
    assert args.online_candidate_gate_metric == "model/control_prediction_loss"
    assert args.online_candidate_min_recent_improvement == 0.01
    assert args.online_candidate_max_anchor_degradation == 0.02
    assert args.control_alignment == "procrustes"
    assert args.control_interface == "procrustes"
    assert args.online_latent_anchor_weight == 0.1
    assert args.online_control_prediction_weight == 0.2


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


def test_single_agent_jepa_cli_accepts_uncertainty_gated_imagination(monkeypatch):
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
            "--dynamics-ensemble-size",
            "3",
            "--uncertainty-penalty",
            "0.2",
            "--uncertainty-latent-weight",
            "1.5",
            "--uncertainty-reward-weight",
            "0.5",
            "--uncertainty-continue-weight",
            "0.25",
            "--uncertainty-threshold",
            "0.75",
            "--uncertainty-budget",
            "2.5",
        ],
    )

    args = train_dmc_jepa.parse_args()

    assert args.dynamics_ensemble_size == 3
    assert args.uncertainty_penalty == 0.2
    assert args.uncertainty_latent_weight == 1.5
    assert args.uncertainty_reward_weight == 0.5
    assert args.uncertainty_continue_weight == 0.25
    assert args.uncertainty_threshold == 0.75
    assert args.uncertainty_budget == 2.5


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


def test_single_agent_jepa_cli_allows_direct_policy_history_context(monkeypatch):
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
            "--model-horizon",
            "3",
            "--target-gradient",
            "symmetric",
            "--no-residual-dynamics",
            "--critic-warmup-steps",
            "0",
        ],
    )

    args = train_dmc_jepa.parse_args()

    assert args.context_window == 2
    assert args.policy_train_steps == 1
    assert args.model_horizon == 3
    assert args.target_gradient == "symmetric"
    assert not args.residual_dynamics
