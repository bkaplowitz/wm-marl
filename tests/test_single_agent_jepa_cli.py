from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from world_marl.jepa.replay import SequenceReplayBuffer
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
    assert args.regularizer == "sigreg"
    assert args.online_reset_replay_env


def test_single_agent_jepa_cli_accepts_shared_validation_seed(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "world-marl-validate-single-agent-world-model",
            "--env",
            "dmc:reacher/easy",
            "--collect-steps",
            "8",
            "--validation-steps",
            "8",
            "--validation-seed",
            "1000042",
            "--chunk-length",
            "4",
            "--open-loop-horizon",
            "2",
        ],
    )

    args = train_dmc_jepa.parse_args()

    assert args.validation_seed == 1_000_042


def test_single_agent_jepa_cli_accepts_reset_rich_bootstrap(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "world-marl-validate-single-agent-world-model",
            "--env",
            "dmc:reacher/easy",
            "--collect-steps",
            "12",
            "--initial-reset-interval",
            "6",
            "--validation-steps",
            "8",
            "--chunk-length",
            "4",
            "--open-loop-horizon",
            "2",
        ],
    )

    args = train_dmc_jepa.parse_args()

    assert args.initial_reset_interval == 6


def test_random_collection_marks_nonterminal_reset_cuts():
    class Adapter:
        num_envs = 1

        def __init__(self):
            self.reset_calls = 0
            self.step_count = 0

        def reset(self):
            self.reset_calls += 1
            self.step_count = 0
            return np.asarray([[[100 * self.reset_calls]]], dtype=np.float32)

        def sample_actions(self, rng):
            del rng
            return np.zeros((1, 1, 1), dtype=np.float32)

        def step(self, actions):
            del actions
            self.step_count += 1
            return SimpleNamespace(
                observations=np.asarray(
                    [[[100 * self.reset_calls + self.step_count]]],
                    dtype=np.float32,
                ),
                rewards=np.zeros((1, 1), dtype=np.float32),
                dones=np.zeros((1, 1), dtype=np.float32),
            )

    adapter = Adapter()
    replay = SequenceReplayBuffer(
        capacity=6,
        num_envs=1,
        observation_shape=(1,),
        action_shape=(1,),
        action_dtype=np.float32,
    )

    observations, env_steps = train_dmc_jepa._collect_random_steps(
        adapter,
        adapter.reset(),
        np.random.default_rng(0),
        replay,
        steps=6,
        reset_interval=3,
        desc="test reset-rich bootstrap",
        quiet=True,
    )

    assert env_steps == 6
    assert adapter.reset_calls == 3
    np.testing.assert_array_equal(
        replay.cuts[:, 0],
        np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
    )
    np.testing.assert_array_equal(replay.dones[:, 0], np.zeros(6))
    np.testing.assert_array_equal(observations, np.asarray([[[300.0]]]))


def test_single_agent_jepa_cli_accepts_wandb_video_controls(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "world-marl-validate-single-agent-world-model",
            "--env",
            "dmc:reacher/easy",
            "--collect-steps",
            "8",
            "--validation-steps",
            "8",
            "--chunk-length",
            "4",
            "--open-loop-horizon",
            "2",
            "--wandb-project",
            "world-marl",
            "--wandb-entity",
            "osaze-obahor",
            "--wandb-tags",
            "jepa",
            "reacher",
            "--wandb-videos",
            "--wandb-video-every-phases",
            "2",
            "--wandb-video-frame-stride",
            "5",
            "--online-checkpoint-interval",
            "5",
            "--value-clip",
            "400",
        ],
    )

    args = train_dmc_jepa.parse_args()

    assert args.wandb_project == "world-marl"
    assert args.wandb_entity == "osaze-obahor"
    assert args.wandb_tags == ["jepa", "reacher"]
    assert args.wandb_videos
    assert args.wandb_video_every_phases == 2
    assert args.wandb_video_frame_stride == 5
    assert args.online_checkpoint_interval == 5
    assert args.value_clip == 400.0


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


def test_single_agent_jepa_cli_accepts_policy_risk_penalties(monkeypatch):
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
            "--policy-selection-std-penalty",
            "0.25",
            "--online-policy-std-penalty",
            "0.5",
            "--policy-selection-failure-penalty",
            "300",
            "--online-policy-failure-penalty",
            "400",
            "--policy-failure-return-threshold",
            "100",
            "--policy-success-return-threshold",
            "900",
            "--policy-action-bound-coef",
            "2.0",
            "--policy-action-bound-limit",
            "0.85",
            "--policy-uncertainty-coef",
            "3.0",
            "--policy-actor-cvar-fraction",
            "0.25",
            "--policy-actor-cvar-coef",
            "0.5",
        ],
    )

    args = train_dmc_jepa.parse_args()

    assert args.policy_selection_std_penalty == 0.25
    assert args.online_policy_std_penalty == 0.5
    assert args.policy_selection_failure_penalty == 300
    assert args.online_policy_failure_penalty == 400
    assert args.policy_failure_return_threshold == 100
    assert args.policy_success_return_threshold == 900
    assert args.policy_action_bound_coef == 2.0
    assert args.policy_action_bound_limit == 0.85
    assert args.policy_uncertainty_coef == 3.0
    assert args.policy_actor_cvar_fraction == 0.25
    assert args.policy_actor_cvar_coef == 0.5


def test_policy_score_penalizes_failure_rate():
    metrics = train_dmc_jepa._return_tail_metrics(
        [0.0, 950.0, 1000.0, 980.0],
        failure_threshold=100.0,
        success_threshold=900.0,
    )
    evaluation = {
        "mean_return": 732.5,
        "std_return": 25.0,
        **metrics,
    }

    score = train_dmc_jepa._policy_evaluation_score(
        evaluation,
        std_penalty=0.5,
        failure_penalty=400.0,
    )

    assert metrics["failure_rate"] == 0.25
    assert metrics["success_rate"] == 0.75
    assert score == pytest.approx(732.5 - 0.5 * 25.0 - 400.0 * 0.25)


def test_collection_reports_use_episode_finish_train_steps():
    rows = []
    logger = SimpleNamespace(append_metrics=rows.append)
    metrics = {
        "returns": [100.0, 950.0],
        "lengths": [1000, 1000],
        "episode_finish_train_env_steps": [16_000, 16_016],
        "mean_return": 525.0,
        "std_return": 425.0,
        "return_p10": 185.0,
        "return_cvar10": 100.0,
        "failure_rate": 0.0,
        "success_rate": 0.5,
        "completed_episodes": 2,
    }

    train_dmc_jepa._log_collection_episode_reports(
        logger,
        metrics,
        online_iteration=3,
        control="none",
    )

    assert [row["budget/train_env_steps"] for row in rows] == [16_000, 16_016]
    assert [row["report/episode_return"] for row in rows] == [100.0, 950.0]
    assert train_dmc_jepa._collection_report_summary(metrics) == {
        "return_mean": 525.0,
        "return_std": 425.0,
        "return_p10": 185.0,
        "return_cvar10": 100.0,
        "failure_rate": 0.0,
        "success_rate": 0.5,
        "completed_episodes": 2,
    }


def test_single_agent_jepa_cli_accepts_candidate_refit_flags(monkeypatch):
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
            "--online-candidate-refit",
            "--online-validation-steps",
            "9",
            "--online-candidate-gate-metric",
            "model/jepa_loss",
            "--online-candidate-min-recent-improvement",
            "0.01",
            "--online-candidate-max-anchor-degradation",
            "0.02",
            "--online-candidate-eval-interval",
            "250",
            "--online-candidate-anchor-penalty",
            "2.0",
            "--online-anchor-batch-fraction",
            "0.75",
            "--online-control-value-weight",
            "0.3",
        ],
    )

    args = train_dmc_jepa.parse_args()

    assert args.online_candidate_refit
    assert args.online_validation_steps == 9
    assert args.online_candidate_gate_metric == "model/jepa_loss"
    assert args.online_candidate_min_recent_improvement == 0.01
    assert args.online_candidate_max_anchor_degradation == 0.02
    assert args.online_candidate_eval_interval == 250
    assert args.online_candidate_anchor_penalty == 2.0
    assert args.online_anchor_batch_fraction == 0.75
    assert args.online_control_value_weight == 0.3


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


def test_single_agent_jepa_cli_accepts_dreamer_control_parity_flags(monkeypatch):
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
            "--stochastic-actor",
            "--policy-gradient-mode",
            "reinforce",
            "--policy-return-normalization",
            "ema-percentile",
            "--policy-return-ema-decay",
            "0.99",
            "--target-critic-ema-decay",
            "0.98",
            "--policy-slow-value-regularization-coef",
            "1.0",
            "--policy-replay-critic-return-mode",
            "lambda",
            "--policy-replay-critic-all-steps",
            "--input-symlog",
            "--activation",
            "silu",
            "--normalization",
            "rms",
            "--actor-output-scale",
            "0.01",
            "--value-output-scale",
            "0",
            "--reward-output-scale",
            "0",
            "--optimizer-warmup-steps",
            "1000",
            "--adaptive-grad-clip",
            "0.3",
            "--optimizer-epsilon",
            "1e-8",
            "--no-online-freeze-encoder",
        ],
    )

    args = train_dmc_jepa.parse_args()

    assert args.policy_gradient_mode == "reinforce"
    assert args.policy_return_normalization == "ema-percentile"
    assert args.policy_return_ema_decay == 0.99
    assert args.policy_replay_critic_return_mode == "lambda"
    assert args.policy_replay_critic_all_steps
    assert args.policy_slow_value_regularization_coef == 1.0
    assert args.input_symlog
    assert args.activation == "silu"
    assert args.normalization == "rms"
    assert args.actor_output_scale == 0.01
    assert args.value_output_scale == 0.0
    assert args.reward_output_scale == 0.0
    assert args.optimizer_warmup_steps == 1000
    assert args.adaptive_grad_clip == 0.3
    assert args.optimizer_epsilon == 1e-8
    assert not args.online_freeze_encoder


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
