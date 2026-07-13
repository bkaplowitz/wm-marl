from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from world_marl.jepa.replay import SequenceReplayBuffer
from world_marl.scripts import train_dmc_jepa


def _minimal_args(*extra: str) -> list[str]:
    return [
        "world-marl-train-dmc-jepa",
        "--collect-steps",
        "8",
        "--initial-reset-interval",
        "8",
        "--validation-steps",
        "8",
        "--chunk-length",
        "4",
        "--model-horizon",
        "2",
        "--open-loop-horizon",
        "2",
        "--context-window",
        "2",
        *extra,
    ]


def test_cli_accepts_dmc_and_brax_environments(monkeypatch):
    for env in ("dmc:reacher/easy", "brax:reacher"):
        monkeypatch.setattr(sys, "argv", _minimal_args("--env", env))
        args = train_dmc_jepa.parse_args()
        assert args.env == env


def test_cli_accepts_shared_validation_seed(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args("--validation-seed", "1000042"),
    )

    args = train_dmc_jepa.parse_args()

    assert args.validation_seed == 1_000_042


def test_cli_accepts_reset_rich_bootstrap(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--collect-steps",
            "12",
            "--initial-reset-interval",
            "6",
        ),
    )

    args = train_dmc_jepa.parse_args()

    assert args.collect_steps == 12
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
        desc="test",
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


def test_cli_accepts_wandb_video_controls(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--wandb-project",
            "world-marl",
            "--wandb-entity",
            "osaze-obahor",
            "--wandb-tags",
            "jepa",
            "reacher",
            "--wandb-videos",
            "--wandb-video-frame-stride",
            "5",
            "--value-clip",
            "400",
        ),
    )

    args = train_dmc_jepa.parse_args()

    assert args.wandb_project == "world-marl"
    assert args.wandb_entity == "osaze-obahor"
    assert args.wandb_tags == ["jepa", "reacher"]
    assert args.wandb_videos
    assert args.wandb_video_frame_stride == 5
    assert args.value_clip == 400.0


def test_cli_uses_regularizer_weight_alias(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args("--sigreg-weight", "0.125"),
    )

    args = train_dmc_jepa.parse_args()

    assert args.regularizer_weight == 0.125


def test_cli_exposes_current_dreamer_stabilizers(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--policy-gradient-mode",
            "reinforce",
            "--policy-return-mode",
            "lambda",
            "--policy-return-normalization",
            "ema-percentile",
            "--target-critic-ema-decay",
            "0.98",
            "--policy-replay-critic-loss-coef",
            "0.3",
            "--policy-slow-value-regularization-coef",
            "1.0",
        ),
    )

    args = train_dmc_jepa.parse_args()

    assert args.policy_gradient_mode == "reinforce"
    assert args.policy_return_mode == "lambda"
    assert args.policy_return_normalization == "ema-percentile"
    assert args.target_critic_ema_decay == 0.98
    assert args.policy_replay_critic_loss_coef == 0.3
    assert args.policy_slow_value_regularization_coef == 1.0


def test_cli_accepts_budget_relative_entropy_decay(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--actor-entropy-coef",
            "0.003",
            "--actor-entropy-final-coef",
            "0.0003",
            "--actor-entropy-decay-start-env-steps",
            "300000",
            "--actor-entropy-decay-end-env-steps",
            "500000",
        ),
    )

    args = train_dmc_jepa.parse_args()

    assert train_dmc_jepa._scheduled_actor_entropy_coef(
        args,
        train_env_steps=299_999,
    ) == pytest.approx(3e-3)
    assert train_dmc_jepa._scheduled_actor_entropy_coef(
        args,
        train_env_steps=400_000,
    ) == pytest.approx(1.65e-3)
    assert train_dmc_jepa._scheduled_actor_entropy_coef(
        args,
        train_env_steps=500_000,
    ) == pytest.approx(3e-4)


def test_cli_accepts_recent_replay_and_curve_evaluation(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--online-recent-replay-fraction",
            "0.5",
            "--online-recent-replay-steps",
            "128",
            "--online-recent-replay-max-oversample",
            "10",
            "--curve-eval-interval-env-steps",
            "50000",
            "--curve-eval-episodes",
            "20",
            "--curve-eval-seed",
            "9000000",
        ),
    )

    args = train_dmc_jepa.parse_args()

    assert args.online_recent_replay_fraction == 0.5
    assert args.online_recent_replay_steps == 128
    assert args.online_recent_replay_max_oversample == 10.0
    assert args.curve_eval_interval_env_steps == 50_000
    assert args.curve_eval_episodes == 20
    assert args.curve_eval_seed == 9_000_000


def test_recent_replay_batch_respects_requested_fraction():
    def replay_with_reward(reward: float) -> SequenceReplayBuffer:
        replay = SequenceReplayBuffer(
            capacity=12,
            num_envs=1,
            observation_shape=(1,),
            action_shape=(1,),
            action_dtype=np.float32,
        )
        for step in range(10):
            replay.add_step(
                observations=np.asarray([[step]], dtype=np.float32),
                actions=np.zeros((1, 1), dtype=np.float32),
                rewards=np.asarray([reward], dtype=np.float32),
                dones=np.zeros((1,), dtype=np.float32),
            )
        return replay

    batch = train_dmc_jepa._sample_replay_batch(
        replay_with_reward(1.0),
        np.random.default_rng(0),
        recent_replay=replay_with_reward(9.0),
        recent_fraction=0.3,
        batch_size=10,
        chunk_length=2,
        max_horizon=1,
    )

    np.testing.assert_array_equal(np.asarray(batch.rewards[:7]), 1.0)
    np.testing.assert_array_equal(np.asarray(batch.rewards[7:]), 9.0)


def test_recent_replay_oversample_cap_decays_fraction_with_replay_size():
    early_fraction = train_dmc_jepa._effective_recent_fraction(
        0.5,
        full_replay_size=3_136,
        recent_replay_size=320,
        max_oversample=10.0,
    )
    late_fraction = train_dmc_jepa._effective_recent_fraction(
        0.5,
        full_replay_size=9_408,
        recent_replay_size=320,
        max_oversample=10.0,
    )

    assert early_fraction == pytest.approx(0.4787234043)
    assert late_fraction == pytest.approx(0.234375)
    assert train_dmc_jepa._recent_oversample_ratio(
        late_fraction,
        full_replay_size=9_408,
        recent_replay_size=320,
    ) == pytest.approx(10.0)


def test_recent_replay_oversample_cap_is_optional():
    assert train_dmc_jepa._effective_recent_fraction(
        0.5,
        full_replay_size=9_408,
        recent_replay_size=320,
        max_oversample=0.0,
    ) == pytest.approx(0.5)


def test_cli_rejects_partial_entropy_decay(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args("--actor-entropy-final-coef", "0.0003"),
    )

    with pytest.raises(SystemExit):
        train_dmc_jepa.parse_args()


def test_cli_rejects_removed_checkpoint_search_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args("--policy-selection-interval", "100"),
    )

    with pytest.raises(SystemExit):
        train_dmc_jepa.parse_args()


def test_collection_reports_use_actual_episode_finish_steps():
    rows = []
    logger = SimpleNamespace(append_metrics=rows.append)
    metrics = {
        "returns": [100.0, 950.0],
        "lengths": [1000, 1000],
        "episode_finish_train_env_steps": [16_000, 16_016],
    }

    train_dmc_jepa._log_collection_episode_reports(
        logger,
        metrics,
        online_iteration=7,
    )

    assert [row["budget/train_env_steps"] for row in rows] == [16_000, 16_016]
    assert [row["report/episode_return"] for row in rows] == [100.0, 950.0]
