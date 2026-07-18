from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from world_marl.jepa.config import canonical_jepa_config
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


def test_direct_cli_defaults_match_the_canonical_500k_configuration(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["world-marl-train-dmc-jepa"])

    args = train_dmc_jepa.parse_args()

    expected = canonical_jepa_config()
    assert {name: getattr(args, name) for name in expected} == expected


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


def test_cli_accepts_phase_aligned_training_snapshot(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--num-envs",
            "2",
            "--online-collect-steps",
            "4",
            "--online-iterations",
            "2",
            "--training-snapshot-env-steps",
            "24",
        ),
    )

    args = train_dmc_jepa.parse_args()

    assert args.training_snapshot_env_steps == [24]


def test_cli_rejects_misaligned_training_snapshot(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--num-envs",
            "2",
            "--online-collect-steps",
            "4",
            "--online-iterations",
            "2",
            "--training-snapshot-env-steps",
            "25",
        ),
    )

    with pytest.raises(SystemExit):
        train_dmc_jepa.parse_args()


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


def test_cli_accepts_temporally_coherent_random_bootstrap(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args("--initial-random-action-hold-steps", "4"),
    )

    args = train_dmc_jepa.parse_args()

    assert args.initial_random_action_hold_steps == 4


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


def test_random_collection_holds_actions_and_resamples_after_forced_reset():
    class Adapter:
        num_envs = 1

        def __init__(self):
            self.action = 0

        def reset(self):
            return np.zeros((1, 1, 1), dtype=np.float32)

        def sample_actions(self, rng):
            del rng
            self.action += 1
            return np.asarray([[[self.action]]], dtype=np.float32)

        def step(self, actions):
            return SimpleNamespace(
                observations=np.asarray(actions, dtype=np.float32),
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

    train_dmc_jepa._collect_random_steps(
        adapter,
        adapter.reset(),
        np.random.default_rng(0),
        replay,
        steps=6,
        reset_interval=3,
        action_hold_steps=2,
        desc="test",
        quiet=True,
    )

    np.testing.assert_array_equal(
        replay.actions[:, 0, 0],
        np.asarray([1.0, 1.0, 2.0, 3.0, 3.0, 4.0]),
    )


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


def test_cli_uses_single_current_actor_critic_objective(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--target-critic-ema-decay",
            "0.98",
            "--policy-replay-critic-loss-coef",
            "0.3",
            "--policy-slow-value-regularization-coef",
            "1.0",
        ),
    )

    args = train_dmc_jepa.parse_args()

    assert not hasattr(args, "policy_gradient_mode")
    assert not hasattr(args, "policy_return_mode")
    assert not hasattr(args, "policy_return_normalization")
    assert args.target_critic_ema_decay == 0.98
    assert args.policy_replay_critic_loss_coef == 0.3
    assert args.policy_slow_value_regularization_coef == 1.0


def test_cli_accepts_budget_relative_value_clip_schedule(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--value-clip",
            "100",
            "--value-clip-final",
            "200",
            "--value-clip-schedule-start-env-steps",
            "100000",
            "--value-clip-schedule-end-env-steps",
            "200000",
        ),
    )

    args = train_dmc_jepa.parse_args()

    assert train_dmc_jepa._scheduled_value_clip(
        args,
        train_env_steps=99_999,
    ) == pytest.approx(100.0)
    assert train_dmc_jepa._scheduled_value_clip(
        args,
        train_env_steps=150_000,
    ) == pytest.approx(150.0)
    assert train_dmc_jepa._scheduled_value_clip(
        args,
        train_env_steps=200_000,
    ) == pytest.approx(200.0)


def test_cli_rejects_partial_value_clip_schedule(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--value-clip-final",
            "200",
        ),
    )

    with pytest.raises(SystemExit):
        train_dmc_jepa.parse_args()


def test_online_actor_update_interval_can_start_after_warmup(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--online-policy-actor-update-interval",
            "2",
            "--online-policy-actor-update-interval-start-env-steps",
            "50000",
        ),
    )

    args = train_dmc_jepa.parse_args()

    assert (
        train_dmc_jepa._scheduled_online_actor_update_interval(
            args,
            train_env_steps=49_999,
        )
        == 1
    )
    assert (
        train_dmc_jepa._scheduled_online_actor_update_interval(
            args,
            train_env_steps=50_000,
        )
        == 2
    )


def test_online_encoder_can_freeze_after_budget_threshold(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--online-freeze-encoder-after-env-steps",
            "100000",
        ),
    )

    args = train_dmc_jepa.parse_args()

    assert not train_dmc_jepa._scheduled_online_encoder_freeze(
        args,
        train_env_steps=99_999,
    )
    assert train_dmc_jepa._scheduled_online_encoder_freeze(
        args,
        train_env_steps=100_000,
    )


def test_cli_accepts_recent_replay_and_curve_evaluation(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--online-recent-world-model-fraction",
            "0.4",
            "--online-recent-world-model-until-env-steps",
            "50000",
            "--online-recent-replay-steps",
            "128",
            "--online-recent-replay-max-oversample",
            "10",
            "--policy-reset-start-fraction",
            "0.05",
            "--policy-reset-start-fraction-start-env-steps",
            "200000",
            "--policy-reset-start-max-age",
            "63",
            "--curve-eval-interval-env-steps",
            "50000",
            "--curve-eval-episodes",
            "20",
            "--curve-eval-seed",
            "9000000",
        ),
    )

    args = train_dmc_jepa.parse_args()

    assert args.online_policy_actor_update_interval == 1
    assert args.online_policy_actor_update_interval_start_env_steps == 0
    assert train_dmc_jepa._scheduled_recent_world_model_fraction(
        args,
        train_env_steps=49_999,
    ) == pytest.approx(0.4)
    assert train_dmc_jepa._scheduled_recent_world_model_fraction(
        args,
        train_env_steps=50_000,
    ) == 0.0
    assert args.online_recent_world_model_until_env_steps == 50_000
    assert args.online_recent_replay_steps == 128
    assert args.online_recent_replay_max_oversample == 10.0
    assert args.policy_reset_start_fraction == 0.05
    assert args.policy_reset_start_fraction_start_env_steps == 200_000
    assert (
        train_dmc_jepa._scheduled_policy_reset_start_fraction(
            args,
            train_env_steps=199_999,
        )
        == 0.0
    )
    assert train_dmc_jepa._scheduled_policy_reset_start_fraction(
        args,
        train_env_steps=200_000,
    ) == pytest.approx(0.05)
    assert args.policy_reset_start_max_age == 63
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


def test_policy_start_mixture_adds_reset_aligned_main_replay_states():
    def replay_with_values(values: list[float]) -> SequenceReplayBuffer:
        replay = SequenceReplayBuffer(
            capacity=12,
            num_envs=1,
            observation_shape=(1,),
            action_shape=(1,),
            action_dtype=np.float32,
        )
        for value in values:
            replay.add_step(
                observations=np.asarray([[value]], dtype=np.float32),
                actions=np.zeros((1, 1), dtype=np.float32),
                rewards=np.zeros((1,), dtype=np.float32),
                dones=np.zeros((1,), dtype=np.float32),
            )
        return replay

    config = train_dmc_jepa.JepaConfig(
        observation_dim=1,
        action_dim=1,
        action_mode="continuous",
        latent_dim=8,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2,
        max_horizon=2,
        context_window=2,
        sigreg_num_proj=4,
        sigreg_knots=3,
        twohot_bins=7,
    )
    replay = replay_with_values([4.0, 4.0, *([1.0] * 8)])
    observations, _ = train_dmc_jepa._sample_policy_starts_with_reset_mix(
        replay,
        np.random.default_rng(0),
        config=config,
        batch_size=10,
        reset_start_indices=(np.asarray([0]), np.asarray([0])),
        reset_start_fraction=0.2,
    )

    endpoint_values = np.asarray(observations[:, -1, 0])
    np.testing.assert_array_equal(endpoint_values[-2:], 4.0)


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
