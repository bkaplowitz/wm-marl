from __future__ import annotations

import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import freeze

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


def test_cli_accepts_slow_policy_bundle(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--policy-bundle-ema-decay",
            "0.995",
            "--policy-bundle-ema-start-env-steps",
            "50000",
            "--policy-bundle-collection-online-action-fraction",
            "1.0",
            "--policy-bundle-eval-online-action-fraction",
            "0.25",
        ),
    )

    args = train_dmc_jepa.parse_args()

    assert args.policy_bundle_ema_decay == pytest.approx(0.995)
    assert args.policy_bundle_ema_start_env_steps == 50_000
    assert args.policy_bundle_collection_online_action_fraction == pytest.approx(1.0)
    assert args.policy_bundle_eval_online_action_fraction == pytest.approx(0.25)


def test_cli_accepts_slow_policy_actor_kl_reference(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--policy-bundle-ema-decay",
            "0.98",
            "--policy-actor-kl-reference-mode",
            "slow-policy",
            "--policy-actor-slow-kl-target-per-dim",
            "0.02",
        ),
    )

    args = train_dmc_jepa.parse_args()

    assert args.policy_actor_kl_reference_mode == "slow-policy"
    assert args.policy_actor_slow_kl_target_per_dim == pytest.approx(0.02)


def test_cli_rejects_slow_policy_actor_kl_reference_without_bundle(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args("--policy-actor-kl-reference-mode", "slow-policy"),
    )

    with pytest.raises(SystemExit):
        train_dmc_jepa.parse_args()


def test_cli_rejects_slow_actor_kl_target_with_phase_reference(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args("--policy-actor-slow-kl-target-per-dim", "0.02"),
    )

    with pytest.raises(SystemExit):
        train_dmc_jepa.parse_args()


def test_cli_rejects_policy_bundle_mix_when_bundle_is_disabled(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args("--policy-bundle-eval-online-action-fraction", "0.25"),
    )

    with pytest.raises(SystemExit):
        train_dmc_jepa.parse_args()


def test_policy_bundle_ema_tracks_encoder_and_actor_together():
    initial = freeze(
        {
            "encoder": {"value": jnp.asarray([1.0])},
            "actor_head": {"value": jnp.asarray([2.0])},
            "value_head": {"value": jnp.asarray([3.0])},
        }
    )
    updated = freeze(
        {
            "encoder": {"value": jnp.asarray([5.0])},
            "actor_head": {"value": jnp.asarray([10.0])},
            "value_head": {"value": jnp.asarray([30.0])},
        }
    )

    slow = train_dmc_jepa._update_policy_bundle_ema(None, initial, decay=0.5)
    slow = train_dmc_jepa._update_policy_bundle_ema(slow, updated, decay=0.75)

    assert set(slow) == {"encoder", "actor_head"}
    np.testing.assert_allclose(slow["encoder"]["value"], [2.0])
    np.testing.assert_allclose(slow["actor_head"]["value"], [4.0])


def test_slow_policy_action_mix_uses_complete_policy_outputs(monkeypatch):
    def fake_select(state, *_args, **_kwargs):
        return state.actions

    monkeypatch.setattr(train_dmc_jepa, "select_continuous_actions", fake_select)
    online = SimpleNamespace(actions=jnp.asarray([[1.0, -1.0]]))
    slow = SimpleNamespace(actions=jnp.asarray([[-1.0, 1.0]]))

    actions = train_dmc_jepa._select_behavior_actions(
        online,
        slow,
        jnp.zeros((1, 1)),
        SimpleNamespace(),
        jnp.asarray([-1.0, -1.0]),
        jnp.asarray([1.0, 1.0]),
        key=jax.random.PRNGKey(0),
        stochastic=False,
        online_action_fraction=0.25,
    )

    np.testing.assert_allclose(actions, [[-0.5, 0.5]])


def test_current_only_collection_does_not_evaluate_slow_policy(monkeypatch):
    calls = []

    def fake_select(state, *_args, **_kwargs):
        calls.append(state.name)
        return state.actions

    monkeypatch.setattr(train_dmc_jepa, "select_continuous_actions", fake_select)
    online = SimpleNamespace(
        name="online",
        actions=jnp.asarray([[0.75, -0.25]]),
    )
    slow = SimpleNamespace(
        name="slow",
        actions=jnp.asarray([[-1.0, 1.0]]),
    )

    actions = train_dmc_jepa._select_behavior_actions(
        online,
        slow,
        jnp.zeros((1, 1)),
        SimpleNamespace(),
        jnp.asarray([-1.0, -1.0]),
        jnp.asarray([1.0, 1.0]),
        key=jax.random.PRNGKey(0),
        stochastic=True,
        online_action_fraction=1.0,
    )

    assert calls == ["online"]
    np.testing.assert_allclose(actions, online.actions)


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


def test_cli_accepts_bounded_online_reset_diversity(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--online-reset-interval",
            "6",
            "--online-reset-until-env-steps",
            "100",
            "--online-reset-fraction",
            "0.25",
        ),
    )

    args = train_dmc_jepa.parse_args()

    assert args.online_reset_interval == 6
    assert args.online_reset_until_env_steps == 100
    assert args.online_reset_fraction == 0.25


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


def test_policy_collection_preserves_online_reset_cadence_across_phases(
    monkeypatch,
):
    class Adapter:
        num_envs = 1

        def __init__(self):
            self.reset_calls = 0
            self.step_count = 0

        def reset(self):
            self.reset_calls += 1
            self.step_count = 0
            return np.asarray([[[100 * self.reset_calls]]], dtype=np.float32)

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
                completed_returns=(),
                completed_lengths=(),
            )

    monkeypatch.setattr(
        train_dmc_jepa,
        "select_continuous_actions",
        lambda *args, **kwargs: jnp.zeros((1, 1), dtype=jnp.float32),
    )
    adapter = Adapter()
    replay = SequenceReplayBuffer(
        capacity=6,
        num_envs=1,
        observation_shape=(1,),
        action_shape=(1,),
        action_dtype=np.float32,
    )
    observations = adapter.reset()
    common = {
        "adapter": adapter,
        "state": None,
        "config": SimpleNamespace(),
        "replay": replay,
        "action_low": np.asarray([-1.0], dtype=np.float32),
        "action_high": np.asarray([1.0], dtype=np.float32),
        "desc": "test",
        "quiet": True,
        "np_rng": np.random.default_rng(0),
        "stochastic_actions": True,
        "failure_return_threshold": 100.0,
        "success_return_threshold": 900.0,
        "reset_interval": 3,
        "reset_until_env_steps": 4,
    }

    observations, _, first = train_dmc_jepa._collect_policy_steps(
        observations=observations,
        steps=2,
        train_env_step_offset=0,
        reset_step_offset=0,
        **common,
    )
    observations, _, second = train_dmc_jepa._collect_policy_steps(
        observations=observations,
        steps=2,
        train_env_step_offset=2,
        reset_step_offset=2,
        **common,
    )
    observations, _, third = train_dmc_jepa._collect_policy_steps(
        observations=observations,
        steps=2,
        train_env_step_offset=4,
        reset_step_offset=4,
        **common,
    )

    assert first["forced_reset_events"] == 0
    assert second["forced_reset_events"] == 1
    assert second["forced_reset_env_segments"] == 1
    assert third["forced_reset_events"] == 0
    assert adapter.reset_calls == 2
    np.testing.assert_array_equal(
        replay.cuts[:, 0],
        np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(replay.dones[:, 0], np.zeros(6))
    np.testing.assert_array_equal(observations, np.asarray([[[203.0]]]))


def test_policy_collection_rotates_partial_resets(monkeypatch):
    class Adapter:
        num_envs = 4

        def __init__(self):
            self.reset_indices_calls = []

        def step(self, actions):
            del actions
            return SimpleNamespace(
                observations=np.zeros((4, 1, 1), dtype=np.float32),
                rewards=np.zeros((4, 1), dtype=np.float32),
                dones=np.zeros((4, 1), dtype=np.float32),
                completed_returns=(),
                completed_lengths=(),
            )

        def reset_indices(self, indices):
            self.reset_indices_calls.append(np.array(indices, copy=True))
            return np.zeros((len(indices), 1, 1), dtype=np.float32)

    monkeypatch.setattr(
        train_dmc_jepa,
        "select_continuous_actions",
        lambda *args, **kwargs: jnp.zeros((4, 1), dtype=jnp.float32),
    )
    adapter = Adapter()
    replay = SequenceReplayBuffer(
        capacity=6,
        num_envs=4,
        observation_shape=(1,),
        action_shape=(1,),
        action_dtype=np.float32,
    )

    _, _, metrics = train_dmc_jepa._collect_policy_steps(
        adapter,
        np.zeros((4, 1, 1), dtype=np.float32),
        None,
        SimpleNamespace(),
        replay,
        steps=6,
        action_low=np.asarray([-1.0], dtype=np.float32),
        action_high=np.asarray([1.0], dtype=np.float32),
        desc="test",
        quiet=True,
        np_rng=np.random.default_rng(0),
        stochastic_actions=True,
        train_env_step_offset=0,
        failure_return_threshold=100.0,
        success_return_threshold=900.0,
        reset_interval=3,
        reset_fraction=0.25,
    )

    assert metrics["forced_reset_events"] == 2
    assert metrics["forced_reset_env_segments"] == 2
    assert metrics["online_reset_envs_per_event"] == 1
    assert len(adapter.reset_indices_calls) == 2
    np.testing.assert_array_equal(adapter.reset_indices_calls[0], np.asarray([0]))
    np.testing.assert_array_equal(adapter.reset_indices_calls[1], np.asarray([1]))
    expected_cuts = np.zeros((6, 4), dtype=np.float32)
    expected_cuts[2, 0] = 1.0
    expected_cuts[5, 1] = 1.0
    np.testing.assert_array_equal(replay.cuts, expected_cuts)
    np.testing.assert_array_equal(replay.dones, np.zeros((6, 4)))


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


def test_online_encoder_update_scale_can_start_at_budget_threshold(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--online-encoder-update-scale",
            "0.1",
            "--online-encoder-update-scale-start-env-steps",
            "50000",
        ),
    )

    args = train_dmc_jepa.parse_args()

    assert (
        train_dmc_jepa._scheduled_online_encoder_update_scale(
            args,
            train_env_steps=49_999,
        )
        == 1.0
    )
    assert (
        train_dmc_jepa._scheduled_online_encoder_update_scale(
            args,
            train_env_steps=50_000,
        )
        == 0.1
    )


def test_cli_accepts_recent_replay_and_curve_evaluation(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--online-recent-replay-fraction",
            "0.5",
            "--online-recent-world-model-fraction",
            "0.4",
            "--online-recent-world-model-until-env-steps",
            "50000",
            "--online-recent-policy-start-fraction",
            "0.0",
            "--online-recent-critic-fraction",
            "0.25",
            "--online-recent-replay-steps",
            "128",
            "--online-recent-replay-max-oversample",
            "10",
            "--policy-bootstrap-start-fraction",
            "0.25",
            "--policy-reset-start-fraction",
            "0.05",
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

    assert args.online_recent_replay_fraction == 0.5
    assert args.online_policy_actor_update_interval == 1
    assert args.online_policy_actor_update_interval_start_env_steps == 0
    assert train_dmc_jepa._requested_recent_fractions(args) == {
        "world_model": 0.4,
        "policy_start": 0.0,
        "critic": 0.25,
    }
    assert train_dmc_jepa._scheduled_recent_fractions(
        args,
        train_env_steps=49_999,
    ) == {
        "world_model": 0.4,
        "policy_start": 0.0,
        "critic": 0.25,
    }
    assert train_dmc_jepa._scheduled_recent_fractions(
        args,
        train_env_steps=50_000,
    ) == {
        "world_model": 0.0,
        "policy_start": 0.0,
        "critic": 0.25,
    }
    assert args.online_recent_world_model_until_env_steps == 50_000
    assert args.online_recent_replay_steps == 128
    assert args.online_recent_replay_max_oversample == 10.0
    assert args.policy_bootstrap_start_fraction == 0.25
    assert args.policy_reset_start_fraction == 0.05
    assert args.policy_reset_start_max_age == 63
    assert args.curve_eval_interval_env_steps == 50_000
    assert args.curve_eval_episodes == 20
    assert args.curve_eval_seed == 9_000_000


def test_policy_interface_drift_separates_rotation_from_behavior_change():
    rng = np.random.default_rng(7)
    latents = rng.normal(size=(8, 3, 4))
    means = rng.normal(size=(8, 2))
    log_stds = np.full((8, 2), -0.5)
    values = rng.normal(size=(8,))
    before = {
        "latents": latents,
        "means": means,
        "log_stds": log_stds,
        "values": values,
    }
    rotated = {
        "latents": latents[..., [1, 0, 3, 2]],
        "means": means,
        "log_stds": log_stds,
        "values": values,
    }
    behavior_changed = {
        **before,
        "means": means + 0.5,
        "log_stds": log_stds - 0.2,
        "values": values + 3.0,
    }

    rotation_metrics = train_dmc_jepa._policy_interface_drift_metrics(
        before,
        rotated,
        prefix="rotation",
    )
    behavior_metrics = train_dmc_jepa._policy_interface_drift_metrics(
        before,
        behavior_changed,
        prefix="behavior",
    )

    assert rotation_metrics["rotation/latent_linear_cka"] == pytest.approx(1.0)
    assert rotation_metrics["rotation/latent_cosine_mean"] < 0.9
    assert rotation_metrics["rotation/policy_kl_per_action_dim_mean"] == pytest.approx(
        0.0
    )
    assert behavior_metrics["behavior/policy_kl_per_action_dim_mean"] > 0.0
    assert behavior_metrics["behavior/normalized_action_mean_abs_delta"] > 0.0
    assert behavior_metrics["behavior/value_abs_delta"] == pytest.approx(3.0)


def test_policy_interface_snapshot_matches_an_unchanged_state():
    config = train_dmc_jepa.JepaConfig(
        observation_dim=3,
        action_dim=2,
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
    state = train_dmc_jepa.create_jepa_train_state(jax.random.PRNGKey(0), config)
    observations = jnp.arange(24, dtype=jnp.float32).reshape((4, 2, 3)) / 10.0

    snapshot = train_dmc_jepa._policy_interface_snapshot(
        state,
        observations,
        config,
    )
    metrics = train_dmc_jepa._policy_interface_drift_metrics(
        snapshot,
        snapshot,
        prefix="same",
    )

    assert snapshot["latents"].shape == (4, 2, 8)
    assert snapshot["means"].shape == (4, 2)
    assert metrics["same/latent_cosine_mean"] == pytest.approx(1.0)
    assert metrics["same/latent_linear_cka"] == pytest.approx(1.0)
    assert metrics["same/policy_kl_per_action_dim_mean"] == pytest.approx(0.0)
    assert metrics["same/normalized_action_mean_abs_delta"] == pytest.approx(0.0)


def test_component_recent_replay_fractions_inherit_shared_default(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _minimal_args(
            "--online-recent-replay-fraction",
            "0.5",
            "--online-recent-replay-steps",
            "128",
        ),
    )

    args = train_dmc_jepa.parse_args()

    assert train_dmc_jepa._requested_recent_fractions(args) == {
        "world_model": 0.5,
        "policy_start": 0.5,
        "critic": 0.5,
    }


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


def test_policy_start_mixture_reuses_frozen_bootstrap_states():
    def replay_with_observation(value: float) -> SequenceReplayBuffer:
        replay = SequenceReplayBuffer(
            capacity=12,
            num_envs=1,
            observation_shape=(1,),
            action_shape=(1,),
            action_dtype=np.float32,
        )
        for _ in range(10):
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
    observations, _ = train_dmc_jepa._sample_mixed_policy_starts(
        replay_with_observation(1.0),
        np.random.default_rng(0),
        config=config,
        batch_size=10,
        recent_replay=replay_with_observation(3.0),
        recent_fraction=0.2,
        bootstrap_replay=replay_with_observation(2.0),
        bootstrap_fraction=0.3,
    )

    endpoint_values = np.asarray(observations[:, -1, 0])
    np.testing.assert_array_equal(endpoint_values[:5], 1.0)
    np.testing.assert_array_equal(endpoint_values[5:8], 2.0)
    np.testing.assert_array_equal(endpoint_values[8:], 3.0)


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
    observations, _ = train_dmc_jepa._sample_mixed_policy_starts(
        replay,
        np.random.default_rng(0),
        config=config,
        batch_size=10,
        recent_replay=replay_with_values([3.0] * 10),
        recent_fraction=0.1,
        bootstrap_replay=replay_with_values([2.0] * 10),
        bootstrap_fraction=0.2,
        reset_start_indices=(np.asarray([0]), np.asarray([0])),
        reset_start_fraction=0.2,
    )

    endpoint_values = np.asarray(observations[:, -1, 0])
    np.testing.assert_array_equal(endpoint_values[5:7], 2.0)
    np.testing.assert_array_equal(endpoint_values[7:9], 4.0)
    np.testing.assert_array_equal(endpoint_values[9:], 3.0)


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
