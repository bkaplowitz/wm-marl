from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.replay import ReplayBatch, SequenceReplayBuffer
from world_marl.jepa.training import (
    action_value_gap,
    continuous_policy_train_step,
    create_jepa_train_state,
    enumerated_policy_train_step,
    evaluate_open_loop,
    isotropy_loss,
    lambda_returns,
    policy_train_step,
    prediction_validity,
    reset_policy_heads,
    select_continuous_actions,
    train_model_step,
)
from world_marl.scripts.train_dmc_jepa import summarize as summarize_dmc_jepa


def _config() -> JepaConfig:
    return JepaConfig(
        observation_dim=4,
        action_dim=2,
        latent_dim=8,
        model_dim=16,
        num_layers=1,
        num_heads=2,
        max_horizon=1,
        context_window=1,
        sigreg_num_proj=32,
    )


def test_sequence_replay_samples_contiguous_chunks():
    replay = SequenceReplayBuffer(capacity=8, num_envs=2, observation_shape=(1,))
    for step in range(6):
        replay.add_step(
            observations=np.asarray([[step], [step + 100]], dtype=np.float32),
            actions=np.asarray([step % 2, (step + 1) % 2]),
            rewards=np.asarray([step, step + 100], dtype=np.float32),
            dones=np.zeros((2,), dtype=np.float32),
        )

    batch = replay.sample(
        np.random.default_rng(0),
        batch_size=4,
        chunk_length=3,
        max_horizon=2,
    )

    assert batch.observations.shape == (4, 5, 1)
    assert batch.actions.shape == (4, 4)
    np.testing.assert_allclose(
        np.diff(np.asarray(batch.observations[:, :, 0]), axis=1),
        1.0,
    )


def test_sequence_replay_supports_continuous_action_vectors():
    replay = SequenceReplayBuffer(
        capacity=8,
        num_envs=2,
        observation_shape=(1,),
        action_shape=(3,),
        action_dtype=np.float32,
    )
    for step in range(6):
        replay.add_step(
            observations=np.asarray([[step], [step + 100]], dtype=np.float32),
            actions=np.asarray(
                [
                    [step, step + 1, step + 2],
                    [step + 100, step + 101, step + 102],
                ],
                dtype=np.float32,
            ),
            rewards=np.asarray([step, step + 100], dtype=np.float32),
            dones=np.zeros((2,), dtype=np.float32),
        )

    batch = replay.sample(
        np.random.default_rng(0),
        batch_size=4,
        chunk_length=3,
        max_horizon=2,
    )

    assert batch.actions.shape == (4, 4, 3)
    assert batch.actions.dtype == jnp.float32


def test_jepa_model_forward_and_model_step_are_finite():
    config = _config()
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    observations = jnp.ones((3, 4, 4), dtype=jnp.float32)
    actions = jnp.zeros((3, 3), dtype=jnp.int32)
    outputs = state.apply_fn(
        {"params": state.params},
        observations,
        actions,
        chunk_length=2,
        method=JepaWorldModel.sequence_outputs,
    )

    assert outputs["predicted_latents"].shape == (3, 2, 1, 8)
    assert outputs["target_latents"].shape == (3, 2, 1, 8)

    replay_batch = _batch(config)
    state, metrics = train_model_step(
        state,
        jax.random.PRNGKey(1),
        replay_batch,
        config,
        chunk_length=2,
    )
    assert jnp.isfinite(metrics["model/total_loss"])


def test_continuous_action_jepa_model_step_is_finite():
    config = JepaConfig(
        observation_dim=4,
        action_dim=3,
        action_mode="continuous",
        latent_dim=8,
        model_dim=16,
        num_layers=1,
        num_heads=2,
        max_horizon=1,
        context_window=1,
        sigreg_num_proj=32,
    )
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    observations = jnp.ones((3, 4, 4), dtype=jnp.float32)
    actions = jnp.zeros((3, 3, 3), dtype=jnp.float32)
    outputs = state.apply_fn(
        {"params": state.params},
        observations,
        actions,
        chunk_length=2,
        method=JepaWorldModel.sequence_outputs,
    )

    assert outputs["predicted_latents"].shape == (3, 2, 1, 8)

    replay_batch = ReplayBatch(
        observations=observations,
        actions=actions,
        rewards=jnp.ones((3, 3), dtype=jnp.float32),
        dones=jnp.zeros((3, 3), dtype=jnp.float32),
    )
    state, metrics = train_model_step(
        state,
        jax.random.PRNGKey(1),
        replay_batch,
        config,
        chunk_length=2,
    )
    assert jnp.isfinite(metrics["model/total_loss"])


def test_continuous_policy_update_freezes_world_model_and_bounds_actions():
    config = JepaConfig(
        observation_dim=4,
        action_dim=3,
        action_mode="continuous",
        latent_dim=8,
        model_dim=16,
        num_layers=1,
        num_heads=2,
        max_horizon=1,
        context_window=1,
        sigreg_num_proj=32,
    )
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    before = state.params
    observations = jnp.ones((8, config.observation_dim), dtype=jnp.float32)
    action_low = -jnp.ones((config.action_dim,), dtype=jnp.float32)
    action_high = jnp.ones((config.action_dim,), dtype=jnp.float32)

    state, metrics = continuous_policy_train_step(
        state,
        jax.random.PRNGKey(1),
        observations,
        config,
        action_low,
        action_high,
        imag_horizon=2,
    )
    actions = select_continuous_actions(
        state,
        observations,
        config,
        action_low,
        action_high,
    )

    assert jnp.isfinite(metrics["policy/total_loss"])
    assert jnp.all(actions <= action_high + 1e-6)
    assert jnp.all(actions >= action_low - 1e-6)
    for group in (
        "encoder",
        "latent_proj",
        "action_encoder_hidden",
        "action_encoder_out",
        "dynamics_norm",
        "predictor",
        "predictor_norm",
        "reward_head",
        "continue_head",
    ):
        before_leaves = jax.tree_util.tree_leaves(before[group])
        after_leaves = jax.tree_util.tree_leaves(state.params[group])
        for left, right in zip(before_leaves, after_leaves, strict=True):
            np.testing.assert_allclose(np.asarray(left), np.asarray(right))


def test_isotropy_detects_collapsed_embeddings():
    collapsed = jnp.ones((4, 3, 8))
    _, metrics = isotropy_loss(collapsed)

    assert metrics["latent_std_min"] <= 1.1e-3
    assert metrics["latent_effective_rank"] <= 1e-6


def test_effective_rank_distinguishes_rank_one_and_isotropic_embeddings():
    dim = 8
    scalars = jnp.linspace(-1.0, 1.0, 16)
    rank_one = scalars[:, None] * jnp.ones((1, dim))
    _, rank_one_metrics = isotropy_loss(rank_one.reshape(4, 4, dim))

    isotropic = jnp.concatenate([jnp.eye(dim), -jnp.eye(dim)], axis=0)
    _, isotropic_metrics = isotropy_loss(isotropic.reshape(4, 4, dim))

    assert rank_one_metrics["latent_effective_rank"] <= 1.01
    assert isotropic_metrics["latent_effective_rank"] >= dim - 0.1


def test_jepa_config_enforces_milestone_one_constraints():
    with pytest.raises(ValueError, match="action_mode"):
        JepaConfig(observation_dim=4, action_dim=2, action_mode="mixed")
    with pytest.raises(ValueError, match="regularizer"):
        JepaConfig(observation_dim=4, action_dim=2, regularizer="made-up")
    with pytest.raises(ValueError, match="max_horizon=1"):
        JepaConfig(observation_dim=4, action_dim=2, max_horizon=2)
    with pytest.raises(ValueError, match="context_window=1"):
        JepaConfig(observation_dim=4, action_dim=2, context_window=2)


def test_lambda_returns_bootstrap_from_next_values():
    rewards = jnp.asarray([[1.0], [2.0]])
    continues = jnp.asarray([[1.0], [1.0]])
    values = jnp.asarray([[10.0], [20.0]])
    last_value = jnp.asarray([30.0])

    returns = lambda_returns(
        rewards,
        continues,
        values,
        last_value,
        gamma=1.0,
        lambda_return=0.5,
    )

    np.testing.assert_allclose(np.asarray(returns[:, 0]), np.asarray([27.0, 32.0]))


def test_prediction_validity_masks_terminal_crossing_targets():
    dones = jnp.asarray([[0.0, 1.0, 0.0, 0.0]])
    validity = prediction_validity(dones, chunk_length=2, max_horizon=2)

    expected = np.asarray([[[1.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(validity), expected)


def test_open_loop_evaluation_masks_terminal_crossing_predictions():
    config = _config()
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    batch = ReplayBatch(
        observations=jnp.zeros((2, 2, config.observation_dim), dtype=jnp.float32),
        actions=jnp.zeros((2, 1), dtype=jnp.int32),
        rewards=jnp.ones((2, 1), dtype=jnp.float32),
        dones=jnp.asarray([[0.0], [1.0]], dtype=jnp.float32),
    )

    metrics = evaluate_open_loop(state, batch, config, horizon=1)

    np.testing.assert_allclose(
        np.asarray(metrics["model/open_loop_valid_fraction"]),
        0.5,
    )
    assert metrics["model/open_loop_finite_fraction"] == 1.0


def test_policy_update_does_not_change_world_model_parameters():
    config = _config()
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    before = state.params
    batch = _batch(config)
    state, _ = policy_train_step(
        state,
        jax.random.PRNGKey(1),
        batch.observations[:, 0],
        config,
        imag_horizon=2,
    )

    for group in (
        "encoder",
        "latent_proj",
        "action_embed",
        "dynamics_norm",
        "predictor",
        "predictor_norm",
    ):
        before_leaves = jax.tree_util.tree_leaves(before[group])
        after_leaves = jax.tree_util.tree_leaves(state.params[group])
        for left, right in zip(before_leaves, after_leaves, strict=True):
            np.testing.assert_allclose(np.asarray(left), np.asarray(right))


def test_enumerated_policy_update_does_not_change_world_model_parameters():
    config = _config()
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    before = state.params
    batch = _batch(config)
    state, metrics = enumerated_policy_train_step(
        state,
        jax.random.PRNGKey(1),
        batch.observations[:, 0],
        config,
    )

    assert jnp.isfinite(metrics["policy/total_loss"])
    for group in (
        "encoder",
        "latent_proj",
        "action_embed",
        "dynamics_norm",
        "predictor",
        "predictor_norm",
        "reward_head",
        "continue_head",
    ):
        before_leaves = jax.tree_util.tree_leaves(before[group])
        after_leaves = jax.tree_util.tree_leaves(state.params[group])
        for left, right in zip(before_leaves, after_leaves, strict=True):
            np.testing.assert_allclose(np.asarray(left), np.asarray(right))


def test_enumerated_no_action_control_leaves_actor_head_unchanged():
    config = _config()
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    before = state.params
    observations = jnp.ones((8, config.observation_dim), dtype=jnp.float32)

    updated, metrics = enumerated_policy_train_step(
        state,
        jax.random.PRNGKey(1),
        observations,
        config,
        control="no-action-world-model",
    )

    np.testing.assert_allclose(
        np.asarray(metrics["policy/enumerated_q_gap"]),
        0.0,
        atol=1e-6,
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(before["actor_head"]),
        jax.tree_util.tree_leaves(updated.params["actor_head"]),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), atol=1e-7)


def test_reset_policy_heads_preserves_model_and_reinitializes_policy_heads():
    config = _config()
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    state, _ = train_model_step(
        state,
        jax.random.PRNGKey(1),
        _batch(config),
        config,
        chunk_length=2,
    )
    reset = reset_policy_heads(state, jax.random.PRNGKey(2), config)

    for group in (
        "encoder",
        "latent_proj",
        "action_embed",
        "dynamics_norm",
        "predictor",
        "predictor_norm",
        "reward_head",
        "continue_head",
    ):
        before_leaves = jax.tree_util.tree_leaves(state.params[group])
        after_leaves = jax.tree_util.tree_leaves(reset.params[group])
        for left, right in zip(before_leaves, after_leaves, strict=True):
            np.testing.assert_allclose(np.asarray(left), np.asarray(right))

    actor_changed = any(
        not np.allclose(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree_util.tree_leaves(state.params["actor_head"]),
            jax.tree_util.tree_leaves(reset.params["actor_head"]),
            strict=True,
        )
    )
    value_changed = any(
        not np.allclose(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree_util.tree_leaves(state.params["value_head"]),
            jax.tree_util.tree_leaves(reset.params["value_head"]),
            strict=True,
        )
    )
    assert actor_changed
    assert value_changed


def test_no_action_control_has_zero_action_value_gap():
    config = _config()
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    observations = jnp.ones((5, config.observation_dim), dtype=jnp.float32)

    gap = action_value_gap(
        state,
        observations,
        config,
        control="no-action-world-model",
    )

    np.testing.assert_allclose(np.asarray(gap), 0.0, atol=1e-6)


def test_dmc_jepa_summary_requires_main_to_beat_controls():
    def outcome(control: str, jepa_loss: float, open_loop_loss: float):
        return {
            "run_index": 0,
            "control": control,
            "passed": True,
            "initial_jepa_loss": 1.0,
            "final_jepa_loss": jepa_loss,
            "initial_open_loop_loss": 1.0,
            "final_open_loop_loss": open_loop_loss,
            "final_model_metrics": {"model/jepa_loss": jepa_loss},
        }

    good = summarize_dmc_jepa(
        [
            outcome("none", 0.01, 0.02),
            outcome("no-action-world-model", 0.03, 0.08),
        ]
    )
    bad = summarize_dmc_jepa(
        [
            outcome("none", 0.05, 0.09),
            outcome("no-action-world-model", 0.03, 0.08),
        ]
    )

    assert good["passed"]
    assert good["main_beats_controls_open_loop"]
    assert good["main_beats_controls_jepa"]
    assert not bad["passed"]


def test_dmc_jepa_summary_tracks_policy_rung():
    def outcome(control: str, policy_improvement: float):
        return {
            "run_index": 0,
            "control": control,
            "passed": True,
            "initial_jepa_loss": 1.0,
            "final_jepa_loss": 0.1 if control == "none" else 0.2,
            "initial_open_loop_loss": 1.0,
            "final_open_loop_loss": 0.1 if control == "none" else 0.2,
            "policy_training_enabled": True,
            "policy_passed": control == "none",
            "policy_random_mean": 0.0,
            "policy_initial_mean": 1.0,
            "policy_trained_mean": 1.0 + policy_improvement,
            "policy_improvement": policy_improvement,
            "policy_trained_minus_random": 1.0 + policy_improvement,
            "final_model_metrics": {"model/jepa_loss": 0.1},
        }

    good = summarize_dmc_jepa(
        [
            outcome("none", 2.0),
            outcome("no-action-world-model", 0.0),
        ]
    )
    bad = summarize_dmc_jepa(
        [
            outcome("none", 0.1),
            outcome("no-action-world-model", 0.2),
        ]
    )

    assert good["passed"]
    assert good["policy_training_enabled"]
    assert good["policy_main_beats_controls"]
    assert not bad["passed"]


def _batch(config: JepaConfig):
    replay = SequenceReplayBuffer(capacity=8, num_envs=1, observation_shape=(4,))
    for step in range(5):
        replay.add_step(
            observations=np.full((1, 4), step, dtype=np.float32),
            actions=np.asarray([step % config.action_dim], dtype=np.int32),
            rewards=np.asarray([1.0], dtype=np.float32),
            dones=np.asarray([0.0], dtype=np.float32),
        )
    return replay.sample(
        np.random.default_rng(0),
        batch_size=2,
        chunk_length=2,
        max_horizon=config.max_horizon,
    )
