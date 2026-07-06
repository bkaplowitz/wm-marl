from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from world_marl.jepa.models import (
    JepaConfig,
    JepaWorldModel,
    apply_rotary_position_embedding,
)
from world_marl.jepa.replay import ReplayBatch, SequenceReplayBuffer
from world_marl.jepa.training import (
    continuous_candidate_distill_step,
    continuous_critic_warmup_step,
    continuous_policy_train_step,
    create_jepa_train_state,
    evaluate_open_loop,
    lambda_returns,
    latent_collapse_metrics,
    masked_mean,
    prediction_validity,
    reset_policy_heads,
    reward_only_returns,
    select_continuous_actions,
    train_model_step,
    transition_start_validity,
)


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


def test_rotary_position_embedding_preserves_norms():
    x = jnp.arange(1 * 3 * 2 * 4, dtype=jnp.float32).reshape((1, 3, 2, 4))
    rotated = apply_rotary_position_embedding(x)

    assert rotated.shape == x.shape
    np.testing.assert_allclose(np.asarray(rotated[:, 0]), np.asarray(x[:, 0]))
    np.testing.assert_allclose(
        np.asarray(jnp.linalg.norm(rotated, axis=-1)),
        np.asarray(jnp.linalg.norm(x, axis=-1)),
        rtol=1e-6,
        atol=1e-6,
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


def test_dynamics_ensemble_model_step_and_policy_metrics_are_finite():
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
        dynamics_ensemble_size=3,
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

    assert outputs["predicted_latents"].shape == (3, 2, 1, 3, 8)
    assert outputs["reward_logits"].shape == (3, 2, 1, 3)
    assert outputs["continue_logits"].shape == (3, 2, 1, 3)

    batch = ReplayBatch(
        observations=observations,
        actions=actions,
        rewards=jnp.ones((3, 3), dtype=jnp.float32),
        dones=jnp.zeros((3, 3), dtype=jnp.float32),
    )
    state, model_metrics = train_model_step(
        state,
        jax.random.PRNGKey(1),
        batch,
        config,
        chunk_length=2,
    )
    assert jnp.isfinite(model_metrics["model/total_loss"])
    assert jnp.isfinite(model_metrics["model/ensemble_latent_disagreement"])

    state, policy_metrics = continuous_policy_train_step(
        state,
        jax.random.PRNGKey(2),
        jnp.ones((5, 1, 4), dtype=jnp.float32),
        config,
        jnp.full((3,), -1.0),
        jnp.full((3,), 1.0),
        imag_horizon=2,
        uncertainty_penalty=0.1,
        uncertainty_threshold=10.0,
        uncertainty_budget=20.0,
    )

    del state
    assert jnp.isfinite(policy_metrics["policy/total_loss"])
    assert jnp.isfinite(policy_metrics["policy/uncertainty"])
    assert jnp.isfinite(policy_metrics["policy/trusted_fraction"])


def test_masked_mean_broadcasts_validity_mask_across_ensemble_axis():
    """A singleton-mask axis must not shrink the denominator: with 2 ensemble
    members at values 1 and 3 over 3 valid positions, the mean is 2, not 4.
    """
    values = jnp.stack(
        [jnp.ones((2, 3)), 3.0 * jnp.ones((2, 3))],
        axis=-1,
    )
    mask = jnp.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])[..., None]

    np.testing.assert_allclose(np.asarray(masked_mean(values, mask)), 2.0, rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(masked_mean(values[..., 0], mask[..., 0])), 1.0, rtol=1e-5
    )


def test_frozen_encoder_model_step_preserves_encoder_and_updates_world_model():
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
    observations = jax.random.normal(jax.random.PRNGKey(1), (4, 4, 4))
    actions = jax.random.normal(jax.random.PRNGKey(2), (4, 3, 3))
    batch = ReplayBatch(
        observations=observations,
        actions=actions,
        rewards=jax.random.normal(jax.random.PRNGKey(3), (4, 3)),
        dones=jnp.zeros((4, 3), dtype=jnp.float32),
    )
    frozen_state, frozen_metrics = train_model_step(
        state,
        jax.random.PRNGKey(4),
        batch,
        config,
        chunk_length=2,
        freeze_encoder=True,
    )
    normal_state, _ = train_model_step(
        state,
        jax.random.PRNGKey(4),
        batch,
        config,
        chunk_length=2,
    )

    assert jnp.isfinite(frozen_metrics["model/total_loss"])
    for left, right in zip(
        jax.tree_util.tree_leaves(state.params["encoder"]),
        jax.tree_util.tree_leaves(frozen_state.params["encoder"]),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right))

    assert _tree_changed(state.params["encoder"], normal_state.params["encoder"])
    assert _tree_changed(state.params["predictor"], frozen_state.params["predictor"])
    assert _tree_changed(state.params["block_0"], frozen_state.params["block_0"])


def test_model_step_supports_control_value_consistency_loss():
    config = JepaConfig(
        observation_dim=4,
        action_dim=3,
        action_mode="continuous",
        latent_dim=8,
        model_dim=16,
        num_layers=1,
        num_heads=2,
        max_horizon=2,
        context_window=1,
        sigreg_num_proj=32,
    )
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    before = state.params
    batch = ReplayBatch(
        observations=jax.random.normal(jax.random.PRNGKey(1), (4, 5, 4)),
        actions=jax.random.normal(jax.random.PRNGKey(2), (4, 4, 3)),
        rewards=jax.random.normal(jax.random.PRNGKey(3), (4, 4)),
        dones=jnp.zeros((4, 4), dtype=jnp.float32),
    )

    updated, metrics = train_model_step(
        state,
        jax.random.PRNGKey(4),
        batch,
        config,
        chunk_length=2,
        control_value_weight=0.25,
    )

    assert jnp.isfinite(metrics["model/total_loss"])
    assert jnp.isfinite(metrics["model/control_value_loss"])
    assert jnp.isfinite(metrics["model/control_value_q_abs_error"])
    assert metrics["model/control_value_weight"] == 0.25
    assert metrics["model/control_value_finite_fraction"] == 1.0
    assert _tree_changed(before["predictor"], updated.params["predictor"])
    for left, right in zip(
        jax.tree_util.tree_leaves(before["value_head"]),
        jax.tree_util.tree_leaves(updated.params["value_head"]),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right))


def test_jepa_model_trains_recursive_overshooting_horizons():
    config = JepaConfig(
        observation_dim=4,
        action_dim=3,
        action_mode="continuous",
        latent_dim=8,
        model_dim=16,
        num_layers=1,
        num_heads=2,
        max_horizon=3,
        context_window=2,
        sigreg_num_proj=32,
    )
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    observations = jnp.ones((3, 6, 4), dtype=jnp.float32)
    actions = jnp.zeros((3, 5, 3), dtype=jnp.float32)
    outputs = state.apply_fn(
        {"params": state.params},
        observations,
        actions,
        chunk_length=3,
        method=JepaWorldModel.sequence_outputs,
    )

    assert outputs["predicted_latents"].shape == (3, 3, 3, 8)
    assert outputs["target_latents"].shape == (3, 3, 3, 8)
    assert outputs["reward_logits"].shape == (3, 3, 3)
    assert outputs["continue_logits"].shape == (3, 3, 3)

    replay_batch = ReplayBatch(
        observations=observations,
        actions=actions,
        rewards=jnp.ones((3, 5), dtype=jnp.float32),
        dones=jnp.zeros((3, 5), dtype=jnp.float32),
    )
    state, metrics = train_model_step(
        state,
        jax.random.PRNGKey(1),
        replay_batch,
        config,
        chunk_length=3,
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


def test_stochastic_continuous_policy_update_reports_entropy_and_samples_actions():
    config = JepaConfig(
        observation_dim=4,
        action_dim=2,
        action_mode="continuous",
        latent_dim=8,
        model_dim=16,
        num_layers=1,
        num_heads=2,
        max_horizon=1,
        context_window=1,
        sigreg_num_proj=32,
        stochastic_actor=True,
    )
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
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
        actor_entropy_coef=0.01,
    )
    deterministic_actions = select_continuous_actions(
        state,
        observations,
        config,
        action_low,
        action_high,
    )
    sampled_actions = select_continuous_actions(
        state,
        observations,
        config,
        action_low,
        action_high,
        key=jax.random.PRNGKey(2),
        stochastic=True,
    )

    assert jnp.isfinite(metrics["policy/total_loss"])
    assert jnp.isfinite(metrics["policy/entropy_bonus"])
    assert metrics["policy/actor_entropy_coef"] == pytest.approx(0.01)
    assert jnp.all(sampled_actions <= action_high + 1e-6)
    assert jnp.all(sampled_actions >= action_low - 1e-6)
    assert not np.allclose(
        np.asarray(deterministic_actions),
        np.asarray(sampled_actions),
    )


def test_continuous_critic_warmup_updates_value_only():
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
    batch = ReplayBatch(
        observations=jnp.ones((8, 4, config.observation_dim), dtype=jnp.float32),
        actions=jnp.zeros((8, 3, config.action_dim), dtype=jnp.float32),
        rewards=jnp.ones((8, 3), dtype=jnp.float32),
        dones=jnp.zeros((8, 3), dtype=jnp.float32),
    )

    updated, metrics = continuous_critic_warmup_step(
        state,
        batch,
        config,
        horizon=3,
        value_clip=10.0,
    )

    assert jnp.isfinite(metrics["critic/total_loss"])
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
        "actor_head",
    ):
        before_leaves = jax.tree_util.tree_leaves(before[group])
        after_leaves = jax.tree_util.tree_leaves(updated.params[group])
        for left, right in zip(before_leaves, after_leaves, strict=True):
            np.testing.assert_allclose(np.asarray(left), np.asarray(right), atol=1e-7)

    value_changed = any(
        not np.allclose(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree_util.tree_leaves(before["value_head"]),
            jax.tree_util.tree_leaves(updated.params["value_head"]),
            strict=True,
        )
    )
    assert value_changed


def test_continuous_candidate_distill_freezes_world_model_and_bounds_actions():
    config = JepaConfig(
        observation_dim=4,
        action_dim=2,
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

    state, metrics = continuous_candidate_distill_step(
        state,
        jax.random.PRNGKey(1),
        observations,
        config,
        action_low,
        action_high,
        imag_horizon=2,
        num_candidates=8,
        candidate_min_gap=0.0,
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
        "value_head",
    ):
        before_leaves = jax.tree_util.tree_leaves(before[group])
        after_leaves = jax.tree_util.tree_leaves(state.params[group])
        for left, right in zip(before_leaves, after_leaves, strict=True):
            np.testing.assert_allclose(np.asarray(left), np.asarray(right), atol=1e-7)


def test_continuous_candidate_distill_no_action_control_keeps_actor_head():
    config = JepaConfig(
        observation_dim=4,
        action_dim=2,
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

    updated, metrics = continuous_candidate_distill_step(
        state,
        jax.random.PRNGKey(1),
        observations,
        config,
        action_low,
        action_high,
        imag_horizon=2,
        control="no-action-world-model",
        num_candidates=8,
        candidate_min_gap=1e-6,
    )

    np.testing.assert_allclose(
        np.asarray(metrics["policy/candidate_active_fraction"]),
        0.0,
        atol=1e-7,
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(before["actor_head"]),
        jax.tree_util.tree_leaves(updated.params["actor_head"]),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right), atol=1e-7)


def test_collapse_metrics_detect_collapsed_embeddings():
    collapsed = jnp.ones((4, 3, 8))
    metrics = latent_collapse_metrics(collapsed)

    assert metrics["latent_std_min"] <= 1.1e-3
    assert metrics["latent_effective_rank"] <= 1e-6


def test_effective_rank_distinguishes_rank_one_and_isotropic_embeddings():
    dim = 8
    scalars = jnp.linspace(-1.0, 1.0, 16)
    rank_one = scalars[:, None] * jnp.ones((1, dim))
    rank_one_metrics = latent_collapse_metrics(rank_one.reshape(4, 4, dim))

    isotropic = jnp.concatenate([jnp.eye(dim), -jnp.eye(dim)], axis=0)
    isotropic_metrics = latent_collapse_metrics(isotropic.reshape(4, 4, dim))

    assert rank_one_metrics["latent_effective_rank"] <= 1.01
    assert isotropic_metrics["latent_effective_rank"] >= dim - 0.1


def test_jepa_config_enforces_world_model_constraints():
    with pytest.raises(ValueError, match="action_mode"):
        JepaConfig(observation_dim=4, action_dim=2, action_mode="mixed")
    with pytest.raises(ValueError, match="regularizer"):
        JepaConfig(observation_dim=4, action_dim=2, regularizer="made-up")
    with pytest.raises(ValueError, match="target_gradient"):
        JepaConfig(observation_dim=4, action_dim=2, target_gradient="ema")
    with pytest.raises(ValueError, match="max_horizon"):
        JepaConfig(observation_dim=4, action_dim=2, max_horizon=0)
    with pytest.raises(ValueError, match="context_window"):
        JepaConfig(observation_dim=4, action_dim=2, context_window=0)
    assert JepaConfig(observation_dim=4, action_dim=2, max_horizon=2)
    assert JepaConfig(observation_dim=4, action_dim=2, context_window=2)


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


def test_reward_only_returns_do_not_bootstrap_from_value_head():
    rewards = jnp.asarray([[1.0], [2.0], [3.0]])
    continues = jnp.asarray([[1.0], [0.0], [1.0]])

    returns = reward_only_returns(rewards, continues, gamma=1.0)

    np.testing.assert_allclose(np.asarray(returns[:, 0]), np.asarray([3.0, 2.0, 3.0]))


def test_prediction_validity_masks_terminal_crossing_targets():
    dones = jnp.asarray([[0.0, 1.0, 0.0, 0.0]])
    validity = prediction_validity(dones, chunk_length=2, max_horizon=2)

    expected = np.asarray([[[1.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(validity), expected)


def test_transition_start_validity_keeps_terminal_transition_labels():
    dones = jnp.asarray([[0.0, 1.0, 0.0, 0.0]])
    validity = transition_start_validity(dones, chunk_length=2, max_horizon=2)

    expected = np.asarray([[[1.0, 1.0], [1.0, 0.0]]], dtype=np.float32)
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


def test_open_loop_evaluation_supports_history_context():
    config = JepaConfig(
        observation_dim=4,
        action_dim=2,
        action_mode="continuous",
        latent_dim=8,
        model_dim=16,
        num_layers=1,
        num_heads=2,
        max_horizon=1,
        context_window=2,
        sigreg_num_proj=32,
    )
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    batch = ReplayBatch(
        observations=jnp.zeros((3, 5, config.observation_dim), dtype=jnp.float32),
        actions=jnp.zeros((3, 4, config.action_dim), dtype=jnp.float32),
        rewards=jnp.ones((3, 4), dtype=jnp.float32),
        dones=jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        ),
    )

    metrics = evaluate_open_loop(state, batch, config, horizon=2)

    np.testing.assert_allclose(
        np.asarray(metrics["model/open_loop_valid_fraction"]),
        1.0 / 3.0,
        atol=1e-6,
    )
    assert metrics["model/open_loop_finite_fraction"] == 1.0


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


def _tree_changed(left, right) -> bool:
    return any(
        not np.allclose(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
            strict=True,
        )
    )
