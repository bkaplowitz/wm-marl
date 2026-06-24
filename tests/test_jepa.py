from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.replay import ReplayBatch, SequenceReplayBuffer
from world_marl.jepa.training import (
    apply_control_alignment,
    continuous_candidate_distill_step,
    continuous_critic_warmup_step,
    continuous_policy_train_step,
    create_jepa_train_state,
    evaluate_open_loop,
    lambda_returns,
    latent_collapse_metrics,
    prediction_validity,
    procrustes_control_alignment,
    reset_policy_heads,
    reward_only_returns,
    select_continuous_actions,
    train_model_step,
    umeyama_control_interface,
)
from world_marl.scripts.train_dmc_jepa import (
    _action_contrast_metrics,
    _candidate_refit_gate_report,
    _control_coordinate_prediction_metrics,
    _merge_online_policy_baseline,
    _online_interface_drift_metrics,
    _online_history_metrics,
    _run_passed as dmc_run_passed,
    summarize as summarize_dmc_jepa,
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


def test_model_step_supports_cosine_latent_anchor_loss():
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
    batch = ReplayBatch(
        observations=jax.random.normal(jax.random.PRNGKey(1), (4, 4, 4)),
        actions=jax.random.normal(jax.random.PRNGKey(2), (4, 3, 3)),
        rewards=jax.random.normal(jax.random.PRNGKey(3), (4, 3)),
        dones=jnp.zeros((4, 3), dtype=jnp.float32),
    )

    updated, metrics = train_model_step(
        state,
        jax.random.PRNGKey(4),
        batch,
        config,
        chunk_length=2,
        latent_anchor_params=state.params,
        latent_anchor_alignment=state.control_alignment,
        latent_anchor_weight=0.1,
        control_prediction_params=state.params,
        control_prediction_alignment=state.control_alignment,
        control_prediction_weight=0.2,
    )

    del updated
    assert jnp.isfinite(metrics["model/total_loss"])
    assert metrics["model/latent_anchor_weight"] == 0.1
    assert metrics["model/control_prediction_train_weight"] == 0.2
    np.testing.assert_allclose(
        np.asarray(metrics["model/latent_anchor_loss"]),
        0.0,
        atol=1e-6,
    )
    assert jnp.isfinite(metrics["model/control_prediction_train_loss"])


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


def test_continuous_policy_behavior_distillation_reports_zero_for_same_teacher():
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
    state, metrics = continuous_policy_train_step(
        state,
        jax.random.PRNGKey(1),
        jnp.ones((5, 1, 4), dtype=jnp.float32),
        config,
        jnp.full((3,), -1.0),
        jnp.full((3,), 1.0),
        imag_horizon=2,
        start_actions=jnp.zeros((5, 1, 3), dtype=jnp.float32),
        behavior_teacher_params=state.params,
        behavior_distill_weight=1.0,
    )

    del state
    assert jnp.isfinite(metrics["policy/total_loss"])
    np.testing.assert_allclose(
        np.asarray(metrics["policy/behavior_distill_loss"]),
        0.0,
        atol=1e-6,
    )
    assert metrics["policy/behavior_distill_weight"] == 1.0


def test_online_interface_drift_metrics_are_zero_for_identical_state():
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
    batch = ReplayBatch(
        observations=jnp.ones((5, 2, 4), dtype=jnp.float32),
        actions=jnp.zeros((5, 1, 3), dtype=jnp.float32),
        rewards=jnp.ones((5, 1), dtype=jnp.float32),
        dones=jnp.zeros((5, 1), dtype=jnp.float32),
    )

    metrics = _online_interface_drift_metrics(
        state,
        state,
        batch,
        config,
        action_low=np.full((3,), -1.0, dtype=np.float32),
        action_high=np.full((3,), 1.0, dtype=np.float32),
    )

    np.testing.assert_allclose(metrics["online/raw_latent_cosine"], 1.0, atol=1e-5)
    np.testing.assert_allclose(metrics["online/raw_latent_drift_l2"], 0.0, atol=1e-6)
    np.testing.assert_allclose(
        metrics["online/control_latent_cosine"],
        1.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        metrics["online/control_latent_drift_l2"],
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        metrics["online/policy_action_drift_l2"],
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        metrics["online/value_drift_abs_mean"],
        0.0,
        atol=1e-6,
    )


def test_procrustes_control_alignment_recovers_orthogonal_interface():
    source = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=jnp.float32,
    )
    rotation = jnp.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    target = source @ rotation

    alignment = procrustes_control_alignment(source, target)
    aligned = apply_control_alignment(source, alignment)

    np.testing.assert_allclose(
        np.asarray(aligned),
        np.asarray(target),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(alignment.T @ alignment),
        np.eye(3),
        atol=1e-5,
    )


def test_umeyama_control_interface_recovers_affine_interface():
    source = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [-1.0, -2.0, -1.0],
            [2.0, -1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    rotation = jnp.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    scale = jnp.asarray(1.7, dtype=jnp.float32)
    bias = jnp.asarray([0.5, -0.25, 0.75], dtype=jnp.float32)
    target = scale * (source @ rotation) + bias

    alignment, fitted_scale, fitted_bias = umeyama_control_interface(source, target)
    aligned = apply_control_alignment(source, alignment, fitted_scale, fitted_bias)

    np.testing.assert_allclose(np.asarray(aligned), np.asarray(target), atol=1e-5)
    np.testing.assert_allclose(np.asarray(fitted_scale), np.asarray(scale), atol=1e-5)
    np.testing.assert_allclose(np.asarray(fitted_bias), np.asarray(bias), atol=1e-5)


def test_select_continuous_actions_uses_control_alignment():
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
    observations = jax.random.normal(jax.random.PRNGKey(1), (6, 4))
    action_low = -jnp.ones((config.action_dim,), dtype=jnp.float32)
    action_high = jnp.ones((config.action_dim,), dtype=jnp.float32)
    identity_actions = select_continuous_actions(
        state,
        observations,
        config,
        action_low,
        action_high,
    )

    flipped = -jnp.eye(config.latent_dim, dtype=jnp.float32)
    aligned_state = state.replace(control_alignment=flipped)
    aligned_actions = select_continuous_actions(
        aligned_state,
        observations,
        config,
        action_low,
        action_high,
    )

    assert not np.allclose(np.asarray(identity_actions), np.asarray(aligned_actions))


def test_candidate_refit_gate_requires_recent_improvement_and_anchor_preservation():
    accepted = _candidate_refit_gate_report(
        {"model/open_loop_loss": 0.40, "model/jepa_loss": 0.10},
        {"model/open_loop_loss": 0.43, "model/jepa_loss": 0.11},
        {"model/open_loop_loss": 0.80, "model/jepa_loss": 0.20},
        {"model/open_loop_loss": 0.65, "model/jepa_loss": 0.18},
        metric="model/open_loop_loss",
        min_recent_improvement=0.05,
        max_anchor_degradation=0.05,
    )
    recent_failed = _candidate_refit_gate_report(
        {"model/open_loop_loss": 0.40},
        {"model/open_loop_loss": 0.43},
        {"model/open_loop_loss": 0.80},
        {"model/open_loop_loss": 0.78},
        metric="model/open_loop_loss",
        min_recent_improvement=0.05,
        max_anchor_degradation=0.05,
    )
    anchor_failed = _candidate_refit_gate_report(
        {"model/open_loop_loss": 0.40},
        {"model/open_loop_loss": 0.50},
        {"model/open_loop_loss": 0.80},
        {"model/open_loop_loss": 0.65},
        metric="model/open_loop_loss",
        min_recent_improvement=0.05,
        max_anchor_degradation=0.05,
    )

    assert accepted["model_update_accepted"]
    assert accepted["recent_validation_improvement"] == pytest.approx(0.15)
    assert accepted["anchor_validation_degradation"] == pytest.approx(0.03)
    assert not recent_failed["model_update_accepted"]
    assert not recent_failed["recent_validation_improved"]
    assert not anchor_failed["model_update_accepted"]
    assert not anchor_failed["anchor_validation_preserved"]


def test_control_coordinate_prediction_metrics_are_finite():
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
    batch = ReplayBatch(
        observations=jax.random.normal(jax.random.PRNGKey(1), (5, 4, 4)),
        actions=jax.random.normal(jax.random.PRNGKey(2), (5, 3, 2)),
        rewards=jnp.zeros((5, 3), dtype=jnp.float32),
        dones=jnp.zeros((5, 3), dtype=jnp.float32),
    )

    metrics = _control_coordinate_prediction_metrics(
        reference_state=state,
        predictor_state=state,
        batch=batch,
        config=config,
        chunk_length=3,
        control="none",
    )

    assert "model/control_prediction_loss" in metrics
    assert np.isfinite(float(metrics["model/control_prediction_loss"]))
    assert float(metrics["model/control_prediction_finite_fraction"]) == 1.0


def test_action_contrast_no_action_control_has_zero_margin():
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
    batch = ReplayBatch(
        observations=jnp.arange(4 * 3 * 4, dtype=jnp.float32).reshape((4, 3, 4)),
        actions=jnp.ones((4, 2, 3), dtype=jnp.float32),
        rewards=jnp.ones((4, 2), dtype=jnp.float32),
        dones=jnp.zeros((4, 2), dtype=jnp.float32),
    )

    metrics = _action_contrast_metrics(
        state,
        jax.random.PRNGKey(1),
        batch,
        config,
        chunk_length=2,
        control="no-action-world-model",
    )

    np.testing.assert_allclose(
        np.asarray(metrics["model/action_contrast_margin"]),
        0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(metrics["model/action_contrast_accuracy"]),
        0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(metrics["model/action_contrast_valid_fraction"]),
        1.0,
        atol=1e-7,
    )
    assert np.asarray(metrics["model/action_contrast_finite_fraction"]) == 1.0


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


def test_dmc_run_passes_when_continue_targets_have_no_terminals():
    initial = {
        "model/jepa_loss": 1.0,
        "model/open_loop_loss": 1.0,
    }
    final = {
        "model/jepa_loss": 0.1,
        "model/open_loop_loss": 0.1,
        "model/open_loop_finite_fraction": 1.0,
        "model/reward_loss": 0.01,
        "model/reward_constant_mse": 0.1,
        "model/continue_loss": 0.001,
        "model/continue_constant_bce": 0.000001,
        "model/terminal_positive_fraction": 0.0,
        "model/nonterminal_recall": 1.0,
    }

    assert dmc_run_passed(initial, final, reload_diff=0.0)


def test_dmc_run_passes_when_continue_targets_have_too_few_terminals():
    initial = {
        "model/jepa_loss": 1.0,
        "model/open_loop_loss": 1.0,
    }
    final = {
        "model/jepa_loss": 0.1,
        "model/open_loop_loss": 0.1,
        "model/open_loop_finite_fraction": 1.0,
        "model/reward_loss": 0.01,
        "model/reward_constant_mse": 0.1,
        "model/continue_loss": 0.01,
        "model/continue_constant_bce": 0.005,
        "model/terminal_positive_fraction": 0.00061,
        "model/nonterminal_recall": 1.0,
    }

    assert dmc_run_passed(initial, final, reload_diff=0.0)


def test_dmc_run_requires_continue_baseline_when_terminals_are_common_enough():
    initial = {
        "model/jepa_loss": 1.0,
        "model/open_loop_loss": 1.0,
    }
    final = {
        "model/jepa_loss": 0.1,
        "model/open_loop_loss": 0.1,
        "model/open_loop_finite_fraction": 1.0,
        "model/reward_loss": 0.01,
        "model/reward_constant_mse": 0.1,
        "model/continue_loss": 0.02,
        "model/continue_constant_bce": 0.01,
        "model/terminal_positive_fraction": 0.02,
        "model/nonterminal_recall": 1.0,
    }

    assert not dmc_run_passed(initial, final, reload_diff=0.0)


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
            "policy_objective": "direct",
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
    assert good["world_model_passed"]
    assert good["policy_training_enabled"]
    assert good["policy_main_beats_controls"]
    assert not bad["passed"]
    assert bad["world_model_passed"]


def test_dmc_jepa_policy_summary_allows_majority_success_with_positive_aggregate():
    def outcome(run_index: int, policy_improvement: float, policy_passed: bool):
        return {
            "run_index": run_index,
            "control": "none",
            "passed": True,
            "initial_jepa_loss": 1.0,
            "final_jepa_loss": 0.1,
            "initial_open_loop_loss": 1.0,
            "final_open_loop_loss": 0.1,
            "policy_training_enabled": True,
            "policy_objective": "direct",
            "policy_passed": policy_passed,
            "policy_random_mean": 0.0,
            "policy_initial_mean": 10.0,
            "policy_trained_mean": 10.0 + policy_improvement,
            "policy_improvement": policy_improvement,
            "policy_trained_minus_random": 10.0 + policy_improvement,
            "final_model_metrics": {"model/jepa_loss": 0.1},
        }

    summary = summarize_dmc_jepa(
        [
            outcome(0, 90.0, True),
            outcome(1, 300.0, True),
            outcome(2, 0.0, False),
        ]
    )

    assert summary["passed"]
    assert summary["policy_main_passed"]
    assert summary["policy_main_successes"] == 2
    assert summary["policy_required_successes"] == 2
    assert summary["policy_aggregate_improved"]


def test_dmc_jepa_summary_marks_candidate_distill_as_diagnostic():
    outcome = {
        "run_index": 0,
        "control": "none",
        "passed": True,
        "initial_jepa_loss": 1.0,
        "final_jepa_loss": 0.1,
        "initial_open_loop_loss": 1.0,
        "final_open_loop_loss": 0.1,
        "policy_training_enabled": True,
        "policy_objective": "candidate-distill",
        "policy_passed": True,
        "policy_random_mean": 0.0,
        "policy_initial_mean": 1.0,
        "policy_trained_mean": 3.0,
        "policy_improvement": 2.0,
        "policy_trained_minus_random": 3.0,
        "final_model_metrics": {"model/jepa_loss": 0.1},
    }

    summary = summarize_dmc_jepa([outcome])

    assert summary["policy_main_passed"]
    assert summary["primary_policy_objective"] == "candidate-distill"
    assert not summary["direct_policy_mainline"]
    assert not summary["passed"]


def test_online_policy_outcome_keeps_original_baseline_for_summary():
    initial = {
        "policy_initial_mean": 10.0,
        "policy_random_mean": 1.0,
        "policy_trained_mean": 40.0,
    }
    final = {
        "policy_initial_mean": 50.0,
        "policy_random_mean": 2.0,
        "policy_trained_mean": 60.0,
        "policy_improvement": 10.0,
        "policy_trained_minus_random": 58.0,
        "policy_final_metrics": {
            "policy/action_saturation_fraction": 0.1,
        },
        "critic_final_metrics": {
            "critic/finite_fraction": 1.0,
        },
    }

    merged = _merge_online_policy_baseline(final, initial)

    assert merged["policy_initial_mean"] == 10.0
    assert merged["policy_random_mean"] == 1.0
    assert merged["policy_online_phase_initial_mean"] == 50.0
    assert merged["policy_online_phase_improvement"] == 10.0
    assert merged["policy_pre_online_trained_mean"] == 40.0
    assert merged["policy_online_total_improvement_vs_pre_online"] == 20.0
    assert merged["policy_improvement"] == 50.0
    assert merged["policy_primary_improvement"] == 10.0
    assert merged["policy_primary_improvement_key"] == "policy_online_phase_improvement"
    assert merged["policy_trained_minus_random"] == 59.0
    assert merged["policy_passed"]


def test_online_policy_regression_fails_even_when_total_return_improves():
    initial = {
        "policy_initial_mean": 10.0,
        "policy_random_mean": 1.0,
        "policy_trained_mean": 45.0,
    }
    final = {
        "policy_initial_mean": 50.0,
        "policy_random_mean": 2.0,
        "policy_trained_mean": 40.0,
        "policy_improvement": -10.0,
        "policy_trained_minus_random": 38.0,
        "policy_final_metrics": {
            "policy/action_saturation_fraction": 0.1,
        },
        "critic_final_metrics": {
            "critic/finite_fraction": 1.0,
        },
    }

    merged = _merge_online_policy_baseline(final, initial)

    assert merged["policy_improvement"] == 30.0
    assert merged["policy_primary_improvement"] == -10.0
    assert merged["policy_online_total_improvement_vs_pre_online"] == -5.0
    assert not merged["policy_passed"]


def test_online_policy_fails_when_phase_improves_but_loses_pre_online_actor():
    initial = {
        "policy_initial_mean": 10.0,
        "policy_random_mean": 1.0,
        "policy_trained_mean": 100.0,
    }
    final = {
        "policy_initial_mean": 50.0,
        "policy_random_mean": 2.0,
        "policy_trained_mean": 70.0,
        "policy_improvement": 20.0,
        "policy_trained_minus_random": 68.0,
        "policy_final_metrics": {
            "policy/action_saturation_fraction": 0.1,
        },
        "critic_final_metrics": {
            "critic/finite_fraction": 1.0,
        },
    }

    merged = _merge_online_policy_baseline(final, initial)

    assert merged["policy_primary_improvement"] == 20.0
    assert merged["policy_online_total_improvement_vs_pre_online"] == -30.0
    assert not merged["policy_passed"]


def test_dmc_jepa_summary_uses_online_phase_as_primary_policy_signal():
    def outcome(control: str, total: float, online: float):
        return {
            "run_index": 0,
            "control": control,
            "passed": True,
            "initial_jepa_loss": 1.0,
            "final_jepa_loss": 0.1 if control == "none" else 0.2,
            "initial_open_loop_loss": 1.0,
            "final_open_loop_loss": 0.1 if control == "none" else 0.2,
            "policy_training_enabled": True,
            "policy_objective": "direct",
            "policy_passed": control == "none" and online > 0.0,
            "policy_random_mean": 0.0,
            "policy_initial_mean": 10.0,
            "policy_trained_mean": 10.0 + total,
            "policy_improvement": total,
            "policy_online_phase_improvement": online,
            "policy_primary_improvement": online,
            "policy_primary_improvement_key": "policy_online_phase_improvement",
            "policy_trained_minus_random": 10.0 + total,
            "final_model_metrics": {"model/jepa_loss": 0.1},
        }

    good = summarize_dmc_jepa(
        [
            outcome("none", 50.0, 5.0),
            outcome("shuffled-action-replay", 40.0, 1.0),
        ]
    )
    bad = summarize_dmc_jepa(
        [
            outcome("none", 50.0, 1.0),
            outcome("shuffled-action-replay", 10.0, 5.0),
        ]
    )

    assert good["passed"]
    assert good["policy_comparison_key"] == "policy_primary_improvement"
    assert good["aggregate_policy_primary_improvement"] == 5.0
    assert good["aggregate_control_policy_primary_improvement"] == 1.0
    assert (
        good["paired_control_differences"]["shuffled-action-replay"][
            "mean_policy_primary_improvement_advantage"
        ]
        == 4.0
    )
    assert not bad["passed"]
    assert not bad["policy_main_beats_controls"]
    assert not bad["paired_policy_ok"]


def test_dmc_jepa_summary_requires_paired_policy_majority():
    def outcome(control: str, run_index: int, primary: float):
        return {
            "run_index": run_index,
            "control": control,
            "passed": True,
            "initial_jepa_loss": 1.0,
            "final_jepa_loss": 0.1 if control == "none" else 0.2,
            "initial_open_loop_loss": 1.0,
            "final_open_loop_loss": 0.1 if control == "none" else 0.2,
            "policy_training_enabled": True,
            "policy_objective": "direct",
            "policy_passed": control == "none",
            "policy_random_mean": 0.0,
            "policy_initial_mean": 10.0,
            "policy_trained_mean": 10.0 + primary,
            "policy_improvement": primary,
            "policy_primary_improvement": primary,
            "policy_trained_minus_random": 10.0 + primary,
            "final_model_metrics": {"model/jepa_loss": 0.1},
        }

    summary = summarize_dmc_jepa(
        [
            outcome("none", 0, 100.0),
            outcome("none", 1, 0.0),
            outcome("none", 2, 0.0),
            outcome("shuffled-action-replay", 0, 0.0),
            outcome("shuffled-action-replay", 1, 1.0),
            outcome("shuffled-action-replay", 2, 1.0),
        ]
    )

    paired = summary["paired_control_differences"]["shuffled-action-replay"]
    assert paired["mean_policy_primary_improvement_advantage"] > 0.0
    assert paired["runs_main_better_policy_primary"] == 1
    assert paired["required_majority_pairs"] == 2
    assert not summary["paired_policy_ok"]
    assert not summary["passed"]


def test_online_history_metrics_tracks_actor_replay_trend():
    metrics = _online_history_metrics(
        [
            {
                "actor_replay": {"mean_return": 10.0},
                "policy": {
                    "policy_training_enabled": True,
                    "policy_improvement": 3.0,
                    "policy_passed": True,
                },
                "model_metrics": {
                    "model/jepa_loss": 0.2,
                    "model/open_loop_loss": 0.4,
                },
                "candidate_refit": {
                    "model_update_accepted": True,
                    "gate": {
                        "recent_validation_improvement": 0.2,
                        "anchor_validation_degradation": 0.01,
                    },
                },
            },
            {
                "actor_replay": {"mean_return": 25.0},
                "policy": {
                    "policy_training_enabled": True,
                    "policy_improvement": 5.0,
                    "policy_passed": True,
                },
                "model_metrics": {
                    "model/jepa_loss": 0.1,
                    "model/open_loop_loss": 0.3,
                },
                "candidate_refit": {
                    "model_update_accepted": False,
                    "gate": {
                        "recent_validation_improvement": -0.1,
                        "anchor_validation_degradation": 0.08,
                    },
                },
            },
        ],
        {"policy_trained_mean": 8.0},
    )

    assert metrics["online_actor_replay_iterations"] == 2
    assert metrics["online_actor_replay_returns"] == [10.0, 25.0]
    assert metrics["online_actor_replay_delta"] == 15.0
    assert metrics["online_actor_replay_vs_initial_policy"] == 17.0
    assert metrics["online_actor_replay_trend_passed"]
    assert metrics["online_policy_phase_improvements"] == [3.0, 5.0]
    assert metrics["online_policy_phase_final_improvement"] == 5.0
    assert metrics["online_policy_phase_passes"] == [True, True]
    assert metrics["online_policy_phase_passed"]
    assert metrics["online_model_jepa_losses"] == [0.2, 0.1]
    assert metrics["online_model_open_loop_losses"] == [0.4, 0.3]
    assert metrics["online_candidate_refit_iterations"] == 2
    assert metrics["online_model_update_acceptances"] == [True, False]
    assert metrics["online_model_update_acceptance_rate"] == 0.5
    assert metrics["online_candidate_recent_validation_improvements"] == [0.2, -0.1]
    assert metrics["online_candidate_anchor_validation_degradations"] == [0.01, 0.08]
    assert metrics["online_candidate_recent_validation_improvement_final"] == -0.1
    assert metrics["online_candidate_anchor_validation_degradation_final"] == 0.08
    assert metrics["online_pipeline_completed"]


def test_online_history_metrics_rejects_actor_replay_regression():
    metrics = _online_history_metrics(
        [
            {"actor_replay": {"mean_return": 5.0}},
        ],
        {"policy_trained_mean": 8.0},
    )

    assert metrics["online_actor_replay_iterations"] == 1
    assert metrics["online_actor_replay_vs_initial_policy"] == -3.0
    assert not metrics["online_actor_replay_trend_passed"]


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
