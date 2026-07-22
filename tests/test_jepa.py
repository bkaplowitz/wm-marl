from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from world_marl.checkpointing import save_checkpoint
from world_marl.jepa.models import (
    JepaConfig,
    JepaWorldModel,
    apply_rotary_position_embedding,
)
from world_marl.jepa.replay import ReplayBatch, SequenceReplayBuffer
from world_marl.jepa.training import (
    continuous_policy_train_step,
    create_jepa_train_state,
    diagonal_gaussian_kl,
    evaluate_open_loop,
    full_policy_kl_penalty,
    lambda_returns,
    latent_collapse_metrics,
    masked_mean,
    prediction_validity,
    replay_return_continues,
    reset_policy_heads,
    select_continuous_actions,
    tanh_normal_entropy_sample,
    train_model_step,
    transition_start_validity,
)
from world_marl.jepa.reproducibility import fingerprint_pytree
from world_marl.scripts.train_dmc_jepa import (
    _reload_and_verify_checkpoint,
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


def _batch(config: JepaConfig):
    replay = SequenceReplayBuffer(capacity=8, num_envs=1, observation_shape=(4,))
    for step in range(5):
        replay.add_step(
            observations=np.full((1, 4), step, dtype=np.float32),
            actions=np.asarray([step % config.action_dim], dtype=np.int32),
            rewards=np.asarray([1.0], dtype=np.float32),
            is_last=np.asarray([0.0], dtype=np.float32),
            is_terminal=np.asarray([0.0], dtype=np.float32),
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


def test_canonical_reacher_parameter_count_is_exact():
    config = JepaConfig(
        observation_dim=6,
        action_dim=2,
        action_mode="continuous",
        latent_dim=144,
        model_dim=144,
        num_layers=2,
        num_heads=4,
        mlp_ratio=4,
        max_horizon=8,
        context_window=8,
        actor_hidden_dim=64,
        critic_hidden_dim=64,
        actor_num_layers=3,
        critic_num_layers=3,
        actor_layer_norm=True,
        critic_layer_norm=True,
        twohot_bins=255,
    )
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    counts = {
        name: sum(
            int(np.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(params)
        )
        for name, params in state.params.items()
    }

    assert counts["horizon_embed"] == 1_296
    assert counts["actor_head"] == 18_004
    assert counts["value_head"] == 34_319
    assert sum(counts.values()) == 926_659


def test_final_checkpoint_verification_covers_every_prediction_head(tmp_path):
    config = JepaConfig(
        observation_dim=4,
        action_dim=2,
        action_mode="continuous",
        latent_dim=8,
        model_dim=16,
        num_layers=1,
        num_heads=2,
        max_horizon=2,
        context_window=1,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        actor_num_layers=2,
        critic_num_layers=2,
        sigreg_num_proj=8,
    )
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    batch = ReplayBatch(
        observations=jax.random.normal(jax.random.PRNGKey(1), (3, 4, 4)),
        actions=jax.random.uniform(
            jax.random.PRNGKey(2),
            (3, 3, 2),
            minval=-1.0,
            maxval=1.0,
        ),
        rewards=jnp.zeros((3, 3), dtype=jnp.float32),
        is_last=jnp.zeros((3, 3), dtype=jnp.float32),
        is_terminal=jnp.zeros((3, 3), dtype=jnp.float32),
    )
    checkpoint_dir = tmp_path / "checkpoint"
    params_sha256 = fingerprint_pytree(state.params)
    save_checkpoint(
        checkpoint_dir,
        state,
        metadata={"params_sha256": params_sha256},
    )

    restored, verification = _reload_and_verify_checkpoint(
        state,
        config,
        checkpoint_dir=checkpoint_dir,
        batch=batch,
        seed=99,
        chunk_length=2,
    )

    assert fingerprint_pytree(restored.params) == params_sha256
    assert verification["parameter_fingerprint_match"]
    assert verification["reload_max_abs_prediction_diff"] == 0.0
    assert verification["reload_max_abs_output_diff"] == 0.0
    assert set(verification["output_max_abs_diffs"]) == {
        "predicted_latents",
        "reward_logits",
        "continue_logits",
        "actor_logits",
        "value_logits",
    }


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


@pytest.mark.parametrize("ensemble_size", [1, 2, 5])
def test_masked_mean_is_ensemble_size_invariant(ensemble_size):
    values = jnp.ones((2, 3, 4, ensemble_size), dtype=jnp.float32)
    mask = jnp.ones((2, 3, 4, 1), dtype=jnp.float32)

    np.testing.assert_allclose(np.asarray(masked_mean(values, mask)), 1.0)


def test_tanh_normal_entropy_penalizes_saturated_action_means():
    log_stds = jnp.zeros((4, 2), dtype=jnp.float32)
    centered = tanh_normal_entropy_sample(jnp.zeros((4, 2)), log_stds)
    saturated = tanh_normal_entropy_sample(jnp.full((4, 2), 4.0), log_stds)

    assert np.all(np.asarray(centered) > np.asarray(saturated))
    assert np.all(np.isfinite(np.asarray(saturated)))


def test_sequence_replay_samples_contiguous_chunks():
    replay = SequenceReplayBuffer(capacity=8, num_envs=2, observation_shape=(1,))
    for step in range(6):
        replay.add_step(
            observations=np.asarray([[step], [step + 100]], dtype=np.float32),
            actions=np.asarray([step % 2, (step + 1) % 2]),
            rewards=np.asarray([step, step + 100], dtype=np.float32),
            is_last=np.zeros((2,), dtype=np.float32),
            is_terminal=np.zeros((2,), dtype=np.float32),
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


def test_sequence_replay_does_not_sample_across_collector_cuts():
    replay = SequenceReplayBuffer(capacity=10, num_envs=1, observation_shape=(1,))
    for step in range(10):
        replay.add_step(
            observations=np.asarray([[step]], dtype=np.float32),
            actions=np.asarray([step % 2]),
            rewards=np.asarray([0.0], dtype=np.float32),
            is_last=np.asarray([0.0], dtype=np.float32),
            is_terminal=np.asarray([0.0], dtype=np.float32),
            cuts=np.asarray([float(step == 4)], dtype=np.float32),
        )

    starts, _ = replay.sample_indices(
        np.random.default_rng(0),
        batch_size=128,
        chunk_length=3,
        max_horizon=2,
    )

    assert set(starts.tolist()) <= {0, 5}


def test_sequence_replay_rejects_indexed_sequence_across_collector_cut():
    replay = SequenceReplayBuffer(capacity=10, num_envs=1, observation_shape=(1,))
    for step in range(10):
        replay.add_step(
            observations=np.asarray([[step]], dtype=np.float32),
            actions=np.asarray([step % 2]),
            rewards=np.asarray([0.0], dtype=np.float32),
            is_last=np.asarray([0.0], dtype=np.float32),
            is_terminal=np.asarray([0.0], dtype=np.float32),
            cuts=np.asarray([float(step == 4)], dtype=np.float32),
        )

    with pytest.raises(ValueError, match="must not cross collector cuts"):
        replay.sample_from_indices(
            np.asarray([2]),
            np.asarray([0]),
            chunk_length=3,
            max_horizon=2,
        )


def test_replay_returns_do_not_leak_rewards_across_nonterminal_reset():
    rewards = jnp.asarray([[1.0], [1.0], [1_000.0]], dtype=jnp.float32)
    is_last = jnp.asarray([[0.0], [1.0], [0.0]], dtype=jnp.float32)
    is_terminal = jnp.zeros_like(is_last)
    continues = replay_return_continues(is_last, is_terminal)

    returns = lambda_returns(
        rewards,
        continues,
        jnp.zeros_like(rewards),
        jnp.zeros((1,), dtype=jnp.float32),
        gamma=1.0,
        lambda_return=1.0,
    )

    np.testing.assert_allclose(np.asarray(returns[:, 0]), [2.0, 1.0, 1_000.0])


def test_sequence_replay_finds_valid_starts_near_episode_boundaries():
    replay = SequenceReplayBuffer(capacity=20, num_envs=1, observation_shape=(1,))
    for step in range(15):
        replay.add_step(
            observations=np.asarray([[step]], dtype=np.float32),
            actions=np.asarray([step % 2]),
            rewards=np.asarray([0.0], dtype=np.float32),
            is_last=np.asarray([float(step == 10)], dtype=np.float32),
            is_terminal=np.asarray([float(step == 10)], dtype=np.float32),
            cuts=np.asarray([float(step == 4)], dtype=np.float32),
        )

    starts, envs = replay.episode_start_indices(
        max_age=1,
        chunk_length=2,
        max_horizon=1,
    )

    np.testing.assert_array_equal(starts, np.asarray([0, 1, 5, 6, 11, 12]))
    np.testing.assert_array_equal(envs, np.zeros((6,), dtype=np.int64))


def test_sequence_replay_reset_starts_remain_correct_after_ring_wrap():
    replay = SequenceReplayBuffer(capacity=10, num_envs=1, observation_shape=(1,))
    for step in range(15):
        replay.add_step(
            observations=np.asarray([[step]], dtype=np.float32),
            actions=np.asarray([step % 2]),
            rewards=np.asarray([0.0], dtype=np.float32),
            is_last=np.asarray([float(step in {6, 11})], dtype=np.float32),
            is_terminal=np.asarray([float(step in {6, 11})], dtype=np.float32),
        )

    starts, envs = replay.episode_start_indices(
        max_age=1,
        chunk_length=2,
        max_horizon=1,
    )

    # The retained replay contains global steps 5..14. Its first partial
    # episode is not a known reset; starts are only admitted after retained
    # terminal boundaries at global steps 6 and 11.
    np.testing.assert_array_equal(starts, np.asarray([2, 3, 7]))
    np.testing.assert_array_equal(envs, np.zeros((3,), dtype=np.int64))


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
            is_last=np.zeros((2,), dtype=np.float32),
            is_terminal=np.zeros((2,), dtype=np.float32),
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


def test_separate_actor_and_value_paths_match_combined_outputs():
    config = _config()
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    model = JepaWorldModel(config)
    latents = jax.random.normal(
        jax.random.PRNGKey(1),
        (7, config.latent_dim),
    )

    means, log_stds, values = state.apply_fn(
        {"params": state.params},
        latents,
        method=model.actor_value_stats_from_latent,
    )
    separate_means, separate_log_stds = state.apply_fn(
        {"params": state.params},
        latents,
        method=model.actor_stats_from_latent,
    )
    separate_values = state.apply_fn(
        {"params": state.params},
        latents,
        method=model.value_from_latent,
    )

    np.testing.assert_array_equal(np.asarray(means), np.asarray(separate_means))
    np.testing.assert_array_equal(
        np.asarray(log_stds),
        np.asarray(separate_log_stds),
    )
    np.testing.assert_array_equal(np.asarray(values), np.asarray(separate_values))


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
        is_last=jnp.zeros((3, 3), dtype=jnp.float32),
        is_terminal=jnp.zeros((3, 3), dtype=jnp.float32),
    )
    state, metrics = train_model_step(
        state,
        jax.random.PRNGKey(1),
        replay_batch,
        config,
        chunk_length=2,
    )
    assert jnp.isfinite(metrics["model/total_loss"])


def test_model_step_can_skip_diagnostics_without_changing_update():
    config = _config()
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    key = jax.random.PRNGKey(1)
    batch = _batch(config)

    detailed_state, detailed_metrics = train_model_step(
        state,
        key,
        batch,
        config,
        chunk_length=2,
        compute_diagnostics=True,
    )
    fast_state, fast_metrics = train_model_step(
        state,
        key,
        batch,
        config,
        chunk_length=2,
        compute_diagnostics=False,
    )

    for detailed, fast in zip(
        jax.tree_util.tree_leaves(detailed_state),
        jax.tree_util.tree_leaves(fast_state),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(detailed), np.asarray(fast), atol=1e-7)
    assert "collapse/latent_effective_rank" in detailed_metrics
    assert "collapse/latent_effective_rank" not in fast_metrics
    assert jnp.isfinite(fast_metrics["model/total_loss"])


def test_model_step_can_freeze_only_the_observation_encoder():
    config = _config()
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    state, _ = train_model_step(
        state,
        jax.random.PRNGKey(1),
        _batch(config),
        config,
        chunk_length=2,
    )
    encoder_before = state.params["encoder"]
    predictor_before = state.params["predictor"]

    state, metrics = train_model_step(
        state,
        jax.random.PRNGKey(2),
        _batch(config),
        config,
        chunk_length=2,
        freeze_encoder=True,
    )

    assert all(
        jnp.array_equal(before, after)
        for before, after in zip(
            jax.tree_util.tree_leaves(encoder_before),
            jax.tree_util.tree_leaves(state.params["encoder"]),
        )
    )
    assert any(
        not jnp.array_equal(before, after)
        for before, after in zip(
            jax.tree_util.tree_leaves(predictor_before),
            jax.tree_util.tree_leaves(state.params["predictor"]),
        )
    )
    assert metrics["model/encoder_frozen"] == 1.0
    assert metrics["model/encoder_grad_norm_unmasked"] > 0.0


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
    assert outputs["reward_logits"].shape == (3, 3, 3, config.twohot_bins)
    assert outputs["continue_logits"].shape == (3, 3, 3)

    replay_batch = ReplayBatch(
        observations=observations,
        actions=actions,
        rewards=jnp.ones((3, 5), dtype=jnp.float32),
        is_last=jnp.zeros((3, 5), dtype=jnp.float32),
        is_terminal=jnp.zeros((3, 5), dtype=jnp.float32),
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


def test_continuous_policy_step_can_update_critic_without_actor():
    config = JepaConfig(
        observation_dim=4,
        action_dim=2,
        action_mode="continuous",
        latent_dim=8,
        model_dim=16,
        num_layers=1,
        num_heads=2,
        max_horizon=2,
        context_window=1,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        actor_num_layers=2,
        critic_num_layers=2,
        sigreg_num_proj=8,
    )
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    before = state.params

    updated, metrics = continuous_policy_train_step(
        state,
        jax.random.PRNGKey(1),
        jax.random.normal(jax.random.PRNGKey(2), (8, 4)),
        config,
        -jnp.ones((2,), dtype=jnp.float32),
        jnp.ones((2,), dtype=jnp.float32),
        imag_horizon=2,
        target_critic_params=state.target_critic_params,
        target_critic_ema_decay=0.98,
        apply_actor_update=False,
    )

    assert metrics["policy/actor_update_applied"] == 0.0
    assert not _tree_changed(before["actor_head"], updated.params["actor_head"])
    assert _tree_changed(before["value_head"], updated.params["value_head"])


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


def test_short_pathwise_reward_auxiliary_changes_only_actor_update():
    config = JepaConfig(
        observation_dim=4,
        action_dim=2,
        action_mode="continuous",
        latent_dim=8,
        model_dim=16,
        num_layers=1,
        num_heads=2,
        max_horizon=2,
        context_window=1,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        actor_num_layers=2,
        critic_num_layers=2,
        sigreg_num_proj=8,
    )
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    observations = jax.random.normal(jax.random.PRNGKey(2), (8, 4))
    action_low = -jnp.ones((2,), dtype=jnp.float32)
    action_high = jnp.ones((2,), dtype=jnp.float32)

    baseline, _ = continuous_policy_train_step(
        state,
        jax.random.PRNGKey(1),
        observations,
        config,
        action_low,
        action_high,
        imag_horizon=4,
    )
    hybrid, metrics = continuous_policy_train_step(
        state,
        jax.random.PRNGKey(1),
        observations,
        config,
        action_low,
        action_high,
        imag_horizon=4,
        pathwise_reward_coef=0.5,
        pathwise_horizon=2,
    )

    assert metrics["policy/gradient_mode_reinforce"] == 1.0
    assert metrics["policy/gradient_mode_pathwise_reward"] == 1.0
    assert metrics["policy/pathwise_reward_coef"] == pytest.approx(0.5)
    assert jnp.isfinite(metrics["policy/pathwise_reward_objective"])
    assert _tree_changed(baseline.params["actor_head"], hybrid.params["actor_head"])
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
        for before, after in zip(
            jax.tree_util.tree_leaves(state.params[group]),
            jax.tree_util.tree_leaves(hybrid.params[group]),
            strict=True,
        ):
            np.testing.assert_allclose(np.asarray(before), np.asarray(after))


def test_dreamer_style_policy_update_is_finite_and_keeps_world_model_frozen():
    config = JepaConfig(
        observation_dim=4,
        action_dim=2,
        action_mode="continuous",
        latent_dim=8,
        model_dim=16,
        num_layers=1,
        num_heads=2,
        max_horizon=2,
        context_window=1,
        actor_log_std_min=-2.302585092994046,
        actor_log_std_max=0.0,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        actor_num_layers=2,
        critic_num_layers=2,
        actor_layer_norm=True,
        critic_layer_norm=True,
        actor_output_scale=0.01,
        value_output_scale=0.0,
        reward_output_scale=0.0,
        twohot_bins=11,
        adaptive_grad_clip=0.3,
        sigreg_num_proj=8,
    )
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    before = state.params
    real_batch = ReplayBatch(
        observations=jax.random.normal(jax.random.PRNGKey(1), (4, 5, 4)),
        actions=jax.random.uniform(
            jax.random.PRNGKey(2),
            (4, 4, 2),
            minval=-1.0,
            maxval=1.0,
        ),
        rewards=jax.random.uniform(jax.random.PRNGKey(3), (4, 4)),
        is_last=jnp.zeros((4, 4), dtype=jnp.float32),
        is_terminal=jnp.zeros((4, 4), dtype=jnp.float32),
    )

    updated, metrics = continuous_policy_train_step(
        state,
        jax.random.PRNGKey(4),
        real_batch.observations[:, :1],
        config,
        -jnp.ones((2,), dtype=jnp.float32),
        jnp.ones((2,), dtype=jnp.float32),
        imag_horizon=2,
        actor_entropy_coef=3e-4,
        target_critic_params=state.target_critic_params,
        target_critic_ema_decay=0.98,
        real_critic_batch=real_batch,
        real_critic_loss_enabled=True,
        real_critic_loss_coef=0.3,
        real_critic_horizon=4,
        slow_value_regularization_coef=1.0,
        value_clip=0.0,
        actor_reference_params=state.params,
        actor_kl_coef=1.0,
        actor_kl_target_per_dim=0.01,
    )

    assert jnp.isfinite(metrics["policy/total_loss"])
    assert metrics["policy/value_clip_enabled"] == 0.0
    assert metrics["policy/value_target_clip_fraction"] == 0.0
    np.testing.assert_allclose(
        np.asarray(metrics["policy/imagined_return"]),
        np.asarray(metrics["policy/clipped_imagined_return"]),
    )
    assert jnp.isfinite(metrics["policy/advantage_std"])
    assert jnp.isfinite(metrics["policy/normalized_advantage_abs_max"])
    assert metrics["policy/actor_kl_enabled"] == 1.0
    assert metrics["policy/actor_kl_penalty"] == pytest.approx(0.0, abs=1e-7)
    assert jnp.isfinite(metrics["policy/reference_full_distribution_kl_mean"])
    assert jnp.isfinite(metrics["policy/update_full_distribution_kl_mean"])
    assert metrics["policy/update_full_distribution_kl_mean"] >= 0.0
    assert 0.0 <= metrics["policy/advantage_positive_fraction"] <= 1.0
    assert metrics["policy/gradient_mode_reinforce"] == 1.0
    assert metrics["policy/action_entropy_tanh_normal"] == 1.0
    assert metrics["policy/return_normalization_ema_percentile"] == 1.0
    assert metrics["policy/replay_critic_lambda_return"] == 1.0
    assert metrics["policy/replay_critic_all_steps"] == 1.0
    assert jnp.isfinite(metrics["policy/slow_value_loss"])
    assert jnp.isfinite(metrics["policy/replay_critic_loss"])
    assert bool(updated.return_range_initialized)
    assert _tree_changed(before["actor_head"], updated.params["actor_head"])
    assert _tree_changed(before["value_head"], updated.params["value_head"])
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
        for left, right in zip(
            jax.tree_util.tree_leaves(before[group]),
            jax.tree_util.tree_leaves(updated.params[group]),
            strict=True,
        ):
            np.testing.assert_allclose(np.asarray(left), np.asarray(right))


def test_diagonal_gaussian_kl_detects_mean_and_scale_updates():
    means = jnp.zeros((2, 3), dtype=jnp.float32)
    log_stds = jnp.zeros_like(means)

    same = diagonal_gaussian_kl(means, log_stds, means, log_stds)
    changed = diagonal_gaussian_kl(
        means,
        log_stds,
        means + 0.25,
        log_stds - 0.1,
    )

    np.testing.assert_allclose(same, 0.0, atol=1e-7)
    assert jnp.all(changed > 0.0)


def test_full_policy_kl_penalty_uses_per_dimension_hinge():
    reference_kl = jnp.asarray([0.01, 0.03], dtype=jnp.float32)
    weights = jnp.ones_like(reference_kl)

    penalty, mean, per_dim, excess, enabled = full_policy_kl_penalty(
        reference_kl,
        weights,
        action_dim=2,
        coef=2.0,
        target_per_dim=0.005,
    )
    disabled = full_policy_kl_penalty(
        reference_kl,
        weights,
        action_dim=2,
        coef=0.0,
        target_per_dim=0.005,
    )

    np.testing.assert_allclose(mean, 0.02, rtol=1e-5)
    np.testing.assert_allclose(per_dim, 0.01, rtol=1e-5)
    np.testing.assert_allclose(excess, 0.005, rtol=1e-5)
    np.testing.assert_allclose(penalty, 0.01, rtol=1e-5)
    assert bool(enabled)
    assert disabled[0] == pytest.approx(0.0)
    assert not bool(disabled[-1])


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
    with pytest.raises(ValueError, match="max_horizon"):
        JepaConfig(observation_dim=4, action_dim=2, max_horizon=0)
    with pytest.raises(ValueError, match="context_window"):
        JepaConfig(observation_dim=4, action_dim=2, context_window=0)
    with pytest.raises(ValueError, match="num_heads"):
        JepaConfig(observation_dim=4, action_dim=2, num_heads=0)
    with pytest.raises(ValueError, match="learning_rate"):
        JepaConfig(observation_dim=4, action_dim=2, learning_rate=0.0)
    with pytest.raises(ValueError, match="gamma"):
        JepaConfig(observation_dim=4, action_dim=2, gamma=1.1)
    with pytest.raises(ValueError, match="lambda_return"):
        JepaConfig(observation_dim=4, action_dim=2, lambda_return=-0.1)
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


def test_prediction_validity_masks_episode_boundary_crossing_targets():
    is_last = jnp.asarray([[0.0, 1.0, 0.0, 0.0]])
    validity = prediction_validity(is_last, chunk_length=2, max_horizon=2)

    expected = np.asarray([[[1.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(validity), expected)


def test_transition_start_validity_keeps_boundary_transition_labels():
    is_last = jnp.asarray([[0.0, 1.0, 0.0, 0.0]])
    validity = transition_start_validity(is_last, chunk_length=2, max_horizon=2)

    expected = np.asarray([[[1.0, 1.0], [1.0, 0.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(validity), expected)


def test_open_loop_evaluation_masks_terminal_crossing_predictions():
    config = _config()
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    batch = ReplayBatch(
        observations=jnp.zeros((2, 2, config.observation_dim), dtype=jnp.float32),
        actions=jnp.zeros((2, 1), dtype=jnp.int32),
        rewards=jnp.ones((2, 1), dtype=jnp.float32),
        is_last=jnp.asarray([[0.0], [1.0]], dtype=jnp.float32),
        is_terminal=jnp.asarray([[0.0], [1.0]], dtype=jnp.float32),
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
        is_last=jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        ),
        is_terminal=jnp.asarray(
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


def test_dmc_jepa_summary_reports_latest_policy_results():
    outcome = {
        "initial_jepa_loss": 2.0,
        "final_jepa_loss": 1.0,
        "initial_open_loop_loss": 2.0,
        "final_open_loop_loss": 1.0,
        "final_policy_eval_mean": 950.0,
        "final_policy_eval_std": 25.0,
        "final_policy_eval_failure_rate": 0.0,
        "final_policy_eval_success_rate": 0.95,
        "real_train_replay_env_steps": 497_664,
        "real_validation_replay_env_steps": 1_280,
        "real_train_plus_validation_env_steps": 498_944,
        "real_policy_eval_env_steps": 20_000,
        "real_total_env_steps": 518_944,
    }

    summary = summarize_dmc_jepa([outcome])

    assert summary["protocol"] == "reset_rich_interleaved_latest_policy"
    assert summary["aggregate_final_policy_eval_mean"] == 950.0
    assert summary["aggregate_real_train_replay_env_steps"] == 497_664


def test_dmc_jepa_summary_separates_seed_variation_from_episode_variation():
    outcomes = [
        {
            "final_policy_eval_mean": 900.0,
            "final_policy_eval_std": 20.0,
            "dreamer_style_train_return_mean": 800.0,
            "dreamer_style_train_return_std": 15.0,
            "dreamer_style_training_score": {
                "curve": [
                    {
                        "bin_start_env_step": 0,
                        "bin_end_env_step": 10_000,
                        "mean_return": 700.0,
                    }
                ]
            },
        },
        {
            "final_policy_eval_mean": 1_000.0,
            "final_policy_eval_std": 40.0,
            "dreamer_style_train_return_mean": 900.0,
            "dreamer_style_train_return_std": 25.0,
            "dreamer_style_training_score": {
                "curve": [
                    {
                        "bin_start_env_step": 0,
                        "bin_end_env_step": 10_000,
                        "mean_return": 900.0,
                    }
                ]
            },
        },
    ]

    summary = summarize_dmc_jepa(outcomes)

    assert summary["aggregate_final_policy_eval_mean"] == 950.0
    assert summary["aggregate_final_policy_eval_std"] == 50.0
    assert summary["aggregate_final_policy_eval_mean_within_seed_episode_std"] == 30.0
    assert summary["aggregate_dreamer_style_train_return_mean"] == 850.0
    assert summary["aggregate_dreamer_style_train_return_std"] == 50.0
    assert summary["aggregate_dreamer_style_curve"] == [
        {
            "bin_start_env_step": 0,
            "bin_end_env_step": 10_000,
            "seed_count": 2,
            "mean_return": 800.0,
            "std_return_across_seed_means": 100.0,
            "seed_mean_returns": [700.0, 900.0],
        }
    ]
