from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from world_marl.jepa.decoder import (
    DecoderConfig,
    create_decoder_train_state,
    decode_open_loop_rollout,
    decoder_reconstruction_mse,
    encode_observations,
    select_display_trajectories,
    train_decoder_step,
)
from world_marl.jepa.models import JepaConfig
from world_marl.jepa.replay import ReplayBatch
from world_marl.jepa.training import create_jepa_train_state


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


def test_decoder_config_validation():
    with pytest.raises(ValueError):
        DecoderConfig(latent_dim=0, observation_dim=4)
    with pytest.raises(ValueError):
        DecoderConfig(latent_dim=8, observation_dim=4, learning_rate=0.0)


def test_decoder_training_reduces_reconstruction_mse():
    config = _config()
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    decoder_config = DecoderConfig(
        latent_dim=config.latent_dim,
        observation_dim=config.observation_dim,
        hidden_dim=32,
        learning_rate=1e-2,
    )
    decoder_state = create_decoder_train_state(jax.random.PRNGKey(1), decoder_config)
    observations = jax.random.normal(
        jax.random.PRNGKey(2),
        (64, config.observation_dim),
    )
    latents = encode_observations(state, observations)
    initial_mse = float(
        decoder_reconstruction_mse(decoder_state, latents, observations)
    )

    for _ in range(200):
        decoder_state, loss = train_decoder_step(
            decoder_state,
            latents,
            observations,
        )

    final_mse = float(decoder_reconstruction_mse(decoder_state, latents, observations))
    assert jnp.isfinite(loss)
    assert final_mse < initial_mse


def test_decoder_training_leaves_world_model_untouched():
    config = _config()
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    decoder_state = create_decoder_train_state(
        jax.random.PRNGKey(1),
        DecoderConfig(
            latent_dim=config.latent_dim,
            observation_dim=config.observation_dim,
            hidden_dim=32,
        ),
    )
    observations = jax.random.normal(
        jax.random.PRNGKey(2),
        (16, config.observation_dim),
    )
    params_before = jax.tree_util.tree_map(lambda leaf: leaf.copy(), state.params)

    latents = encode_observations(state, observations)
    train_decoder_step(decoder_state, latents, observations)

    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        jax.device_get(params_before),
        jax.device_get(state.params),
    )


def test_decode_open_loop_rollout_matches_plot_contract():
    config = _config()
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    decoder_state = create_decoder_train_state(
        jax.random.PRNGKey(1),
        DecoderConfig(
            latent_dim=config.latent_dim,
            observation_dim=config.observation_dim,
            hidden_dim=32,
        ),
    )
    horizon = 3
    transition_count = config.context_window + horizon - 1
    batch = ReplayBatch(
        observations=jax.random.normal(
            jax.random.PRNGKey(2),
            (2, config.context_window + horizon, config.observation_dim),
        ),
        actions=jnp.zeros((2, transition_count), dtype=jnp.int32),
        rewards=jnp.zeros((2, transition_count), dtype=jnp.float32),
        is_last=jnp.zeros((2, transition_count), dtype=jnp.float32),
        is_terminal=jnp.zeros((2, transition_count), dtype=jnp.float32),
    )

    rollout = decode_open_loop_rollout(
        state,
        decoder_state,
        batch,
        config,
        horizon=horizon,
    )

    observation_dim = config.observation_dim
    assert rollout["context_observations"].shape == (
        2,
        config.context_window,
        observation_dim,
    )
    assert rollout["real_observations"].shape == (2, horizon, observation_dim)
    assert rollout["decoded_context"].shape == (
        2,
        config.context_window,
        observation_dim,
    )
    assert rollout["reconstructed_observations"].shape == (
        2,
        horizon,
        observation_dim,
    )
    assert rollout["imagined_observations"].shape == (
        2,
        horizon,
        observation_dim,
    )
    assert rollout["open_loop_cosine"].shape == (2, horizon)
    assert rollout["validity"].shape == (2, horizon)
    assert np.asarray(rollout["validity"]).all()
    for value in rollout.values():
        assert np.isfinite(np.asarray(value)).all()


def test_select_display_trajectories_prefers_reset_free_windows():
    observations = jnp.arange(4 * 4 * 2, dtype=jnp.float32).reshape((4, 4, 2))
    is_last = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    batch = ReplayBatch(
        observations=observations,
        actions=jnp.zeros((4, 3), dtype=jnp.int32),
        rewards=jnp.zeros((4, 3), dtype=jnp.float32),
        is_last=is_last,
        is_terminal=is_last,
    )

    chosen = select_display_trajectories(
        batch,
        context_window=1,
        horizon=3,
        count=2,
    )

    assert chosen.observations.shape == (2, 4, 2)
    np.testing.assert_array_equal(
        np.asarray(chosen.observations),
        np.asarray(observations[jnp.asarray([1, 3])]),
    )
