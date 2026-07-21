"""Tests for the single-agent generative world-model arms (world_marl.genwm)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from world_marl.genwm import (
    GENWM_ARMS,
    ContinuousTokenTransformer,
    GenWMConfig,
    PPOConfig,
    create_genwm_state,
    create_head_state,
    create_policy_state,
    decode_tokens,
    encode_tokens,
    fit_quantile_tokenizer,
    genwm_predict_next,
    genwm_train_step,
    head_train_step,
    imagined_rollout,
    ppo_update,
)

OBS_DIM = 4
OBS_BINS = 5
ACTION_BINS = 3


def _config(arm: str, action_mode: str) -> GenWMConfig:
    return GenWMConfig(
        arm=arm,
        obs_dim=OBS_DIM,
        action_dim=2,
        action_mode=action_mode,
        obs_bins=OBS_BINS,
        action_bins=ACTION_BINS,
        model_dim=16,
        num_heads=2,
        num_layers=1,
        integration_steps=3,
        block_size=2,
        steps_per_block=2,
    )


def _tokenizers(config: GenWMConfig, rng: np.random.Generator):
    obs_tokenizer = fit_quantile_tokenizer(
        rng.normal(size=(256, config.obs_dim)), config.obs_bins
    )
    action_tokenizer = None
    if config.action_mode == "continuous":
        action_tokenizer = fit_quantile_tokenizer(
            rng.uniform(-1.0, 1.0, size=(256, config.action_dim)), config.action_bins
        )
    return obs_tokenizer, action_tokenizer


def _transition_batch(config: GenWMConfig, rng: np.random.Generator, batch: int = 16):
    observations = jnp.asarray(
        rng.normal(size=(batch, config.obs_dim)), dtype=jnp.float32
    )
    next_observations = jnp.asarray(
        rng.normal(size=(batch, config.obs_dim)), dtype=jnp.float32
    )
    if config.action_mode == "discrete":
        actions = jnp.asarray(
            rng.integers(0, config.action_dim, size=(batch,)), dtype=jnp.int32
        )
    else:
        actions = jnp.asarray(
            rng.uniform(-1.0, 1.0, size=(batch, config.action_dim)),
            dtype=jnp.float32,
        )
    return observations, actions, next_observations


def test_quantile_tokenizer_roundtrip_stays_within_bins() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(512, 3))
    tokenizer = fit_quantile_tokenizer(samples, 8)

    values = jnp.asarray(samples[:64], dtype=jnp.float32)
    tokens = encode_tokens(tokenizer, values)
    assert tokens.shape == values.shape
    assert tokens.dtype == jnp.int32
    assert int(tokens.min()) >= 0
    assert int(tokens.max()) < 8

    decoded = decode_tokens(tokenizer, tokens)
    assert decoded.shape == values.shape
    # Re-encoding the decoded centers must land back in the same bins.
    assert jnp.array_equal(encode_tokens(tokenizer, decoded), tokens)


def test_quantile_tokenizer_constant_dimension_decodes_constant() -> None:
    rng = np.random.default_rng(1)
    samples = np.stack(
        [rng.normal(size=128), np.full(128, 3.5)],
        axis=1,
    )
    tokenizer = fit_quantile_tokenizer(samples, 4)
    values = jnp.asarray(samples[:16], dtype=jnp.float32)
    decoded = decode_tokens(tokenizer, encode_tokens(tokenizer, values))
    np.testing.assert_allclose(np.asarray(decoded[:, 1]), 3.5, rtol=1e-6)


def test_continuous_token_transformer_contract() -> None:
    model = ContinuousTokenTransformer(model_dim=16, num_heads=2, ffn_hidden_dims=(32,))
    key = jax.random.PRNGKey(0)
    x = jnp.ones((5, OBS_DIM))
    t = jnp.full((5, 1), 0.3)
    cond = jnp.ones((5, 6))
    params = model.init(key, x, t, cond)["params"]
    velocity = model.apply({"params": params}, x, t, cond)
    assert velocity.shape == (5, OBS_DIM)
    assert bool(jnp.all(jnp.isfinite(velocity)))


@pytest.mark.parametrize("arm", GENWM_ARMS)
@pytest.mark.parametrize("action_mode", ["discrete", "continuous"])
def test_genwm_train_and_predict(arm: str, action_mode: str) -> None:
    config = _config(arm, action_mode)
    rng = np.random.default_rng(2)
    obs_tokenizer, action_tokenizer = _tokenizers(config, rng)
    observations, actions, next_observations = _transition_batch(config, rng)

    key = jax.random.PRNGKey(0)
    state = create_genwm_state(key, config)
    for _ in range(3):
        key, step_key = jax.random.split(key)
        state, loss = genwm_train_step(
            state,
            step_key,
            observations,
            actions,
            next_observations,
            obs_tokenizer,
            action_tokenizer,
            config,
        )
        assert bool(jnp.isfinite(loss))

    key, sample_key = jax.random.split(key)
    predicted = genwm_predict_next(
        state,
        sample_key,
        observations,
        actions,
        obs_tokenizer,
        action_tokenizer,
        config,
    )
    assert predicted.shape == (observations.shape[0], config.obs_dim)
    assert bool(jnp.all(jnp.isfinite(predicted)))
    if arm != "continuous-transformer":
        # Token arms must emit values on the tokenizer's decode grid.
        tokens = encode_tokens(obs_tokenizer, predicted)
        np.testing.assert_allclose(
            np.asarray(decode_tokens(obs_tokenizer, tokens)),
            np.asarray(predicted),
            rtol=1e-5,
        )


def test_head_train_step_reduces_loss() -> None:
    config = _config("continuous-transformer", "continuous")
    rng = np.random.default_rng(3)
    observations = jnp.asarray(rng.normal(size=(64, config.obs_dim)), jnp.float32)
    action_feats = jnp.asarray(
        rng.uniform(-1.0, 1.0, size=(64, config.action_dim)), jnp.float32
    )
    rewards = observations.sum(axis=-1)
    continues = (observations[:, 0] > 0.0).astype(jnp.float32)

    head_state = create_head_state(jax.random.PRNGKey(0), config)
    losses = []
    for _ in range(50):
        head_state, metrics = head_train_step(
            head_state, observations, action_feats, rewards, continues
        )
        losses.append(float(metrics["head_total_loss"]))
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]


@pytest.mark.parametrize(
    ("arm", "action_mode"),
    [("discrete-transformer", "discrete"), ("continuous-transformer", "continuous")],
)
def test_imagined_rollout_and_ppo_update(arm: str, action_mode: str) -> None:
    config = _config(arm, action_mode)
    ppo_config = PPOConfig(num_minibatches=4, update_epochs=2)
    rng = np.random.default_rng(4)
    obs_tokenizer, action_tokenizer = _tokenizers(config, rng)

    key = jax.random.PRNGKey(0)
    key, wm_key, head_key, policy_key = jax.random.split(key, 4)
    wm_state = create_genwm_state(wm_key, config)
    head_state = create_head_state(head_key, config)
    policy_state = create_policy_state(policy_key, config, ppo_config)

    start = jnp.asarray(rng.normal(size=(8, config.obs_dim)), jnp.float32)
    key, rollout_key = jax.random.split(key)
    batch, last_values = imagined_rollout(
        policy_state,
        wm_state,
        head_state,
        obs_tokenizer,
        action_tokenizer,
        start,
        rollout_key,
        horizon=4,
        config=config,
        ppo_config=ppo_config,
    )

    assert batch.observations.shape == (4, 8, config.obs_dim)
    assert batch.log_probs.shape == (4, 8)
    assert batch.rewards.shape == (4, 8)
    assert last_values.shape == (8,)
    assert bool(jnp.all((batch.dones == 0.0) | (batch.dones == 1.0)))
    if action_mode == "discrete":
        assert batch.actions.shape == (4, 8)
        assert int(batch.actions.min()) >= 0
        assert int(batch.actions.max()) < config.action_dim
    else:
        assert batch.actions.shape == (4, 8, config.action_dim)

    key, update_key = jax.random.split(key)
    before = jax.tree_util.tree_leaves(policy_state.params)[0]
    policy_state, metrics = ppo_update(
        policy_state, batch, last_values, update_key, ppo_config
    )
    after = jax.tree_util.tree_leaves(policy_state.params)[0]
    for name, value in metrics.items():
        assert bool(jnp.isfinite(value)), name
    assert not jnp.array_equal(before, after)
