"""Latent-space genwm plumbing: frozen jepa encoder feeding train_single_genwm."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from world_marl.checkpointing import save_checkpoint
from world_marl.genwm import GenWMConfig, PPOConfig, create_policy_state
from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.training import create_jepa_train_state, load_frozen_encoder
from world_marl.scripts.train_single_genwm import (
    _encode_replay,
    _make_scan_action_fns,
    _resolve_latent_encoder,
)

OBS_DIM = 4
LATENT_DIM = 8


class _AdapterStub:
    action_dim = 2
    action_low = np.array([-1.0, -1.0], dtype=np.float32)
    action_high = np.array([1.0, 1.0], dtype=np.float32)
    observation_shape = (OBS_DIM,)

    def scan_rollout(self):
        raise NotImplementedError


def _tiny_jepa_config() -> JepaConfig:
    return JepaConfig(
        observation_dim=OBS_DIM,
        action_dim=2,
        action_mode="continuous",
        latent_dim=LATENT_DIM,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2,
        dynamics_ensemble_size=2,
        max_horizon=2,
        context_window=2,
    )


def _save_tiny_checkpoint(tmp_path):
    config = _tiny_jepa_config()
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    checkpoint_dir = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint_dir,
        state,
        metadata={"jepa_config": dataclasses.asdict(config)},
    )
    return config, state, checkpoint_dir


def _linear_encoder():
    weights = jnp.asarray(
        np.random.default_rng(1).normal(size=(OBS_DIM, LATENT_DIM)),
        dtype=jnp.float32,
    )
    return lambda observations: jnp.asarray(observations, jnp.float32) @ weights


def test_load_frozen_encoder_matches_saved_params(tmp_path):
    config, state, checkpoint_dir = _save_tiny_checkpoint(tmp_path)
    encode_fn, latent_dim = load_frozen_encoder(checkpoint_dir)
    assert latent_dim == config.latent_dim
    observations = jnp.asarray(
        np.random.default_rng(0).normal(size=(5, config.observation_dim)),
        dtype=jnp.float32,
    )
    expected = JepaWorldModel(config).apply(
        {"params": state.params}, observations, method=JepaWorldModel.encode
    )
    assert np.asarray(encode_fn(observations)).shape == (5, config.latent_dim)
    np.testing.assert_allclose(
        np.asarray(encode_fn(observations)), np.asarray(expected), rtol=1e-5
    )


def test_scan_action_fns_apply_encoder_before_policy():
    config = GenWMConfig(
        arm="llada2",
        obs_dim=LATENT_DIM,
        action_dim=2,
        action_mode="continuous",
        obs_bins=4,
        action_bins=4,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        mlp_ratio=2,
        block_size=2,
        steps_per_block=1,
    )
    policy_state = create_policy_state(jax.random.PRNGKey(0), config, PPOConfig())
    encode_fn = _linear_encoder()
    fns = _make_scan_action_fns(_AdapterStub(), "continuous", encode_fn=encode_fn)
    observations = jnp.asarray(
        np.random.default_rng(2).normal(size=(3, OBS_DIM)), dtype=jnp.float32
    )
    actions, _, values, _ = fns["mode"](
        policy_state, jax.random.PRNGKey(3), observations
    )
    policy, expected_values = policy_state.apply_fn(
        {"params": policy_state.params}, encode_fn(observations)
    )
    np.testing.assert_allclose(
        np.asarray(actions), np.asarray(policy.mode()), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(values), np.asarray(expected_values), rtol=1e-6
    )
    random_actions, _, _, _ = fns["random"](None, jax.random.PRNGKey(4), observations)
    assert random_actions.shape == (3, 2)


def test_encode_replay_maps_observation_keys_only():
    encode_fn = _linear_encoder()
    rng = np.random.default_rng(3)
    replay = {
        "observations": rng.normal(size=(6, OBS_DIM)).astype(np.float32),
        "actions": np.zeros((6, 2), dtype=np.float32),
        "rewards": np.arange(6, dtype=np.float32),
        "dones": np.zeros((6,), dtype=np.float32),
        "next_observations": rng.normal(size=(6, OBS_DIM)).astype(np.float32),
    }
    encoded = _encode_replay(encode_fn, replay)
    assert encoded["observations"].shape == (6, LATENT_DIM)
    assert encoded["next_observations"].shape == (6, LATENT_DIM)
    assert encoded["observations"].dtype == np.float32
    np.testing.assert_allclose(
        encoded["observations"],
        np.asarray(encode_fn(replay["observations"])),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(encoded["rewards"], replay["rewards"])
    np.testing.assert_array_equal(encoded["actions"], replay["actions"])


def test_resolve_latent_encoder_passthrough_without_flag():
    encode_fn, obs_dim = _resolve_latent_encoder(
        None, _AdapterStub(), OBS_DIM, arm="llada2"
    )
    assert encode_fn is None
    assert obs_dim == OBS_DIM


def test_resolve_latent_encoder_loads_checkpoint(tmp_path):
    config, _, checkpoint_dir = _save_tiny_checkpoint(tmp_path)
    encode_fn, obs_dim = _resolve_latent_encoder(
        str(checkpoint_dir), _AdapterStub(), OBS_DIM, arm="llada2"
    )
    assert obs_dim == config.latent_dim
    observations = jnp.zeros((2, OBS_DIM), dtype=jnp.float32)
    assert np.asarray(encode_fn(observations)).shape == (2, config.latent_dim)


def test_resolve_latent_encoder_rejects_loop_adapters(tmp_path):
    _, _, checkpoint_dir = _save_tiny_checkpoint(tmp_path)

    class _LoopAdapter:
        observation_shape = (OBS_DIM,)

    with pytest.raises(ValueError, match="scan_rollout"):
        _resolve_latent_encoder(
            str(checkpoint_dir), _LoopAdapter(), OBS_DIM, arm="llada2"
        )


def test_resolve_latent_encoder_rejects_model_free(tmp_path):
    _, _, checkpoint_dir = _save_tiny_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="model-free"):
        _resolve_latent_encoder(
            str(checkpoint_dir), _AdapterStub(), OBS_DIM, arm="model-free"
        )


def test_parse_args_accepts_latent_encoder(tmp_path):
    from world_marl.scripts.train_single_genwm import parse_args

    base = ["--env", "brax:reacher", "--arm", "llada2"]
    args = parse_args([*base, "--latent-encoder", str(tmp_path)])
    assert args.latent_encoder == str(tmp_path)
    assert parse_args(base).latent_encoder is None
