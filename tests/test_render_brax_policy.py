"""Checkpoint discovery and actor reconstruction for render_brax_policy.

Covers the pure/flax-only pieces: discovery + skip rules on synthetic run
trees, and save -> reload -> deterministic-action round-trips for every arm
family (jepa, plain genwm, genie tokenizer, frozen latent encoder). No brax
env or mujoco rendering is exercised here.
"""

import dataclasses
import json
from argparse import Namespace

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.checkpointing import save_checkpoint
from world_marl.genwm import (
    GenieTokenizer,
    GenWMConfig,
    PPOConfig,
    create_genie_state,
    create_policy_state,
)
from world_marl.jepa.models import JepaConfig
from world_marl.jepa.training import create_jepa_train_state
from world_marl.scripts.render_brax_policy import discover_checkpoints, load_actor
from world_marl.scripts.train_single_genwm import _save_policy_checkpoint


def _tiny_jepa_config():
    return JepaConfig(
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


def _write_metadata(checkpoint_dir, **overrides):
    metadata = {
        "algorithm": "single_agent_sigreg_jepa_world_model",
        "env": "brax:reacher",
        "control": "none",
        "policy_trained": True,
    }
    metadata.update(overrides)
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata))


def _genwm_args(**overrides):
    args = Namespace(
        arm="discrete-transformer",
        env="brax:reacher",
        tokenizer="quantile",
        latent_encoder=None,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _tiny_genwm_config(**overrides):
    base = GenWMConfig(
        arm="discrete-transformer",
        obs_dim=3,
        action_dim=2,
        action_mode="continuous",
        model_dim=16,
        num_heads=2,
        num_layers=1,
    )
    return dataclasses.replace(base, **overrides)


def test_discovery_skip_rules(tmp_path):
    _write_metadata(tmp_path / "a" / "run_000" / "checkpoint")
    _write_metadata(
        tmp_path / "b" / "run_000" / "checkpoint", control="shuffle-rewards"
    )
    _write_metadata(tmp_path / "c" / "run_000" / "checkpoint", env="gymnax:CartPole-v1")
    _write_metadata(
        tmp_path / "d" / "run_000" / "policy_checkpoint",
        algorithm="single_genwm_policy",
    )
    _write_metadata(tmp_path / "e" / "run_000" / "checkpoint", policy_trained=False)

    refs, skipped = discover_checkpoints([tmp_path])
    assert [ref.checkpoint_dir.parts[-3] for ref in refs] == ["a", "d"]
    assert len(skipped) == 3

    refs, _ = discover_checkpoints([tmp_path], include_controls=True)
    assert len(refs) == 3

    refs, skipped = discover_checkpoints([tmp_path / "a" / "run_000" / "checkpoint"])
    assert len(refs) == 1 and not skipped


def test_jepa_actor_round_trip(tmp_path):
    config = _tiny_jepa_config()
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    save_checkpoint(
        tmp_path / "run_000" / "checkpoint",
        state,
        metadata={
            "algorithm": "single_agent_sigreg_jepa_world_model",
            "env": "brax:reacher",
            "control": "none",
            "policy_trained": True,
            "jepa_config": dataclasses.asdict(config),
        },
    )
    refs, skipped = discover_checkpoints([tmp_path])
    assert len(refs) == 1 and not skipped
    act = load_actor(refs[0])
    actions = act(jnp.ones((1, config.observation_dim), dtype=jnp.float32))
    assert actions.shape == (1, config.action_dim)
    assert bool(jnp.all(jnp.abs(actions) <= 1.0))


def test_genwm_policy_round_trip(tmp_path):
    config = _tiny_genwm_config()
    ppo_config = PPOConfig()
    policy_state = create_policy_state(jax.random.PRNGKey(0), config, ppo_config)
    run_dir = tmp_path / "run_000"
    _save_policy_checkpoint(
        run_dir,
        policy_state,
        args=_genwm_args(),
        config=config,
        ppo_config=ppo_config,
        genie_module=None,
        genie_state=None,
        action_mode="continuous",
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        seed=0,
    )
    refs, skipped = discover_checkpoints([tmp_path])
    assert len(refs) == 1 and not skipped
    act = load_actor(refs[0])
    observations = jnp.ones((1, config.obs_dim), dtype=jnp.float32)
    actions = act(observations)
    policy, _ = policy_state.apply_fn({"params": policy_state.params}, observations)
    expected = jnp.clip(policy.mode(), -1.0, 1.0)
    # jit vs eager float32 reassociation leaves ~1e-9 absolute noise
    np.testing.assert_allclose(
        np.asarray(actions), np.asarray(expected), rtol=1e-4, atol=1e-7
    )


def test_genie_policy_round_trip(tmp_path):
    genie_module = GenieTokenizer(
        obs_dim=3,
        codebook_size=8,
        code_dim=2,
        model_dim=16,
        num_heads=2,
        num_layers=1,
        mlp_ratio=2,
    )
    genie_state = create_genie_state(
        jax.random.PRNGKey(1), genie_module, learning_rate=1e-3
    )
    config = _tiny_genwm_config(code_dim=2)
    ppo_config = PPOConfig()
    policy_state = create_policy_state(jax.random.PRNGKey(0), config, ppo_config)
    _save_policy_checkpoint(
        tmp_path / "run_000",
        policy_state,
        args=_genwm_args(tokenizer="genie"),
        config=config,
        ppo_config=ppo_config,
        genie_module=genie_module,
        genie_state=genie_state,
        action_mode="continuous",
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        seed=0,
    )
    refs, skipped = discover_checkpoints([tmp_path])
    assert len(refs) == 1 and not skipped
    act = load_actor(refs[0])
    actions = act(jnp.ones((1, config.obs_dim), dtype=jnp.float32))
    assert actions.shape == (1, config.action_dim)
    assert bool(jnp.all(jnp.abs(actions) <= 1.0))


def test_latent_encoder_policy_round_trip(tmp_path):
    jepa_config = _tiny_jepa_config()
    jepa_state = create_jepa_train_state(jax.random.PRNGKey(0), jepa_config)
    encoder_dir = tmp_path / "jepa_source" / "checkpoint"
    save_checkpoint(
        encoder_dir,
        jepa_state,
        metadata={"jepa_config": dataclasses.asdict(jepa_config)},
    )
    config = _tiny_genwm_config(obs_dim=jepa_config.latent_dim)
    ppo_config = PPOConfig()
    policy_state = create_policy_state(jax.random.PRNGKey(0), config, ppo_config)
    _save_policy_checkpoint(
        tmp_path / "runs" / "run_000",
        policy_state,
        args=_genwm_args(latent_encoder=str(encoder_dir)),
        config=config,
        ppo_config=ppo_config,
        genie_module=None,
        genie_state=None,
        action_mode="continuous",
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        seed=0,
    )
    refs, skipped = discover_checkpoints([tmp_path / "runs"])
    assert len(refs) == 1 and not skipped
    act = load_actor(refs[0])
    actions = act(jnp.ones((1, jepa_config.observation_dim), dtype=jnp.float32))
    assert actions.shape == (1, config.action_dim)
    assert bool(jnp.all(jnp.abs(actions) <= 1.0))
