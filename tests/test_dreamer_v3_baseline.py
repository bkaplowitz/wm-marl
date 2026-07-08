from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from world_marl.dreamer_v3_baseline.config import DreamerV3Config
from world_marl.dreamer_v3_baseline.losses import (
    categorical_kl_loss,
    symexp,
    symlog,
    two_hot,
)
from world_marl.dreamer_v3_baseline.models import (
    ContinueHead,
    DreamerDecoder,
    DreamerEncoder,
    RewardHead,
)
from world_marl.dreamer_v3_baseline.rssm import (
    DreamerRSSM,
    categorical_straight_through,
    flatten_rssm_state,
    initial_rssm_state,
)
from world_marl.dreamer_v3_baseline.training import (
    create_dreamer_train_state,
    dreamer_train_step,
)
from world_marl.scripts.train_dreamer_v3_baseline import main as train_dreamer_main
from world_marl.world_model_foundation.collect import synthetic_sequence_collector


def test_config_defaults_lock_categorical_rssm_contract() -> None:
    config = DreamerV3Config(action_dim=4, observation_shape=(8, 8, 3))

    assert config.rssm.deterministic_size > 0
    assert config.rssm.stochastic_size > 0
    assert config.rssm.discrete_classes > 1
    assert config.rssm.latent_size == (
        config.rssm.deterministic_size
        + config.rssm.stochastic_size * config.rssm.discrete_classes
    )
    assert config.reward_head.distribution == "symlog_two_hot"
    assert config.continue_head.distribution == "bernoulli"


def test_categorical_straight_through_returns_one_hot_forward_values() -> None:
    logits = jnp.asarray([[[0.0, 1.0, -1.0], [2.0, 0.0, -2.0]]], dtype=jnp.float32)

    stoch, probs = categorical_straight_through(logits)

    assert stoch.shape == logits.shape
    assert probs.shape == logits.shape
    assert bool(jnp.allclose(jnp.sum(stoch, axis=-1), 1.0))
    assert bool(jnp.allclose(jnp.sum(probs, axis=-1), 1.0))


def test_rssm_prior_and_posterior_shapes_and_finite_kl() -> None:
    config = DreamerV3Config(action_dim=4, observation_shape=(8, 8, 3))
    rssm = DreamerRSSM(config.rssm, action_dim=config.action_dim)
    prev_state = initial_rssm_state(batch_size=3, config=config.rssm)
    actions = jax.nn.one_hot(jnp.asarray([0, 1, 2]), config.action_dim)
    embed = jnp.ones((3, config.encoder.embedding_dim), dtype=jnp.float32)
    params = rssm.init(jax.random.PRNGKey(0), prev_state, actions, embed)

    prior, posterior = rssm.apply(params, prev_state, actions, embed)
    kl = categorical_kl_loss(posterior.logits, prior.logits, free_nats=0.0)

    assert prior.deterministic.shape == (3, config.rssm.deterministic_size)
    assert posterior.stochastic.shape == (
        3,
        config.rssm.stochastic_size,
        config.rssm.discrete_classes,
    )
    assert flatten_rssm_state(posterior).shape == (3, config.rssm.latent_size)
    assert bool(jnp.isfinite(kl))


def test_encoder_decoder_reward_continue_heads_match_world_model_shapes() -> None:
    config = DreamerV3Config(action_dim=4, observation_shape=(8, 8, 3))
    observations = jnp.ones((2, *config.observation_shape), dtype=jnp.float32)
    features = jnp.ones((2, config.rssm.latent_size), dtype=jnp.float32)

    encoder = DreamerEncoder(config.encoder.embedding_dim)
    encoder_params = encoder.init(jax.random.PRNGKey(1), observations)
    embeddings = encoder.apply(encoder_params, observations)

    decoder = DreamerDecoder(config.observation_shape)
    decoder_params = decoder.init(jax.random.PRNGKey(2), features)
    reconstructions = decoder.apply(decoder_params, features)

    reward_head = RewardHead(config.reward_head.bins)
    reward_params = reward_head.init(jax.random.PRNGKey(3), features)
    reward_logits = reward_head.apply(reward_params, features)

    continue_head = ContinueHead()
    continue_params = continue_head.init(jax.random.PRNGKey(4), features)
    continue_logits = continue_head.apply(continue_params, features)

    assert embeddings.shape == (2, config.encoder.embedding_dim)
    assert reconstructions.shape == observations.shape
    assert reward_logits.shape == (2, config.reward_head.bins)
    assert continue_logits.shape == (2,)
    assert bool(jnp.all((reconstructions >= 0.0) & (reconstructions <= 1.0)))


def test_symlog_symexp_and_two_hot_reward_targets() -> None:
    values = jnp.asarray([-2.0, 0.0, 3.0], dtype=jnp.float32)
    encoded = symlog(values)
    decoded = symexp(encoded)
    targets = two_hot(encoded, num_bins=9, lower=-4.0, upper=4.0)

    assert bool(jnp.allclose(decoded, values, atol=1e-5))
    assert targets.shape == (3, 9)
    assert bool(jnp.allclose(jnp.sum(targets, axis=-1), 1.0))


def test_world_model_train_step_updates_params_and_returns_finite_metrics() -> None:
    config = DreamerV3Config(action_dim=3, observation_shape=(6, 6, 3))
    batch = synthetic_sequence_collector(
        env_name="synthetic:image-grid",
        time_steps=4,
        batch_size=2,
        observation_shape=config.observation_shape,
        action_dim=config.action_dim,
    )
    state = create_dreamer_train_state(
        jax.random.PRNGKey(5), config, learning_rate=1e-3
    )

    updated, metrics = dreamer_train_step(state, batch, config)

    assert updated.step == state.step + 1
    for key in (
        "loss",
        "reconstruction_loss",
        "reward_loss",
        "continue_loss",
        "kl_loss",
    ):
        assert key in metrics
        assert bool(jnp.isfinite(metrics[key]))


def test_dreamer_cli_smoke_writes_expected_artifacts(tmp_path) -> None:
    exit_code = train_dreamer_main(
        [
            "--env",
            "synthetic:image-grid",
            "--out-dir",
            str(tmp_path),
            "--train-steps",
            "2",
            "--policy-train-steps",
            "2",
            "--time-steps",
            "4",
            "--batch-size",
            "2",
            "--image-size",
            "6",
            "--allow-fail",
        ]
    )

    assert exit_code == 0
    for name in (
        "config.json",
        "sources.json",
        "world_model_metrics.jsonl",
        "actor_critic_metrics.jsonl",
        "open_loop_reconstruction.png",
        "imagined_rollout.png",
        "outcome.json",
        "summary.json",
    ):
        assert (tmp_path / name).exists()
    outcome = json.loads((tmp_path / "outcome.json").read_text())
    assert outcome["status"] in {"ok", "learning_gate_failed"}
