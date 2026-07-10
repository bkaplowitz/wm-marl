from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import freeze, unfreeze

from world_marl.genie2_continuous_jax.action_bridge import (
    fit_linear_action_bridge,
)
from world_marl.genie2_continuous_jax.autoencoder import (
    ContinuousLatentAutoencoder,
    reconstruction_loss,
)
from world_marl.genie2_continuous_jax.config import (
    AutoencoderConfig,
    DynamicsConfig,
    Genie2ContinuousConfig,
    LAMConfig,
)
from world_marl.genie2_continuous_jax.dynamics import (
    CausalLatentDynamics,
    classifier_free_guidance,
    dynamics_mse_loss,
)
from world_marl.genie2_continuous_jax.lam import (
    ContinuousLAM,
    lam_kl_loss,
    sample_latent_actions,
)
from world_marl.genie2_continuous_jax.rl_heads import RewardContinueHead
from world_marl.genie2_continuous_jax.sampling import sample_next_observation
from world_marl.genie2_continuous_jax.policy import train_genie2_latent_policy
from world_marl.genie2_continuous_jax.training import (
    create_genie2_train_state,
    genie2_train_step,
)
from world_marl.scripts.train_genie2_continuous_jax import main as train_genie2_main
from world_marl.world_model_foundation.collect import synthetic_sequence_collector
from world_marl.world_model_foundation.replay import WorldModelSequenceBatch


def test_config_defaults_use_continuous_latents_not_vq_primary() -> None:
    config = Genie2ContinuousConfig()

    assert config.representation == "continuous_latent"
    assert config.lam.kind == "continuous"
    assert config.dynamics.objective in {"diffusion_velocity", "flow_matching"}
    assert config.dynamics.sampling_steps > 0
    assert config.lam.kl_scale > 0.0
    assert config.vq_maskgit_ablation_enabled is False
    assert config.autoencoder.latent_dim > 0


def test_autoencoder_returns_continuous_latents_and_finite_reconstruction_loss() -> (
    None
):
    observations = jnp.ones((2, 8, 8, 3), dtype=jnp.float32) * 0.5
    model = ContinuousLatentAutoencoder(latent_dim=12, hidden_dims=(32,))
    params = model.init(jax.random.PRNGKey(0), observations)

    latents, reconstructions = model.apply(params, observations)
    loss = reconstruction_loss(observations, reconstructions)

    assert latents.shape == (2, 12)
    assert reconstructions.shape == observations.shape
    assert bool(jnp.isfinite(loss))
    assert float(jnp.min(reconstructions)) >= 0.0
    assert float(jnp.max(reconstructions)) <= 1.0


def test_autoencoder_vector_decoder_is_unbounded() -> None:
    observations = jnp.zeros((2, 5), dtype=jnp.float32)
    model = ContinuousLatentAutoencoder(latent_dim=4, hidden_dims=(8,))
    params = model.init(jax.random.PRNGKey(40), observations)
    mutable = unfreeze(params)
    mutable["params"]["decoder"]["kernel"] = jnp.zeros_like(
        mutable["params"]["decoder"]["kernel"]
    )
    mutable["params"]["decoder"]["bias"] = jnp.full_like(
        mutable["params"]["decoder"]["bias"], -2.0
    )

    _, reconstructions = model.apply(freeze(mutable), observations)

    assert bool(jnp.allclose(reconstructions, -2.0))


def test_continuous_lam_samples_latent_actions_from_latent_transitions() -> None:
    prev_latents = jnp.zeros((3, 10), dtype=jnp.float32)
    next_latents = jnp.ones((3, 10), dtype=jnp.float32)
    model = ContinuousLAM(latent_action_dim=5, hidden_dims=(32,))
    params = model.init(jax.random.PRNGKey(1), prev_latents, next_latents)

    mean, log_std = model.apply(params, prev_latents, next_latents)
    actions = sample_latent_actions(jax.random.PRNGKey(2), mean, log_std)
    loss = lam_kl_loss(mean, log_std)

    assert mean.shape == (3, 5)
    assert log_std.shape == (3, 5)
    assert actions.shape == (3, 5)
    assert bool(jnp.all(jnp.isfinite(actions)))
    assert bool(jnp.isfinite(loss))


def test_causal_dynamics_predicts_next_latent_and_cfg_combines_predictions() -> None:
    latent_history = jnp.zeros((2, 4, 10), dtype=jnp.float32)
    latent_actions = jnp.ones((2, 4, 5), dtype=jnp.float32)
    noise_level = jnp.full((2,), 0.25, dtype=jnp.float32)
    model = CausalLatentDynamics(
        latent_dim=10,
        latent_action_dim=5,
        model_dim=32,
        num_heads=4,
        num_layers=1,
    )
    params = model.init(
        jax.random.PRNGKey(3), latent_history, latent_actions, noise_level
    )

    conditioned = model.apply(params, latent_history, latent_actions, noise_level)
    unconditioned = model.apply(
        params, latent_history, jnp.zeros_like(latent_actions), noise_level
    )
    guided = classifier_free_guidance(
        conditioned=conditioned,
        unconditioned=unconditioned,
        guidance_scale=1.5,
    )
    loss = dynamics_mse_loss(conditioned, jnp.ones_like(conditioned))

    assert conditioned.shape == (2, 10)
    assert guided.shape == (2, 10)
    assert bool(jnp.all(jnp.isfinite(guided)))
    assert bool(jnp.isfinite(loss))


def test_linear_action_bridge_recovers_known_action_mapping() -> None:
    latent_actions = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
    )
    real_actions = np.asarray([[0.5], [2.5], [-0.5], [1.5]], dtype=np.float32)

    bridge = fit_linear_action_bridge(latent_actions, real_actions, ridge=1e-6)
    predicted = bridge.predict(latent_actions)

    np.testing.assert_allclose(predicted, real_actions, atol=1e-4)
    assert bridge.latent_action_dim == 2
    assert bridge.real_action_dim == 1


def test_reward_continue_head_and_sampler_shapes_are_finite() -> None:
    latents = jnp.ones((3, 12), dtype=jnp.float32)
    latent_actions = jnp.zeros((3, 5), dtype=jnp.float32)
    head = RewardContinueHead(hidden_dims=(16,))
    params = head.init(jax.random.PRNGKey(4), latents, latent_actions)

    reward, continue_logit = head.apply(params, latents, latent_actions)

    assert reward.shape == (3,)
    assert continue_logit.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(reward)))

    autoencoder = ContinuousLatentAutoencoder(latent_dim=12, hidden_dims=(16,))
    obs = jnp.ones((3, 6, 6, 3), dtype=jnp.float32) * 0.25
    ae_params = autoencoder.init(jax.random.PRNGKey(5), obs)
    decoded = sample_next_observation(autoencoder.apply, ae_params, latents)
    decoded_zeros = sample_next_observation(
        autoencoder.apply,
        ae_params,
        jnp.zeros_like(latents),
    )

    assert decoded.shape == obs.shape
    assert bool(jnp.all((decoded >= 0.0) & (decoded <= 1.0)))
    assert not bool(jnp.allclose(decoded, decoded_zeros))


def test_genie2_train_step_updates_params_and_returns_finite_metrics() -> None:
    config = Genie2ContinuousConfig()
    batch = synthetic_sequence_collector(
        env_name="synthetic:image-grid",
        time_steps=4,
        batch_size=2,
        observation_shape=(6, 6, 3),
        action_dim=3,
    )
    state = create_genie2_train_state(
        jax.random.PRNGKey(6),
        observation_shape=(6, 6, 3),
        config=config,
        learning_rate=1e-3,
    )

    updated, metrics = genie2_train_step(state, batch, config)

    assert updated.step == state.step + 1
    for key in (
        "loss",
        "reconstruction_loss",
        "lam_kl_loss",
        "lam_reconstruction_loss",
        "dynamics_loss",
        "flow_velocity_loss",
        "reward_loss",
        "continue_loss",
    ):
        assert key in metrics
        assert bool(jnp.isfinite(metrics[key]))

    outputs = state.apply_fn(
        state.params,
        jnp.asarray(batch.observations, dtype=jnp.float32),
        jnp.asarray(batch.rewards, dtype=jnp.float32),
        jnp.asarray(batch.continues, dtype=jnp.float32),
    )
    assert outputs["flow_time"].shape == (batch.batch_size,)
    assert bool(jnp.all((outputs["flow_time"] > 0.0) & (outputs["flow_time"] < 1.0)))
    assert outputs["predicted_velocity"].shape == outputs["target_velocity"].shape


def test_genie2_train_step_accepts_vector_adapter_replay() -> None:
    config = Genie2ContinuousConfig()
    batch = WorldModelSequenceBatch(
        observations=np.linspace(0.0, 1.0, num=4 * 2 * 5, dtype=np.float32).reshape(
            (4, 2, 5)
        ),
        actions=np.zeros((4, 2, 2), dtype=np.float32),
        rewards=np.zeros((4, 2), dtype=np.float32),
        continues=np.ones((4, 2), dtype=np.float32),
        is_first=np.array(
            [[True, True], [False, False], [False, False], [False, False]]
        ),
        is_terminal=np.zeros((4, 2), dtype=bool),
        metadata={"action_mode": "continuous", "env": "fake:continuous"},
    )
    state = create_genie2_train_state(
        jax.random.PRNGKey(7),
        observation_shape=batch.observation_shape,
        config=config,
        learning_rate=1e-3,
    )

    updated, metrics = genie2_train_step(state, batch, config)

    assert updated.step == state.step + 1
    assert bool(jnp.isfinite(metrics["loss"]))


def test_latent_policy_training_returns_finite_learned_simulator_rollout() -> None:
    config = Genie2ContinuousConfig(
        autoencoder=AutoencoderConfig(latent_dim=8, hidden_dims=(16,)),
        lam=LAMConfig(latent_action_dim=4, hidden_dims=(16,)),
        dynamics=DynamicsConfig(
            model_dim=16,
            num_heads=4,
            num_layers=1,
            max_context=4,
        ),
        reward_continue_hidden_dims=(16,),
    )
    batch = WorldModelSequenceBatch(
        observations=np.linspace(0.0, 1.0, num=4 * 2 * 5, dtype=np.float32).reshape(
            (4, 2, 5)
        ),
        actions=np.zeros((4, 2, 2), dtype=np.float32),
        rewards=np.zeros((4, 2), dtype=np.float32),
        continues=np.ones((4, 2), dtype=np.float32),
        is_first=np.array(
            [[True, True], [False, False], [False, False], [False, False]]
        ),
        is_terminal=np.zeros((4, 2), dtype=bool),
        metadata={"action_mode": "continuous", "env": "fake:policy"},
    )
    world_model_state = create_genie2_train_state(
        jax.random.PRNGKey(30),
        observation_shape=batch.observation_shape,
        config=config,
        learning_rate=1e-3,
    )

    actor_state, critic_state, metrics, rollout = train_genie2_latent_policy(
        world_model_state=world_model_state,
        batch=batch,
        observation_shape=batch.observation_shape,
        config=config,
        train_steps=2,
        learning_rate=1e-3,
        imagination_horizon=3,
        seed=31,
    )

    assert actor_state.step == 2
    assert critic_state.step == 2
    assert len(metrics) == 2
    assert rollout.latents.shape == (3, 2, config.autoencoder.latent_dim)
    assert rollout.latent_actions.shape == (3, 2, config.lam.latent_action_dim)
    for row in metrics:
        for key in ("actor_loss", "critic_loss", "imagined_reward", "imagined_value"):
            assert np.isfinite(row[key])


def test_genie2_cli_smoke_writes_expected_artifacts(tmp_path) -> None:
    exit_code = train_genie2_main(
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
        "autoencoder_metrics.jsonl",
        "lam_metrics.jsonl",
        "dynamics_metrics.jsonl",
        "reward_continue_metrics.jsonl",
        "policy_metrics.jsonl",
        "latent_action_bridge.json",
        "latent_action_usage.json",
        "open_loop_rollout.png",
        "latent_action_grid.png",
        "outcome.json",
        "summary.json",
    ):
        assert (tmp_path / name).exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["model"] == "genie2_continuous_jax"
    assert summary["policy_source"] == "latent_policy_bridge"


def test_genie2_cli_brax_smoke_writes_real_env_bridge_artifacts(tmp_path) -> None:
    pytest.importorskip("brax")

    exit_code = train_genie2_main(
        [
            "--env",
            "brax:reacher",
            "--out-dir",
            str(tmp_path),
            "--num-envs",
            "2",
            "--collect-steps",
            "4",
            "--max-cycles",
            "4",
            "--train-steps",
            "2",
            "--policy-train-steps",
            "2",
            "--eval-episodes",
            "1",
            "--allow-fail",
        ]
    )

    assert exit_code == 0
    for name in (
        "config.json",
        "autoencoder_metrics.jsonl",
        "lam_metrics.jsonl",
        "dynamics_metrics.jsonl",
        "reward_continue_metrics.jsonl",
        "policy_metrics.jsonl",
        "latent_action_bridge.json",
        "latent_action_usage.json",
        "real_env_metrics.jsonl",
        "open_loop_rollout.png",
        "latent_action_grid.png",
        "outcome.json",
        "summary.json",
    ):
        assert (tmp_path / name).exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["env"] == "brax:reacher"
    assert summary["action_mode"] == "continuous"
    assert summary["policy_source"] == "latent_policy_bridge"
    assert "real_env_bridged_return" in summary
    real_env_row = json.loads(
        (tmp_path / "real_env_metrics.jsonl").read_text().splitlines()[0]
    )
    assert real_env_row["policy_source"] == "latent_policy_bridge"
