from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.genie2_continuous_jax.action_bridge import (
    fit_linear_action_bridge,
)
from world_marl.genie2_continuous_jax.autoencoder import (
    ContinuousLatentAutoencoder,
    reconstruction_loss,
)
from world_marl.genie2_continuous_jax.config import Genie2ContinuousConfig
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
from world_marl.genie2_continuous_jax.training import (
    create_genie2_train_state,
    genie2_train_step,
)
from world_marl.scripts.train_genie2_continuous_jax import main as train_genie2_main
from world_marl.world_model_foundation.collect import synthetic_sequence_collector


def test_config_defaults_use_continuous_latents_not_vq_primary() -> None:
    config = Genie2ContinuousConfig()

    assert config.representation == "continuous_latent"
    assert config.lam.kind == "continuous"
    assert config.dynamics.objective in {"diffusion_velocity", "flow_matching"}
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

    assert decoded.shape == obs.shape
    assert bool(jnp.all((decoded >= 0.0) & (decoded <= 1.0)))


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
        "dynamics_loss",
        "reward_loss",
        "continue_loss",
    ):
        assert key in metrics
        assert bool(jnp.isfinite(metrics[key]))


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
        "latent_action_bridge.json",
        "outcome.json",
        "summary.json",
    ):
        assert (tmp_path / name).exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["model"] == "genie2_continuous_jax"
