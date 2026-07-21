from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from world_marl.genie2_continuous_jax import dynamics as dynamics_module
from world_marl.genie2_continuous_jax import policy as policy_module
from world_marl.genie2_continuous_jax.autoencoder import (
    ContinuousVideoTokenizer,
)
from world_marl.genie2_continuous_jax.config import Genie2ContinuousConfig
from world_marl.genie2_continuous_jax.dynamics import (
    ActionConditionedLatentDiffusion,
    autoregressive_sample,
    classifier_free_guidance,
    diffusion_forcing_loss,
)
from world_marl.genie2_continuous_jax.training import (
    create_genie2_train_state,
    genie2_transition_targets,
    scan_genie2_training_phases,
)
from world_marl.genie2_continuous_jax.policy import (
    create_latent_policy_states,
    simulate_latent_policy_rollout,
)
from world_marl.genie2_continuous_jax.st_transformer import AxialTransformerBlock
from world_marl.world_model_foundation.collect import synthetic_sequence_collector
from world_marl.world_model_foundation.replay import sequence_batch_to_jax


def test_real_action_heads_target_the_outcome_record_and_mask_resets() -> None:
    batch = sequence_batch_to_jax(
        synthetic_sequence_collector(
            env_name="synthetic:image-grid",
            time_steps=4,
            batch_size=1,
            observation_shape=(16, 16, 3),
            action_dim=3,
        )
    )
    batch = batch._replace(
        rewards=jnp.asarray([[0.0], [10.0], [20.0], [30.0]]),
        continues=jnp.asarray([[1.0], [1.0], [0.0], [1.0]]),
        is_first=jnp.asarray([[True], [False], [False], [True]]),
    )

    targets = genie2_transition_targets(batch)

    assert jnp.array_equal(targets["actions"], batch.actions[:-1])
    assert jnp.array_equal(targets["rewards"], batch.rewards[1:])
    assert jnp.array_equal(targets["continues"], batch.continues[1:])
    assert targets["valid"][:, 0].tolist() == [True, True, False]


def test_public_genie2_and_jasmine_substitution_defaults_are_locked() -> None:
    config = Genie2ContinuousConfig()

    assert config.conditioning_mode == "real_action"
    assert config.autoencoder.representation == "continuous_patch_grid"
    assert config.autoencoder.patch_size == 16
    assert config.autoencoder.latent_patch_dim == 32
    assert config.autoencoder.model_dim == 512
    assert config.autoencoder.ffn_dim == 2048
    assert config.autoencoder.num_blocks == 4
    assert config.autoencoder.num_heads == 8
    assert config.autoencoder.max_mask_ratio == 0.9
    assert config.dynamics.objective == "diffusion_forcing_x_prediction"
    assert config.dynamics.model_dim == 512
    assert config.dynamics.ffn_dim == 2048
    assert config.dynamics.num_blocks == 6
    assert config.dynamics.num_heads == 8
    assert config.dynamics.denoising_steps == 25
    assert config.dynamics.ramp_weight is True
    assert config.lam.enabled is False


def test_visual_tokenizer_preserves_continuous_spatiotemporal_patch_grid() -> None:
    config = Genie2ContinuousConfig.debug(
        action_dim=3,
        observation_shape=(16, 16, 3),
    )
    tokenizer = ContinuousVideoTokenizer(config.autoencoder)
    videos = jnp.linspace(0.0, 1.0, 2 * 3 * 16 * 16 * 3).reshape((2, 3, 16, 16, 3))

    variables = tokenizer.init(
        jax.random.PRNGKey(0),
        videos,
        key=jax.random.PRNGKey(1),
        training=True,
    )
    latents, reconstructions, mask = tokenizer.apply(
        variables,
        videos,
        key=jax.random.PRNGKey(2),
        training=True,
    )

    assert latents.shape == (2, 3, 16, config.autoencoder.latent_patch_dim)
    assert reconstructions.shape == videos.shape
    assert mask.shape == (2, 3, 16)
    assert bool(jnp.all(jnp.abs(latents) <= 1.0))
    assert bool(jnp.all(jnp.isfinite(reconstructions)))


def test_diffusion_forcing_uses_direct_actions_and_independent_frame_noise() -> None:
    config = Genie2ContinuousConfig.debug(
        action_dim=3,
        observation_shape=(16, 16, 3),
    )
    dynamics = ActionConditionedLatentDiffusion(
        latent_patch_dim=config.autoencoder.latent_patch_dim,
        action_dim=config.action_dim,
        config=config.dynamics,
    )
    latents = jnp.zeros((2, 4, 16, config.autoencoder.latent_patch_dim))
    actions = jnp.ones((2, 3, config.action_dim))

    variables = dynamics.init(
        jax.random.PRNGKey(0),
        latents,
        actions,
        key=jax.random.PRNGKey(1),
        training=True,
    )
    outputs = dynamics.apply(
        variables,
        latents,
        actions,
        key=jax.random.PRNGKey(2),
        training=True,
    )
    loss = diffusion_forcing_loss(outputs, ramp_weight=True)

    assert outputs["x_prediction"].shape == latents.shape
    assert outputs["x_target"].shape == latents.shape
    assert outputs["signal_level"].shape == latents.shape[:2]
    assert outputs["noise"].shape == latents.shape
    assert jnp.isfinite(loss)


def test_classifier_free_guidance_uses_standard_conditional_difference() -> None:
    conditioned = jnp.asarray([3.0, 5.0])
    unconditioned = jnp.asarray([1.0, 2.0])
    guided = classifier_free_guidance(conditioned, unconditioned, 1.5)
    assert jnp.allclose(guided, unconditioned + 1.5 * (conditioned - unconditioned))


def test_classifier_free_dropout_drops_the_whole_action_sequence() -> None:
    config = Genie2ContinuousConfig.debug(
        action_dim=2,
        observation_shape=(16, 16, 3),
    )
    dynamics_config = replace(config.dynamics, classifier_free_dropout=0.5)
    dynamics = ActionConditionedLatentDiffusion(
        latent_patch_dim=config.autoencoder.latent_patch_dim,
        action_dim=config.action_dim,
        config=dynamics_config,
    )
    latents = jnp.zeros((16, 6, 16, config.autoencoder.latent_patch_dim))
    actions = jnp.ones((16, 5, config.action_dim))
    variables = dynamics.init(
        jax.random.PRNGKey(0),
        latents,
        actions,
        key=jax.random.PRNGKey(1),
        training=True,
    )

    outputs = dynamics.apply(
        variables,
        latents,
        actions,
        key=jax.random.PRNGKey(2),
        training=True,
    )
    keep = outputs["condition_keep_mask"]

    assert bool(jnp.all(keep == keep[:, :1]))
    assert bool(jnp.any(keep == 0.0))
    assert bool(jnp.any(keep == 1.0))


def test_jasmine_context_corruption_uses_nearest_discrete_denoising_level() -> None:
    assert hasattr(dynamics_module, "quantized_context_signal_level")

    level = dynamics_module.quantized_context_signal_level(
        denoising_steps=25,
        context_corruption=0.1,
    )

    assert jnp.allclose(level, 22.0 / 25.0)


def test_visual_tokenizer_and_dynamics_are_temporally_causal() -> None:
    config = Genie2ContinuousConfig.debug(
        action_dim=2,
        observation_shape=(16, 16, 3),
    )
    tokenizer = ContinuousVideoTokenizer(config.autoencoder)
    videos = jnp.zeros((1, 3, 16, 16, 3), dtype=jnp.float32)
    changed_future = videos.at[:, -1].set(1.0)
    tokenizer_variables = tokenizer.init(
        jax.random.PRNGKey(0),
        videos,
        training=False,
    )
    base_latents, _ = tokenizer.apply(
        tokenizer_variables,
        videos,
        training=False,
        method=ContinuousVideoTokenizer.encode,
    )
    future_changed_latents, _ = tokenizer.apply(
        tokenizer_variables,
        changed_future,
        training=False,
        method=ContinuousVideoTokenizer.encode,
    )

    assert jnp.allclose(base_latents[:, :-1], future_changed_latents[:, :-1])

    dynamics = ActionConditionedLatentDiffusion(
        latent_patch_dim=config.autoencoder.latent_patch_dim,
        action_dim=config.action_dim,
        config=config.dynamics,
    )
    actions = jnp.zeros((1, 2, config.action_dim), dtype=jnp.float32)
    changed_actions = actions.at[:, -1].set(1.0)
    denoising_steps = jnp.zeros((1, 3), dtype=jnp.int32)
    condition_keep = jnp.ones((1, 2), dtype=jnp.float32)
    dynamics_variables = dynamics.init(
        jax.random.PRNGKey(1),
        base_latents,
        actions,
        denoising_steps,
        condition_keep_mask=condition_keep,
        training=False,
        method=ActionConditionedLatentDiffusion.predict_x,
    )
    base_prediction = dynamics.apply(
        dynamics_variables,
        base_latents,
        actions,
        denoising_steps,
        condition_keep_mask=condition_keep,
        training=False,
        method=ActionConditionedLatentDiffusion.predict_x,
    )
    changed_prediction = dynamics.apply(
        dynamics_variables,
        future_changed_latents,
        changed_actions,
        denoising_steps,
        condition_keep_mask=condition_keep,
        training=False,
        method=ActionConditionedLatentDiffusion.predict_x,
    )

    assert jnp.allclose(base_prediction[:, :-1], changed_prediction[:, :-1])


def test_axial_transformer_blocks_rematerialize_during_backpropagation() -> None:
    block = AxialTransformerBlock(
        model_dim=8,
        ffn_dim=16,
        num_heads=2,
        compute_dtype="float32",
    )
    inputs = jnp.zeros((1, 2, 2, 8), dtype=jnp.float32)
    variables = block.init(jax.random.PRNGKey(0), inputs, True)

    def loss(params, values):
        outputs = block.apply({"params": params}, values, True)
        return jnp.sum(outputs)

    backward_jaxpr = str(jax.make_jaxpr(jax.grad(loss))(variables["params"], inputs))

    assert "remat" in backward_jaxpr


def test_policy_observation_history_shifts_and_resets_per_environment() -> None:
    assert hasattr(policy_module, "update_observation_history")
    history = jnp.asarray(
        [
            [[1.0], [4.0]],
            [[2.0], [5.0]],
            [[3.0], [6.0]],
        ]
    )
    observations = jnp.asarray([[10.0], [20.0]])

    updated = policy_module.update_observation_history(
        history,
        observations,
        jnp.asarray([False, True]),
    )

    assert jnp.array_equal(updated[:, 0, 0], jnp.asarray([2.0, 3.0, 10.0]))
    assert jnp.array_equal(updated[:, 1, 0], jnp.asarray([20.0, 20.0, 20.0]))


def test_repository_policy_lambda_returns_include_task_discount() -> None:
    returns = policy_module._lambda_returns(
        rewards=jnp.zeros((2, 1)),
        continues=jnp.ones((2, 1)),
        values=jnp.zeros((2, 1)),
        bootstrap=jnp.ones((1,)),
        discount=0.5,
        discount_lambda=1.0,
    )

    assert jnp.allclose(returns[:, 0], jnp.asarray([0.25, 0.5]))


def test_continuous_policy_is_squashed_and_preserves_score_gradient() -> None:
    bounded_config = Genie2ContinuousConfig.debug(
        action_dim=1,
        observation_shape=(4,),
        action_low=(-1.0,),
        action_high=(1.0,),
    )
    actor_state, _ = create_latent_policy_states(
        jax.random.PRNGKey(0),
        bounded_config,
        learning_rate=1e-3,
    )
    pooled = jnp.zeros(
        (4096, bounded_config.autoencoder.latent_patch_dim), dtype=jnp.float32
    )
    sample = policy_module._policy_sample(
        actor_state,
        actor_state.params,
        pooled,
        bounded_config,
        jax.random.PRNGKey(1),
        deterministic=False,
    )

    assert bool(jnp.all(sample.env_action > -1.0))
    assert bool(jnp.all(sample.env_action < 1.0))
    assert bool(jnp.all(jnp.isfinite(sample.log_probability)))

    score_config = Genie2ContinuousConfig.debug(
        action_dim=1,
        observation_shape=(4,),
        action_low=(-100.0,),
        action_high=(100.0,),
    )
    score_actor_state, _ = create_latent_policy_states(
        jax.random.PRNGKey(2),
        score_config,
        learning_rate=1e-3,
    )
    score_pooled = jnp.zeros(
        (1, score_config.autoencoder.latent_patch_dim), dtype=jnp.float32
    )

    def log_probability(params):
        score_sample = policy_module._policy_sample(
            score_actor_state,
            params,
            score_pooled,
            score_config,
            jax.random.PRNGKey(3),
            deterministic=False,
        )
        return jnp.sum(score_sample.log_probability)

    gradients = jax.grad(log_probability)(score_actor_state.params)
    mean_bias_gradient = gradients["params"]["action_mean"]["bias"]
    assert bool(jnp.any(jnp.abs(mean_bias_gradient) > 1e-5))


def test_autoregressive_diffusion_sampler_lowers_to_nested_scan() -> None:
    config = Genie2ContinuousConfig.debug(
        action_dim=2,
        observation_shape=(16, 16, 3),
    )
    dynamics = ActionConditionedLatentDiffusion(
        latent_patch_dim=config.autoencoder.latent_patch_dim,
        action_dim=config.action_dim,
        config=config.dynamics,
    )
    context = jnp.zeros((1, 2, 16, config.autoencoder.latent_patch_dim))
    actions = jnp.zeros((1, 3, config.action_dim))
    variables = dynamics.init(
        jax.random.PRNGKey(0),
        context,
        actions[:, :1],
        key=jax.random.PRNGKey(1),
        training=False,
    )

    def sample(key: jax.Array) -> jax.Array:
        return autoregressive_sample(
            dynamics.apply,
            variables,
            context,
            actions,
            key=key,
            num_future_frames=2,
            config=config.dynamics,
        )

    sampled = jax.jit(sample)(jax.random.PRNGKey(3))
    jaxpr = str(jax.make_jaxpr(sample)(jax.random.PRNGKey(4)))
    assert sampled.shape == (1, 4, 16, config.autoencoder.latent_patch_dim)
    assert jaxpr.count("scan[") >= 2


def test_staged_training_uses_compiled_scans_and_frozen_stage_boundaries() -> None:
    config = Genie2ContinuousConfig.debug(
        action_dim=2,
        observation_shape=(16, 16, 3),
        action_mode="discrete",
    )
    replay = sequence_batch_to_jax(
        synthetic_sequence_collector(
            time_steps=4,
            batch_size=2,
            observation_shape=config.observation_shape,
            action_dim=config.action_dim,
            env_name="synthetic:image-grid",
        )
    )
    state = create_genie2_train_state(
        jax.random.PRNGKey(0),
        config=config,
    )

    def train(train_state, key):
        return scan_genie2_training_phases(
            train_state,
            replay,
            key,
            config=config,
            tokenizer_steps=1,
            dynamics_steps=1,
            reward_continue_steps=1,
            sequence_length=4,
            batch_size=2,
        )

    updated, metrics, validation = jax.jit(train)(state, jax.random.PRNGKey(1))
    jaxpr = str(jax.make_jaxpr(train)(state, jax.random.PRNGKey(2)))
    assert updated.tokenizer.step == 1
    assert updated.dynamics.step == 1
    assert updated.heads.step == 1
    assert jaxpr.count("scan[") >= 3
    assert all(
        bool(jnp.all(jnp.isfinite(value)))
        for phase_metrics in metrics.values()
        for value in phase_metrics.values()
    )
    assert all(
        bool(jnp.all(jnp.isfinite(value)))
        for phase_metrics in validation.values()
        for value in phase_metrics.values()
    )


def test_primary_simulator_policy_chooses_real_actions_without_a_bridge() -> None:
    config = Genie2ContinuousConfig.debug(
        action_dim=3,
        observation_shape=(16, 16, 3),
        action_mode="discrete",
    )
    world_state = create_genie2_train_state(jax.random.PRNGKey(0), config=config)
    actor_state, critic_state = create_latent_policy_states(
        jax.random.PRNGKey(1),
        config,
        learning_rate=1e-3,
    )
    start_latents = jnp.zeros(
        (1, 1, 16, config.autoencoder.latent_patch_dim), dtype=jnp.float32
    )
    rollout = jax.jit(
        lambda key: simulate_latent_policy_rollout(
            world_model_state=world_state,
            actor_state=actor_state,
            critic_state=critic_state,
            start_latents=start_latents,
            config=config,
            horizon=1,
            key=key,
        )
    )(jax.random.PRNGKey(2))

    assert rollout.environment_actions.shape == (1, 1)
    assert rollout.model_actions.shape == (1, 1, config.action_dim)
    assert rollout.environment_actions.dtype == jnp.int32
    assert bool(jnp.all(jnp.isfinite(rollout.returns)))
