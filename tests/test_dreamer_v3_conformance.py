from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.traverse_util import flatten_dict

from world_marl.dreamer_v3_baseline.config import DreamerV3Config
from world_marl.dreamer_v3_baseline.imagination import (
    decode_two_hot_logits,
    discount_continuation,
    ema_critic_parameters,
    reinforce_actor_loss,
    sample_actor_output,
    update_return_normalization,
)
from world_marl.dreamer_v3_baseline.losses import (
    reconstruction_loss,
    symexp_two_hot,
    symexp_two_hot_support,
)
from world_marl.dreamer_v3_baseline.models import (
    DreamerActor,
    DreamerCritic,
    DreamerDecoder,
    DreamerEncoder,
    RewardHead,
)
from world_marl.dreamer_v3_baseline.optimizer import dreamer_laprop
from world_marl.dreamer_v3_baseline.rssm import (
    DreamerRSSM,
    categorical_straight_through,
    initial_rssm_state,
)
from world_marl.dreamer_v3_baseline.training import (
    DreamerWorldModel,
    build_dreamer_online_learner_step,
    create_dreamer_online_carry,
    create_dreamer_agent_state,
    create_dreamer_train_state,
    dreamer_agent_loss_arrays,
    dreamer_agent_train_step,
    dreamer_replay_critic_returns,
    dreamer_world_model_loss_arrays,
    observe_dreamer_sequence,
)
from world_marl.dreamer_v3_baseline.replay import (
    append_dreamer_replay,
    initialize_dreamer_replay,
    sample_dreamer_replay,
    update_dreamer_replay_latents,
)
from world_marl.world_model_foundation.collect import synthetic_sequence_collector
from world_marl.world_model_foundation.replay import sequence_batch_to_jax


def _parameter_paths(variables: dict) -> set[str]:
    return {"/".join(path) for path in flatten_dict(variables["params"]).keys()}


def test_paper_12m_profile_and_hyperparameters_are_exact() -> None:
    config = DreamerV3Config(action_dim=6, observation_shape=(64, 64, 3))

    assert config.rssm.deterministic_size == 2048
    assert config.rssm.hidden_size == 256
    assert config.rssm.stochastic_size == 32
    assert config.rssm.discrete_classes == 16
    assert config.rssm.blocks == 8
    assert config.rssm.unimix == 0.01
    assert config.encoder.hidden_dims == (256, 256, 256)
    assert config.encoder.cnn_depth == 16
    assert config.encoder.cnn_multipliers == (2, 3, 4, 4)
    assert config.encoder.cnn_kernel == 5
    assert config.encoder.cnn_outer_stride == 2
    assert config.reward_head.hidden_dims == (256,)
    assert config.continue_head.hidden_dims == (256,)
    assert config.actor_critic.hidden_dims == (256, 256, 256)
    assert config.kl_free_nats == 1.0
    assert config.dynamics_kl_scale == 1.0
    assert config.representation_kl_scale == 0.1
    assert config.actor_critic.imagination_horizon == 15
    assert config.actor_critic.discount == 1.0 - 1.0 / 333.0
    assert config.actor_critic.discount_lambda == 0.95
    assert config.actor_critic.entropy_scale == 3e-4
    assert config.actor_critic.actor_unimix == 0.01
    assert config.actor_critic.critic_replay_scale == 0.3
    assert config.actor_critic.critic_ema_decay == 0.98
    assert config.optimizer.learning_rate == 4e-5
    assert config.optimizer.agc == 0.3
    assert config.optimizer.epsilon == 1e-20
    assert config.optimizer.beta1 == 0.9
    assert config.optimizer.beta2 == 0.99
    assert config.replay.capacity == 5_000_000
    assert config.replay.batch_size == 16
    assert config.replay.batch_length == 64
    assert config.replay.train_ratio == 32.0


def test_visual_model_uses_stride_cnn_and_transpose_cnn() -> None:
    config = DreamerV3Config.debug(
        action_dim=3,
        observation_shape=(64, 64, 3),
    )
    observations = jnp.zeros((2, 64, 64, 3), dtype=jnp.float32)
    features = jnp.zeros((2, config.rssm.latent_size), dtype=jnp.float32)
    encoder = DreamerEncoder(
        config.observation_shape,
        hidden_dims=config.encoder.hidden_dims,
        cnn_depth=config.encoder.cnn_depth,
        cnn_multipliers=config.encoder.cnn_multipliers,
        cnn_kernel=config.encoder.cnn_kernel,
        cnn_outer_stride=config.encoder.cnn_outer_stride,
    )
    decoder = DreamerDecoder(
        config.observation_shape,
        hidden_dims=config.encoder.hidden_dims,
        cnn_depth=config.encoder.cnn_depth,
        cnn_multipliers=config.encoder.cnn_multipliers,
        cnn_kernel=config.encoder.cnn_kernel,
        cnn_outer_stride=config.encoder.cnn_outer_stride,
        deterministic_size=config.rssm.deterministic_size,
        stochastic_size=config.rssm.stochastic_size,
        discrete_classes=config.rssm.discrete_classes,
        blocks=config.rssm.blocks,
        hidden_size=config.rssm.hidden_size,
    )

    encoder_vars = encoder.init(jax.random.PRNGKey(0), observations)
    decoder_vars = decoder.init(jax.random.PRNGKey(1), features)
    encoded = encoder.apply(encoder_vars, observations)
    decoded = decoder.apply(decoder_vars, features)

    assert encoded.shape == (2, 4 * 4 * 16)
    assert decoded.shape == observations.shape
    assert all("cnn_" in path for path in _parameter_paths(encoder_vars))
    assert any("cnn_transpose_" in path for path in _parameter_paths(decoder_vars))
    assert any("image_output" in path for path in _parameter_paths(decoder_vars))
    assert any(
        "deterministic_space/kernel" in path for path in _parameter_paths(decoder_vars)
    )
    assert any(
        "stochastic_space/kernel" in path for path in _parameter_paths(decoder_vars)
    )
    assert decoder_vars["params"]["deterministic_space"]["kernel"].ndim == 3

    encoder_shapes = jax.eval_shape(
        lambda variables, inputs: encoder.apply(
            variables,
            inputs,
            capture_intermediates=lambda module, _: isinstance(module, nn.Conv),
            mutable=["intermediates"],
        ),
        encoder_vars,
        observations,
    )[1]["intermediates"]
    encoder_spatial_shapes = [
        values["__call__"][0].shape[1:3]
        for name, values in sorted(encoder_shapes.items())
        if name.startswith("cnn_")
    ]
    assert encoder_spatial_shapes == [(32, 32), (16, 16), (8, 8), (4, 4)]

    decoder_shapes = jax.eval_shape(
        lambda variables, inputs: decoder.apply(
            variables,
            inputs,
            capture_intermediates=lambda module, _: isinstance(
                module, nn.ConvTranspose
            ),
            mutable=["intermediates"],
        ),
        decoder_vars,
        features,
    )[1]["intermediates"]
    decoder_spatial_shapes = [
        values["__call__"][0].shape[1:3]
        for name, values in sorted(decoder_shapes.items())
        if name.startswith("cnn_transpose_")
    ]
    decoder_spatial_shapes.append(
        decoder_shapes["image_output"]["__call__"][0].shape[1:3]
    )
    assert decoder_spatial_shapes == [(8, 8), (16, 16), (32, 32), (64, 64)]


def test_imagination_does_not_run_observation_decoder() -> None:
    config = DreamerV3Config.debug(
        action_dim=2,
        observation_shape=(64, 64, 3),
    )
    state = create_dreamer_train_state(jax.random.PRNGKey(60), config)
    model = DreamerWorldModel(config)
    previous = initial_rssm_state(batch_size=2, config=config.rssm)
    actions = jax.nn.one_hot(jnp.asarray([0, 1]), config.action_dim)

    _, prediction = model.apply(
        state.params,
        previous,
        actions,
        jax.random.PRNGKey(61),
        method=model.imagine_step,
    )

    assert set(prediction) == {"features", "reward_logits", "continue_logits"}


def test_rssm_uses_block_gru_and_keyed_unimix_sampling() -> None:
    config = DreamerV3Config.debug(action_dim=3, observation_shape=(5,))
    rssm = DreamerRSSM(config.rssm, action_dim=config.action_dim)
    previous = initial_rssm_state(batch_size=4, config=config.rssm)
    assert bool(jnp.all(previous.stochastic == 0.0))
    actions = jax.nn.one_hot(jnp.arange(4) % 3, 3)
    embed = jnp.zeros((4, config.encoder.hidden_dims[-1]), dtype=jnp.float32)
    variables = rssm.init(
        jax.random.PRNGKey(2),
        previous,
        actions,
        embed,
        jax.random.PRNGKey(3),
    )

    _, posterior_a = rssm.apply(
        variables,
        previous,
        actions,
        embed,
        jax.random.PRNGKey(4),
    )
    _, posterior_b = rssm.apply(
        variables,
        previous,
        actions,
        embed,
        jax.random.PRNGKey(5),
    )
    paths = _parameter_paths(variables)

    assert any("block_gru_hidden" in path for path in paths)
    assert any("block_gru_gates" in path for path in paths)
    assert not any("GRUCell" in path for path in paths)
    assert not np.array_equal(posterior_a.stochastic, posterior_b.stochastic)
    assert bool(jnp.allclose(jnp.exp(posterior_a.logits).sum(-1), 1.0))
    assert (
        float(jnp.exp(posterior_a.logits).min())
        >= (config.rssm.unimix / config.rssm.discrete_classes) - 1e-7
    )


def test_categorical_straight_through_requires_key_and_is_stochastic() -> None:
    logits = jnp.zeros((64, 4, 8), dtype=jnp.float32)
    sample_a, probs_a = categorical_straight_through(
        logits,
        jax.random.PRNGKey(6),
        unimix=0.01,
    )
    sample_b, probs_b = categorical_straight_through(
        logits,
        jax.random.PRNGKey(7),
        unimix=0.01,
    )

    assert not np.array_equal(sample_a, sample_b)
    assert bool(jnp.allclose(probs_a, probs_b))
    assert bool(jnp.allclose(sample_a.sum(-1), 1.0))
    assert bool(jnp.allclose(probs_a.sum(-1), 1.0))


def test_rssm_masks_previous_action_on_episode_reset() -> None:
    config = DreamerV3Config.debug(action_dim=3, observation_shape=(5,))
    state = create_dreamer_train_state(jax.random.PRNGKey(50), config)
    observations = jnp.zeros((3, 2, 5), dtype=jnp.float32)
    resets = jnp.ones((3, 2), dtype=bool)
    zero_actions = jnp.zeros((3, 2), dtype=jnp.int32)
    nonzero_actions = jnp.full((3, 2), 2, dtype=jnp.int32)

    zero_outputs = observe_dreamer_sequence(
        state.params,
        observations,
        zero_actions,
        resets,
        config,
        jax.random.PRNGKey(51),
    )
    nonzero_outputs = observe_dreamer_sequence(
        state.params,
        observations,
        nonzero_actions,
        resets,
        config,
        jax.random.PRNGKey(51),
    )

    assert bool(
        jnp.allclose(
            zero_outputs["deterministic"],
            nonzero_outputs["deterministic"],
        )
    )
    assert bool(
        jnp.allclose(
            zero_outputs["posterior_logits"],
            nonzero_outputs["posterior_logits"],
        )
    )


def test_reward_and_critic_outputs_are_zero_initialized() -> None:
    features = jnp.ones((3, 16), dtype=jnp.float32)
    reward = RewardHead(255)
    critic = DreamerCritic(255)
    reward_vars = reward.init(jax.random.PRNGKey(8), features)
    critic_vars = critic.init(jax.random.PRNGKey(9), features)

    assert bool(jnp.all(reward.apply(reward_vars, features) == 0.0))
    assert bool(jnp.all(critic.apply(critic_vars, features) == 0.0))


def test_paper_optimizer_is_agc_then_laprop_without_adam() -> None:
    config = DreamerV3Config(action_dim=2, observation_shape=(3,))
    optimizer = dreamer_laprop(config.optimizer)
    params = {"weight": jnp.ones((2, 2)), "bias": jnp.ones((2,))}
    state = optimizer.init(params)
    gradients = jax.tree.map(lambda value: 10.0 * value, params)
    updates, _ = optimizer.update(gradients, state, params)

    assert type(state).__name__ == "tuple"
    assert len(state) == 4
    assert bool(jnp.allclose(updates["weight"], -4e-5, atol=1e-9))
    assert bool(jnp.allclose(updates["bias"], -4e-5, atol=1e-9))


def test_discrete_actor_sampling_applies_one_percent_unimix() -> None:
    config = DreamerV3Config.debug(action_dim=3, observation_shape=(4,))
    outputs = {"logits": jnp.asarray([[100.0, -100.0, -100.0]], dtype=jnp.float32)}

    sample = sample_actor_output(
        outputs,
        config,
        jax.random.PRNGKey(10),
        deterministic=False,
    )

    assert sample.env_action.shape == (1,)
    assert sample.model_action.shape == (1, 3)
    assert bool(jnp.allclose(sample.model_action.sum(-1), 1.0))
    assert float(sample.probabilities.min()) >= 0.01 / 3.0 - 1e-7
    expected_log_prob = jnp.take_along_axis(
        jnp.log(sample.probabilities),
        sample.env_action[:, None],
        axis=-1,
    )[:, 0]
    assert bool(jnp.allclose(sample.log_prob, expected_log_prob))


def test_actor_output_weights_use_paper_point_zero_one_outscale() -> None:
    config = DreamerV3Config.debug(action_dim=3, observation_shape=(4,))
    actor = DreamerActor(
        config.action_dim,
        config.action_mode,
        hidden_dims=config.actor_critic.hidden_dims,
    )
    variables = actor.init(
        jax.random.PRNGKey(52),
        jnp.zeros((1, config.rssm.latent_size), dtype=jnp.float32),
    )

    output_kernel = variables["params"]["logits"]["kernel"]
    assert float(jnp.std(output_kernel)) < 0.005


def test_continuous_actor_samples_plain_bounded_mean_normal() -> None:
    config = DreamerV3Config.debug(
        action_dim=2,
        observation_shape=(4,),
        action_mode="continuous",
    )
    outputs = {
        "mean": jnp.zeros((256, 2), dtype=jnp.float32),
        "stddev": jnp.ones((256, 2), dtype=jnp.float32),
    }

    sample = sample_actor_output(
        outputs,
        config,
        jax.random.PRNGKey(11),
        deterministic=False,
    )

    assert sample.env_action.shape == (256, 2)
    assert bool(jnp.any(jnp.abs(sample.env_action) > 1.0))
    assert bool(jnp.all(jnp.isfinite(sample.log_prob)))
    assert bool(jnp.all(jnp.isfinite(sample.entropy)))


def test_imagined_continuation_multiplies_probability_by_gamma() -> None:
    config = DreamerV3Config(action_dim=2, observation_shape=(4,))
    logits = jnp.asarray([-2.0, 0.0, 2.0], dtype=jnp.float32)

    discounts = discount_continuation(logits, config)

    assert bool(
        jnp.allclose(
            discounts,
            config.actor_critic.discount * jax.nn.sigmoid(logits),
        )
    )


def test_reconstruction_loss_sums_event_dimensions_and_symlogs_vectors() -> None:
    vector_config = DreamerV3Config.debug(action_dim=2, observation_shape=(5,))
    vector_observations = jnp.full((2, 3, 5), 3.0, dtype=jnp.float32)
    exact_vector_prediction = jnp.full_like(
        vector_observations,
        jnp.log1p(3.0),
    )
    zero_vector_prediction = jnp.zeros_like(vector_observations)

    assert (
        reconstruction_loss(
            exact_vector_prediction,
            vector_observations,
            vector_config,
        )
        == 0.0
    )
    assert bool(
        jnp.allclose(
            reconstruction_loss(
                zero_vector_prediction,
                jnp.ones_like(vector_observations),
                vector_config,
            ),
            5.0 * jnp.square(jnp.log(2.0)),
        )
    )

    image_config = DreamerV3Config.debug(
        action_dim=2,
        observation_shape=(16, 16, 3),
    )
    image_observations = jnp.ones((2, 3, 16, 16, 3), dtype=jnp.float32)
    assert (
        reconstruction_loss(
            jnp.zeros_like(image_observations),
            image_observations,
            image_config,
        )
        == 16 * 16 * 3
    )


def test_prediction_losses_include_reset_records_as_in_paper() -> None:
    config = DreamerV3Config.debug(action_dim=2, observation_shape=(4,))
    replay = sequence_batch_to_jax(
        synthetic_sequence_collector(
            env_name="synthetic:vector",
            time_steps=4,
            batch_size=2,
            observation_shape=(4,),
            action_dim=2,
        )
    )
    state = create_dreamer_train_state(jax.random.PRNGKey(4), config)
    _, metrics = dreamer_world_model_loss_arrays(
        state.params,
        replay,
        config,
        jax.random.PRNGKey(5),
    )
    previous_actions = jnp.concatenate(
        [jnp.zeros_like(replay.actions[:1]), replay.actions[:-1]],
        axis=0,
    )
    outputs = observe_dreamer_sequence(
        state.params,
        replay.observations,
        previous_actions,
        replay.is_first,
        config,
        jax.random.PRNGKey(5),
    )
    expected = jnp.mean(
        optax.sigmoid_binary_cross_entropy(outputs["continue_logits"], replay.continues)
    )

    assert bool(jnp.allclose(metrics["continue_loss"], expected))


def test_joint_loss_reuses_its_posterior_rollout() -> None:
    config = DreamerV3Config.debug(action_dim=2, observation_shape=(4,))
    batch = sequence_batch_to_jax(
        synthetic_sequence_collector(
            env_name="synthetic:vector",
            time_steps=4,
            batch_size=2,
            observation_shape=(4,),
            action_dim=2,
        )
    )
    state = create_dreamer_agent_state(jax.random.PRNGKey(62), config)

    _, (metrics, _, _, _, posterior) = dreamer_agent_loss_arrays(
        state.params,
        batch,
        state.slow_critic_params,
        state.return_low,
        state.return_high,
        config,
        jax.random.PRNGKey(63),
        imagination_horizon=2,
    )
    expected_reconstruction = reconstruction_loss(
        posterior["reconstructions"],
        batch.observations,
        config,
    )

    assert bool(
        jnp.allclose(
            metrics["reconstruction_loss"],
            expected_reconstruction,
        )
    )


def test_uniform_two_hot_logits_decode_to_exact_zero() -> None:
    logits = jnp.zeros((32, 255), dtype=jnp.float32)

    values = decode_two_hot_logits(logits)

    assert bool(jnp.all(values == 0.0))


def test_symexp_two_hot_uses_original_space_bin_distances() -> None:
    support = symexp_two_hot_support(num_bins=255, lower=-20.0, upper=20.0)
    lower_index = 180
    upper_index = lower_index + 1
    target = 0.25 * support[lower_index] + 0.75 * support[upper_index]

    encoded = symexp_two_hot(
        target,
        num_bins=255,
        lower=-20.0,
        upper=20.0,
    )

    assert bool(jnp.allclose(encoded.sum(), 1.0))
    assert bool(jnp.allclose(encoded[lower_index], 0.25))
    assert bool(jnp.allclose(encoded[upper_index], 0.75))
    assert int(jnp.count_nonzero(encoded)) == 2


def test_symexp_two_hot_prediction_is_original_space_expectation() -> None:
    support = symexp_two_hot_support(num_bins=255, lower=-20.0, upper=20.0)
    lower_index = 180
    upper_index = lower_index + 1
    logits = jnp.full((255,), -100.0, dtype=jnp.float32)
    logits = logits.at[lower_index].set(0.0)
    logits = logits.at[upper_index].set(0.0)

    prediction = decode_two_hot_logits(logits)

    expected = 0.5 * (support[lower_index] + support[upper_index])
    assert bool(jnp.allclose(prediction, expected, rtol=1e-5))


def test_actor_loss_is_reinforce_with_stopped_return_advantage() -> None:
    log_probs = jnp.asarray([[-0.2, -0.4], [-0.1, -0.3]], dtype=jnp.float32)
    entropies = jnp.zeros_like(log_probs)
    returns = jnp.asarray([[3.0, 5.0], [7.0, 9.0]], dtype=jnp.float32)
    values = jnp.ones_like(returns)
    weights = jnp.asarray([[1.0, 0.5], [0.25, 0.125]], dtype=jnp.float32)

    def loss_for(logp, ret):
        return reinforce_actor_loss(
            logp,
            entropies,
            ret,
            values,
            weights,
            return_scale=jnp.asarray(2.0),
            entropy_scale=3e-4,
        )

    log_prob_gradient, return_gradient = jax.grad(loss_for, argnums=(0, 1))(
        log_probs,
        returns,
    )
    expected = -weights * ((returns - values) / 2.0) / returns.size

    assert bool(jnp.allclose(log_prob_gradient, expected))
    assert bool(jnp.all(return_gradient == 0.0))


def test_return_normalization_and_slow_critic_use_paper_ema_rates() -> None:
    config = DreamerV3Config(action_dim=2, observation_shape=(4,))
    returns = jnp.arange(100, dtype=jnp.float32)

    low, high, scale = update_return_normalization(
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        returns,
        config,
    )
    expected_low = 0.01 * jnp.percentile(returns, 5.0)
    expected_high = 0.01 * jnp.percentile(returns, 95.0)

    assert bool(jnp.allclose(low, expected_low))
    assert bool(jnp.allclose(high, expected_high))
    expected_scale = jnp.asarray(1.0)
    assert bool(jnp.allclose(scale, expected_scale))

    online = {"kernel": jnp.full((2, 2), 3.0)}
    slow = {"kernel": jnp.ones((2, 2))}
    updated = ema_critic_parameters(online, slow, config)
    assert bool(jnp.allclose(updated["kernel"], 1.04))


def test_replay_critic_uses_only_transitions_with_a_next_replay_state() -> None:
    config = DreamerV3Config.debug(
        action_dim=2,
        observation_shape=(4,),
    )
    rewards = jnp.asarray([[0.0], [1.0], [2.0]])
    continues = jnp.ones_like(rewards)
    imagination_annotations = jnp.zeros_like(rewards)

    returns = dreamer_replay_critic_returns(
        rewards,
        continues,
        jnp.zeros_like(rewards, dtype=bool),
        imagination_annotations,
        config,
    )

    expected_first = 1.0 + (
        config.actor_critic.discount * config.actor_critic.discount_lambda * 2.0
    )
    assert returns.shape == (2, 1)
    assert jnp.allclose(returns[:, 0], jnp.asarray([expected_first, 2.0]))


def test_replay_critic_bootstraps_at_truncation_without_cross_episode_lambda() -> None:
    config = DreamerV3Config.debug(action_dim=2, observation_shape=(4,))
    rewards = jnp.asarray([[0.0], [1.0], [2.0]])
    continues = jnp.ones_like(rewards)
    is_last = jnp.asarray([[False], [True], [False]])
    bootstrap = jnp.asarray([[10.0], [20.0], [30.0]])

    returns = dreamer_replay_critic_returns(
        rewards,
        continues,
        is_last,
        bootstrap,
        config,
    )

    assert jnp.allclose(
        returns[:, 0],
        jnp.asarray(
            [
                1.0 + config.actor_critic.discount * 20.0,
                2.0 + config.actor_critic.discount * 30.0,
            ]
        ),
    )


def test_online_queue_precedes_uniform_replay_and_latents_are_written_back() -> None:
    config = DreamerV3Config.debug(action_dim=2, observation_shape=(4,))
    sequence = sequence_batch_to_jax(
        synthetic_sequence_collector(
            env_name="synthetic:vector",
            time_steps=9,
            batch_size=1,
            observation_shape=(4,),
            action_dim=2,
        )
    )
    state = create_dreamer_agent_state(jax.random.PRNGKey(20), config)
    replay = initialize_dreamer_replay(
        sequence,
        state.params["world_model"],
        config,
        jax.random.PRNGKey(21),
        sequence_length=4,
    )

    replay, sample = sample_dreamer_replay(
        replay,
        jax.random.PRNGKey(22),
        sequence_length=4,
        batch_size=3,
    )

    assert int(sample.online_items) == 1
    assert sample.context_starts[:1].tolist() == [0]
    assert jnp.allclose(
        sample.initial_state.deterministic,
        replay.deterministic[
            sample.context_starts,
            sample.sequence_indices,
        ],
    )
    assert jnp.array_equal(
        sample.previous_actions[0],
        sequence.actions[
            sample.context_starts,
            sample.sequence_indices,
        ],
    )

    fresh = {
        "deterministic": jnp.full(
            sample.batch.observations.shape[:2] + (config.rssm.deterministic_size,),
            7.0,
        ),
        "stochastic": jnp.full(
            sample.batch.observations.shape[:2]
            + (
                config.rssm.stochastic_size,
                config.rssm.discrete_classes,
            ),
            7.0,
        ),
        "posterior_logits": jnp.full(
            sample.batch.observations.shape[:2]
            + (
                config.rssm.stochastic_size,
                config.rssm.discrete_classes,
            ),
            7.0,
        ),
    }
    replay = update_dreamer_replay_latents(replay, sample, fresh)
    assert jnp.all(
        replay.deterministic[sample.time_indices, sample.sequence_indices] == 7.0
    )

    _, second = sample_dreamer_replay(
        replay,
        jax.random.PRNGKey(23),
        sequence_length=4,
        batch_size=3,
    )
    assert int(second.online_items) == 0


def test_replay_windows_cross_resets_and_online_stride_includes_context() -> None:
    config = DreamerV3Config.debug(action_dim=2, observation_shape=(4,))
    sequence = sequence_batch_to_jax(
        synthetic_sequence_collector(
            env_name="synthetic:vector",
            time_steps=11,
            batch_size=1,
            observation_shape=(4,),
            action_dim=2,
        )
    )
    sequence = sequence._replace(
        is_first=sequence.is_first.at[3, 0].set(True).at[8, 0].set(True)
    )
    state = create_dreamer_agent_state(jax.random.PRNGKey(24), config)

    replay = initialize_dreamer_replay(
        sequence,
        state.params["world_model"],
        config,
        jax.random.PRNGKey(25),
        sequence_length=4,
    )

    assert bool(jnp.all(jnp.isfinite(replay.valid_start_logits)))
    assert int(replay.online_count) == 2
    assert replay.online_starts[:2].tolist() == [0, 5]


def test_replay_append_wraps_and_samples_contiguous_logical_windows() -> None:
    config = DreamerV3Config.debug(action_dim=2, observation_shape=(4,))
    initial = sequence_batch_to_jax(
        synthetic_sequence_collector(
            env_name="synthetic:initial",
            time_steps=6,
            batch_size=1,
            observation_shape=(4,),
            action_dim=2,
        )
    )
    state = create_dreamer_agent_state(jax.random.PRNGKey(26), config)
    replay = initialize_dreamer_replay(
        initial,
        state.params["world_model"],
        config,
        jax.random.PRNGKey(27),
        sequence_length=2,
        capacity_time=8,
    )
    appended = sequence_batch_to_jax(
        synthetic_sequence_collector(
            env_name="synthetic:appended",
            time_steps=4,
            batch_size=1,
            observation_shape=(4,),
            action_dim=2,
        )
    )
    appended = appended._replace(
        observations=100.0 + appended.observations,
    )
    posterior = {
        "deterministic": jnp.full((4, 1, config.rssm.deterministic_size), 7.0),
        "stochastic": jnp.full(
            (
                4,
                1,
                config.rssm.stochastic_size,
                config.rssm.discrete_classes,
            ),
            7.0,
        ),
        "posterior_logits": jnp.full(
            (
                4,
                1,
                config.rssm.stochastic_size,
                config.rssm.discrete_classes,
            ),
            7.0,
        ),
    }

    replay = append_dreamer_replay(
        replay,
        appended,
        posterior,
        sequence_length=2,
    )
    replay = replay._replace(online_cursor=replay.online_count)
    _, sample = sample_dreamer_replay(
        replay,
        jax.random.PRNGKey(28),
        sequence_length=2,
        batch_size=3,
    )

    assert int(replay.size) == 8
    assert int(replay.total_inserted) == 10
    assert int(replay.write_index) == 2
    assert jnp.allclose(
        replay.sequence.observations[jnp.asarray([6, 7, 0, 1]), 0],
        appended.observations[:, 0],
    )
    physical_windows = jnp.concatenate(
        [sample.context_starts[None], sample.time_indices], axis=0
    )
    assert jnp.all((physical_windows[1:] - physical_windows[:-1]) % 8 == 1)


def test_online_scheduler_collects_with_actor_and_starts_ratio_when_ready() -> None:
    config = DreamerV3Config.debug(
        action_dim=2,
        observation_shape=(4,),
        action_mode="continuous",
    )
    carry = create_dreamer_online_carry(
        config=config,
        num_envs=2,
        capacity_time=12,
        sequence_length=3,
        seed=29,
    )
    learner_step = build_dreamer_online_learner_step(
        config=config,
        num_envs=2,
        sequence_length=3,
        batch_size=2,
        train_ratio=32.0,
        max_train_steps=2,
        imagination_horizon=3,
    )
    observations = jnp.arange(5 * 2 * 4, dtype=jnp.float32).reshape((5, 2, 4))
    rewards = jnp.zeros((5, 2), dtype=jnp.float32)
    terminals = jnp.zeros((5, 2), dtype=bool)
    lasts = jnp.zeros((5, 2), dtype=bool)
    firsts = jnp.zeros((5, 2), dtype=bool).at[0].set(True)

    def step(train_carry, arrivals):
        obs, reward, terminal, last, first = arrivals
        train_carry, actions, metrics = learner_step(
            train_carry,
            obs,
            reward,
            terminal,
            last,
            first,
        )
        return train_carry, (actions, metrics)

    carry, (actions, metrics) = jax.jit(
        lambda initial: jax.lax.scan(
            step,
            initial,
            (observations, rewards, terminals, lasts, firsts),
        )
    )(carry)

    assert int(carry.replay.total_inserted) == 5
    assert int(carry.completed_updates) == 2
    assert int(carry.agent_state.step) == 2
    assert actions.shape == (5, 2, 2)
    assert np.asarray(metrics["update_executed"]).sum() == 2
    assert np.asarray(metrics["update_executed"][:3]).sum() == 0
    assert np.asarray(metrics["update_executed"][3]).sum() == 1


def test_joint_agent_update_contains_all_paper_losses_and_one_optimizer() -> None:
    config = DreamerV3Config.debug(action_dim=3, observation_shape=(5,))
    batch = synthetic_sequence_collector(
        env_name="synthetic:dreamer-joint",
        time_steps=4,
        batch_size=2,
        observation_shape=config.observation_shape,
        action_dim=config.action_dim,
    )
    state = create_dreamer_agent_state(jax.random.PRNGKey(12), config)

    updated, metrics, rollout = dreamer_agent_train_step(
        state,
        batch,
        config,
        jax.random.PRNGKey(13),
        imagination_horizon=3,
    )

    assert set(state.params) == {"world_model", "actor", "critic"}
    assert state.step == 0
    assert updated.step == 1
    assert set(metrics) >= {
        "loss",
        "world_model_loss",
        "actor_loss",
        "critic_loss",
        "replay_critic_loss",
        "reconstruction_loss",
        "reward_loss",
        "continue_loss",
        "dynamics_kl_loss",
        "representation_kl_loss",
    }
    expected_total = (
        metrics["world_model_loss"]
        + metrics["actor_loss"]
        + metrics["critic_loss"]
        + config.actor_critic.critic_replay_scale * metrics["replay_critic_loss"]
    )
    assert bool(jnp.allclose(metrics["loss"], expected_total))
    assert rollout.features.shape == (3, 8, config.rssm.latent_size)
