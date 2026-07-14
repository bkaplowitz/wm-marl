from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from world_marl.dreamer_v3_baseline.config import DreamerV3Config
from world_marl.dreamer_v3_baseline.losses import (
    balanced_categorical_kl_loss,
    reconstruction_loss,
    symexp_two_hot,
)
from world_marl.dreamer_v3_baseline.models import (
    ContinueHead,
    DreamerActor,
    DreamerCritic,
    DreamerDecoder,
    DreamerEncoder,
    RewardHead,
)
from world_marl.dreamer_v3_baseline.optimizer import dreamer_laprop
from world_marl.dreamer_v3_baseline.rssm import (
    DreamerRSSM,
    RSSMState,
    flatten_rssm_state,
    initial_rssm_state,
    reset_rssm_state,
)
from world_marl.world_model_foundation.replay import WorldModelSequenceBatch
from world_marl.world_model_foundation.replay import (
    JaxSequenceBatch,
    sample_sequence_windows,
    sequence_batch_to_jax,
)


class DreamerWorldModel(nn.Module):
    config: DreamerV3Config

    def setup(self) -> None:
        self.encoder = DreamerEncoder(
            self.config.observation_shape,
            hidden_dims=self.config.encoder.hidden_dims,
            cnn_depth=self.config.encoder.cnn_depth,
            cnn_multipliers=self.config.encoder.cnn_multipliers,
            cnn_kernel=self.config.encoder.cnn_kernel,
            cnn_outer_stride=self.config.encoder.cnn_outer_stride,
            name="encoder",
        )
        self.rssm = DreamerRSSM(
            self.config.rssm,
            action_dim=self.config.action_dim,
            name="rssm",
        )
        self.decoder = DreamerDecoder(
            self.config.observation_shape,
            hidden_dims=self.config.encoder.hidden_dims,
            cnn_depth=self.config.encoder.cnn_depth,
            cnn_multipliers=self.config.encoder.cnn_multipliers,
            cnn_kernel=self.config.encoder.cnn_kernel,
            cnn_outer_stride=self.config.encoder.cnn_outer_stride,
            deterministic_size=self.config.rssm.deterministic_size,
            stochastic_size=self.config.rssm.stochastic_size,
            discrete_classes=self.config.rssm.discrete_classes,
            blocks=self.config.rssm.blocks,
            hidden_size=self.config.rssm.hidden_size,
            name="decoder",
        )
        self.reward_head = RewardHead(
            self.config.reward_head.bins,
            hidden_dims=self.config.reward_head.hidden_dims,
            name="reward_head",
        )
        self.continue_head = ContinueHead(
            hidden_dims=self.config.continue_head.hidden_dims,
            name="continue_head",
        )

    def observe_step(
        self,
        prev_state: RSSMState,
        action_features: jax.Array,
        observations: jax.Array,
        key: jax.Array,
    ) -> tuple[RSSMState, RSSMState, dict[str, jax.Array]]:
        embed = self.encoder(observations)
        prior, posterior = self.rssm(prev_state, action_features, embed, key)
        feature = flatten_rssm_state(posterior)
        return prior, posterior, self.predict(feature)

    def imagine_step(
        self,
        prev_state: RSSMState,
        action_features: jax.Array,
        key: jax.Array,
    ) -> tuple[RSSMState, dict[str, jax.Array]]:
        prior = self.rssm.prior(prev_state, action_features, key)
        features = flatten_rssm_state(prior)
        return prior, {
            "features": features,
            "reward_logits": self.reward_head(features),
            "continue_logits": self.continue_head(features),
        }

    def predict(self, features: jax.Array) -> dict[str, jax.Array]:
        return {
            "features": features,
            "reconstructions": self.decoder(features),
            "reward_logits": self.reward_head(features),
            "continue_logits": self.continue_head(features),
        }

    def encode(self, observations: jax.Array) -> jax.Array:
        return self.encoder(observations)

    def rssm_observe(
        self,
        previous_state: RSSMState,
        action_features: jax.Array,
        embedding: jax.Array,
        key: jax.Array,
    ) -> tuple[RSSMState, RSSMState]:
        return self.rssm(previous_state, action_features, embedding, key)


class DreamerAgentState(TrainState):
    slow_critic_params: Any
    return_low: jax.Array
    return_high: jax.Array


class DreamerOnlineCarry(NamedTuple):
    agent_state: DreamerAgentState
    replay: Any
    policy_state: RSSMState
    previous_action: jax.Array
    key: jax.Array
    ratio_credit: jax.Array
    ratio_started: jax.Array
    completed_updates: jax.Array


def dreamer_action_features(actions: jax.Array, config: DreamerV3Config) -> jax.Array:
    if config.action_mode == "discrete":
        return jax.nn.one_hot(actions.astype(jnp.int32), config.action_dim)
    return actions.astype(jnp.float32).reshape((actions.shape[0], config.action_dim))


def observe_dreamer_sequence(
    params: Any,
    observations: jax.Array,
    actions: jax.Array,
    is_first: jax.Array,
    config: DreamerV3Config,
    key: jax.Array,
    *,
    initial_state: RSSMState | None = None,
) -> dict[str, jax.Array]:
    model = DreamerWorldModel(config)
    time_steps, batch_size = observations.shape[:2]
    flat_observations = observations.reshape(
        (time_steps * batch_size, *config.observation_shape)
    )
    embeddings = model.apply(
        params,
        flat_observations,
        method=model.encode,
    )
    embeddings = embeddings.reshape((time_steps, batch_size, -1))
    action_features = jax.vmap(lambda action: dreamer_action_features(action, config))(
        actions
    )

    def step(
        carry: tuple[RSSMState, jax.Array],
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[RSSMState, jax.Array], dict[str, jax.Array]]:
        previous_state, sequence_key = carry
        sequence_key, sample_key = jax.random.split(sequence_key)
        embedding, action, first = inputs
        previous_state = reset_rssm_state(
            previous_state,
            first,
            config=config.rssm,
        )
        action = jnp.where(first[:, None], jnp.zeros_like(action), action)
        prior, posterior = model.apply(
            params,
            previous_state,
            action,
            embedding,
            sample_key,
            method=model.rssm_observe,
        )
        outputs = {
            "prior_logits": prior.logits,
            "posterior_logits": posterior.logits,
            "deterministic": posterior.deterministic,
            "stochastic": posterior.stochastic,
        }
        return (posterior, sequence_key), outputs

    if initial_state is None:
        initial_state = initial_rssm_state(batch_size=batch_size, config=config.rssm)
    _, outputs = jax.lax.scan(
        step,
        (initial_state, key),
        (embeddings, action_features, is_first),
    )
    flat_state = RSSMState(
        deterministic=outputs["deterministic"].reshape(
            (time_steps * batch_size, config.rssm.deterministic_size)
        ),
        stochastic=outputs["stochastic"].reshape(
            (
                time_steps * batch_size,
                config.rssm.stochastic_size,
                config.rssm.discrete_classes,
            )
        ),
        logits=outputs["posterior_logits"].reshape(
            (
                time_steps * batch_size,
                config.rssm.stochastic_size,
                config.rssm.discrete_classes,
            )
        ),
    )
    flat_predictions = model.apply(
        params,
        flatten_rssm_state(flat_state),
        method=model.predict,
    )
    outputs.update(
        {
            "features": flat_predictions["features"].reshape(
                (time_steps, batch_size, config.rssm.latent_size)
            ),
            "reconstructions": flat_predictions["reconstructions"].reshape(
                (time_steps, batch_size, *config.observation_shape)
            ),
            "reward_logits": flat_predictions["reward_logits"].reshape(
                (time_steps, batch_size, config.reward_head.bins)
            ),
            "continue_logits": flat_predictions["continue_logits"].reshape(
                (time_steps, batch_size)
            ),
        }
    )
    return outputs


def create_dreamer_train_state(
    key: jax.Array,
    config: DreamerV3Config,
    *,
    learning_rate: float | None = None,
) -> TrainState:
    model = DreamerWorldModel(config)
    dummy_obs = jnp.zeros((1, 1, *config.observation_shape), dtype=jnp.float32)
    if config.action_mode == "discrete":
        dummy_actions = jnp.zeros((1, 1), dtype=jnp.int32)
    else:
        dummy_actions = jnp.zeros((1, 1, config.action_dim), dtype=jnp.float32)
    params_key, sample_key = jax.random.split(key)
    initial_state = initial_rssm_state(batch_size=1, config=config.rssm)
    dummy_action_features = dreamer_action_features(dummy_actions[0], config)
    params = model.init(
        params_key,
        initial_state,
        dummy_action_features,
        dummy_obs[0],
        sample_key,
        method=model.observe_step,
    )
    optimizer_config = config.optimizer
    if learning_rate is not None:
        optimizer_config = replace(optimizer_config, learning_rate=learning_rate)
    return TrainState.create(
        apply_fn=lambda variables, observations, actions, is_first, sample_key: (
            observe_dreamer_sequence(
                variables,
                observations,
                actions,
                is_first,
                config,
                sample_key,
            )
        ),
        params=params,
        tx=dreamer_laprop(optimizer_config),
    )


def create_dreamer_agent_state(
    key: jax.Array,
    config: DreamerV3Config,
    *,
    learning_rate: float | None = None,
) -> DreamerAgentState:
    from world_marl.dreamer_v3_baseline.imagination import (
        create_dreamer_actor_critic_states,
    )

    world_key, policy_key = jax.random.split(key)
    world_state = create_dreamer_train_state(world_key, config)
    actor_state, critic_state = create_dreamer_actor_critic_states(
        policy_key,
        config,
    )
    optimizer_config = config.optimizer
    if learning_rate is not None:
        optimizer_config = replace(optimizer_config, learning_rate=learning_rate)
    params = {
        "world_model": world_state.params,
        "actor": actor_state.params,
        "critic": critic_state.params,
    }
    return DreamerAgentState.create(
        apply_fn=DreamerWorldModel(config).apply,
        params=params,
        tx=dreamer_laprop(optimizer_config),
        slow_critic_params=critic_state.params,
        return_low=jnp.asarray(0.0, dtype=jnp.float32),
        return_high=jnp.asarray(0.0, dtype=jnp.float32),
    )


def create_dreamer_online_carry(
    *,
    config: DreamerV3Config,
    num_envs: int,
    capacity_time: int,
    sequence_length: int,
    seed: int,
    learning_rate: float | None = None,
) -> DreamerOnlineCarry:
    from world_marl.dreamer_v3_baseline.replay import (
        initialize_empty_dreamer_replay,
    )

    agent_state = create_dreamer_agent_state(
        jax.random.PRNGKey(seed),
        config,
        learning_rate=learning_rate,
    )
    replay = initialize_empty_dreamer_replay(
        config,
        num_sequences=num_envs,
        capacity_time=capacity_time,
        sequence_length=sequence_length,
    )
    if config.action_mode == "discrete":
        previous_action = jnp.zeros((num_envs,), dtype=jnp.int32)
    else:
        previous_action = jnp.zeros(
            (num_envs, config.action_dim),
            dtype=jnp.float32,
        )
    return DreamerOnlineCarry(
        agent_state=agent_state,
        replay=replay,
        policy_state=initial_rssm_state(batch_size=num_envs, config=config.rssm),
        previous_action=previous_action,
        key=jax.random.PRNGKey(seed + 1),
        ratio_credit=jnp.zeros((), dtype=jnp.float32),
        ratio_started=jnp.zeros((), dtype=bool),
        completed_updates=jnp.zeros((), dtype=jnp.int32),
    )


def dreamer_agent_views(
    state: DreamerAgentState,
    config: DreamerV3Config,
) -> tuple[TrainState, Any, Any]:
    from world_marl.dreamer_v3_baseline.imagination import (
        DreamerActorState,
        DreamerCriticState,
    )

    identity = optax.identity()
    world_params = state.params["world_model"]
    actor_params = state.params["actor"]
    critic_params = state.params["critic"]
    world_state = TrainState(
        step=state.step,
        apply_fn=lambda variables, observations, actions, is_first, sample_key: (
            observe_dreamer_sequence(
                variables,
                observations,
                actions,
                is_first,
                config,
                sample_key,
            )
        ),
        params=world_params,
        tx=identity,
        opt_state=identity.init(world_params),
    )
    actor = DreamerActor(
        config.action_dim,
        config.action_mode,
        hidden_dims=config.actor_critic.hidden_dims,
        min_std=config.actor_critic.min_std,
        max_std=config.actor_critic.max_std,
    )
    actor_state = DreamerActorState(
        step=state.step,
        apply_fn=actor.apply,
        params=actor_params,
        tx=identity,
        opt_state=identity.init(actor_params),
        return_low=state.return_low,
        return_high=state.return_high,
    )
    critic = DreamerCritic(
        config.actor_critic.value_bins,
        hidden_dims=config.actor_critic.hidden_dims,
    )
    critic_state = DreamerCriticState(
        step=state.step,
        apply_fn=critic.apply,
        params=critic_params,
        tx=identity,
        opt_state=identity.init(critic_params),
        slow_params=state.slow_critic_params,
    )
    return world_state, actor_state, critic_state


def dreamer_world_model_loss(
    params: Any,
    state: TrainState,
    batch: WorldModelSequenceBatch,
    config: DreamerV3Config,
    key: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    del state
    return dreamer_world_model_loss_arrays(
        params,
        sequence_batch_to_jax(batch),
        config,
        key,
    )


def dreamer_world_model_loss_arrays(
    params: Any,
    batch: JaxSequenceBatch,
    config: DreamerV3Config,
    key: jax.Array,
    *,
    initial_state: RSSMState | None = None,
    previous_actions: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    observations = batch.observations.astype(jnp.float32)
    action_dtype = jnp.int32 if config.action_mode == "discrete" else jnp.float32
    actions = batch.actions.astype(action_dtype)
    is_first = batch.is_first.astype(bool)
    if previous_actions is None:
        previous_actions = jnp.concatenate(
            [jnp.zeros_like(actions[:1]), actions[:-1]],
            axis=0,
        )
    outputs = observe_dreamer_sequence(
        params,
        observations,
        previous_actions,
        is_first,
        config,
        key,
        initial_state=initial_state,
    )
    return _dreamer_world_model_loss_from_outputs(outputs, batch, config)


def _dreamer_world_model_loss_from_outputs(
    outputs: dict[str, jax.Array],
    batch: JaxSequenceBatch,
    config: DreamerV3Config,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    observations = batch.observations.astype(jnp.float32)
    rewards = batch.rewards.astype(jnp.float32)
    continues = batch.continues.astype(jnp.float32)
    reconstruction = reconstruction_loss(
        outputs["reconstructions"],
        observations,
        config,
    )
    reward_targets = symexp_two_hot(
        rewards,
        num_bins=config.reward_head.bins,
        lower=-20.0,
        upper=20.0,
    )
    reward_losses = -jnp.sum(
        reward_targets * jax.nn.log_softmax(outputs["reward_logits"], axis=-1),
        axis=-1,
    )
    continue_losses = optax.sigmoid_binary_cross_entropy(
        outputs["continue_logits"], continues
    )
    reward_loss = jnp.mean(reward_losses)
    continue_loss = jnp.mean(continue_losses)
    kl_loss, dynamics_kl_loss, representation_kl_loss = balanced_categorical_kl_loss(
        outputs["posterior_logits"],
        outputs["prior_logits"],
        free_nats=config.kl_free_nats,
        dynamics_scale=config.dynamics_kl_scale,
        representation_scale=config.representation_kl_scale,
    )
    loss = reconstruction + reward_loss + continue_loss + kl_loss
    metrics = {
        "loss": loss,
        "reconstruction_loss": reconstruction,
        "reward_loss": reward_loss,
        "continue_loss": continue_loss,
        "kl_loss": kl_loss,
        "dynamics_kl_loss": dynamics_kl_loss,
        "representation_kl_loss": representation_kl_loss,
    }
    return loss, metrics


def dreamer_replay_critic_returns(
    rewards: jax.Array,
    continues: jax.Array,
    is_last: jax.Array,
    imagination_annotations: jax.Array,
    config: DreamerV3Config,
) -> jax.Array:
    if (
        rewards.shape != continues.shape
        or rewards.shape != is_last.shape
        or rewards.shape != imagination_annotations.shape
    ):
        raise ValueError(
            "rewards, continues, is_last, and annotations must share shape"
        )
    if rewards.shape[0] < 2:
        raise ValueError("replay critic requires at least two replay states")

    live = config.actor_critic.discount * continues[1:].astype(jnp.float32)
    lambda_continue = (~is_last[1:].astype(bool)).astype(
        jnp.float32
    ) * config.actor_critic.discount_lambda
    bootstrap = imagination_annotations[1:].astype(jnp.float32)
    intermediate = (
        rewards[1:].astype(jnp.float32) + (1.0 - lambda_continue) * live * bootstrap
    )

    def step(next_return: jax.Array, inputs):
        reward_bootstrap, step_live, step_lambda = inputs
        current_return = reward_bootstrap + step_live * step_lambda * next_return
        return current_return, current_return

    _, reversed_returns = jax.lax.scan(
        step,
        imagination_annotations[-1].astype(jnp.float32),
        (
            intermediate[::-1],
            live[::-1],
            lambda_continue[::-1],
        ),
    )
    return reversed_returns[::-1]


def _critic_distribution_loss(
    logits: jax.Array,
    targets: jax.Array,
    slow_logits: jax.Array,
    weights: jax.Array,
    config: DreamerV3Config,
) -> jax.Array:
    from world_marl.dreamer_v3_baseline.imagination import decode_two_hot_logits

    target_probabilities = symexp_two_hot(
        jax.lax.stop_gradient(targets),
        num_bins=config.actor_critic.value_bins,
        lower=-20.0,
        upper=20.0,
    )
    target_loss = -jnp.sum(
        target_probabilities * jax.nn.log_softmax(logits, axis=-1),
        axis=-1,
    )
    slow_values = jax.lax.stop_gradient(decode_two_hot_logits(slow_logits))
    slow_probabilities = symexp_two_hot(
        slow_values,
        num_bins=config.actor_critic.value_bins,
        lower=-20.0,
        upper=20.0,
    )
    slow_loss = -jnp.sum(
        slow_probabilities * jax.nn.log_softmax(logits, axis=-1),
        axis=-1,
    )
    total = target_loss + config.actor_critic.critic_ema_regularizer_scale * slow_loss
    return jnp.mean(jax.lax.stop_gradient(weights) * total)


def dreamer_agent_loss_arrays(
    params: Any,
    batch: JaxSequenceBatch,
    slow_critic_params: Any,
    return_low: jax.Array,
    return_high: jax.Array,
    config: DreamerV3Config,
    key: jax.Array,
    *,
    imagination_horizon: int,
    initial_state: RSSMState | None = None,
    previous_actions: jax.Array | None = None,
) -> tuple[
    jax.Array,
    tuple[dict[str, jax.Array], Any, jax.Array, jax.Array, jax.Array],
]:
    from world_marl.dreamer_v3_baseline.imagination import (
        evaluate_actor_output,
        imagine_dreamer_rollout_from_params,
        reinforce_actor_loss,
        update_return_normalization,
    )

    posterior_key, rollout_key = jax.random.split(key)
    action_dtype = jnp.int32 if config.action_mode == "discrete" else jnp.float32
    actions = batch.actions.astype(action_dtype)
    if previous_actions is None:
        previous_actions = jnp.concatenate(
            [jnp.zeros_like(actions[:1]), actions[:-1]],
            axis=0,
        )
    posterior = observe_dreamer_sequence(
        params["world_model"],
        batch.observations.astype(jnp.float32),
        previous_actions,
        batch.is_first.astype(bool),
        config,
        posterior_key,
        initial_state=initial_state,
    )
    world_model_loss, world_metrics = _dreamer_world_model_loss_from_outputs(
        posterior,
        batch,
        config,
    )
    time_steps, batch_size = posterior["deterministic"].shape[:2]
    start_state = RSSMState(
        deterministic=jax.lax.stop_gradient(
            posterior["deterministic"].reshape(
                (time_steps * batch_size, config.rssm.deterministic_size)
            )
        ),
        stochastic=jax.lax.stop_gradient(
            posterior["stochastic"].reshape(
                (
                    time_steps * batch_size,
                    config.rssm.stochastic_size,
                    config.rssm.discrete_classes,
                )
            )
        ),
        logits=jax.lax.stop_gradient(
            posterior["posterior_logits"].reshape(
                (
                    time_steps * batch_size,
                    config.rssm.stochastic_size,
                    config.rssm.discrete_classes,
                )
            )
        ),
    )
    actor = DreamerActor(
        config.action_dim,
        config.action_mode,
        hidden_dims=config.actor_critic.hidden_dims,
        min_std=config.actor_critic.min_std,
        max_std=config.actor_critic.max_std,
    )
    critic = DreamerCritic(
        config.actor_critic.value_bins,
        hidden_dims=config.actor_critic.hidden_dims,
    )
    rollout = imagine_dreamer_rollout_from_params(
        world_model_params=params["world_model"],
        actor_apply=actor.apply,
        actor_params=params["actor"],
        critic_apply=critic.apply,
        critic_params=params["critic"],
        start_state=start_state,
        config=config,
        horizon=imagination_horizon,
        key=rollout_key,
        start_continues=batch.continues.reshape((time_steps * batch_size,)),
    )

    return_low, return_high, return_scale = update_return_normalization(
        return_low,
        return_high,
        rollout.returns,
        config,
    )
    actor_outputs = actor.apply(
        {"params": params["actor"]},
        jax.lax.stop_gradient(rollout.features),
    )
    log_probs, entropies = evaluate_actor_output(
        actor_outputs,
        jax.lax.stop_gradient(rollout.actions),
        config,
    )
    actor_loss = reinforce_actor_loss(
        log_probs,
        entropies,
        rollout.returns,
        rollout.values,
        rollout.weights,
        return_scale=return_scale,
        entropy_scale=config.actor_critic.entropy_scale,
    )

    imagined_logits = critic.apply(
        {"params": params["critic"]},
        jax.lax.stop_gradient(rollout.features),
    )
    imagined_slow_logits = critic.apply(
        {"params": slow_critic_params},
        jax.lax.stop_gradient(rollout.features),
    )
    critic_loss = _critic_distribution_loss(
        imagined_logits,
        rollout.returns,
        imagined_slow_logits,
        rollout.weights,
        config,
    )

    replay_bootstrap = jax.lax.stop_gradient(rollout.returns[0]).reshape(
        (time_steps, batch_size)
    )
    replay_returns = dreamer_replay_critic_returns(
        batch.rewards,
        batch.continues,
        batch.is_last,
        replay_bootstrap,
        config,
    )
    replay_features = posterior["features"][:-1]
    replay_logits = critic.apply(
        {"params": params["critic"]},
        replay_features,
    )
    replay_slow_logits = critic.apply(
        {"params": slow_critic_params},
        jax.lax.stop_gradient(replay_features),
    )
    replay_critic_loss = _critic_distribution_loss(
        replay_logits,
        replay_returns,
        replay_slow_logits,
        (~batch.is_last[:-1].astype(bool)).astype(jnp.float32),
        config,
    )

    loss = (
        world_model_loss
        + actor_loss
        + config.actor_critic.critic_imagination_scale * critic_loss
        + config.actor_critic.critic_replay_scale * replay_critic_loss
    )
    metrics = {
        **world_metrics,
        "loss": loss,
        "world_model_loss": world_model_loss,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "replay_critic_loss": replay_critic_loss,
        "return_scale": return_scale,
        "imagined_reward": jnp.mean(rollout.rewards),
        "imagined_value": jnp.mean(rollout.values),
        "actor_entropy": jnp.mean(entropies),
    }
    return loss, (metrics, rollout, return_low, return_high, posterior)


def _dreamer_agent_train_step_arrays(
    state: DreamerAgentState,
    batch: JaxSequenceBatch,
    config: DreamerV3Config,
    key: jax.Array,
    *,
    imagination_horizon: int,
) -> tuple[DreamerAgentState, dict[str, jax.Array], Any]:
    from world_marl.dreamer_v3_baseline.imagination import ema_critic_parameters

    (
        (
            _,
            (metrics, rollout, return_low, return_high, _),
        ),
        grads,
    ) = jax.value_and_grad(
        dreamer_agent_loss_arrays,
        has_aux=True,
    )(
        state.params,
        batch,
        state.slow_critic_params,
        state.return_low,
        state.return_high,
        config,
        key,
        imagination_horizon=imagination_horizon,
    )
    state = state.apply_gradients(grads=grads)
    state = state.replace(
        slow_critic_params=ema_critic_parameters(
            state.params["critic"],
            state.slow_critic_params,
            config,
        ),
        return_low=return_low,
        return_high=return_high,
    )
    return state, metrics, rollout


def dreamer_agent_train_step(
    state: DreamerAgentState,
    batch: WorldModelSequenceBatch,
    config: DreamerV3Config,
    key: jax.Array,
    *,
    imagination_horizon: int | None = None,
) -> tuple[DreamerAgentState, dict[str, jax.Array], Any]:
    horizon = imagination_horizon or config.actor_critic.imagination_horizon
    arrays = sequence_batch_to_jax(batch)
    return jax.jit(
        lambda train_state, train_batch, train_key: _dreamer_agent_train_step_arrays(
            train_state,
            train_batch,
            config,
            train_key,
            imagination_horizon=horizon,
        )
    )(state, arrays, key)


def scan_dreamer_agent_updates(
    state: DreamerAgentState,
    replay: JaxSequenceBatch,
    key: jax.Array,
    *,
    config: DreamerV3Config,
    train_steps: int,
    sequence_length: int,
    batch_size: int,
    imagination_horizon: int,
) -> tuple[DreamerAgentState, dict[str, jax.Array], Any]:
    def update(
        carry: tuple[DreamerAgentState, jax.Array],
        _: None,
    ) -> tuple[tuple[DreamerAgentState, jax.Array], dict[str, jax.Array]]:
        train_state, update_key = carry
        update_key, sample_key, loss_key = jax.random.split(update_key, 3)
        train_batch = sample_sequence_windows(
            replay,
            sample_key,
            sequence_length=sequence_length,
            batch_size=batch_size,
        )
        train_state, metrics, _ = _dreamer_agent_train_step_arrays(
            train_state,
            train_batch,
            config,
            loss_key,
            imagination_horizon=imagination_horizon,
        )
        return (train_state, update_key), metrics

    (state, key), metrics = jax.lax.scan(
        update,
        (state, key),
        None,
        length=train_steps,
    )
    key, sample_key, rollout_key = jax.random.split(key, 3)
    final_batch = sample_sequence_windows(
        replay,
        sample_key,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )
    _, (_, rollout, _, _, _) = dreamer_agent_loss_arrays(
        state.params,
        final_batch,
        state.slow_critic_params,
        state.return_low,
        state.return_high,
        config,
        rollout_key,
        imagination_horizon=imagination_horizon,
    )
    return state, metrics, rollout


def _empty_dreamer_agent_metrics() -> dict[str, jax.Array]:
    zero = jnp.zeros((), dtype=jnp.float32)
    return {
        "loss": zero,
        "world_model_loss": zero,
        "reconstruction_loss": zero,
        "reward_loss": zero,
        "continue_loss": zero,
        "kl_loss": zero,
        "dynamics_kl_loss": zero,
        "representation_kl_loss": zero,
        "actor_loss": zero,
        "critic_loss": zero,
        "replay_critic_loss": zero,
        "return_scale": zero,
        "imagined_reward": zero,
        "imagined_value": zero,
        "actor_entropy": zero,
        "online_replay_items": jnp.zeros((), dtype=jnp.int32),
    }


def _dreamer_replay_update(
    state: DreamerAgentState,
    replay,
    key: jax.Array,
    *,
    config: DreamerV3Config,
    sequence_length: int,
    batch_size: int,
    imagination_horizon: int,
):
    from world_marl.dreamer_v3_baseline.imagination import ema_critic_parameters
    from world_marl.dreamer_v3_baseline.replay import (
        sample_dreamer_replay,
        update_dreamer_replay_latents,
    )

    key, sample_key, loss_key = jax.random.split(key, 3)
    replay, sample = sample_dreamer_replay(
        replay,
        sample_key,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )
    (
        (
            _,
            (metrics, _, return_low, return_high, posterior),
        ),
        grads,
    ) = jax.value_and_grad(
        dreamer_agent_loss_arrays,
        has_aux=True,
    )(
        state.params,
        sample.batch,
        state.slow_critic_params,
        state.return_low,
        state.return_high,
        config,
        loss_key,
        imagination_horizon=imagination_horizon,
        initial_state=sample.initial_state,
        previous_actions=sample.previous_actions,
    )
    state = state.apply_gradients(grads=grads)
    state = state.replace(
        slow_critic_params=ema_critic_parameters(
            state.params["critic"],
            state.slow_critic_params,
            config,
        ),
        return_low=return_low,
        return_high=return_high,
    )
    replay = update_dreamer_replay_latents(replay, sample, posterior)
    metrics = {**metrics, "online_replay_items": sample.online_items}
    return state, replay, key, metrics


def build_dreamer_online_learner_step(
    *,
    config: DreamerV3Config,
    num_envs: int,
    sequence_length: int,
    batch_size: int,
    train_ratio: float,
    max_train_steps: int,
    imagination_horizon: int,
):
    from world_marl.dreamer_v3_baseline.imagination import sample_actor_output
    from world_marl.dreamer_v3_baseline.replay import append_dreamer_replay

    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if sequence_length <= 0 or batch_size <= 0:
        raise ValueError("sequence_length and batch_size must be positive")
    if train_ratio < 0:
        raise ValueError("train_ratio must be nonnegative")
    if max_train_steps <= 0:
        raise ValueError("max_train_steps must be positive")
    update_ratio = num_envs * train_ratio / float(batch_size * sequence_length)
    max_updates_per_step = max(1, math.ceil(update_ratio))
    world_model = DreamerWorldModel(config)
    actor = DreamerActor(
        config.action_dim,
        config.action_mode,
        hidden_dims=config.actor_critic.hidden_dims,
        min_std=config.actor_critic.min_std,
        max_std=config.actor_critic.max_std,
    )

    def learner_step(
        carry: DreamerOnlineCarry,
        observations: jax.Array,
        rewards: jax.Array,
        is_terminal: jax.Array,
        is_last: jax.Array,
        is_first: jax.Array,
    ):
        key, observe_key, action_key = jax.random.split(carry.key, 3)
        policy_state = reset_rssm_state(
            carry.policy_state,
            is_first,
            config=config.rssm,
        )
        if config.action_mode == "discrete":
            previous_action = jnp.where(
                is_first,
                jnp.zeros_like(carry.previous_action),
                carry.previous_action,
            )
        else:
            previous_action = jnp.where(
                is_first[:, None],
                jnp.zeros_like(carry.previous_action),
                carry.previous_action,
            )
        _, posterior, _ = world_model.apply(
            carry.agent_state.params["world_model"],
            policy_state,
            dreamer_action_features(previous_action, config),
            observations.reshape((num_envs, *config.observation_shape)),
            observe_key,
            method=world_model.observe_step,
        )
        actor_outputs = actor.apply(
            {"params": carry.agent_state.params["actor"]},
            flatten_rssm_state(posterior),
        )
        actor_sample = sample_actor_output(
            actor_outputs,
            config,
            action_key,
            deterministic=False,
        )
        actions = actor_sample.env_action
        if config.action_mode == "discrete":
            actions = jnp.where(is_last, jnp.zeros_like(actions), actions)
        else:
            actions = jnp.where(is_last[:, None], jnp.zeros_like(actions), actions)
        record = JaxSequenceBatch(
            observations=observations.reshape(
                (1, num_envs, *config.observation_shape)
            ).astype(jnp.float32),
            actions=actions[None],
            rewards=rewards.reshape((1, num_envs)).astype(jnp.float32),
            continues=(~is_terminal.astype(bool))[None].astype(jnp.float32),
            is_first=is_first.reshape((1, num_envs)).astype(bool),
            is_terminal=is_terminal.reshape((1, num_envs)).astype(bool),
            is_last=is_last.reshape((1, num_envs)).astype(bool),
        )
        replay = append_dreamer_replay(
            carry.replay,
            record,
            {
                "deterministic": posterior.deterministic[None],
                "stochastic": posterior.stochastic[None],
                "posterior_logits": posterior.logits[None],
            },
            sequence_length=sequence_length,
        )
        ready = jnp.logical_and(
            replay.size > sequence_length,
            replay.size * num_envs >= batch_size * sequence_length,
        )
        can_start = jnp.logical_and(ready, train_ratio > 0.0)
        first_update = jnp.logical_and(can_start, ~carry.ratio_started)
        accrued_credit = jnp.where(
            carry.ratio_started,
            carry.ratio_credit + update_ratio,
            carry.ratio_credit,
        )
        scheduled = jnp.floor(accrued_credit).astype(jnp.int32)
        due = jnp.where(
            first_update,
            jnp.ones((), dtype=jnp.int32),
            jnp.where(carry.ratio_started, scheduled, 0),
        )
        remaining = jnp.maximum(max_train_steps - carry.completed_updates, 0)
        due = jnp.minimum(due, remaining)
        ratio_credit = jnp.where(
            first_update,
            jnp.zeros((), dtype=jnp.float32),
            jnp.where(
                carry.ratio_started,
                accrued_credit - scheduled.astype(jnp.float32),
                carry.ratio_credit,
            ),
        )

        def update(carry_values, index):
            agent_state, train_replay, update_key = carry_values

            def run_update(values):
                return _dreamer_replay_update(
                    *values,
                    config=config,
                    sequence_length=sequence_length,
                    batch_size=batch_size,
                    imagination_horizon=imagination_horizon,
                )

            def skip_update(values):
                return (*values, _empty_dreamer_agent_metrics())

            agent_state, train_replay, update_key, metrics = jax.lax.cond(
                index < due,
                run_update,
                skip_update,
                (agent_state, train_replay, update_key),
            )
            return (agent_state, train_replay, update_key), (
                metrics,
                index < due,
            )

        (agent_state, replay, key), (update_metrics, update_executed) = jax.lax.scan(
            update,
            (carry.agent_state, replay, key),
            jnp.arange(max_updates_per_step, dtype=jnp.int32),
        )
        next_carry = DreamerOnlineCarry(
            agent_state=agent_state,
            replay=replay,
            policy_state=posterior,
            previous_action=actions,
            key=key,
            ratio_credit=ratio_credit,
            ratio_started=jnp.logical_or(carry.ratio_started, can_start),
            completed_updates=carry.completed_updates + due,
        )
        metrics = {
            "agent": update_metrics,
            "update_executed": update_executed,
            "replay_size": replay.size,
            "completed_updates": next_carry.completed_updates,
        }
        return next_carry, actions, metrics

    return learner_step


def scan_dreamer_replay_updates(
    state: DreamerAgentState,
    replay,
    key: jax.Array,
    *,
    config: DreamerV3Config,
    train_steps: int,
    sequence_length: int,
    batch_size: int,
    imagination_horizon: int,
):
    from world_marl.dreamer_v3_baseline.replay import sample_dreamer_replay

    def update(carry, _):
        train_state, train_replay, update_key = carry
        train_state, train_replay, update_key, metrics = _dreamer_replay_update(
            train_state,
            train_replay,
            update_key,
            config=config,
            sequence_length=sequence_length,
            batch_size=batch_size,
            imagination_horizon=imagination_horizon,
        )
        return (train_state, train_replay, update_key), metrics

    (state, replay, key), metrics = jax.lax.scan(
        update,
        (state, replay, key),
        None,
        length=train_steps,
    )
    key, sample_key, rollout_key = jax.random.split(key, 3)
    uniform_replay = replay._replace(online_cursor=replay.online_count)
    _, final_sample = sample_dreamer_replay(
        uniform_replay,
        sample_key,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )
    _, (_, rollout, _, _, _) = dreamer_agent_loss_arrays(
        state.params,
        final_sample.batch,
        state.slow_critic_params,
        state.return_low,
        state.return_high,
        config,
        rollout_key,
        imagination_horizon=imagination_horizon,
        initial_state=final_sample.initial_state,
        previous_actions=final_sample.previous_actions,
    )
    return state, replay, metrics, rollout


def train_dreamer_agent(
    *,
    batch: WorldModelSequenceBatch,
    config: DreamerV3Config,
    train_steps: int,
    seed: int,
    learning_rate: float | None = None,
    sequence_length: int | None = None,
    batch_size: int | None = None,
    imagination_horizon: int | None = None,
) -> tuple[DreamerAgentState, list[dict[str, float]], Any]:
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if batch.time_steps < 2:
        raise ValueError("Dreamer replay requires at least two records")
    resolved_sequence_length = min(
        batch.time_steps - 1,
        config.replay.batch_length if sequence_length is None else sequence_length,
    )
    resolved_batch_size = (
        min(batch.batch_size, config.replay.batch_size)
        if batch_size is None
        else batch_size
    )
    horizon = imagination_horizon or config.actor_critic.imagination_horizon
    state = create_dreamer_agent_state(
        jax.random.PRNGKey(seed),
        config,
        learning_rate=learning_rate,
    )
    replay_sequence = sequence_batch_to_jax(batch)
    from world_marl.dreamer_v3_baseline.replay import initialize_dreamer_replay

    replay = jax.jit(
        lambda sequence, params, replay_key: initialize_dreamer_replay(
            sequence,
            params,
            config,
            replay_key,
            sequence_length=resolved_sequence_length,
        )
    )(
        replay_sequence,
        state.params["world_model"],
        jax.random.PRNGKey(seed + 1),
    )
    state, _, device_metrics, rollout = jax.jit(
        lambda train_state, train_replay, train_key: scan_dreamer_replay_updates(
            train_state,
            train_replay,
            train_key,
            config=config,
            train_steps=train_steps,
            sequence_length=resolved_sequence_length,
            batch_size=resolved_batch_size,
            imagination_horizon=horizon,
        )
    )(state, replay, jax.random.PRNGKey(seed + 2))
    host_metrics = jax.device_get(device_metrics)
    metrics = [
        {
            "step": step,
            **{name: float(values[step]) for name, values in host_metrics.items()},
        }
        for step in range(train_steps)
    ]
    return state, metrics, rollout


def _dreamer_replay_diagnostic_rollout(
    state: DreamerAgentState,
    replay,
    key: jax.Array,
    *,
    config: DreamerV3Config,
    sequence_length: int,
    batch_size: int,
    imagination_horizon: int,
):
    from world_marl.dreamer_v3_baseline.replay import sample_dreamer_replay

    sample_key, rollout_key = jax.random.split(key)
    uniform_replay = replay._replace(online_cursor=replay.online_count)
    _, sample = sample_dreamer_replay(
        uniform_replay,
        sample_key,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )
    _, (_, rollout, _, _, _) = dreamer_agent_loss_arrays(
        state.params,
        sample.batch,
        state.slow_critic_params,
        state.return_low,
        state.return_high,
        config,
        rollout_key,
        imagination_horizon=imagination_horizon,
        initial_state=sample.initial_state,
        previous_actions=sample.previous_actions,
    )
    return rollout


def train_dreamer_agent_online(
    *,
    adapter: Any,
    observations: np.ndarray,
    config: DreamerV3Config,
    environment_steps: int,
    max_train_steps: int,
    seed: int,
    train_ratio: float | None = None,
    learning_rate: float | None = None,
    sequence_length: int | None = None,
    batch_size: int | None = None,
    imagination_horizon: int | None = None,
) -> tuple[
    DreamerAgentState,
    list[dict[str, float]],
    Any,
    WorldModelSequenceBatch,
]:
    if environment_steps <= 1:
        raise ValueError("environment_steps must exceed one")
    scan_online_rollout = getattr(adapter, "scan_online_rollout", None)
    if scan_online_rollout is None:
        raise RuntimeError("Dreamer online training requires scan_online_rollout")
    num_envs = int(adapter.num_envs)
    resolved_sequence_length = min(
        environment_steps - 1,
        config.replay.batch_length if sequence_length is None else sequence_length,
    )
    resolved_batch_size = config.replay.batch_size if batch_size is None else batch_size
    resolved_train_ratio = (
        config.replay.train_ratio if train_ratio is None else train_ratio
    )
    horizon = imagination_horizon or config.actor_critic.imagination_horizon
    capacity_limit = max(
        config.replay.capacity // num_envs, resolved_sequence_length + 1
    )
    capacity_time = min(
        capacity_limit,
        max(environment_steps, resolved_sequence_length + 1),
    )
    carry = create_dreamer_online_carry(
        config=config,
        num_envs=num_envs,
        capacity_time=capacity_time,
        sequence_length=resolved_sequence_length,
        seed=seed,
        learning_rate=learning_rate,
    )
    learner_step = build_dreamer_online_learner_step(
        config=config,
        num_envs=num_envs,
        sequence_length=resolved_sequence_length,
        batch_size=resolved_batch_size,
        train_ratio=resolved_train_ratio,
        max_train_steps=max_train_steps,
        imagination_horizon=horizon,
    )
    ys, _, carry = scan_online_rollout(
        learner_step,
        carry,
        environment_steps,
        observations=observations,
    )
    rollout = jax.jit(
        lambda state, replay, rollout_key: _dreamer_replay_diagnostic_rollout(
            state,
            replay,
            rollout_key,
            config=config,
            sequence_length=resolved_sequence_length,
            batch_size=resolved_batch_size,
            imagination_horizon=horizon,
        )
    )(
        carry.agent_state,
        carry.replay,
        jax.random.PRNGKey(seed + 2),
    )
    host_ys, host_rollout, completed_updates = jax.device_get(
        (ys, rollout, carry.completed_updates)
    )
    if int(completed_updates) == 0:
        raise RuntimeError(
            "online replay did not become ready for a Dreamer learner update"
        )
    (
        collected_observations,
        collected_actions,
        collected_rewards,
        collected_terminals,
        collected_lasts,
        collected_firsts,
        learner_metrics,
    ) = host_ys
    executed = np.asarray(learner_metrics["update_executed"], dtype=bool)
    agent_metrics = learner_metrics["agent"]
    metrics: list[dict[str, float]] = []
    for outer_index, inner_index in np.argwhere(executed):
        metrics.append(
            {
                "step": len(metrics),
                "environment_step": int(outer_index),
                **{
                    name: float(np.asarray(values)[outer_index, inner_index])
                    for name, values in agent_metrics.items()
                },
            }
        )
    environment_metadata = dict(getattr(adapter, "environment_metadata", {}))
    namespace = str(getattr(adapter, "substrate", "adapter")).split(":", 1)[0]
    environment_metadata.setdefault(
        "environment_backend",
        {
            "brax": "brax",
            "dmc": "mujoco_playground",
            "gymnax": "gymnax",
        }.get(namespace, "unknown"),
    )
    environment_metadata.setdefault(
        "observation_mode",
        "pixels" if config.is_image_observation else "vector",
    )
    batch = WorldModelSequenceBatch(
        observations=np.asarray(collected_observations, dtype=np.float32).reshape(
            (environment_steps, num_envs, *config.observation_shape)
        ),
        actions=np.asarray(collected_actions),
        rewards=np.asarray(collected_rewards, dtype=np.float32),
        continues=(~np.asarray(collected_terminals, dtype=bool)).astype(np.float32),
        is_first=np.asarray(collected_firsts, dtype=bool),
        is_terminal=np.asarray(collected_terminals, dtype=bool),
        is_last=np.asarray(collected_lasts, dtype=bool),
        metadata={
            "env": str(getattr(adapter, "substrate", "adapter")),
            "action_mode": config.action_mode,
            "action_dim": config.action_dim,
            "collection_execution": "jax_scan",
            "collection_policy": "dreamer_actor",
            "train_ratio": resolved_train_ratio,
            "environment_transitions": environment_steps * num_envs,
            "real_env_transitions": environment_steps * num_envs,
            **environment_metadata,
        },
    )
    return carry.agent_state, metrics, host_rollout, batch


def dreamer_train_step(
    state: TrainState,
    batch: WorldModelSequenceBatch,
    config: DreamerV3Config,
    key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    arrays = sequence_batch_to_jax(batch)
    return jax.jit(
        lambda train_state, train_batch: _dreamer_train_step_arrays(
            train_state,
            train_batch,
            config,
            key,
        )
    )(state, arrays)


def _dreamer_train_step_arrays(
    state: TrainState,
    batch: JaxSequenceBatch,
    config: DreamerV3Config,
    key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    (_, metrics), grads = jax.value_and_grad(
        dreamer_world_model_loss_arrays,
        has_aux=True,
    )(state.params, batch, config, key)
    return state.apply_gradients(grads=grads), metrics


def scan_dreamer_world_model_updates(
    state: TrainState,
    replay: JaxSequenceBatch,
    key: jax.Array,
    *,
    config: DreamerV3Config,
    train_steps: int,
    sequence_length: int,
    batch_size: int,
) -> tuple[TrainState, dict[str, jax.Array]]:
    def update(
        carry: tuple[TrainState, jax.Array],
        _: None,
    ) -> tuple[tuple[TrainState, jax.Array], dict[str, jax.Array]]:
        train_state, update_key = carry
        update_key, sample_key = jax.random.split(update_key)
        update_key, loss_key = jax.random.split(update_key)
        train_batch = sample_sequence_windows(
            replay,
            sample_key,
            sequence_length=sequence_length,
            batch_size=batch_size,
        )
        train_state, metrics = _dreamer_train_step_arrays(
            train_state,
            train_batch,
            config,
            loss_key,
        )
        return (train_state, update_key), metrics

    (state, _), metrics = jax.lax.scan(
        update,
        (state, key),
        None,
        length=train_steps,
    )
    return state, metrics


def train_dreamer_world_model(
    *,
    batch: WorldModelSequenceBatch,
    config: DreamerV3Config,
    train_steps: int,
    learning_rate: float,
    seed: int,
    sequence_length: int | None = None,
    batch_size: int | None = None,
) -> tuple[TrainState, list[dict[str, float]]]:
    state = create_dreamer_train_state(
        jax.random.PRNGKey(seed),
        config,
        learning_rate=learning_rate,
    )
    resolved_sequence_length = min(
        batch.time_steps,
        64 if sequence_length is None else sequence_length,
    )
    resolved_batch_size = (
        min(batch.batch_size, 16) if batch_size is None else batch_size
    )
    replay = sequence_batch_to_jax(batch)
    state, device_metrics = jax.jit(
        lambda train_state, train_replay, train_key: scan_dreamer_world_model_updates(
            train_state,
            train_replay,
            train_key,
            config=config,
            train_steps=train_steps,
            sequence_length=resolved_sequence_length,
            batch_size=resolved_batch_size,
        )
    )(state, replay, jax.random.PRNGKey(seed + 1))
    host_metrics = jax.device_get(device_metrics)
    metrics = [
        {
            "step": step,
            **{name: float(values[step]) for name, values in host_metrics.items()},
        }
        for step in range(train_steps)
    ]
    return state, metrics
