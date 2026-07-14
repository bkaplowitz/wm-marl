from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from world_marl.dreamer_v3_baseline.config import DreamerV3Config
from world_marl.dreamer_v3_baseline.losses import (
    symexp_two_hot,
    symexp_two_hot_support,
)
from world_marl.dreamer_v3_baseline.models import DreamerActor, DreamerCritic
from world_marl.dreamer_v3_baseline.optimizer import dreamer_laprop
from world_marl.dreamer_v3_baseline.rssm import RSSMState, flatten_rssm_state
from world_marl.dreamer_v3_baseline.training import (
    DreamerWorldModel,
    observe_dreamer_sequence,
)
from world_marl.world_model_foundation.replay import WorldModelSequenceBatch


class DreamerActorState(TrainState):
    return_low: jax.Array
    return_high: jax.Array


class DreamerCriticState(TrainState):
    slow_params: Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class DreamerActorSample:
    env_action: jax.Array
    model_action: jax.Array
    log_prob: jax.Array
    entropy: jax.Array
    probabilities: jax.Array

    def tree_flatten(self):
        return (
            (
                self.env_action,
                self.model_action,
                self.log_prob,
                self.entropy,
                self.probabilities,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


def _discrete_probabilities(
    logits: jax.Array,
    config: DreamerV3Config,
) -> jax.Array:
    probabilities = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    uniform = jnp.full_like(probabilities, 1.0 / probabilities.shape[-1])
    return (
        1.0 - config.actor_critic.actor_unimix
    ) * probabilities + config.actor_critic.actor_unimix * uniform


def evaluate_actor_output(
    outputs: dict[str, jax.Array],
    actions: jax.Array,
    config: DreamerV3Config,
) -> tuple[jax.Array, jax.Array]:
    if config.action_mode == "discrete":
        probabilities = _discrete_probabilities(outputs["logits"], config)
        one_hot = jax.nn.one_hot(
            actions.astype(jnp.int32),
            config.action_dim,
            dtype=jnp.float32,
        )
        log_probabilities = jnp.log(probabilities)
        log_prob = jnp.sum(log_probabilities * one_hot, axis=-1)
        entropy = -jnp.sum(probabilities * log_probabilities, axis=-1)
        return log_prob, entropy

    mean = outputs["mean"]
    stddev = outputs["stddev"]
    standardized = (actions - mean) / stddev
    element_log_prob = -0.5 * (
        jnp.square(standardized) + 2.0 * jnp.log(stddev) + jnp.log(2.0 * jnp.pi)
    )
    log_prob = jnp.sum(element_log_prob, axis=-1)
    entropy = jnp.sum(
        jnp.log(stddev) + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e),
        axis=-1,
    )
    return log_prob, entropy


def sample_actor_output(
    outputs: dict[str, jax.Array],
    config: DreamerV3Config,
    key: jax.Array,
    *,
    deterministic: bool,
) -> DreamerActorSample:
    if config.action_mode == "discrete":
        probabilities = _discrete_probabilities(outputs["logits"], config)
        mixed_logits = jnp.log(probabilities)
        if deterministic:
            action_ids = jnp.argmax(mixed_logits, axis=-1)
        else:
            action_ids = jax.random.categorical(key, mixed_logits, axis=-1)
        hard = jax.nn.one_hot(
            action_ids,
            config.action_dim,
            dtype=jnp.float32,
        )
        model_action = jax.lax.stop_gradient(hard) + (
            probabilities - jax.lax.stop_gradient(probabilities)
        )
        log_prob, entropy = evaluate_actor_output(
            outputs,
            action_ids,
            config,
        )
        return DreamerActorSample(
            env_action=action_ids.astype(jnp.int32),
            model_action=model_action,
            log_prob=log_prob,
            entropy=entropy,
            probabilities=probabilities,
        )

    mean = outputs["mean"]
    stddev = outputs["stddev"]
    action = (
        mean if deterministic else mean + stddev * jax.random.normal(key, mean.shape)
    )
    log_prob, entropy = evaluate_actor_output(outputs, action, config)
    probabilities = jnp.empty((*mean.shape[:-1], 0), dtype=jnp.float32)
    return DreamerActorSample(
        env_action=action,
        model_action=action,
        log_prob=log_prob,
        entropy=entropy,
        probabilities=probabilities,
    )


def discount_continuation(
    continue_logits: jax.Array,
    config: DreamerV3Config,
) -> jax.Array:
    return config.actor_critic.discount * jax.nn.sigmoid(continue_logits)


def imagination_weights(
    predicted_continues: jax.Array,
    start_continues: jax.Array,
) -> jax.Array:
    if predicted_continues.ndim < 2:
        raise ValueError("predicted_continues must have time and batch axes")
    if start_continues.shape != predicted_continues.shape[1:]:
        raise ValueError("start_continues must match the imagined batch shape")
    preceding = jnp.concatenate(
        [
            jnp.ones_like(predicted_continues[:1]),
            jnp.cumprod(predicted_continues[:-1], axis=0),
        ],
        axis=0,
    )
    return start_continues[None].astype(preceding.dtype) * preceding


def reinforce_actor_loss(
    log_probs: jax.Array,
    entropies: jax.Array,
    returns: jax.Array,
    values: jax.Array,
    weights: jax.Array,
    *,
    return_scale: jax.Array,
    entropy_scale: float,
) -> jax.Array:
    advantage = jax.lax.stop_gradient(
        (returns - values) / jnp.maximum(1.0, return_scale)
    )
    objective = log_probs * advantage + entropy_scale * entropies
    return -jnp.mean(jax.lax.stop_gradient(weights) * objective)


def update_return_normalization(
    low: jax.Array,
    high: jax.Array,
    returns: jax.Array,
    config: DreamerV3Config,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    actor_config = config.actor_critic
    stopped_returns = jax.lax.stop_gradient(returns.astype(jnp.float32))
    batch_low = jnp.percentile(stopped_returns, actor_config.return_percentile_low)
    batch_high = jnp.percentile(stopped_returns, actor_config.return_percentile_high)
    decay = actor_config.return_norm_decay
    low = decay * low + (1.0 - decay) * batch_low
    high = decay * high + (1.0 - decay) * batch_high
    scale = jnp.maximum(actor_config.return_scale_min, high - low)
    return jax.tree.map(
        jax.lax.stop_gradient,
        (low, high, scale),
    )


def ema_critic_parameters(
    online: Any,
    slow: Any,
    config: DreamerV3Config,
) -> Any:
    decay = config.actor_critic.critic_ema_decay
    return jax.tree.map(
        lambda online_value, slow_value: (
            decay * slow_value + (1.0 - decay) * online_value
        ),
        online,
        slow,
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class DreamerImaginedRollout:
    actions: jax.Array
    features: jax.Array
    rewards: jax.Array
    continues: jax.Array
    values: jax.Array
    returns: jax.Array
    log_probs: jax.Array
    entropies: jax.Array
    weights: jax.Array

    def tree_flatten(self):
        return (
            (
                self.actions,
                self.features,
                self.rewards,
                self.continues,
                self.values,
                self.returns,
                self.log_probs,
                self.entropies,
                self.weights,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


def decode_two_hot_logits(
    logits: jax.Array,
    *,
    lower: float = -20.0,
    upper: float = 20.0,
) -> jax.Array:
    bins = logits.shape[-1]
    support = symexp_two_hot_support(
        num_bins=bins,
        lower=lower,
        upper=upper,
    )
    probabilities = jax.nn.softmax(logits, axis=-1)
    if bins % 2:
        midpoint = (bins - 1) // 2
        left_probs = probabilities[..., :midpoint]
        center_probs = probabilities[..., midpoint : midpoint + 1]
        right_probs = probabilities[..., midpoint + 1 :]
        left_support = support[:midpoint]
        center_support = support[midpoint : midpoint + 1]
        right_support = support[midpoint + 1 :]
        weighted = jnp.sum(center_probs * center_support, axis=-1)
        weighted += jnp.sum(
            (left_probs * left_support)[..., ::-1] + right_probs * right_support,
            axis=-1,
        )
    else:
        midpoint = bins // 2
        weighted = jnp.sum(
            (probabilities[..., :midpoint] * support[:midpoint])[..., ::-1]
            + probabilities[..., midpoint:] * support[midpoint:],
            axis=-1,
        )
    return weighted


def lambda_returns(
    rewards: jax.Array,
    continues: jax.Array,
    values: jax.Array,
    bootstrap: jax.Array,
    *,
    discount_lambda: float,
) -> jax.Array:
    next_values = jnp.concatenate([values[1:], bootstrap[None]], axis=0)

    def step(last: jax.Array, inputs: tuple[jax.Array, jax.Array, jax.Array]):
        reward, continue_, next_value = inputs
        last = reward + continue_ * (
            (1.0 - discount_lambda) * next_value + discount_lambda * last
        )
        return last, last

    _, targets = jax.lax.scan(
        step,
        bootstrap,
        (rewards[::-1], continues[::-1], next_values[::-1]),
    )
    return targets[::-1]


def create_dreamer_actor_critic_states(
    key: jax.Array,
    config: DreamerV3Config,
    *,
    learning_rate: float | None = None,
) -> tuple[DreamerActorState, DreamerCriticState]:
    actor_key, critic_key = jax.random.split(key)
    dummy_features = jnp.zeros((1, config.rssm.latent_size), dtype=jnp.float32)
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
    actor_params = actor.init(actor_key, dummy_features)["params"]
    critic_params = critic.init(critic_key, dummy_features)["params"]
    optimizer_config = config.optimizer
    if learning_rate is not None:
        optimizer_config = replace(optimizer_config, learning_rate=learning_rate)
    return (
        DreamerActorState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=dreamer_laprop(optimizer_config),
            return_low=jnp.asarray(0.0, dtype=jnp.float32),
            return_high=jnp.asarray(0.0, dtype=jnp.float32),
        ),
        DreamerCriticState.create(
            apply_fn=critic.apply,
            params=critic_params,
            tx=dreamer_laprop(optimizer_config),
            slow_params=critic_params,
        ),
    )


def _actor_action(
    actor_apply: Any,
    actor_params: Any,
    features: jax.Array,
    config: DreamerV3Config,
    key: jax.Array,
    *,
    deterministic: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    sample = sample_actor_output(
        actor_apply({"params": actor_params}, features),
        config,
        key,
        deterministic=deterministic,
    )
    return sample.env_action, sample.model_action, sample.entropy


def dreamer_policy_action(
    actor_state: DreamerActorState,
    features: jax.Array,
    config: DreamerV3Config,
    key: jax.Array,
    *,
    deterministic: bool = False,
) -> jax.Array:
    actions, _, _ = _actor_action(
        actor_state.apply_fn,
        actor_state.params,
        features,
        config,
        key,
        deterministic=deterministic,
    )
    return actions


def _posterior_outputs(
    world_model_state: TrainState,
    batch: WorldModelSequenceBatch,
    config: DreamerV3Config,
    key: jax.Array,
) -> dict[str, jax.Array]:
    action_dtype = jnp.int32 if config.action_mode == "discrete" else jnp.float32
    actions = jnp.asarray(batch.actions, dtype=action_dtype)
    previous_actions = jnp.concatenate(
        [jnp.zeros_like(actions[:1]), actions[:-1]],
        axis=0,
    )
    return observe_dreamer_sequence(
        world_model_state.params,
        jnp.asarray(batch.observations, dtype=jnp.float32),
        previous_actions,
        jnp.asarray(batch.is_first, dtype=bool),
        config,
        key,
    )


def _posterior_start_state(
    world_model_state: TrainState,
    batch: WorldModelSequenceBatch,
    config: DreamerV3Config,
    key: jax.Array,
) -> RSSMState:
    outputs = _posterior_outputs(world_model_state, batch, config, key)
    time_steps, batch_size = outputs["deterministic"].shape[:2]
    return RSSMState(
        deterministic=jax.lax.stop_gradient(
            outputs["deterministic"].reshape(
                (time_steps * batch_size, config.rssm.deterministic_size)
            )
        ),
        stochastic=jax.lax.stop_gradient(
            outputs["stochastic"].reshape(
                (
                    time_steps * batch_size,
                    config.rssm.stochastic_size,
                    config.rssm.discrete_classes,
                )
            )
        ),
        logits=jax.lax.stop_gradient(
            outputs["posterior_logits"].reshape(
                (
                    time_steps * batch_size,
                    config.rssm.stochastic_size,
                    config.rssm.discrete_classes,
                )
            )
        ),
    )


def imagine_dreamer_rollout(
    *,
    world_model_state: TrainState,
    actor_state: DreamerActorState,
    critic_state: DreamerCriticState,
    start_state: RSSMState,
    config: DreamerV3Config,
    horizon: int,
    key: jax.Array,
    start_continues: jax.Array | None = None,
    actor_params: Any | None = None,
    critic_params: Any | None = None,
) -> DreamerImaginedRollout:
    actor_params = actor_state.params if actor_params is None else actor_params
    critic_params = critic_state.params if critic_params is None else critic_params
    return imagine_dreamer_rollout_from_params(
        world_model_params=world_model_state.params,
        actor_apply=actor_state.apply_fn,
        actor_params=actor_params,
        critic_apply=critic_state.apply_fn,
        critic_params=critic_params,
        start_state=start_state,
        config=config,
        horizon=horizon,
        key=key,
        start_continues=start_continues,
    )


def imagine_dreamer_rollout_from_params(
    *,
    world_model_params: Any,
    actor_apply: Any,
    actor_params: Any,
    critic_apply: Any,
    critic_params: Any,
    start_state: RSSMState,
    config: DreamerV3Config,
    horizon: int,
    key: jax.Array,
    start_continues: jax.Array | None = None,
) -> DreamerImaginedRollout:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    model = DreamerWorldModel(config)

    def step(
        carry: tuple[RSSMState, jax.Array],
        _: None,
    ) -> tuple[tuple[RSSMState, jax.Array], tuple[jax.Array, ...]]:
        state, rollout_key = carry
        rollout_key, action_key, model_key = jax.random.split(rollout_key, 3)
        features = flatten_rssm_state(state)
        actor_outputs = actor_apply({"params": actor_params}, features)
        sample = sample_actor_output(
            actor_outputs,
            config,
            action_key,
            deterministic=False,
        )
        next_state, prediction = model.apply(
            world_model_params,
            state,
            jax.lax.stop_gradient(sample.model_action),
            model_key,
            method=model.imagine_step,
        )
        value_logits = critic_apply({"params": critic_params}, features)
        outputs = (
            jax.lax.stop_gradient(sample.env_action),
            jax.lax.stop_gradient(features),
            jax.lax.stop_gradient(decode_two_hot_logits(prediction["reward_logits"])),
            jax.lax.stop_gradient(
                discount_continuation(prediction["continue_logits"], config)
            ),
            jax.lax.stop_gradient(decode_two_hot_logits(value_logits)),
            jax.lax.stop_gradient(sample.log_prob),
            jax.lax.stop_gradient(sample.entropy),
        )
        return (next_state, rollout_key), outputs

    (final_state, _), outputs = jax.lax.scan(
        step,
        (start_state, key),
        None,
        length=horizon,
    )
    (
        actions,
        features,
        rewards,
        continues,
        values,
        log_probs,
        entropies,
    ) = outputs
    final_features = flatten_rssm_state(final_state)
    bootstrap_logits = critic_apply(
        {"params": critic_params},
        final_features,
    )
    bootstrap = decode_two_hot_logits(bootstrap_logits)
    returns = lambda_returns(
        rewards,
        continues,
        values,
        bootstrap,
        discount_lambda=config.actor_critic.discount_lambda,
    )
    if start_continues is None:
        start_continues = jnp.ones_like(continues[0])
    weights = imagination_weights(continues, start_continues)
    return DreamerImaginedRollout(
        actions=actions,
        features=features,
        rewards=rewards,
        continues=continues,
        values=values,
        returns=returns,
        log_probs=log_probs,
        entropies=entropies,
        weights=weights,
    )


def _actor_loss(
    actor_params: Any,
    *,
    actor_state: DreamerActorState,
    rollout: DreamerImaginedRollout,
    config: DreamerV3Config,
    return_scale: jax.Array,
) -> jax.Array:
    features = jax.lax.stop_gradient(rollout.features)
    actions = jax.lax.stop_gradient(rollout.actions)
    outputs = actor_state.apply_fn({"params": actor_params}, features)
    log_probs, entropies = evaluate_actor_output(outputs, actions, config)
    return reinforce_actor_loss(
        log_probs,
        entropies,
        rollout.returns,
        rollout.values,
        rollout.weights,
        return_scale=return_scale,
        entropy_scale=config.actor_critic.entropy_scale,
    )


def _critic_loss(
    critic_params: Any,
    *,
    critic_state: DreamerCriticState,
    rollout: DreamerImaginedRollout,
    config: DreamerV3Config,
) -> jax.Array:
    features = jax.lax.stop_gradient(rollout.features)
    return_targets = symexp_two_hot(
        jax.lax.stop_gradient(rollout.returns),
        num_bins=config.actor_critic.value_bins,
        lower=-20.0,
        upper=20.0,
    )
    logits = critic_state.apply_fn({"params": critic_params}, features)
    return_loss = -jnp.sum(
        return_targets * jax.nn.log_softmax(logits, axis=-1),
        axis=-1,
    )
    slow_logits = critic_state.apply_fn(
        {"params": critic_state.slow_params},
        features,
    )
    slow_targets = symexp_two_hot(
        jax.lax.stop_gradient(decode_two_hot_logits(slow_logits)),
        num_bins=config.actor_critic.value_bins,
        lower=-20.0,
        upper=20.0,
    )
    slow_regularizer = -jnp.sum(
        slow_targets * jax.nn.log_softmax(logits, axis=-1),
        axis=-1,
    )
    losses = (
        config.actor_critic.critic_imagination_scale * return_loss
        + config.actor_critic.critic_ema_regularizer_scale * slow_regularizer
    )
    return jnp.mean(jax.lax.stop_gradient(rollout.weights) * losses)


def train_dreamer_actor_critic(
    *,
    world_model_state: TrainState,
    batch: WorldModelSequenceBatch,
    config: DreamerV3Config,
    train_steps: int,
    learning_rate: float | None = None,
    imagination_horizon: int | None = None,
    seed: int,
) -> tuple[
    DreamerActorState,
    DreamerCriticState,
    list[dict[str, float]],
    DreamerImaginedRollout,
]:
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    horizon = imagination_horizon or config.actor_critic.imagination_horizon
    key = jax.random.PRNGKey(seed)
    key, init_key, posterior_key = jax.random.split(key, 3)
    actor_state, critic_state = create_dreamer_actor_critic_states(
        init_key,
        config,
        learning_rate=learning_rate,
    )
    start_state = _posterior_start_state(
        world_model_state,
        batch,
        config,
        posterior_key,
    )

    def update(
        carry: tuple[DreamerActorState, DreamerCriticState, jax.Array],
        _: None,
    ) -> tuple[
        tuple[DreamerActorState, DreamerCriticState, jax.Array],
        dict[str, jax.Array],
    ]:
        actor_state, critic_state, update_key = carry
        update_key, rollout_key = jax.random.split(update_key)
        rollout = imagine_dreamer_rollout(
            world_model_state=world_model_state,
            actor_state=actor_state,
            critic_state=critic_state,
            start_state=start_state,
            config=config,
            horizon=horizon,
            key=rollout_key,
        )
        return_low, return_high, return_scale = update_return_normalization(
            actor_state.return_low,
            actor_state.return_high,
            rollout.returns,
            config,
        )
        actor_loss, actor_grads = jax.value_and_grad(_actor_loss)(
            actor_state.params,
            actor_state=actor_state,
            rollout=rollout,
            config=config,
            return_scale=return_scale,
        )
        actor_state = actor_state.apply_gradients(grads=actor_grads).replace(
            return_low=return_low,
            return_high=return_high,
        )
        critic_loss, critic_grads = jax.value_and_grad(_critic_loss)(
            critic_state.params,
            critic_state=critic_state,
            rollout=rollout,
            config=config,
        )
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        critic_state = critic_state.replace(
            slow_params=ema_critic_parameters(
                critic_state.params,
                critic_state.slow_params,
                config,
            )
        )
        metrics = {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "imagined_reward": jnp.mean(rollout.rewards),
            "imagined_value": jnp.mean(rollout.values),
            "imagined_continue": jnp.mean(rollout.continues),
            "actor_entropy": jnp.mean(rollout.entropies),
            "return_scale": return_scale,
        }
        return (actor_state, critic_state, update_key), metrics

    def run_updates(actor_state, critic_state, update_key):
        (actor_state, critic_state, update_key), metrics = jax.lax.scan(
            update,
            (actor_state, critic_state, update_key),
            None,
            length=train_steps,
        )
        update_key, final_key = jax.random.split(update_key)
        rollout = imagine_dreamer_rollout(
            world_model_state=world_model_state,
            actor_state=actor_state,
            critic_state=critic_state,
            start_state=start_state,
            config=config,
            horizon=horizon,
            key=final_key,
        )
        return actor_state, critic_state, metrics, rollout

    actor_state, critic_state, device_metrics, rollout = jax.jit(run_updates)(
        actor_state,
        critic_state,
        key,
    )
    host_metrics = jax.device_get(device_metrics)
    metrics = [
        {
            "step": step,
            **{name: float(values[step]) for name, values in host_metrics.items()},
        }
        for step in range(train_steps)
    ]
    return actor_state, critic_state, metrics, rollout


def open_loop_diagnostic(features: jax.Array, horizon: int) -> DreamerImaginedRollout:
    clipped = features[:horizon]
    rewards = jnp.mean(clipped, axis=-1)
    continues = jnp.ones_like(rewards)
    values = jnp.cumsum(rewards[::-1], axis=0)[::-1]
    actions = jnp.zeros((*rewards.shape, 0), dtype=jnp.float32)
    return DreamerImaginedRollout(
        actions=actions,
        features=clipped,
        rewards=rewards,
        continues=continues,
        values=values,
        returns=values,
        log_probs=jnp.zeros_like(rewards),
        entropies=jnp.zeros_like(rewards),
        weights=jnp.ones_like(rewards),
    )
