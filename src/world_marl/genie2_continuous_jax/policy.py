from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from world_marl.genie2_continuous_jax.config import Genie2ContinuousConfig
from world_marl.genie2_continuous_jax.training import Genie2WorldModel
from world_marl.world_model_foundation.replay import WorldModelSequenceBatch


class LatentActionPolicy(nn.Module):
    latent_action_dim: int
    hidden_dims: tuple[int, ...] = (128, 128)

    @nn.compact
    def __call__(self, latents: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = latents.astype(jnp.float32)
        for dim in self.hidden_dims:
            x = nn.silu(nn.LayerNorm()(nn.Dense(dim)(x)))
        mean = nn.Dense(self.latent_action_dim, name="mean")(x)
        log_std = jnp.clip(
            nn.Dense(self.latent_action_dim, name="log_std")(x),
            -5.0,
            2.0,
        )
        return mean, log_std


class LatentValue(nn.Module):
    hidden_dims: tuple[int, ...] = (128, 128)

    @nn.compact
    def __call__(self, latents: jax.Array) -> jax.Array:
        x = latents.astype(jnp.float32)
        for dim in self.hidden_dims:
            x = nn.silu(nn.LayerNorm()(nn.Dense(dim)(x)))
        return nn.Dense(1, name="value")(x)[..., 0]


@dataclass(frozen=True, slots=True)
class Genie2PolicyRollout:
    latents: jax.Array
    latent_actions: jax.Array
    rewards: jax.Array
    continues: jax.Array
    values: jax.Array
    returns: jax.Array
    entropies: jax.Array


def _lambda_returns(
    rewards: jax.Array,
    continues: jax.Array,
    values: jax.Array,
    bootstrap: jax.Array,
    *,
    discount_lambda: float,
) -> jax.Array:
    next_values = jnp.concatenate([values[1:], bootstrap[None]], axis=0)
    last = bootstrap
    targets = []
    for index in range(rewards.shape[0] - 1, -1, -1):
        last = rewards[index] + continues[index] * (
            (1.0 - discount_lambda) * next_values[index] + discount_lambda * last
        )
        targets.append(last)
    return jnp.stack(targets[::-1])


def create_latent_policy_states(
    key: jax.Array,
    config: Genie2ContinuousConfig,
    *,
    learning_rate: float,
) -> tuple[TrainState, TrainState]:
    actor_key, critic_key = jax.random.split(key)
    dummy_latents = jnp.zeros(
        (1, config.autoencoder.latent_dim),
        dtype=jnp.float32,
    )
    actor = LatentActionPolicy(
        config.lam.latent_action_dim,
        hidden_dims=config.latent_policy.hidden_dims,
    )
    critic = LatentValue(hidden_dims=config.latent_policy.hidden_dims)
    actor_params = actor.init(actor_key, dummy_latents)["params"]
    critic_params = critic.init(critic_key, dummy_latents)["params"]
    return (
        TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=optax.adam(learning_rate),
        ),
        TrainState.create(
            apply_fn=critic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate),
        ),
    )


def _policy_action(
    actor_state: TrainState,
    actor_params: Any,
    latents: jax.Array,
    key: jax.Array,
    *,
    deterministic: bool,
) -> tuple[jax.Array, jax.Array]:
    mean, log_std = actor_state.apply_fn({"params": actor_params}, latents)
    if deterministic:
        pre_tanh = mean
    else:
        pre_tanh = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
    actions = jnp.tanh(pre_tanh)
    entropies = jnp.sum(
        log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e),
        axis=-1,
    )
    return actions, entropies


def latent_policy_action(actor_state: TrainState, latents: jax.Array) -> jax.Array:
    actions, _ = _policy_action(
        actor_state,
        actor_state.params,
        latents,
        jax.random.PRNGKey(0),
        deterministic=True,
    )
    return actions


def _start_latents(
    world_model_state: TrainState,
    batch: WorldModelSequenceBatch,
    observation_shape: tuple[int, ...],
    config: Genie2ContinuousConfig,
) -> jax.Array:
    model = Genie2WorldModel(observation_shape=observation_shape, config=config)
    return jax.lax.stop_gradient(
        model.apply(
            world_model_state.params,
            jnp.asarray(batch.observations[-1], dtype=jnp.float32),
            method=model.encode,
        )
    )


def simulate_latent_policy_rollout(
    *,
    world_model_state: TrainState,
    actor_state: TrainState,
    critic_state: TrainState,
    start_latents: jax.Array,
    observation_shape: tuple[int, ...],
    config: Genie2ContinuousConfig,
    horizon: int,
    key: jax.Array,
    actor_params: Any | None = None,
    critic_params: Any | None = None,
) -> Genie2PolicyRollout:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    actor_params = actor_state.params if actor_params is None else actor_params
    critic_params = critic_state.params if critic_params is None else critic_params
    model = Genie2WorldModel(observation_shape=observation_shape, config=config)
    history = [start_latents]
    action_history = []
    predicted_latents = []
    rewards = []
    continues = []
    values = []
    entropies = []
    for _ in range(horizon):
        key, action_key = jax.random.split(key)
        latent_action, entropy = _policy_action(
            actor_state,
            actor_params,
            history[-1],
            action_key,
            deterministic=False,
        )
        action_history.append(latent_action)
        context_length = min(len(action_history), config.dynamics.max_context)
        latent_context = jnp.stack(history[-context_length:], axis=1)
        action_context = jnp.stack(action_history[-context_length:], axis=1)
        next_latent = model.apply(
            world_model_state.params,
            latent_context,
            action_context,
            method=model.predict_next,
        )
        reward, continue_logit = model.apply(
            world_model_state.params,
            next_latent,
            latent_action,
            method=model.predict_reward_continue,
        )
        value = critic_state.apply_fn({"params": critic_params}, next_latent)
        history.append(next_latent)
        predicted_latents.append(next_latent)
        rewards.append(reward)
        continues.append(jax.nn.sigmoid(continue_logit))
        values.append(value)
        entropies.append(entropy)

    stacked_latents = jnp.stack(predicted_latents)
    stacked_actions = jnp.stack(action_history)
    stacked_rewards = jnp.stack(rewards)
    stacked_continues = jnp.stack(continues)
    stacked_values = jnp.stack(values)
    bootstrap = critic_state.apply_fn(
        {"params": critic_params},
        stacked_latents[-1],
    )
    returns = _lambda_returns(
        stacked_rewards,
        stacked_continues,
        stacked_values,
        bootstrap,
        discount_lambda=config.latent_policy.discount_lambda,
    )
    return Genie2PolicyRollout(
        latents=stacked_latents,
        latent_actions=stacked_actions,
        rewards=stacked_rewards,
        continues=stacked_continues,
        values=stacked_values,
        returns=returns,
        entropies=jnp.stack(entropies),
    )


def _actor_loss(
    actor_params: Any,
    *,
    world_model_state: TrainState,
    actor_state: TrainState,
    critic_state: TrainState,
    start_latents: jax.Array,
    observation_shape: tuple[int, ...],
    config: Genie2ContinuousConfig,
    horizon: int,
    key: jax.Array,
) -> jax.Array:
    rollout = simulate_latent_policy_rollout(
        world_model_state=world_model_state,
        actor_state=actor_state,
        critic_state=critic_state,
        start_latents=start_latents,
        observation_shape=observation_shape,
        config=config,
        horizon=horizon,
        key=key,
        actor_params=actor_params,
    )
    weights = jnp.concatenate(
        [
            jnp.ones_like(rollout.continues[:1]),
            jnp.cumprod(rollout.continues[:-1], axis=0),
        ],
        axis=0,
    )
    objective = jnp.mean(weights * rollout.returns)
    entropy = jnp.mean(rollout.entropies)
    action_norm = jnp.mean(jnp.square(rollout.latent_actions))
    return (
        -objective
        - config.latent_policy.entropy_scale * entropy
        + (config.latent_policy.action_penalty * action_norm)
    )


def _critic_loss(
    critic_params: Any,
    *,
    critic_state: TrainState,
    rollout: Genie2PolicyRollout,
) -> jax.Array:
    predictions = critic_state.apply_fn(
        {"params": critic_params},
        jax.lax.stop_gradient(rollout.latents),
    )
    return jnp.mean(jnp.square(predictions - jax.lax.stop_gradient(rollout.returns)))


def train_genie2_latent_policy(
    *,
    world_model_state: TrainState,
    batch: WorldModelSequenceBatch,
    observation_shape: tuple[int, ...],
    config: Genie2ContinuousConfig,
    train_steps: int,
    learning_rate: float,
    imagination_horizon: int | None = None,
    seed: int,
) -> tuple[TrainState, TrainState, list[dict[str, float]], Genie2PolicyRollout]:
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    horizon = imagination_horizon or config.latent_policy.imagination_horizon
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    actor_state, critic_state = create_latent_policy_states(
        init_key,
        config,
        learning_rate=learning_rate,
    )
    start_latents = _start_latents(
        world_model_state,
        batch,
        observation_shape,
        config,
    )
    metrics: list[dict[str, float]] = []
    rollout = None
    for step in range(train_steps):
        key, actor_key, rollout_key = jax.random.split(key, 3)
        actor_loss, actor_grads = jax.value_and_grad(_actor_loss)(
            actor_state.params,
            world_model_state=world_model_state,
            actor_state=actor_state,
            critic_state=critic_state,
            start_latents=start_latents,
            observation_shape=observation_shape,
            config=config,
            horizon=horizon,
            key=actor_key,
        )
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        rollout = simulate_latent_policy_rollout(
            world_model_state=world_model_state,
            actor_state=actor_state,
            critic_state=critic_state,
            start_latents=start_latents,
            observation_shape=observation_shape,
            config=config,
            horizon=horizon,
            key=rollout_key,
        )
        critic_loss, critic_grads = jax.value_and_grad(_critic_loss)(
            critic_state.params,
            critic_state=critic_state,
            rollout=rollout,
        )
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        metrics.append(
            {
                "step": step,
                "actor_loss": float(actor_loss),
                "critic_loss": float(critic_loss),
                "imagined_reward": float(jnp.mean(rollout.rewards)),
                "imagined_value": float(jnp.mean(rollout.values)),
                "imagined_continue": float(jnp.mean(rollout.continues)),
                "actor_entropy": float(jnp.mean(rollout.entropies)),
            }
        )
    assert rollout is not None
    key, final_key = jax.random.split(key)
    rollout = simulate_latent_policy_rollout(
        world_model_state=world_model_state,
        actor_state=actor_state,
        critic_state=critic_state,
        start_latents=start_latents,
        observation_shape=observation_shape,
        config=config,
        horizon=horizon,
        key=final_key,
    )
    return actor_state, critic_state, metrics, rollout
