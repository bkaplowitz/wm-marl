from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from world_marl.dreamer_v3_baseline.config import DreamerV3Config
from world_marl.dreamer_v3_baseline.losses import symexp, symlog, two_hot
from world_marl.dreamer_v3_baseline.models import DreamerActor, DreamerCritic
from world_marl.dreamer_v3_baseline.rssm import RSSMState, flatten_rssm_state
from world_marl.dreamer_v3_baseline.training import DreamerWorldModel
from world_marl.world_model_foundation.replay import WorldModelSequenceBatch


@dataclass(frozen=True, slots=True)
class DreamerImaginedRollout:
    actions: jax.Array
    features: jax.Array
    rewards: jax.Array
    continues: jax.Array
    values: jax.Array
    returns: jax.Array
    entropies: jax.Array


def decode_two_hot_logits(
    logits: jax.Array,
    *,
    lower: float = -20.0,
    upper: float = 20.0,
) -> jax.Array:
    support = jnp.linspace(lower, upper, logits.shape[-1], dtype=jnp.float32)
    return symexp(jnp.sum(jax.nn.softmax(logits, axis=-1) * support, axis=-1))


def lambda_returns(
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


def create_dreamer_actor_critic_states(
    key: jax.Array,
    config: DreamerV3Config,
    *,
    learning_rate: float,
) -> tuple[TrainState, TrainState]:
    actor_key, critic_key = jax.random.split(key)
    dummy_features = jnp.zeros((1, config.rssm.latent_size), dtype=jnp.float32)
    actor = DreamerActor(
        config.action_dim,
        config.action_mode,
        hidden_dims=config.actor_critic.hidden_dims,
    )
    critic = DreamerCritic(
        config.actor_critic.value_bins,
        hidden_dims=config.actor_critic.hidden_dims,
    )
    actor_params = actor.init(actor_key, dummy_features)["params"]
    critic_params = critic.init(critic_key, dummy_features)["params"]
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


def _actor_action(
    actor_apply: Any,
    actor_params: Any,
    features: jax.Array,
    config: DreamerV3Config,
    key: jax.Array,
    *,
    deterministic: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    outputs = actor_apply({"params": actor_params}, features)
    if config.action_mode == "discrete":
        logits = outputs["logits"]
        probs = jax.nn.softmax(logits, axis=-1)
        if deterministic:
            action_ids = jnp.argmax(logits, axis=-1)
        else:
            action_ids = jax.random.categorical(key, logits, axis=-1)
        hard = jax.nn.one_hot(action_ids, config.action_dim, dtype=jnp.float32)
        action_features = hard - jax.lax.stop_gradient(probs) + probs
        entropy = -jnp.sum(
            probs * jax.nn.log_softmax(logits, axis=-1),
            axis=-1,
        )
        return action_ids.astype(jnp.int32), action_features, entropy

    mean = outputs["mean"]
    log_std = outputs["log_std"]
    if deterministic:
        pre_tanh = mean
    else:
        pre_tanh = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
    action = jnp.tanh(pre_tanh)
    entropy = jnp.sum(
        log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e),
        axis=-1,
    )
    return action, action, entropy


def dreamer_policy_action(
    actor_state: TrainState,
    features: jax.Array,
    config: DreamerV3Config,
) -> jax.Array:
    actions, _, _ = _actor_action(
        actor_state.apply_fn,
        actor_state.params,
        features,
        config,
        jax.random.PRNGKey(0),
        deterministic=True,
    )
    return actions


def _posterior_start_state(
    world_model_state: TrainState,
    batch: WorldModelSequenceBatch,
    config: DreamerV3Config,
) -> RSSMState:
    action_dtype = jnp.int32 if config.action_mode == "discrete" else jnp.float32
    outputs = world_model_state.apply_fn(
        world_model_state.params,
        jnp.asarray(batch.observations, dtype=jnp.float32),
        jnp.asarray(batch.actions, dtype=action_dtype),
        jnp.asarray(batch.is_first, dtype=bool),
    )
    return RSSMState(
        deterministic=jax.lax.stop_gradient(outputs["deterministic"][-1]),
        stochastic=jax.lax.stop_gradient(outputs["stochastic"][-1]),
        logits=jax.lax.stop_gradient(outputs["posterior_logits"][-1]),
    )


def imagine_dreamer_rollout(
    *,
    world_model_state: TrainState,
    actor_state: TrainState,
    critic_state: TrainState,
    start_state: RSSMState,
    config: DreamerV3Config,
    horizon: int,
    key: jax.Array,
    actor_params: Any | None = None,
    critic_params: Any | None = None,
) -> DreamerImaginedRollout:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    actor_params = actor_state.params if actor_params is None else actor_params
    critic_params = critic_state.params if critic_params is None else critic_params
    model = DreamerWorldModel(config)
    state = start_state
    actions = []
    features = []
    rewards = []
    continues = []
    values = []
    entropies = []
    for _ in range(horizon):
        key, action_key = jax.random.split(key)
        current_features = flatten_rssm_state(state)
        action, action_features, entropy = _actor_action(
            actor_state.apply_fn,
            actor_params,
            current_features,
            config,
            action_key,
            deterministic=False,
        )
        state, prediction = model.apply(
            world_model_state.params,
            state,
            action_features,
            method=model.imagine_step,
        )
        imagined_features = prediction["features"]
        value_logits = critic_state.apply_fn(
            {"params": critic_params}, imagined_features
        )
        actions.append(action)
        features.append(imagined_features)
        rewards.append(decode_two_hot_logits(prediction["reward_logits"]))
        continues.append(jax.nn.sigmoid(prediction["continue_logits"]))
        values.append(decode_two_hot_logits(value_logits))
        entropies.append(entropy)

    stacked_features = jnp.stack(features)
    stacked_rewards = jnp.stack(rewards)
    stacked_continues = jnp.stack(continues)
    stacked_values = jnp.stack(values)
    bootstrap_logits = critic_state.apply_fn(
        {"params": critic_params}, stacked_features[-1]
    )
    bootstrap = decode_two_hot_logits(bootstrap_logits)
    returns = lambda_returns(
        stacked_rewards,
        stacked_continues,
        stacked_values,
        bootstrap,
        discount_lambda=config.actor_critic.discount_lambda,
    )
    return DreamerImaginedRollout(
        actions=jnp.stack(actions),
        features=stacked_features,
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
    start_state: RSSMState,
    config: DreamerV3Config,
    horizon: int,
    key: jax.Array,
) -> jax.Array:
    rollout = imagine_dreamer_rollout(
        world_model_state=world_model_state,
        actor_state=actor_state,
        critic_state=critic_state,
        start_state=start_state,
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
    return -(objective + config.actor_critic.entropy_scale * entropy)


def _critic_loss(
    critic_params: Any,
    *,
    critic_state: TrainState,
    rollout: DreamerImaginedRollout,
    config: DreamerV3Config,
) -> jax.Array:
    features = jax.lax.stop_gradient(rollout.features)
    targets = two_hot(
        symlog(jax.lax.stop_gradient(rollout.returns)),
        num_bins=config.actor_critic.value_bins,
        lower=-20.0,
        upper=20.0,
    )
    logits = critic_state.apply_fn({"params": critic_params}, features)
    return -jnp.mean(jnp.sum(targets * jax.nn.log_softmax(logits, axis=-1), axis=-1))


def train_dreamer_actor_critic(
    *,
    world_model_state: TrainState,
    batch: WorldModelSequenceBatch,
    config: DreamerV3Config,
    train_steps: int,
    learning_rate: float,
    imagination_horizon: int | None = None,
    seed: int,
) -> tuple[
    TrainState,
    TrainState,
    list[dict[str, float]],
    DreamerImaginedRollout,
]:
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    horizon = imagination_horizon or config.actor_critic.imagination_horizon
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    actor_state, critic_state = create_dreamer_actor_critic_states(
        init_key,
        config,
        learning_rate=learning_rate,
    )
    start_state = _posterior_start_state(world_model_state, batch, config)
    metrics: list[dict[str, float]] = []
    rollout = None
    for step in range(train_steps):
        key, actor_key, rollout_key = jax.random.split(key, 3)
        actor_loss, actor_grads = jax.value_and_grad(_actor_loss)(
            actor_state.params,
            world_model_state=world_model_state,
            actor_state=actor_state,
            critic_state=critic_state,
            start_state=start_state,
            config=config,
            horizon=horizon,
            key=actor_key,
        )
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        rollout = imagine_dreamer_rollout(
            world_model_state=world_model_state,
            actor_state=actor_state,
            critic_state=critic_state,
            start_state=start_state,
            config=config,
            horizon=horizon,
            key=rollout_key,
        )
        critic_loss, critic_grads = jax.value_and_grad(_critic_loss)(
            critic_state.params,
            critic_state=critic_state,
            rollout=rollout,
            config=config,
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
        entropies=jnp.zeros_like(rewards),
    )
