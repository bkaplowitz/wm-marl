"""Reusable MAPPO update code with a centralized critic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from world_marl.algs.gae import compute_gae
from world_marl.algs.networks import CNNMAPPOActorCritic


@dataclass(frozen=True)
class MAPPOConfig:
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    activation: str = "relu"
    shuffle_rewards: bool = False
    zero_advantages: bool = False


class MAPPORolloutBatch(NamedTuple):
    observations: jnp.ndarray
    central_observations: jnp.ndarray
    actions: jnp.ndarray
    log_probs: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    values: jnp.ndarray


def create_train_state(
    rng: jax.Array,
    observation_shape: tuple[int, int, int],
    central_observation_shape: tuple[int, int, int],
    action_dim: int,
    config: MAPPOConfig,
) -> TrainState:
    """Initialize the MAPPO actor-critic network and optimizer."""
    network = CNNMAPPOActorCritic(action_dim=action_dim, activation=config.activation)
    init_observation = jnp.zeros((1, *observation_shape), dtype=jnp.float32)
    init_central_observation = jnp.zeros(
        (1, *central_observation_shape),
        dtype=jnp.float32,
    )
    params = network.init(rng, init_observation, init_central_observation)["params"]
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate, eps=1e-5),
    )
    return TrainState.create(apply_fn=network.apply, params=params, tx=tx)


def apply_actor_critic(
    train_state: TrainState,
    observations: jnp.ndarray,
    central_observations: jnp.ndarray,
):
    """Apply the MAPPO actor and centralized critic."""
    return train_state.apply_fn(
        {"params": train_state.params},
        observations,
        central_observations,
    )


def select_actions(
    train_state: TrainState,
    rng: jax.Array,
    observations: jnp.ndarray,
    central_observations: jnp.ndarray,
    *,
    deterministic: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Select actions with local observations and value them centrally."""
    policy, values = apply_actor_critic(
        train_state,
        observations,
        central_observations,
    )
    if deterministic:
        actions = jnp.argmax(policy.logits, axis=-1)
    else:
        actions = policy.sample(seed=rng)
    return actions.astype(jnp.int32), policy.log_prob(actions), values


def mappo_update(
    train_state: TrainState,
    batch: MAPPORolloutBatch,
    last_values: jnp.ndarray,
    rng: jax.Array,
    config: MAPPOConfig,
) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    """Run one MAPPO update over a rollout batch."""
    rewards = batch.rewards
    if config.shuffle_rewards:
        flat_rewards = rewards.reshape((-1,))
        flat_rewards = jax.random.permutation(rng, flat_rewards)
        rewards = flat_rewards.reshape(rewards.shape)

    advantages, targets = compute_gae(
        rewards=rewards,
        values=batch.values,
        dones=batch.dones,
        last_values=last_values,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    if config.zero_advantages:
        advantages = jnp.zeros_like(advantages)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    batch_size = batch.actions.shape[0] * batch.actions.shape[1]
    if batch_size % config.num_minibatches != 0:
        raise ValueError("rollout actors * steps must divide evenly into minibatches")
    minibatch_size = batch_size // config.num_minibatches

    flat = {
        "observations": batch.observations.reshape(
            (batch_size, *batch.observations.shape[2:])
        ),
        "central_observations": batch.central_observations.reshape(
            (batch_size, *batch.central_observations.shape[2:])
        ),
        "actions": batch.actions.reshape((batch_size,)),
        "old_log_probs": batch.log_probs.reshape((batch_size,)),
        "old_values": batch.values.reshape((batch_size,)),
        "advantages": advantages.reshape((batch_size,)),
        "targets": targets.reshape((batch_size,)),
    }

    def loss_fn(params, minibatch):
        policy, values = train_state.apply_fn(
            {"params": params},
            minibatch["observations"],
            minibatch["central_observations"],
        )
        log_probs = policy.log_prob(minibatch["actions"])
        ratio = jnp.exp(log_probs - minibatch["old_log_probs"])

        actor_loss_1 = ratio * minibatch["advantages"]
        actor_loss_2 = (
            jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps)
            * minibatch["advantages"]
        )
        actor_loss = -jnp.mean(jnp.minimum(actor_loss_1, actor_loss_2))

        value_pred_clipped = minibatch["old_values"] + jnp.clip(
            values - minibatch["old_values"],
            -config.clip_eps,
            config.clip_eps,
        )
        value_losses = jnp.square(values - minibatch["targets"])
        value_losses_clipped = jnp.square(value_pred_clipped - minibatch["targets"])
        value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))
        entropy = jnp.mean(policy.entropy())
        total_loss = (
            actor_loss + config.vf_coef * value_loss - config.ent_coef * entropy
        )

        approx_kl = jnp.mean(minibatch["old_log_probs"] - log_probs)
        clip_fraction = jnp.mean(
            (jnp.abs(ratio - 1.0) > config.clip_eps).astype(jnp.float32)
        )
        return total_loss, {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    def update_minibatch(state, minibatch):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params,
            minibatch,
        )
        del loss
        return state.apply_gradients(grads=grads), metrics

    def update_epoch(carry, epoch_rng):
        state = carry
        permutation = jax.random.permutation(epoch_rng, batch_size)
        shuffled = jax.tree_util.tree_map(lambda x: x[permutation], flat)
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((config.num_minibatches, minibatch_size, *x.shape[1:])),
            shuffled,
        )
        state, metrics = jax.lax.scan(update_minibatch, state, minibatches)
        return state, metrics

    epoch_rngs = jax.random.split(rng, config.update_epochs)
    train_state, metrics = jax.lax.scan(update_epoch, train_state, epoch_rngs)
    metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics)
    metrics["advantages_mean"] = jnp.mean(advantages)
    metrics["targets_mean"] = jnp.mean(targets)
    return train_state, metrics
