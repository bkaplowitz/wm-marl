"""Imagined-rollout PPO for the generative world-model arms.

The coins pipeline's ``_imagined_rollout`` takes ``reward_done_fn`` as a static
jit argument, which structurally rules out learned reward/continue heads —
their parameters change every update. This module re-implements the imagination
loop for the single-agent setting with the head TrainState threaded through as
a traced argument instead. Known caveat shared with all state-conditioned done
heads: time-limit truncations are not predictable from (s, a), so the continue
head only models genuine terminations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from world_marl.algs.gae import compute_gae
from world_marl.algs.networks import MLPActorCritic
from world_marl.genwm.models import GaussianMLPActorCritic, RewardContinueHead
from world_marl.genwm.tokenizer import QuantileTokenizer
from world_marl.genwm.world_model import (
    GenWMConfig,
    action_features,
    genwm_predict_next,
)


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    action_low: float = -1.0
    action_high: float = 1.0


class ImaginedBatch(NamedTuple):
    observations: jnp.ndarray  # [T, N, obs_dim]
    actions: jnp.ndarray  # [T, N] int32 or [T, N, action_dim] float32
    log_probs: jnp.ndarray  # [T, N]
    rewards: jnp.ndarray  # [T, N]
    dones: jnp.ndarray  # [T, N]
    values: jnp.ndarray  # [T, N]


def create_policy_state(
    key: jax.Array,
    config: GenWMConfig,
    ppo_config: PPOConfig,
) -> TrainState:
    if config.action_mode == "discrete":
        network = MLPActorCritic(action_dim=config.action_dim)
    else:
        network = GaussianMLPActorCritic(action_dim=config.action_dim)
    params = network.init(key, jnp.zeros((1, config.obs_dim), dtype=jnp.float32))[
        "params"
    ]
    tx = optax.chain(
        optax.clip_by_global_norm(ppo_config.max_grad_norm),
        optax.adam(ppo_config.learning_rate, eps=1e-5),
    )
    return TrainState.create(apply_fn=network.apply, params=params, tx=tx)


def create_head_state(
    key: jax.Array,
    config: GenWMConfig,
    *,
    hidden_dims: tuple[int, ...] = (256, 256),
    learning_rate: float = 1e-3,
) -> TrainState:
    network = RewardContinueHead(hidden_dims=hidden_dims)
    params = network.init(
        key,
        jnp.zeros((1, config.obs_dim), dtype=jnp.float32),
        jnp.zeros((1, config.action_dim), dtype=jnp.float32),
    )["params"]
    return TrainState.create(
        apply_fn=network.apply, params=params, tx=optax.adam(learning_rate)
    )


@jax.jit
def head_train_step(
    head_state: TrainState,
    observations: jax.Array,
    action_feats: jax.Array,
    rewards: jax.Array,
    continues: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """One update of the reward/continue head on real (s, a, r, 1-done) data."""

    def loss_fn(params):
        pred_reward, continue_logit = head_state.apply_fn(
            {"params": params}, observations, action_feats
        )
        reward_loss = jnp.mean(jnp.square(pred_reward - rewards))
        continue_loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(continue_logit, continues)
        )
        total = reward_loss + continue_loss
        return total, {
            "head_total_loss": total,
            "head_reward_loss": reward_loss,
            "head_continue_loss": continue_loss,
        }

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(head_state.params)
    return head_state.apply_gradients(grads=grads), metrics


@partial(jax.jit, static_argnames=("horizon", "config", "ppo_config"))
def imagined_rollout(
    policy_state: TrainState,
    wm_state: TrainState,
    head_state: TrainState,
    obs_tokenizer: QuantileTokenizer,
    action_tokenizer: QuantileTokenizer | None,
    start_observations: jax.Array,
    rng: jax.Array,
    *,
    horizon: int,
    config: GenWMConfig,
    ppo_config: PPOConfig,
) -> tuple[ImaginedBatch, jax.Array]:
    """Roll the policy through the generative model for ``horizon`` steps.

    Starts from real replay observations ``(N, obs_dim)``; the world model
    supplies next observations, the learned head supplies rewards and dones.
    Continuous actions are clipped to the PPO bounds before entering the model,
    while the stored actions/log-probs stay pre-clip so the PPO ratio is exact.
    """

    def step(carry, _):
        observations, key = carry
        key, action_key, model_key = jax.random.split(key, 3)
        policy, values = policy_state.apply_fn(
            {"params": policy_state.params}, observations
        )
        actions = policy.sample(seed=action_key)
        log_probs = policy.log_prob(actions)
        if config.action_mode == "discrete":
            env_actions = actions
        else:
            env_actions = jnp.clip(
                actions, ppo_config.action_low, ppo_config.action_high
            )
        next_observations = genwm_predict_next(
            wm_state,
            model_key,
            observations,
            env_actions,
            obs_tokenizer,
            action_tokenizer,
            config,
        )
        reward, continue_logit = head_state.apply_fn(
            {"params": head_state.params},
            observations,
            action_features(env_actions, config),
        )
        done = (jax.nn.sigmoid(continue_logit) < 0.5).astype(jnp.float32)
        ys = (observations, actions, log_probs, reward, done, values)
        return (next_observations, key), ys

    (last_observations, _), stacked = jax.lax.scan(
        step, (start_observations, rng), None, length=horizon
    )
    _, last_values = policy_state.apply_fn(
        {"params": policy_state.params}, last_observations
    )
    return ImaginedBatch(*stacked), last_values


@partial(jax.jit, static_argnames=("config",))
def ppo_update(
    policy_state: TrainState,
    batch: ImaginedBatch,
    last_values: jax.Array,
    rng: jax.Array,
    config: PPOConfig,
) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    """Single-agent clipped-PPO update; action-shape-agnostic twin of ippo's.

    Differs from ``algs.ippo.ppo_update`` only in dropping the negative-control
    knobs and reshaping actions as ``(batch, *action_shape)`` so diag-Gaussian
    action vectors work alongside categorical scalars.
    """
    advantages, targets = compute_gae(
        rewards=batch.rewards,
        values=batch.values,
        dones=batch.dones,
        last_values=last_values,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    batch_size = batch.actions.shape[0] * batch.actions.shape[1]
    if batch_size % config.num_minibatches != 0:
        raise ValueError("rollout actors * steps must divide evenly into minibatches")
    minibatch_size = batch_size // config.num_minibatches

    flat = {
        "observations": batch.observations.reshape(
            (batch_size, *batch.observations.shape[2:])
        ),
        "actions": batch.actions.reshape((batch_size, *batch.actions.shape[2:])),
        "old_log_probs": batch.log_probs.reshape((batch_size,)),
        "old_values": batch.values.reshape((batch_size,)),
        "advantages": advantages.reshape((batch_size,)),
        "targets": targets.reshape((batch_size,)),
    }

    def loss_fn(params, minibatch):
        policy, values = policy_state.apply_fn(
            {"params": params},
            minibatch["observations"],
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
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params,
            minibatch,
        )
        return state.apply_gradients(grads=grads), metrics

    def update_epoch(state, epoch_rng):
        permutation = jax.random.permutation(epoch_rng, batch_size)
        shuffled = jax.tree_util.tree_map(lambda x: x[permutation], flat)
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((config.num_minibatches, minibatch_size, *x.shape[1:])),
            shuffled,
        )
        return jax.lax.scan(update_minibatch, state, minibatches)

    epoch_rngs = jax.random.split(rng, config.update_epochs)
    policy_state, metrics = jax.lax.scan(update_epoch, policy_state, epoch_rngs)
    return policy_state, jax.tree_util.tree_map(lambda x: x.mean(), metrics)
