"""Vector-state world model utilities for model-based PPO rollouts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from flow_matching.paths import conditional_vector_field, sample_conditional_path
from world_marl.algs.ippo import RolloutBatch
from world_marl.training import RolloutResult


class VectorTransitionBatch(NamedTuple):
    states: jnp.ndarray
    actions: jnp.ndarray
    next_states: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    policy_ids: jnp.ndarray


@dataclass(frozen=True)
class VectorWorldModelConfig:
    state_dim: int
    num_agents: int
    action_dim: int
    num_policies: int = 1
    hidden_dims: tuple[int, ...] = (128, 128)
    learning_rate: float = 1e-3
    reward_loss_coef: float = 1.0
    done_loss_coef: float = 1.0
    integration_steps: int = 8


class VectorWorldModel(nn.Module):
    state_dim: int
    num_agents: int
    action_dim: int
    num_policies: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(
        self,
        xt: jnp.ndarray,
        t: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        policy_ids: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        flat_xt = xt.reshape((xt.shape[0], -1))
        flat_states = states.reshape((states.shape[0], -1))
        action_features = jax.nn.one_hot(actions, self.action_dim).reshape(
            (actions.shape[0], -1)
        )
        policy_features = jax.nn.one_hot(policy_ids, self.num_policies)
        x = jnp.concatenate(
            [flat_xt, t, flat_states, action_features, policy_features],
            axis=-1,
        )
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.silu(x)

        flow = nn.Dense(self.num_agents * self.state_dim, name="flow")(x)
        reward = nn.Dense(self.num_agents, name="reward")(x)
        done_logits = nn.Dense(self.num_agents, name="done_logits")(x)
        return flow.reshape((-1, self.num_agents, self.state_dim)), reward, done_logits


def create_world_model_state(
    key: jax.Array,
    config: VectorWorldModelConfig,
) -> TrainState:
    model = VectorWorldModel(
        state_dim=config.state_dim,
        num_agents=config.num_agents,
        action_dim=config.action_dim,
        num_policies=config.num_policies,
        hidden_dims=config.hidden_dims,
    )
    init_xt = jnp.zeros((1, config.num_agents, config.state_dim), dtype=jnp.float32)
    init_t = jnp.zeros((1, 1), dtype=jnp.float32)
    init_actions = jnp.zeros((1, config.num_agents), dtype=jnp.int32)
    init_policy_ids = jnp.zeros((1,), dtype=jnp.int32)
    params = model.init(
        key,
        init_xt,
        init_t,
        init_xt,
        init_actions,
        init_policy_ids,
    )["params"]
    tx = optax.adam(config.learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def world_model_loss(
    params: Any,
    apply_fn: Any,
    key: jax.Array,
    batch: VectorTransitionBatch,
    config: VectorWorldModelConfig,
) -> jnp.ndarray:
    key_t, key_xt = jax.random.split(key)
    batch_size = batch.states.shape[0]
    t = jax.random.uniform(key_t, shape=(batch_size, 1))
    state_t = t.reshape((batch_size, 1, 1))
    xt = sample_conditional_path(key_xt, batch.next_states, state_t)
    target_flow = conditional_vector_field(xt, batch.next_states, state_t)
    pred_flow, pred_rewards, pred_done_logits = apply_fn(
        {"params": params},
        xt,
        t,
        batch.states,
        batch.actions,
        batch.policy_ids,
    )
    flow_loss = jnp.mean(jnp.square(pred_flow - target_flow))
    reward_loss = jnp.mean(jnp.square(pred_rewards - batch.rewards))
    done_loss = jnp.mean(
        optax.sigmoid_binary_cross_entropy(pred_done_logits, batch.dones)
    )
    return (
        flow_loss
        + config.reward_loss_coef * reward_loss
        + config.done_loss_coef * done_loss
    )


@partial(jax.jit, static_argnames="config")
def train_world_model_step(
    state: TrainState,
    key: jax.Array,
    batch: VectorTransitionBatch,
    config: VectorWorldModelConfig,
) -> tuple[TrainState, jnp.ndarray]:
    loss, grads = jax.value_and_grad(world_model_loss)(
        state.params,
        state.apply_fn,
        key,
        batch,
        config,
    )
    return state.apply_gradients(grads=grads), loss


def predict_next(
    state: TrainState,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    policy_ids: jnp.ndarray,
    config: VectorWorldModelConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    xt = jax.random.normal(jax.random.PRNGKey(0), states.shape)
    ts = jnp.linspace(0.0, 1.0, config.integration_steps + 1)
    current = xt
    for step_index in range(config.integration_steps):
        t = jnp.full((states.shape[0], 1), ts[step_index])
        dt = ts[step_index + 1] - ts[step_index]
        flow, _, _ = state.apply_fn(
            {"params": state.params},
            current,
            t,
            states,
            actions,
            policy_ids,
        )
        current = current + flow * dt
    flow, rewards, done_logits = state.apply_fn(
        {"params": state.params},
        current,
        jnp.ones((states.shape[0], 1)),
        states,
        actions,
        policy_ids,
    )
    del flow
    return current, rewards, jax.nn.sigmoid(done_logits)


def simulate_ippo_model_rollout(
    model_state: TrainState,
    policy_state: TrainState,
    initial_states: jnp.ndarray,
    rng: jax.Array,
    *,
    rollout_steps: int,
    config: VectorWorldModelConfig,
    policy_id: int = 0,
) -> RolloutResult:
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")

    obs_rows = []
    action_rows = []
    log_prob_rows = []
    reward_rows = []
    done_rows = []
    value_rows = []
    current_states = initial_states
    num_envs = current_states.shape[0]
    num_actors = num_envs * config.num_agents

    for _ in range(rollout_steps):
        flat_states = current_states.reshape((num_actors, config.state_dim))
        rng, action_key = jax.random.split(rng)
        policy, values = policy_state.apply_fn(
            {"params": policy_state.params},
            flat_states,
        )
        actions = policy.sample(seed=action_key).astype(jnp.int32)
        log_probs = policy.log_prob(actions)
        env_actions = actions.reshape((num_envs, config.num_agents))
        policy_ids = jnp.full((num_envs,), policy_id, dtype=jnp.int32)
        next_states, rewards, done_probs = predict_next(
            model_state,
            current_states,
            env_actions,
            policy_ids,
            config,
        )

        obs_rows.append(flat_states)
        action_rows.append(actions)
        log_prob_rows.append(log_probs)
        reward_rows.append(rewards.reshape((num_actors,)))
        done_rows.append((done_probs > 0.5).astype(jnp.float32).reshape((num_actors,)))
        value_rows.append(values)
        current_states = next_states

    last_values = policy_state.apply_fn(
        {"params": policy_state.params},
        current_states.reshape((num_actors, config.state_dim)),
    )[1]
    batch = RolloutBatch(
        observations=jnp.stack(obs_rows, axis=0),
        actions=jnp.stack(action_rows, axis=0),
        log_probs=jnp.stack(log_prob_rows, axis=0),
        rewards=jnp.stack(reward_rows, axis=0),
        dones=jnp.stack(done_rows, axis=0),
        values=jnp.stack(value_rows, axis=0),
    )
    return RolloutResult(
        batch=batch,
        next_observations=current_states,
        last_values=last_values,
        metrics={"model_rollout_mean_reward": float(jnp.mean(batch.rewards))},
    )
