"""Vector-state world-model glue for model-based PPO rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from flow_matching.models import MLPVectorField
from flow_matching.paths import conditional_vector_field, sample_conditional_path
from flow_matching.simulate import euler_integrate
from flow_matching.train import create_train_state as create_flow_train_state
from world_marl.algs.ippo import RolloutBatch
from world_marl.algs.mappo import MAPPORolloutBatch
from world_marl.training import RolloutResult


class VectorTransitionBatch(NamedTuple):
    states: jnp.ndarray
    actions: jnp.ndarray
    next_states: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray


@dataclass(frozen=True)
class VectorWorldModelConfig:
    state_dim: int
    num_agents: int
    action_dim: int
    hidden_dims: tuple[int, ...] = (128, 128)
    learning_rate: float = 1e-3
    reward_loss_coef: float = 1.0
    done_loss_coef: float = 1.0
    integration_steps: int = 8


def create_world_model_state(
    key: jax.Array,
    config: VectorWorldModelConfig,
) -> TrainState:
    model = MLPVectorField(hidden_dims=config.hidden_dims)
    return create_flow_train_state(
        key,
        model,
        config.learning_rate,
        dim=_model_input_dim(config),
    )


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
    target_transition = _pack_transition(
        batch.next_states,
        batch.rewards,
        batch.dones,
        config,
    )
    xt_transition = sample_conditional_path(key_xt, target_transition, t)
    target_flow = conditional_vector_field(xt_transition, target_transition, t)
    context = _pack_context(batch.states, batch.actions, config)
    model_input = jnp.concatenate([xt_transition, context], axis=-1)
    pred_flow = apply_fn({"params": params}, model_input, t)[
        :, : _transition_dim(config)
    ]

    state_end = _flat_state_dim(config)
    reward_end = state_end + config.num_agents
    state_loss = jnp.mean(
        jnp.square(pred_flow[:, :state_end] - target_flow[:, :state_end])
    )
    reward_loss = jnp.mean(
        jnp.square(
            pred_flow[:, state_end:reward_end] - target_flow[:, state_end:reward_end]
        )
    )
    done_loss = jnp.mean(
        jnp.square(pred_flow[:, reward_end:] - target_flow[:, reward_end:])
    )
    return (
        state_loss
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
    key: jax.Array,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    config: VectorWorldModelConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    context = _pack_context(states, actions, config)
    x0 = jax.random.normal(key, (states.shape[0], _transition_dim(config)))
    ts = jnp.linspace(0.0, 1.0, config.integration_steps + 1)

    def drift(transition_xt: jax.Array, t: jax.Array) -> jax.Array:
        model_input = jnp.concatenate([transition_xt, context], axis=-1)
        tt = jnp.full((transition_xt.shape[0], 1), t)
        return state.apply_fn({"params": state.params}, model_input, tt)[
            :, : _transition_dim(config)
        ]

    transition = euler_integrate(drift, x0, ts)[-1]
    next_states, rewards, dones = _unpack_transition(transition, config)
    return next_states, rewards, jnp.clip(dones, 0.0, 1.0)


def simulate_ippo_model_rollout(
    model_state: TrainState,
    policy_state: TrainState,
    initial_states: jnp.ndarray,
    rng: jax.Array,
    *,
    rollout_steps: int,
    config: VectorWorldModelConfig,
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
        rng, action_key, model_key = jax.random.split(rng, 3)
        # Compute distribution of actions and values for the current state from current policy.
        policy, values = policy_state.apply_fn(
            {"params": policy_state.params},
            flat_states,
        )
        # Sample an action from the policy distribution for each agent.
        actions = policy.sample(seed=action_key).astype(jnp.int32)
        log_probs = policy.log_prob(actions)
        env_actions = actions.reshape((num_envs, config.num_agents))
        # Use model to predict next states, rewards, and done probabilities.
        next_states, rewards, done_probs = predict_next(
            model_state,
            model_key,
            current_states,
            env_actions,
            config,
        )
        # Store the current state, action, log probability, reward, done probability, and value.
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
        metrics={
            "rollout_mean_reward": float(jnp.mean(batch.rewards)),
            "model_rollout_mean_reward": float(jnp.mean(batch.rewards)),
        },
    )


def simulate_mappo_model_rollout(
    model_state: TrainState,
    policy_state: TrainState,
    initial_states: jnp.ndarray,
    rng: jax.Array,
    *,
    rollout_steps: int,
    config: VectorWorldModelConfig,
) -> RolloutResult:
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")

    obs_rows = []
    central_obs_rows = []
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
        central_states = _vector_central_states(current_states).reshape(
            (num_actors, -1)
        )
        rng, action_key, model_key = jax.random.split(rng, 3)
        policy, values = policy_state.apply_fn(
            {"params": policy_state.params},
            flat_states,
            central_states,
        )
        actions = policy.sample(seed=action_key).astype(jnp.int32)
        log_probs = policy.log_prob(actions)
        env_actions = actions.reshape((num_envs, config.num_agents))
        next_states, rewards, done_probs = predict_next(
            model_state,
            model_key,
            current_states,
            env_actions,
            config,
        )

        obs_rows.append(flat_states)
        central_obs_rows.append(central_states)
        action_rows.append(actions)
        log_prob_rows.append(log_probs)
        reward_rows.append(rewards.reshape((num_actors,)))
        done_rows.append((done_probs > 0.5).astype(jnp.float32).reshape((num_actors,)))
        value_rows.append(values)
        current_states = next_states

    last_central_states = _vector_central_states(current_states).reshape(
        (num_actors, -1)
    )
    last_values = policy_state.apply_fn(
        {"params": policy_state.params},
        current_states.reshape((num_actors, config.state_dim)),
        last_central_states,
    )[1]
    batch = MAPPORolloutBatch(
        observations=jnp.stack(obs_rows, axis=0),
        central_observations=jnp.stack(central_obs_rows, axis=0),
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
        metrics={
            "rollout_mean_reward": float(jnp.mean(batch.rewards)),
            "model_rollout_mean_reward": float(jnp.mean(batch.rewards)),
        },
    )


def _pack_context(
    states: jnp.ndarray,
    actions: jnp.ndarray,
    config: VectorWorldModelConfig,
) -> jnp.ndarray:
    flat_states = states.reshape((states.shape[0], _flat_state_dim(config)))
    action_features = jax.nn.one_hot(actions, config.action_dim).reshape(
        (actions.shape[0], config.num_agents * config.action_dim)
    )
    return jnp.concatenate([flat_states, action_features], axis=-1)


def _pack_transition(
    next_states: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    config: VectorWorldModelConfig,
) -> jnp.ndarray:
    flat_next_states = next_states.reshape(
        (next_states.shape[0], _flat_state_dim(config))
    )
    return jnp.concatenate(
        [
            flat_next_states,
            rewards.reshape((rewards.shape[0], config.num_agents)),
            dones.reshape((dones.shape[0], config.num_agents)),
        ],
        axis=-1,
    )


def _unpack_transition(
    transition: jnp.ndarray,
    config: VectorWorldModelConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    state_end = _flat_state_dim(config)
    reward_end = state_end + config.num_agents
    next_states = transition[:, :state_end].reshape(
        (transition.shape[0], config.num_agents, config.state_dim)
    )
    rewards = transition[:, state_end:reward_end]
    dones = transition[:, reward_end : reward_end + config.num_agents]
    return next_states, rewards, dones


def _flat_state_dim(config: VectorWorldModelConfig) -> int:
    return config.num_agents * config.state_dim


def _transition_dim(config: VectorWorldModelConfig) -> int:
    return _flat_state_dim(config) + 2 * config.num_agents


def _context_dim(config: VectorWorldModelConfig) -> int:
    return _flat_state_dim(config) + config.num_agents * config.action_dim


def _model_input_dim(config: VectorWorldModelConfig) -> int:
    return _transition_dim(config) + _context_dim(config)


def _vector_central_states(states: jnp.ndarray) -> jnp.ndarray:
    num_envs, num_agents, state_dim = states.shape
    central = states.reshape((num_envs, num_agents * state_dim))
    central = jnp.repeat(central[:, None, :], repeats=num_agents, axis=1)
    target_ids = jnp.broadcast_to(
        jnp.eye(num_agents, dtype=jnp.float32)[None, :, :],
        (num_envs, num_agents, num_agents),
    )
    return jnp.concatenate([central, target_ids], axis=-1)
