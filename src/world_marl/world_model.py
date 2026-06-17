"""Vector-state world-model glue for model-based PPO rollouts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from flow_matching.models import MLPVectorField
from flow_matching.simulate import sample_conditioned_flow
from flow_matching.train import (
    conditioned_flow_matching_loss,
    conditioned_train_step,
    create_conditioned_train_state,
)
from world_marl.algs.ippo import RolloutBatch
from world_marl.algs.mappo import MAPPORolloutBatch
from world_marl.training import RolloutResult, build_vector_central

# (states, env_actions, next_states) -> (rewards, dones), each [env, agent].
RewardDoneFn = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, jnp.ndarray],
]


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
    integration_steps: int = 8
    flow_type: str = "gaussian"


def create_world_model_state(
    key: jax.Array,
    config: VectorWorldModelConfig,
) -> TrainState:
    model = MLPVectorField(hidden_dims=config.hidden_dims)
    return create_conditioned_train_state(
        key,
        model,
        config.learning_rate,
        dim=_transition_dim(config),
        cond_dim=_cond_dim(config),
    )


def world_model_loss(
    params: Any,
    apply_fn: Any,
    key: jax.Array,
    batch: VectorTransitionBatch,
    config: VectorWorldModelConfig,
) -> jnp.ndarray:
    x1 = _pack_transition(batch.next_states, config)
    cond_vars = _pack_cond_vars(batch.states, batch.actions, config)
    return conditioned_flow_matching_loss(
        params, apply_fn, key, x1, cond_vars, config.flow_type
    )


@partial(jax.jit, static_argnames="config")
def train_world_model_step(
    state: TrainState,
    key: jax.Array,
    batch: VectorTransitionBatch,
    config: VectorWorldModelConfig,
) -> tuple[TrainState, jnp.ndarray]:
    x1 = _pack_transition(batch.next_states, config)
    cond_vars = _pack_cond_vars(batch.states, batch.actions, config)
    return conditioned_train_step(state, key, x1, cond_vars, config.flow_type)


def predict_next(
    state: TrainState,
    key: jax.Array,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    config: VectorWorldModelConfig,
) -> jnp.ndarray:
    """Sample next-states from the conditioned flow (next-state only)."""
    cond_vars = _pack_cond_vars(states, actions, config)
    transition = sample_conditioned_flow(
        state.apply_fn,
        state.params,
        key,
        cond_vars,
        dim=_transition_dim(config),
        steps=config.integration_steps,
    )
    return _unpack_transition(transition, config)


def simulate_ippo_model_rollout(
    model_state: TrainState,
    policy_state: TrainState,
    initial_states: jnp.ndarray,
    rng: jax.Array,
    *,
    rollout_steps: int,
    config: VectorWorldModelConfig,
    reward_done_fn: RewardDoneFn,
) -> RolloutResult:
    return _simulate_model_rollout(
        model_state,
        policy_state,
        initial_states,
        rng,
        rollout_steps=rollout_steps,
        config=config,
        algorithm="ippo",
        reward_done_fn=reward_done_fn,
    )


def simulate_mappo_model_rollout(
    model_state: TrainState,
    policy_state: TrainState,
    initial_states: jnp.ndarray,
    rng: jax.Array,
    *,
    rollout_steps: int,
    config: VectorWorldModelConfig,
    reward_done_fn: RewardDoneFn,
) -> RolloutResult:
    return _simulate_model_rollout(
        model_state,
        policy_state,
        initial_states,
        rng,
        rollout_steps=rollout_steps,
        config=config,
        algorithm="mappo",
        reward_done_fn=reward_done_fn,
    )


def _simulate_model_rollout(
    model_state: TrainState,
    policy_state: TrainState,
    initial_states: jnp.ndarray,
    rng: jax.Array,
    *,
    rollout_steps: int,
    config: VectorWorldModelConfig,
    algorithm: str,
    reward_done_fn: RewardDoneFn,
) -> RolloutResult:
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")
    if algorithm not in {"ippo", "mappo"}:
        raise ValueError(f"unsupported algorithm {algorithm!r}")
    is_mappo = algorithm == "mappo"

    stacked, final_states, last_values = _imagined_rollout(
        model_state,
        policy_state,
        initial_states,
        rng,
        rollout_steps=rollout_steps,
        config=config,
        is_mappo=is_mappo,
        reward_done_fn=reward_done_fn,
    )

    common = {
        "observations": stacked["observations"],
        "actions": stacked["actions"],
        "log_probs": stacked["log_probs"],
        "rewards": stacked["rewards"],
        "dones": stacked["dones"],
        "values": stacked["values"],
    }
    if is_mappo:
        batch = MAPPORolloutBatch(
            central_observations=stacked["central_observations"],
            **common,
        )
    else:
        batch = RolloutBatch(**common)
    # float() metrics stay on the host, outside the jitted scan above.
    mean_reward = float(jnp.mean(batch.rewards))
    return RolloutResult(
        batch=batch,
        next_observations=final_states,
        last_values=last_values,
        metrics={
            "rollout_mean_reward": mean_reward,
            "model_rollout_mean_reward": mean_reward,
        },
    )


@partial(
    jax.jit,
    static_argnames=("rollout_steps", "config", "is_mappo", "reward_done_fn"),
)
def _imagined_rollout(
    model_state: TrainState,
    policy_state: TrainState,
    initial_states: jnp.ndarray,
    rng: jax.Array,
    *,
    rollout_steps: int,
    config: VectorWorldModelConfig,
    is_mappo: bool,
    reward_done_fn: RewardDoneFn,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Fused imagined rollout: one ``lax.scan`` step per imagined timestep. This prevents interdevice unloading and loading that would otherwise slow down the runtime substantially and keeps all training on the GPU.

    The carry is ``(rng, current_states)`` and ``scan`` stacks every per-step
    transition along axis 0.``.
    This is safe:
    - ``reward_done_fn`` is a static argument because it is a plain callable, not a
    pytree
    - a module-level provider such as ``coin_game_reward_done`` keeps a
    stable identity, so this compiles once and reuses the cache across every PPO
    update
    - The inner Euler integrator in ``predict_next`` is itself a
    ``lax.scan``, so the two nest.

    """
    num_envs = initial_states.shape[0]
    num_actors = num_envs * config.num_agents

    def step(carry, _):
        rng, current_states = carry
        flat_states = current_states.reshape((num_actors, config.state_dim))
        central_states = (
            build_vector_central(current_states, jnp).reshape((num_actors, -1))
            if is_mappo
            else None
        )
        rng, action_key, model_key = jax.random.split(rng, 3)
        # Distribution over actions and value estimates from the current policy.
        policy, values = _apply_vector_policy(policy_state, flat_states, central_states)
        actions = policy.sample(seed=action_key).astype(jnp.int32)
        log_probs = policy.log_prob(actions)
        env_actions = actions.reshape((num_envs, config.num_agents))
        # World model supplies next-states; rewards/dones come from the callback.
        next_states = predict_next(
            model_state, model_key, current_states, env_actions, config
        )
        rewards, dones = _reward_done(
            reward_done_fn, current_states, env_actions, next_states, config
        )
        outputs = {
            "observations": flat_states,
            "actions": actions,
            "log_probs": log_probs,
            "rewards": rewards.reshape((num_actors,)),
            "dones": dones.reshape((num_actors,)),
            "values": values,
        }
        if is_mappo:
            outputs["central_observations"] = central_states
        return (rng, next_states), outputs

    (rng, final_states), stacked = jax.lax.scan(
        step, (rng, initial_states), xs=None, length=rollout_steps
    )

    last_flat = final_states.reshape((num_actors, config.state_dim))
    last_central = (
        build_vector_central(final_states, jnp).reshape((num_actors, -1))
        if is_mappo
        else None
    )
    last_values = _apply_vector_policy(policy_state, last_flat, last_central)[1]
    return stacked, final_states, last_values


def _apply_vector_policy(
    policy_state: TrainState,
    flat_states: jnp.ndarray,
    central_states: jnp.ndarray | None,
) -> tuple[Any, jnp.ndarray]:
    """Apply an MLP policy, passing central observations only for MAPPO."""
    if central_states is None:
        return policy_state.apply_fn({"params": policy_state.params}, flat_states)
    return policy_state.apply_fn(
        {"params": policy_state.params}, flat_states, central_states
    )


def _reward_done(
    reward_done_fn: RewardDoneFn,
    states: jnp.ndarray,
    env_actions: jnp.ndarray,
    next_states: jnp.ndarray,
    config: VectorWorldModelConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rewards/dones from the callback. None is not acceptable"""
    rewards, dones = reward_done_fn(states, env_actions, next_states)
    return (
        jnp.asarray(rewards, dtype=jnp.float32),
        jnp.asarray(dones, dtype=jnp.float32),
    )


def _pack_cond_vars(
    states: jnp.ndarray,
    actions: jnp.ndarray,
    config: VectorWorldModelConfig,
) -> jnp.ndarray:
    flat_actions_dim = config.num_agents * config.action_dim
    flat_states = states.reshape((states.shape[0], _flat_state_dim(config)))
    action_features = jax.nn.one_hot(actions, config.action_dim).reshape(
        (actions.shape[0], flat_actions_dim)
    )
    return jnp.concatenate([flat_states, action_features], axis=-1)


def _pack_transition(
    next_states: jnp.ndarray,
    config: VectorWorldModelConfig,
) -> jnp.ndarray:
    return next_states.reshape((next_states.shape[0], _flat_state_dim(config)))


def _unpack_transition(
    transition: jnp.ndarray,
    config: VectorWorldModelConfig,
) -> jnp.ndarray:
    return transition.reshape(
        (transition.shape[0], config.num_agents, config.state_dim)
    )


def _flat_state_dim(config: VectorWorldModelConfig) -> int:
    return config.num_agents * config.state_dim


def _transition_dim(config: VectorWorldModelConfig) -> int:
    return _flat_state_dim(config)


def _cond_dim(config: VectorWorldModelConfig) -> int:
    return _flat_state_dim(config) + config.num_agents * config.action_dim
