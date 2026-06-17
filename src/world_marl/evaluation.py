"""Evaluation loops for vectorized multi-agent adapters."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from world_marl.algs.ippo import select_actions
from world_marl.algs.mappo import select_actions as select_mappo_actions
from world_marl.envs.meltingpot_adapter import (
    flatten_agent_batch,
    unflatten_agent_actions,
)
from world_marl.training import build_central_observations
from world_marl.training import ObservationMode

if TYPE_CHECKING:
    from world_marl.scripts.train_e2e import TrainingAdapter


PolicyFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class EvaluationResult:
    returns: np.ndarray
    lengths: np.ndarray
    mean_return_per_agent: float
    episodes: int
    steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "steps": self.steps,
            "mean_return_per_agent": self.mean_return_per_agent,
            "returns_mean_by_agent": self.returns.mean(axis=0).tolist(),
            "returns": self.returns.tolist(),
            "lengths": self.lengths.tolist(),
        }


def evaluate_policy(
    adapter: TrainingAdapter,
    policy_fn: PolicyFn,
    *,
    episodes: int,
    max_steps: int | None = None,
) -> EvaluationResult:
    """Evaluate a policy until ``episodes`` complete episodes are collected."""
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    max_steps = max_steps or (
        math.ceil(episodes / adapter.num_envs)
        * adapter.max_cycles
        * adapter.num_envs
        * 2
    )
    observations = adapter.reset()
    completed_returns: list[tuple[float, ...]] = []
    completed_lengths: list[int] = []

    steps = 0
    while len(completed_returns) < episodes and steps < max_steps:
        actions = np.asarray(policy_fn(observations), dtype=np.int32)
        step = adapter.step(actions)
        observations = step.observations
        completed_returns.extend(step.completed_returns)
        completed_lengths.extend(step.completed_lengths)
        steps += adapter.num_envs

    if len(completed_returns) < episodes:
        raise RuntimeError(
            f"only collected {len(completed_returns)} of {episodes} episodes "
            f"after {max_steps} vector steps"
        )

    returns = np.asarray(completed_returns[:episodes], dtype=np.float32)
    lengths = np.asarray(completed_lengths[:episodes], dtype=np.int32)
    return EvaluationResult(
        returns=returns,
        lengths=lengths,
        mean_return_per_agent=float(returns.mean()),
        episodes=episodes,
        steps=steps,
    )


def random_policy(adapter: Any, rng: np.random.Generator) -> PolicyFn:
    """Create a random action policy for an adapter."""

    def act(observations: np.ndarray) -> np.ndarray:
        del observations
        return adapter.sample_actions(rng)

    return act


def constant_policy(action: int = 0) -> PolicyFn:
    """Create a fixed action policy, useful for evaluation tests."""

    def act(observations: np.ndarray) -> np.ndarray:
        return np.full(
            (observations.shape[0], observations.shape[1]),
            action,
            dtype=np.int32,
        )

    return act


def train_state_policy(
    train_state: TrainState,
    *,
    num_envs: int,
    num_agents: int,
    deterministic: bool = True,
    seed: int = 0,
    observation_mode: ObservationMode = "image",
) -> PolicyFn:
    """Create a numpy policy function backed by a Flax TrainState."""
    key = jax.random.PRNGKey(seed)
    infer_fn = jax.jit(
        lambda state, action_key, flat_obs: select_actions(
            state,
            action_key,
            flat_obs,
            deterministic=deterministic,
        )[0]
    )

    def act(observations: np.ndarray) -> np.ndarray:
        nonlocal key
        flat_observations = jnp.asarray(
            _policy_observations(observations, observation_mode)
        )
        key, action_key = jax.random.split(key)
        actions = infer_fn(
            train_state,
            action_key,
            flat_observations,
        )
        return unflatten_agent_actions(
            np.asarray(actions),
            num_envs=num_envs,
            num_agents=num_agents,
        )

    return act


def mappo_train_state_policy(
    train_state: TrainState,
    *,
    num_envs: int,
    num_agents: int,
    deterministic: bool = True,
    seed: int = 0,
    observation_mode: ObservationMode = "image",
) -> PolicyFn:
    """Create a MAPPO policy function backed by a Flax TrainState."""
    key = jax.random.PRNGKey(seed)
    infer_fn = jax.jit(
        lambda state, action_key, flat_obs, flat_central_obs: select_mappo_actions(
            state,
            action_key,
            flat_obs,
            flat_central_obs,
            deterministic=deterministic,
        )[0]
    )

    def act(observations: np.ndarray) -> np.ndarray:
        nonlocal key
        central_observations = build_central_observations(
            observations,
            observation_mode=observation_mode,
        )
        flat_observations = jnp.asarray(
            _policy_observations(observations, observation_mode)
        )
        flat_central_observations = jnp.asarray(
            flatten_agent_batch(central_observations)
        )
        key, action_key = jax.random.split(key)
        actions = infer_fn(
            train_state,
            action_key,
            flat_observations,
            flat_central_observations,
        )
        return unflatten_agent_actions(
            np.asarray(actions),
            num_envs=num_envs,
            num_agents=num_agents,
        )

    return act


def _policy_observations(
    observations: np.ndarray,
    observation_mode: ObservationMode,
) -> np.ndarray:
    if observation_mode == "vector":
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim < 3:
            raise ValueError("expected observations shaped [env, agent, ...]")
        return observations.reshape((-1, int(np.prod(observations.shape[2:]))))
    if observation_mode == "image":
        return flatten_agent_batch(observations)
    raise ValueError(f"unsupported observation_mode {observation_mode!r}")
