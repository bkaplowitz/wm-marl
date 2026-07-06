"""Evaluation loops for vectorized Melting Pot adapters."""

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
from world_marl.training import build_central_observations, build_vector_central
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


def evaluate_policy_scan(
    adapter: TrainingAdapter,
    train_state: TrainState,
    *,
    episodes: int,
    deterministic: bool = True,
    observation_mode: ObservationMode = "vector",
    seed: int = 0,
    algorithm: str = "ippo",
) -> EvaluationResult:
    """On-device equivalent of ``evaluate_policy`` for vector-mode policies.

    Drives the policy through ``adapter.scan_rewards_dones`` (a single jitted
    ``lax.scan``) so the whole eval rollout stays on the accelerator -- no
    per-step host round-trips. MAPPO rebuilds its centralized critic input on
    device with the same ``build_vector_central`` the training scan uses, after
    the float32 cast the loop path applies.
    """
    if observation_mode != "vector":
        raise ValueError("scan eval is only wired for vector observations")

    num_envs = adapter.num_envs
    num_agents = adapter.num_agents

    if algorithm == "ippo":

        def action_fn(
            observations: jnp.ndarray, action_key: jax.Array
        ) -> jnp.ndarray:
            flat_obs = observations.reshape((num_envs * num_agents, -1))
            actions = select_actions(
                train_state, action_key, flat_obs, deterministic=deterministic
            )[0]
            return actions.reshape((num_envs, num_agents))

    elif algorithm == "mappo":

        def action_fn(
            observations: jnp.ndarray, action_key: jax.Array
        ) -> jnp.ndarray:
            observations = observations.astype(jnp.float32)
            flat_obs = observations.reshape((num_envs * num_agents, -1))
            flat_central = build_vector_central(observations, jnp).reshape(
                (num_envs * num_agents, -1)
            )
            actions = select_mappo_actions(
                train_state,
                action_key,
                flat_obs,
                flat_central,
                deterministic=deterministic,
            )[0]
            return actions.reshape((num_envs, num_agents))

    else:
        raise ValueError(f"unsupported algorithm {algorithm!r}")

    return _scan_eval(
        adapter, action_fn, episodes=episodes, policy_key=jax.random.PRNGKey(seed)
    )


def evaluate_random_policy_scan(
    adapter: TrainingAdapter,
    *,
    episodes: int,
    seed: int = 0,
) -> EvaluationResult:
    """On-device random baseline: uniform actions from the jax PRNG stream.

    Replaces the loop-based ``evaluate_policy(random_policy(...))`` baseline on
    scannable adapters; statistically equivalent to (not bit-comparable with)
    the numpy-RNG loop baseline.
    """
    num_envs = adapter.num_envs
    num_agents = adapter.num_agents
    action_dim = adapter.action_dim

    def action_fn(observations: jnp.ndarray, action_key: jax.Array) -> jnp.ndarray:
        del observations
        return jax.random.randint(
            action_key, (num_envs, num_agents), 0, action_dim
        )

    return _scan_eval(
        adapter, action_fn, episodes=episodes, policy_key=jax.random.PRNGKey(seed)
    )


def _scan_eval(
    adapter: TrainingAdapter,
    action_fn: Callable[[jnp.ndarray, jax.Array], jnp.ndarray],
    *,
    episodes: int,
    policy_key: jax.Array,
) -> EvaluationResult:
    """Run ``adapter.scan_rewards_dones`` and reconstruct per-episode returns.

    Coins is lockstep (all envs reset together every ``max_cycles`` steps), so
    ``ceil(episodes/num_envs)`` waves of ``max_cycles`` steps yield exactly
    ``episodes`` fixed-length episodes and the per-wave block sum of rewards
    reproduces the loop accumulator bit-for-bit.
    """
    if episodes < 1:
        raise ValueError("episodes must be >= 1")

    num_envs = adapter.num_envs
    num_agents = adapter.num_agents
    max_cycles = adapter.max_cycles
    waves = math.ceil(episodes / num_envs)
    num_steps = waves * max_cycles

    rewards, dones_all = adapter.scan_rewards_dones(
        action_fn, num_steps, policy_key=policy_key
    )
    rewards = np.asarray(rewards)  # [T, E, A]
    dones_all = np.asarray(dones_all)  # [T, E]

    boundary = np.zeros(num_steps, dtype=bool)
    boundary[np.arange(max_cycles - 1, num_steps, max_cycles)] = True
    if not np.array_equal(
        dones_all, np.broadcast_to(boundary[:, None], dones_all.shape)
    ):
        raise RuntimeError(
            "scan eval requires lockstep episodes aligned to max_cycles (coins); "
            "dones did not fire on the expected block boundaries"
        )

    block_returns = rewards.reshape((waves, max_cycles, num_envs, num_agents)).sum(
        axis=1
    )  # [waves, E, A]
    returns = block_returns.reshape((waves * num_envs, num_agents))[:episodes].astype(
        np.float32
    )
    lengths = np.full((episodes,), max_cycles, dtype=np.int32)
    return EvaluationResult(
        returns=returns,
        lengths=lengths,
        mean_return_per_agent=float(returns.mean()),
        episodes=episodes,
        steps=num_steps * num_envs,
    )


def random_policy(adapter: TrainingAdapter, rng: np.random.Generator) -> PolicyFn:
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
    get_action = jax.jit(
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
        actions = get_action(
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
    get_action = jax.jit(
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
        actions = get_action(
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


# TODO: Generate evaluation from fit model of policy.
