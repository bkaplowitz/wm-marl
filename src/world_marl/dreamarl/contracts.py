"""Tensor and lifecycle contracts shared by all DreaMARL environments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class JaxMultiAgentSequenceBatch(NamedTuple):
    """Time-major multi-agent data with explicit environment and agent axes."""

    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    team_rewards: jax.Array
    is_first: jax.Array
    is_last: jax.Array
    is_terminal: jax.Array
    agent_alive: jax.Array


@dataclass(slots=True)
class MultiAgentSequenceBatch:
    """Validated replay input shaped ``[time, env, agent, ...]``.

    Environment lifecycle flags are intentionally shaped ``[time, env]``.
    Agent lifecycle is represented separately by ``agent_alive`` so an agent
    can disappear without falsely terminating or resetting the joint world.
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    team_rewards: np.ndarray
    is_first: np.ndarray
    is_last: np.ndarray
    is_terminal: np.ndarray
    agent_alive: np.ndarray
    agent_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations)
        self.actions = np.asarray(self.actions)
        self.rewards = np.asarray(self.rewards, dtype=np.float32)
        self.team_rewards = np.asarray(self.team_rewards, dtype=np.float32)
        self.is_first = np.asarray(self.is_first, dtype=bool)
        self.is_last = np.asarray(self.is_last, dtype=bool)
        self.is_terminal = np.asarray(self.is_terminal, dtype=bool)
        self.agent_alive = np.asarray(self.agent_alive, dtype=bool)
        self.agent_ids = tuple(self.agent_ids)

        if self.observations.ndim < 4:
            raise ValueError(
                "observations must have shape [time, env, agent, ...]"
            )
        time_steps, num_envs, num_agents = self.observations.shape[:3]
        prefix = (time_steps, num_envs, num_agents)
        self._require_agent_prefix("actions", self.actions, prefix)
        self._require_exact("rewards", self.rewards, prefix)
        self._require_exact("agent_alive", self.agent_alive, prefix)
        env_prefix = (time_steps, num_envs)
        self._require_exact("team_rewards", self.team_rewards, env_prefix)
        self._require_exact("is_first", self.is_first, env_prefix)
        self._require_exact("is_last", self.is_last, env_prefix)
        self._require_exact("is_terminal", self.is_terminal, env_prefix)

        if len(self.agent_ids) != num_agents:
            raise ValueError(
                f"agent_ids has length {len(self.agent_ids)}, expected {num_agents}"
            )
        if len(set(self.agent_ids)) != num_agents:
            raise ValueError("agent_ids must be unique and ordered")
        if np.any(self.is_terminal & ~self.is_last):
            raise ValueError("every terminal transition must also be is_last")

    @staticmethod
    def _require_exact(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")

    @staticmethod
    def _require_agent_prefix(
        name: str, value: np.ndarray, prefix: tuple[int, int, int]
    ) -> None:
        if value.ndim < 3 or value.shape[:3] != prefix:
            raise ValueError(f"{name} must start with shape {prefix}, got {value.shape}")

    @property
    def time_steps(self) -> int:
        return int(self.observations.shape[0])

    @property
    def num_envs(self) -> int:
        return int(self.observations.shape[1])

    @property
    def num_agents(self) -> int:
        return int(self.observations.shape[2])

    @property
    def continues(self) -> np.ndarray:
        return 1.0 - self.is_terminal.astype(np.float32)


def sequence_batch_to_jax(
    batch: MultiAgentSequenceBatch,
) -> JaxMultiAgentSequenceBatch:
    """Move a validated batch to JAX without collapsing the agent axis."""

    return JaxMultiAgentSequenceBatch(
        observations=jnp.asarray(batch.observations),
        actions=jnp.asarray(batch.actions),
        rewards=jnp.asarray(batch.rewards, dtype=jnp.float32),
        team_rewards=jnp.asarray(batch.team_rewards, dtype=jnp.float32),
        is_first=jnp.asarray(batch.is_first, dtype=bool),
        is_last=jnp.asarray(batch.is_last, dtype=bool),
        is_terminal=jnp.asarray(batch.is_terminal, dtype=bool),
        agent_alive=jnp.asarray(batch.agent_alive, dtype=bool),
    )


def stack_agent_actions(
    actions: Mapping[str, np.ndarray], agent_ids: Sequence[str]
) -> np.ndarray:
    """Assemble decentralized actions into one ordered joint action."""

    missing = set(agent_ids) - set(actions)
    extra = set(actions) - set(agent_ids)
    if missing or extra:
        raise ValueError(
            f"joint action keys differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    rows = [np.asarray(actions[agent]) for agent in agent_ids]
    expected = rows[0].shape
    if any(row.shape != expected for row in rows[1:]):
        raise ValueError("all decentralized action batches must share a shape")
    return np.stack(rows, axis=1)
