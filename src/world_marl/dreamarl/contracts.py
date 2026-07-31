"""Environment-neutral trajectory contracts for DreaMARL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class JaxMultiAgentSequenceBatch(NamedTuple):
    """Time-major joint transitions with explicit agent and lifecycle axes."""

    observations: jax.Array
    next_observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    team_rewards: jax.Array
    is_first: jax.Array
    is_last: jax.Array
    is_terminal: jax.Array
    valid: jax.Array
    agent_alive: jax.Array
    next_agent_alive: jax.Array
    action_mask: jax.Array
    next_action_mask: jax.Array


@dataclass(slots=True)
class MultiAgentSequenceBatch:
    """Validated transitions shaped ``[time, env, agent, ...]``.

    ``next_observations`` is explicit because the real successor of a terminal
    or truncated transition is not the auto-reset observation used by the next
    replay record. Missing action masks are represented on device by a
    zero-width final axis, keeping the pytree static for continuous-action or
    unconstrained environments.
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    team_rewards: np.ndarray
    is_first: np.ndarray
    is_last: np.ndarray
    is_terminal: np.ndarray
    agent_alive: np.ndarray
    next_agent_alive: np.ndarray
    agent_ids: tuple[str, ...]
    valid: np.ndarray | None = None
    action_mask: np.ndarray | None = None
    next_action_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations)
        self.next_observations = np.asarray(self.next_observations)
        self.actions = np.asarray(self.actions)
        self.rewards = np.asarray(self.rewards, dtype=np.float32)
        self.team_rewards = np.asarray(self.team_rewards, dtype=np.float32)
        self.is_first = np.asarray(self.is_first, dtype=bool)
        self.is_last = np.asarray(self.is_last, dtype=bool)
        self.is_terminal = np.asarray(self.is_terminal, dtype=bool)
        self.agent_alive = np.asarray(self.agent_alive, dtype=bool)
        self.next_agent_alive = np.asarray(self.next_agent_alive, dtype=bool)
        self.agent_ids = tuple(self.agent_ids)

        if self.observations.ndim < 4:
            raise ValueError(
                "observations must have shape [time, env, agent, ...]"
            )
        if self.next_observations.shape != self.observations.shape:
            raise ValueError(
                "next_observations must match observations exactly, got "
                f"{self.next_observations.shape} and {self.observations.shape}"
            )
        time_steps, num_envs, num_agents = self.observations.shape[:3]
        agent_prefix = (time_steps, num_envs, num_agents)
        self._require_agent_prefix("actions", self.actions, agent_prefix)
        self._require_exact("rewards", self.rewards, agent_prefix)
        self._require_exact("agent_alive", self.agent_alive, agent_prefix)
        self._require_exact(
            "next_agent_alive", self.next_agent_alive, agent_prefix
        )
        env_prefix = (time_steps, num_envs)
        self._require_exact("team_rewards", self.team_rewards, env_prefix)
        self._require_exact("is_first", self.is_first, env_prefix)
        self._require_exact("is_last", self.is_last, env_prefix)
        self._require_exact("is_terminal", self.is_terminal, env_prefix)

        if self.valid is None:
            self.valid = np.ones(env_prefix, dtype=bool)
        else:
            self.valid = np.asarray(self.valid, dtype=bool)
            self._require_exact("valid", self.valid, env_prefix)

        self.action_mask = self._normalize_action_mask(
            "action_mask", self.action_mask, agent_prefix
        )
        self.next_action_mask = self._normalize_action_mask(
            "next_action_mask", self.next_action_mask, agent_prefix
        )
        if self.action_mask.shape != self.next_action_mask.shape:
            raise ValueError(
                "action_mask and next_action_mask must have identical shapes"
            )

        if len(self.agent_ids) != num_agents:
            raise ValueError(
                f"agent_ids has length {len(self.agent_ids)}, expected {num_agents}"
            )
        if len(set(self.agent_ids)) != num_agents:
            raise ValueError("agent_ids must be unique and ordered")
        if np.any(self.is_terminal & ~self.is_last):
            raise ValueError("every terminal transition must also be is_last")
        if time_steps > 1 and np.any(self.is_first[1:] != self.is_last[:-1]):
            raise ValueError(
                "is_first[t+1] must equal is_last[t] in contiguous collection"
            )
        if np.any(self.action_mask & ~self.agent_alive[..., None]):
            raise ValueError("inactive agents cannot have legal current actions")
        if np.any(self.next_action_mask & ~self.next_agent_alive[..., None]):
            raise ValueError("inactive agents cannot have legal next actions")

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

    @staticmethod
    def _normalize_action_mask(
        name: str,
        value: np.ndarray | None,
        prefix: tuple[int, int, int],
    ) -> np.ndarray:
        if value is None:
            return np.zeros((*prefix, 0), dtype=bool)
        result = np.asarray(value, dtype=bool)
        if result.ndim != 4 or result.shape[:3] != prefix:
            raise ValueError(f"{name} must have shape {prefix} + [action]")
        return result

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
    """Move a validated batch to JAX without collapsing semantic axes."""

    return JaxMultiAgentSequenceBatch(
        observations=jnp.asarray(batch.observations),
        next_observations=jnp.asarray(batch.next_observations),
        actions=jnp.asarray(batch.actions),
        rewards=jnp.asarray(batch.rewards, dtype=jnp.float32),
        team_rewards=jnp.asarray(batch.team_rewards, dtype=jnp.float32),
        is_first=jnp.asarray(batch.is_first, dtype=bool),
        is_last=jnp.asarray(batch.is_last, dtype=bool),
        is_terminal=jnp.asarray(batch.is_terminal, dtype=bool),
        valid=jnp.asarray(batch.valid, dtype=bool),
        agent_alive=jnp.asarray(batch.agent_alive, dtype=bool),
        next_agent_alive=jnp.asarray(batch.next_agent_alive, dtype=bool),
        action_mask=jnp.asarray(batch.action_mask, dtype=bool),
        next_action_mask=jnp.asarray(batch.next_action_mask, dtype=bool),
    )


def sequence_batch_to_numpy(
    batch: JaxMultiAgentSequenceBatch,
    agent_ids: Sequence[str],
) -> MultiAgentSequenceBatch:
    """Transfer a device batch to validated host replay storage."""

    return MultiAgentSequenceBatch(
        **{
            field: np.asarray(getattr(batch, field))
            for field in JaxMultiAgentSequenceBatch._fields
        },
        agent_ids=tuple(agent_ids),
    )


def stack_sequence_batches(
    batches: Sequence[JaxMultiAgentSequenceBatch],
) -> JaxMultiAgentSequenceBatch:
    """Stack update batches on a new leading learner-update axis."""

    if not batches:
        raise ValueError("at least one sequence batch is required")
    return jax.tree.map(lambda *values: jnp.stack(values), *batches)


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
