"""Environment adapters for the first-party DreaMARL runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import jaxmarl


@dataclass(frozen=True, slots=True)
class MultiAgentEnvSpec:
    """Static geometry and lifecycle metadata for one environment."""

    name: str
    agent_ids: tuple[str, ...]
    action_dim: int
    max_episode_steps: int

    @property
    def num_agents(self) -> int:
        return len(self.agent_ids)


class AdapterReset(NamedTuple):
    observations: jax.Array
    env_state: Any
    agent_alive: jax.Array
    action_mask: jax.Array


class AdapterTransition(NamedTuple):
    next_observations: jax.Array
    env_state: Any
    rewards: jax.Array
    is_terminal: jax.Array
    next_agent_alive: jax.Array
    next_action_mask: jax.Array


class MultiAgentAdapter(Protocol):
    """Pure-JAX adapter boundary implemented by every DreaMARL benchmark."""

    spec: MultiAgentEnvSpec

    def reset(self, keys: jax.Array) -> AdapterReset: ...

    def step(
        self,
        keys: jax.Array,
        env_state: Any,
        actions: jax.Array,
    ) -> AdapterTransition: ...


class CoinGameAdapter:
    """Vectorized CoinGame adapter that preserves real final successors."""

    def __init__(self, *, max_episode_steps: int = 64) -> None:
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        # CoinGame resets internally at num_inner_steps. Keep that boundary one
        # step beyond ours so the runtime can retain the real final successor.
        self._env = jaxmarl.make(
            "coin_game", num_inner_steps=max_episode_steps + 1
        )
        agent_ids = tuple(str(agent) for agent in self._env.agents)
        self.spec = MultiAgentEnvSpec(
            name="coin_game",
            agent_ids=agent_ids,
            action_dim=int(self._env.action_space(agent_ids[0]).n),
            max_episode_steps=max_episode_steps,
        )

    def reset(self, keys: jax.Array) -> AdapterReset:
        observations, env_state = jax.vmap(self._env.reset)(keys)
        stacked = self._stack_observations(observations)
        batch = keys.shape[0]
        alive = jnp.ones((batch, self.spec.num_agents), bool)
        action_mask = jnp.ones(
            (batch, self.spec.num_agents, self.spec.action_dim), bool
        )
        return AdapterReset(stacked, env_state, alive, action_mask)

    def step(
        self,
        keys: jax.Array,
        env_state: Any,
        actions: jax.Array,
    ) -> AdapterTransition:
        action_dict = {
            agent: actions[:, index]
            for index, agent in enumerate(self.spec.agent_ids)
        }
        observations, next_state, rewards, dones, _ = jax.vmap(
            self._env.step
        )(keys, env_state, action_dict)
        stacked_rewards = jnp.stack(
            [rewards[agent] for agent in self.spec.agent_ids], axis=1
        ).astype(jnp.float32)
        batch = actions.shape[0]
        alive = jnp.ones((batch, self.spec.num_agents), bool)
        action_mask = jnp.ones(
            (batch, self.spec.num_agents, self.spec.action_dim), bool
        )
        return AdapterTransition(
            next_observations=self._stack_observations(observations),
            env_state=next_state,
            rewards=stacked_rewards,
            is_terminal=jnp.asarray(dones["__all__"], bool),
            next_agent_alive=alive,
            next_action_mask=action_mask,
        )

    def _stack_observations(self, observations: dict[str, jax.Array]) -> jax.Array:
        return jnp.stack(
            [
                observations[agent].reshape((observations[agent].shape[0], -1))
                for agent in self.spec.agent_ids
            ],
            axis=1,
        ).astype(jnp.float32)
