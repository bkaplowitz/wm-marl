"""Expose JaxMARL's CoinGame through the MeltingPot vector-adapter interface."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np

from world_marl.envs.meltingpot_adapter import VectorStep


class JaxMARLCoinGameVectorAdapter:
    """Wrap JaxMARL CoinGame as a synchronous, numpy-batched vector environment.

    Mirrors ``MeltingPotVectorAdapter``'s duck-typed interface so the same
    IPPO/MAPPO training and world-model collection code runs unchanged: per-env
    PRNG keys and ``EnvState`` are held in plain lists, and the dict-keyed JaxMARL
    observations/rewards/dones are stacked into ``[env, agent, ...]`` arrays.
    """

    def __init__(
        self,
        *,
        num_envs: int = 1,
        max_cycles: int = 10,
        seed: int = 0,
        auto_reset: bool = True,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")

        self.num_envs = num_envs
        self.max_cycles = max_cycles
        self.auto_reset = auto_reset
        self.env = jaxmarl.make("coin_game", num_inner_steps=max_cycles)
        self.agents = list(self.env.agents)
        self.num_agents = len(self.agents)
        self.action_dim = int(self.env.action_space(self.agents[0]).n)

        probe_obs, _ = self.env.reset(jax.random.PRNGKey(seed))
        self.observation_shape = (
            int(np.asarray(probe_obs[self.agents[0]]).reshape(-1).shape[0]),
        )

        self._keys = list(jax.random.split(jax.random.PRNGKey(seed), num_envs))
        self._states: list = [None] * num_envs
        self._episode_returns = np.zeros((num_envs, self.num_agents), dtype=np.float32)
        self._episode_lengths = np.zeros((num_envs,), dtype=np.int32)

    def reset(self) -> np.ndarray:
        observations = []
        for env_index in range(self.num_envs):
            self._keys[env_index], reset_key = jax.random.split(self._keys[env_index])
            obs, state = self.env.reset(reset_key)
            self._states[env_index] = state
            self._episode_returns[env_index] = 0.0
            self._episode_lengths[env_index] = 0
            observations.append(self._stack_agents(obs))
        return np.stack(observations, axis=0).astype(np.float32)

    def step(self, actions: np.ndarray) -> VectorStep:
        actions = np.asarray(actions, dtype=np.int32).reshape(
            (self.num_envs, self.num_agents)
        )
        observations = np.zeros(
            (self.num_envs, self.num_agents, self.observation_shape[0]),
            dtype=np.float32,
        )
        rewards = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        dones = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        completed_returns: list[tuple[float, ...]] = []
        completed_lengths: list[int] = []

        for env_index in range(self.num_envs):
            next_key, step_key = jax.random.split(self._keys[env_index])
            action_dict = {
                agent: jnp.asarray(actions[env_index, agent_index], dtype=jnp.int32)
                for agent_index, agent in enumerate(self.agents)
            }
            obs, state, reward, done, _ = self.env.step(
                step_key, self._states[env_index], action_dict
            )
            self._keys[env_index] = next_key
            self._states[env_index] = state

            reward_row = np.asarray([reward[a] for a in self.agents], dtype=np.float32)
            observations[env_index] = self._stack_agents(obs)
            rewards[env_index] = reward_row
            dones[env_index] = np.asarray(
                [done[a] for a in self.agents], dtype=np.float32
            )
            self._episode_returns[env_index] += reward_row
            self._episode_lengths[env_index] += 1

            if bool(done["__all__"]):
                completed_returns.append(
                    tuple(float(x) for x in self._episode_returns[env_index])
                )
                completed_lengths.append(int(self._episode_lengths[env_index]))
                self._episode_returns[env_index] = 0.0
                self._episode_lengths[env_index] = 0
                if self.auto_reset:
                    self._keys[env_index], reset_key = jax.random.split(
                        self._keys[env_index]
                    )
                    obs, self._states[env_index] = self.env.reset(reset_key)
                    observations[env_index] = self._stack_agents(obs)

        return VectorStep(
            observations=observations,
            rewards=rewards,
            dones=dones,
            completed_returns=tuple(completed_returns),
            completed_lengths=tuple(completed_lengths),
            step_infos=tuple({} for _ in range(self.num_envs)),
            infos=tuple({} for _ in range(self.num_envs)),
        )

    def sample_actions(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(
            low=0,
            high=self.action_dim,
            size=(self.num_envs, self.num_agents),
        ).astype(np.int32)

    def close(self) -> None:
        return None

    def _stack_agents(self, obs) -> np.ndarray:
        return np.stack(
            [np.asarray(obs[a], dtype=np.float32).reshape(-1) for a in self.agents],
            axis=0,
        )
