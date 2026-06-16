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
    IPPO/MAPPO training and world-model collection code runs unchanged. Because
    CoinGame is pure JAX, every env is stepped together under ``jax.vmap``: the
    per-env PRNG keys are a single ``(num_envs, 2)`` array, the ``EnvState`` is a
    single batched pytree, and the dict-keyed JaxMARL observations/rewards/dones
    are stacked into ``[env, agent, ...]`` arrays.

    ``auto_reset`` is accepted for signature parity with
    ``MeltingPotVectorAdapter`` but is **not honored**: CoinGame's ``step`` always
    resets the grid internally at the episode boundary, so the environment is
    inherently continuing and the returned boundary observation is the env's own
    fresh-episode observation.
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

        self._split = jax.vmap(jax.random.split)
        self._reset = jax.vmap(self.env.reset)
        self._step = jax.vmap(self.env.step)

        self._keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
        self._state = None
        self._episode_returns = np.zeros((num_envs, self.num_agents), dtype=np.float32)
        self._episode_lengths = np.zeros((num_envs,), dtype=np.int32)

    def reset(self) -> np.ndarray:
        split_keys = self._split(self._keys)
        self._keys = split_keys[:, 0]
        observations, self._state = self._reset(split_keys[:, 1])
        self._episode_returns[:] = 0.0
        self._episode_lengths[:] = 0
        return self._stack_agents(observations)

    def step(self, actions: np.ndarray) -> VectorStep:
        actions = np.asarray(actions, dtype=np.int32).reshape(
            (self.num_envs, self.num_agents)
        )
        action_dict = {
            agent: jnp.asarray(actions[:, agent_index], dtype=jnp.int32)
            for agent_index, agent in enumerate(self.agents)
        }

        split_keys = self._split(self._keys)
        self._keys = split_keys[:, 0]
        obs, self._state, reward, done, _ = self._step(
            split_keys[:, 1], self._state, action_dict
        )

        observations = self._stack_agents(obs)
        rewards = np.stack(
            [np.asarray(reward[a], dtype=np.float32) for a in self.agents], axis=1
        )
        dones = np.stack(
            [np.asarray(done[a], dtype=np.float32) for a in self.agents], axis=1
        )
        done_all = np.asarray(done["__all__"])

        self._episode_returns += rewards
        self._episode_lengths += 1

        completed_returns: list[tuple[float, ...]] = []
        completed_lengths: list[int] = []
        for env_index in np.flatnonzero(done_all):
            completed_returns.append(
                tuple(float(x) for x in self._episode_returns[env_index])
            )
            completed_lengths.append(int(self._episode_lengths[env_index]))
            self._episode_returns[env_index] = 0.0
            self._episode_lengths[env_index] = 0

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
            [
                np.asarray(obs[a], dtype=np.float32).reshape((self.num_envs, -1))
                for a in self.agents
            ],
            axis=1,
        )
