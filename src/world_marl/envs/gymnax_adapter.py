"""Vector adapter for single-agent Gymnax environments.

Gymnax environments are native JAX environments. This adapter exposes them with
the same small vector-env contract used by the Melting Pot and JaxMARL CoinGame
adapters: observations are shaped ``[env, agent, ...]`` and actions are shaped
``[env, agent]``. For Gymnax, ``agent`` is always a singleton axis.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from world_marl.envs.meltingpot_adapter import VectorStep


GymnaxFactory = Callable[[], tuple[Any, Any]]


class GymnaxVectorAdapter:
    """Wrap a single-agent Gymnax environment as a vectorized training adapter.

    ``auto_reset`` is accepted for signature parity with the other adapters but
    is **not honored**: gymnax's ``step`` always auto-resets internally at the
    episode boundary, so the returned boundary observation is the env's own
    fresh-episode observation.
    """

    def __init__(
        self,
        env_name: str = "CartPole-v1",
        *,
        num_envs: int = 1,
        max_cycles: int = 500,
        seed: int = 0,
        env_factory: GymnaxFactory | None = None,
        auto_reset: bool = True,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")

        self.substrate = f"gymnax:{env_name}"
        self.env_name = env_name
        self.num_envs = num_envs
        self.max_cycles = max_cycles
        self.auto_reset = auto_reset
        self.agents = ("agent_0",)
        self.num_agents = 1

        if env_factory is None:
            import gymnax

            self.env, self.env_params = gymnax.make(env_name)
        else:
            self.env, self.env_params = env_factory()
        self.env_params = _with_max_cycles(self.env_params, max_cycles)

        action_space = self.env.action_space(self.env_params)
        if not hasattr(action_space, "n"):
            raise TypeError("only discrete Gymnax action spaces are supported")
        self.action_dim = int(action_space.n)

        observation_shape = tuple(
            int(dim) for dim in self.env.observation_space(self.env_params).shape
        )
        self.observation_shape = observation_shape or (1,)
        self.raw_observation_shape = self.observation_shape
        self.observation_size = None
        self.include_observation_scalars = False
        self.scalar_observation_keys: tuple[str, ...] = ()
        self.append_agent_id = False

        self._split = jax.vmap(jax.random.split)
        self._reset = jax.jit(jax.vmap(self.env.reset, in_axes=(0, None)))
        self._step = jax.jit(
            jax.vmap(self.env.step, in_axes=(0, 0, 0, None)),
        )

        self._keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)
        self._state = None
        self._episode_returns = np.zeros((num_envs, 1), dtype=np.float32)
        self._episode_lengths = np.zeros((num_envs,), dtype=np.int32)

    def reset(self) -> np.ndarray:
        split_keys = self._split(self._keys)
        self._keys = split_keys[:, 0]
        observations, self._state = self._reset(split_keys[:, 1], self.env_params)
        self._episode_returns[:] = 0.0
        self._episode_lengths[:] = 0
        return self._stack_observations(observations)

    def step(self, actions: np.ndarray) -> VectorStep:
        actions = np.asarray(actions, dtype=np.int32).reshape((self.num_envs, 1))
        split_keys = self._split(self._keys)
        self._keys = split_keys[:, 0]
        observations, self._state, reward, done, _ = self._step(
            split_keys[:, 1],
            self._state,
            actions[:, 0],
            self.env_params,
        )

        rewards = np.asarray(reward, dtype=np.float32).reshape((self.num_envs, 1))
        dones = np.asarray(done, dtype=np.float32).reshape((self.num_envs, 1))
        done_all = np.asarray(done, dtype=bool)
        self._episode_returns += rewards
        self._episode_lengths += 1

        completed_returns: list[tuple[float, ...]] = []
        completed_lengths: list[int] = []
        infos: list[dict[str, Any]] = []
        for env_index in np.flatnonzero(done_all):
            completed_returns.append((float(self._episode_returns[env_index, 0]),))
            completed_lengths.append(int(self._episode_lengths[env_index]))
            infos.append(
                {
                    "env_index": int(env_index),
                    "terminated": True,
                    "truncated": False,
                    "agent_infos": {},
                }
            )
            self._episode_returns[env_index] = 0.0
            self._episode_lengths[env_index] = 0

        return VectorStep(
            observations=self._stack_observations(observations),
            rewards=rewards,
            dones=dones,
            completed_returns=tuple(completed_returns),
            completed_lengths=tuple(completed_lengths),
            step_infos=tuple({} for _ in range(self.num_envs)),
            infos=tuple(infos),
        )

    def sample_actions(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(
            low=0,
            high=self.action_dim,
            size=(self.num_envs, 1),
            dtype=np.int32,
        )

    def close(self) -> None:
        return None

    def _stack_observations(self, observations: Any) -> np.ndarray:
        array = np.asarray(observations, dtype=np.float32)
        array = array.reshape((self.num_envs, *self.observation_shape))
        return array[:, None, ...]


def is_gymnax_substrate(substrate: str) -> bool:
    return substrate.startswith("gymnax:")


def gymnax_env_name(substrate: str) -> str:
    if not is_gymnax_substrate(substrate):
        raise ValueError(f"not a Gymnax substrate: {substrate!r}")
    env_name = substrate.split(":", 1)[1]
    if not env_name:
        raise ValueError("Gymnax substrates must be formatted as 'gymnax:<env_id>'")
    return env_name


def _with_max_cycles(env_params: Any, max_cycles: int) -> Any:
    """Align Gymnax episode horizon with the adapter's max_cycles when possible."""
    if dataclasses.is_dataclass(env_params) and hasattr(
        env_params,
        "max_steps_in_episode",
    ):
        return dataclasses.replace(env_params, max_steps_in_episode=max_cycles)
    return env_params
