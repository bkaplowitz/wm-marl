"""Vector adapter for Brax single-agent continuous-control environments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.envs.meltingpot_adapter import VectorStep

BraxEnvFactory = Callable[[], Any]


def make_brax_env(
    env_id: str,
    *,
    backend: str | None = None,
    episode_length: int = 1000,
):
    """Build a Brax environment by name."""

    from brax import envs

    kwargs = {}
    if backend is not None:
        kwargs["backend"] = backend
    return envs.create(env_name=env_id, episode_length=episode_length, **kwargs)


class BraxVectorAdapter:
    """Wrap Brax environments in the repo's single-agent vector contract.

    ``auto_reset=False`` is **not honored** for the underlying dynamics:
    ``brax.envs.create`` always applies Brax's ``AutoResetWrapper``, which
    restores a done env to its original fixed initial state on the next step.
    Setting ``auto_reset=False`` only skips the adapter's re-randomized resets;
    the underlying environment keeps restarting either way.
    """

    def __init__(
        self,
        env_id: str = "reacher",
        *,
        num_envs: int = 1,
        max_cycles: int = 1000,
        seed: int = 0,
        env_factory: BraxEnvFactory | None = None,
        auto_reset: bool = True,
        backend: str | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")

        self.env_id = env_id
        self.substrate = f"brax:{env_id}"
        self.num_envs = int(num_envs)
        self.max_cycles = int(max_cycles)
        self.auto_reset = auto_reset
        self.backend = backend
        self.agents = ("agent_0",)
        self.num_agents = 1

        self._env = (
            env_factory
            or (
                lambda: make_brax_env(
                    env_id,
                    backend=backend,
                    episode_length=max_cycles,
                )
            )
        )()
        self._reset = jax.jit(jax.vmap(self._env.reset))
        self._step = jax.jit(jax.vmap(self._env.step))
        self._base_key = jax.random.PRNGKey(seed)
        self._reset_counter = 0
        self._state = self._reset(self._next_reset_keys())

        observations = np.asarray(jax.device_get(self._state.obs), dtype=np.float32)
        self.observation_shape = tuple(observations.shape[1:]) or (1,)
        self.raw_observation_shape = self.observation_shape
        self.observation_size = None
        self.include_observation_scalars = False
        self.scalar_observation_keys: tuple[str, ...] = ()
        self.append_agent_id = False

        self.action_shape = (int(getattr(self._env, "action_size")),)
        self.action_dim = self.action_shape[0]
        self.action_low = -np.ones((self.action_dim,), dtype=np.float32)
        self.action_high = np.ones((self.action_dim,), dtype=np.float32)

        self._episode_returns = np.zeros((self.num_envs, 1), dtype=np.float32)
        self._episode_lengths = np.zeros((self.num_envs,), dtype=np.int32)

    def reset(self) -> np.ndarray:
        self._episode_returns[:] = 0.0
        self._episode_lengths[:] = 0
        self._state = self._reset(self._next_reset_keys())
        return self._observations()

    def reset_indices(self, indices: np.ndarray) -> np.ndarray:
        """Reset selected vector members and return their new observations."""

        reset_indices = np.asarray(indices, dtype=np.int64).reshape((-1,))
        if np.any((reset_indices < 0) | (reset_indices >= self.num_envs)):
            raise IndexError("reset index is outside the vector environment")
        if np.unique(reset_indices).size != reset_indices.size:
            raise ValueError("reset indices must be unique")
        if reset_indices.size == 0:
            return np.empty((0, 1, *self.observation_shape), dtype=np.float32)

        reset_mask = np.zeros((self.num_envs,), dtype=bool)
        reset_mask[reset_indices] = True
        reset_state = self._reset(self._next_reset_keys())
        self._state = _select_reset_state(
            reset_state,
            self._state,
            jnp.asarray(reset_mask),
            num_envs=self.num_envs,
        )
        self._episode_returns[reset_mask] = 0.0
        self._episode_lengths[reset_mask] = 0
        return self._observations()[reset_indices]

    def step(self, actions: np.ndarray) -> VectorStep:
        action_batch = np.asarray(actions, dtype=np.float32).reshape(
            (self.num_envs, self.action_dim)
        )
        action_batch = np.clip(action_batch, self.action_low, self.action_high)

        next_state = self._step(self._state, jnp.asarray(action_batch))
        rewards_flat = np.asarray(jax.device_get(next_state.reward), dtype=np.float32)
        env_done = np.asarray(jax.device_get(next_state.done), dtype=np.float32) > 0.5

        self._episode_returns[:, 0] += rewards_flat.reshape((self.num_envs,))
        self._episode_lengths += 1
        truncated = self._episode_lengths >= self.max_cycles
        done_mask = np.logical_or(env_done.reshape((self.num_envs,)), truncated)

        completed_returns: list[tuple[float, ...]] = []
        completed_lengths: list[int] = []
        infos: list[dict[str, Any]] = []
        for env_index, done in enumerate(done_mask):
            if not bool(done):
                continue
            completed_returns.append((float(self._episode_returns[env_index, 0]),))
            completed_lengths.append(int(self._episode_lengths[env_index]))
            infos.append(
                {
                    "env_index": int(env_index),
                    "terminated": bool(env_done[env_index]),
                    "truncated": bool(truncated[env_index]),
                    "agent_infos": {},
                }
            )

        if self.auto_reset and bool(np.any(done_mask)):
            reset_state = self._reset(self._next_reset_keys())
            next_state = _select_reset_state(
                reset_state,
                next_state,
                jnp.asarray(done_mask),
                num_envs=self.num_envs,
            )
            self._episode_returns[done_mask] = 0.0
            self._episode_lengths[done_mask] = 0

        self._state = next_state
        return VectorStep(
            observations=self._observations(),
            rewards=rewards_flat.reshape((self.num_envs, 1)).astype(np.float32),
            dones=done_mask.astype(np.float32).reshape((self.num_envs, 1)),
            completed_returns=tuple(completed_returns),
            completed_lengths=tuple(completed_lengths),
            step_infos=tuple({} for _ in range(self.num_envs)),
            infos=tuple(infos),
        )

    def sample_actions(self, rng: np.random.Generator) -> np.ndarray:
        actions = rng.uniform(
            low=self.action_low,
            high=self.action_high,
            size=(self.num_envs, self.action_dim),
        ).astype(np.float32)
        return actions[:, None, :]

    def close(self) -> None:
        return None

    def _next_reset_keys(self) -> jax.Array:
        key = jax.random.fold_in(self._base_key, self._reset_counter)
        self._reset_counter += 1
        return jax.random.split(key, self.num_envs)

    def _observations(self) -> np.ndarray:
        observations = np.asarray(jax.device_get(self._state.obs), dtype=np.float32)
        return observations.reshape((self.num_envs, -1))[:, None, :]


def is_brax_substrate(substrate: str) -> bool:
    return substrate.startswith("brax:")


def brax_env_name(substrate: str) -> str:
    if not is_brax_substrate(substrate):
        raise ValueError(f"not a Brax substrate: {substrate!r}")
    env_id = substrate.split(":", 1)[1]
    if not env_id:
        raise ValueError("Brax substrates must be formatted as 'brax:<env_name>'")
    return env_id


def _select_reset_state(
    reset_state, step_state, done_mask: jax.Array, *, num_envs: int
):
    def select(reset_leaf, step_leaf):
        if not hasattr(step_leaf, "shape") or not hasattr(reset_leaf, "shape"):
            return step_leaf
        if not step_leaf.shape or step_leaf.shape[0] != num_envs:
            return step_leaf
        mask = done_mask
        while mask.ndim < step_leaf.ndim:
            mask = mask[..., None]
        return jnp.where(mask, reset_leaf, step_leaf)

    return jax.tree_util.tree_map(select, reset_state, step_state)
