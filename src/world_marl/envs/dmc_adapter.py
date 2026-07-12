"""Vector adapter for DeepMind Control Suite state observations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from world_marl.envs.meltingpot_adapter import VectorStep


DMCEnvFactory = Callable[[int], Any]


def make_dmc_env(env_id: str, *, seed: int):
    """Build a DeepMind Control Suite environment.

    ``env_id`` is formatted as ``"<domain>/<task>"``, for example
    ``"cartpole/swingup"``.
    """

    from dm_control import suite

    domain_name, task_name = _split_env_id(env_id)
    return suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={"random": seed},
    )


class DMCVectorAdapter:
    """Wrap DeepMind Control tasks in the repo's single-agent vector contract."""

    def __init__(
        self,
        env_id: str = "cartpole/swingup",
        *,
        num_envs: int = 1,
        max_cycles: int = 1000,
        seed: int = 0,
        env_factory: DMCEnvFactory | None = None,
        auto_reset: bool = True,
        num_workers: int = 1,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")

        self.env_id = env_id
        self.substrate = f"dmc:{env_id}"
        self.num_envs = int(num_envs)
        self.max_cycles = int(max_cycles)
        self.auto_reset = auto_reset
        self.num_workers = int(num_workers)
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self.agents = ("agent_0",)
        self.num_agents = 1

        factory = env_factory or (lambda env_seed: make_dmc_env(env_id, seed=env_seed))
        self._envs = [factory(seed + index) for index in range(self.num_envs)]
        self._executor = (
            ThreadPoolExecutor(max_workers=min(self.num_workers, self.num_envs))
            if self.num_workers > 1
            else None
        )

        first_env = self._envs[0]
        self._observation_keys = _observation_keys(first_env.observation_spec())
        self.observation_shape = (
            _flatten_observation_spec(
                first_env.observation_spec(),
                self._observation_keys,
            ),
        )
        self.raw_observation_shape = self.observation_shape
        self.observation_size = None
        self.include_observation_scalars = False
        self.scalar_observation_keys: tuple[str, ...] = ()
        self.append_agent_id = False

        action_spec = first_env.action_spec()
        self.action_shape = tuple(int(dim) for dim in action_spec.shape) or (1,)
        self.action_dim = int(np.prod(self.action_shape))
        self.action_low = np.broadcast_to(
            np.asarray(action_spec.minimum, dtype=np.float32),
            self.action_shape,
        ).reshape((self.action_dim,))
        self.action_high = np.broadcast_to(
            np.asarray(action_spec.maximum, dtype=np.float32),
            self.action_shape,
        ).reshape((self.action_dim,))

        self._episode_returns = np.zeros((self.num_envs, 1), dtype=np.float32)
        self._episode_lengths = np.zeros((self.num_envs,), dtype=np.int32)

    def reset(self) -> np.ndarray:
        self._episode_returns[:] = 0.0
        self._episode_lengths[:] = 0
        if self._executor is None:
            timesteps = [env.reset() for env in self._envs]
        else:
            timesteps = list(self._executor.map(lambda env: env.reset(), self._envs))
        observations = [
            self._flatten_observation(timestep.observation) for timestep in timesteps
        ]
        return np.asarray(observations, dtype=np.float32)[:, None, :]

    def step(self, actions: np.ndarray) -> VectorStep:
        action_batch = np.asarray(actions, dtype=np.float32).reshape(
            (self.num_envs, self.action_dim)
        )
        observations = []
        rewards = np.zeros((self.num_envs, 1), dtype=np.float32)
        dones = np.zeros((self.num_envs, 1), dtype=np.float32)
        completed_returns: list[tuple[float, ...]] = []
        completed_lengths: list[int] = []
        infos: list[dict[str, Any]] = []

        if self._executor is None:
            timesteps = [
                env.step(flat_action.reshape(self.action_shape))
                for env, flat_action in zip(self._envs, action_batch, strict=True)
            ]
        else:
            timesteps = list(
                self._executor.map(
                    lambda item: item[0].step(item[1].reshape(self.action_shape)),
                    zip(self._envs, action_batch, strict=True),
                )
            )

        for env_index, (env, timestep) in enumerate(
            zip(self._envs, timesteps, strict=True)
        ):
            reward = 0.0 if timestep.reward is None else float(timestep.reward)
            self._episode_returns[env_index, 0] += reward
            self._episode_lengths[env_index] += 1

            done = bool(timestep.last()) or (
                self._episode_lengths[env_index] >= self.max_cycles
            )
            rewards[env_index, 0] = reward
            dones[env_index, 0] = float(done)

            if done:
                completed_returns.append((float(self._episode_returns[env_index, 0]),))
                completed_lengths.append(int(self._episode_lengths[env_index]))
                infos.append(
                    {
                        "env_index": int(env_index),
                        "terminated": bool(timestep.last()),
                        "truncated": self._episode_lengths[env_index]
                        >= self.max_cycles,
                        "agent_infos": {},
                    }
                )
                if self.auto_reset:
                    timestep = env.reset()
                self._episode_returns[env_index] = 0.0
                self._episode_lengths[env_index] = 0

            observations.append(self._flatten_observation(timestep.observation))

        return VectorStep(
            observations=np.asarray(observations, dtype=np.float32)[:, None, :],
            rewards=rewards,
            dones=dones,
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

    def render(
        self,
        env_index: int = 0,
        *,
        height: int = 64,
        width: int = 64,
        camera_id: int = 0,
    ) -> np.ndarray:
        """Render one vector member through its dm_control physics object."""

        if not 0 <= env_index < self.num_envs:
            raise IndexError(f"env_index must be in [0, {self.num_envs})")
        if height < 1 or width < 1:
            raise ValueError("render height and width must be >= 1")
        physics = getattr(self._envs[env_index], "physics", None)
        if physics is None or not hasattr(physics, "render"):
            raise RuntimeError("DMC environment does not expose physics.render")
        frame = physics.render(
            height=int(height),
            width=int(width),
            camera_id=int(camera_id),
        )
        return np.asarray(frame, dtype=np.uint8)

    def close(self) -> None:
        for env in self._envs:
            close = getattr(env, "close", None)
            if close is not None:
                close()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _flatten_observation(self, observation: Any) -> np.ndarray:
        return _flatten_observation(observation, self._observation_keys)


def is_dmc_substrate(substrate: str) -> bool:
    return substrate.startswith("dmc:")


def dmc_env_name(substrate: str) -> str:
    if not is_dmc_substrate(substrate):
        raise ValueError(f"not a DMC substrate: {substrate!r}")
    env_id = substrate.split(":", 1)[1]
    _split_env_id(env_id)
    return env_id


def _split_env_id(env_id: str) -> tuple[str, str]:
    if "/" not in env_id:
        raise ValueError("DMC substrates must be formatted as 'dmc:<domain>/<task>'")
    domain_name, task_name = env_id.split("/", 1)
    if not domain_name or not task_name:
        raise ValueError("DMC substrates must be formatted as 'dmc:<domain>/<task>'")
    return domain_name, task_name


def _observation_keys(observation_spec: Any) -> tuple[str, ...]:
    if isinstance(observation_spec, dict):
        return tuple(observation_spec.keys())
    return ()


def _flatten_observation_spec(
    observation_spec: Any,
    keys: Sequence[str],
) -> int:
    if isinstance(observation_spec, dict):
        return int(
            sum(np.prod(tuple(observation_spec[key].shape) or (1,)) for key in keys)
        )
    return int(np.prod(tuple(observation_spec.shape) or (1,)))


def _flatten_observation(observation: Any, keys: Sequence[str]) -> np.ndarray:
    if isinstance(observation, dict):
        parts = [
            np.asarray(observation[key], dtype=np.float32).reshape((-1,))
            for key in keys
        ]
        return np.concatenate(parts, axis=0)
    return np.asarray(observation, dtype=np.float32).reshape((-1,))
