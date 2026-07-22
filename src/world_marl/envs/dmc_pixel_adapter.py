"""Pixel-observation adapter for official DeepMind Control Suite tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from world_marl.envs.dmc_adapter import dmc_boundary_flags
from world_marl.envs.meltingpot_adapter import VectorStep


DMCPixelEnvFactory = Callable[[int], Any]


def make_dmc_pixel_env(
    env_id: str,
    *,
    seed: int,
    height: int,
    width: int,
    camera_id: int = 0,
):
    """Build an official DMC task whose observation is a rendered RGB frame."""

    from dm_control import suite
    from dm_control.suite.wrappers import pixels

    domain_name, task_name = _split_env_id(env_id)
    base_env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={"random": seed},
    )
    return pixels.Wrapper(
        base_env,
        pixels_only=True,
        render_kwargs={
            "height": int(height),
            "width": int(width),
            "camera_id": int(camera_id),
        },
    )


class DMCPixelAdapter:
    """Expose official DMC rendered observations through the repo adapter API."""

    def __init__(
        self,
        env_id: str = "point_mass/easy",
        *,
        num_envs: int = 1,
        max_cycles: int = 1000,
        seed: int = 0,
        image_size: int = 64,
        camera_id: int = 0,
        env_factory: DMCPixelEnvFactory | None = None,
        auto_reset: bool = True,
        num_workers: int = 1,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")
        if image_size < 1:
            raise ValueError("image_size must be >= 1")
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")

        domain_name, task_name = _split_env_id(env_id)
        self.env_id = env_id
        self.substrate = f"dmc-pixels:{env_id}"
        self.num_envs = int(num_envs)
        self.max_cycles = int(max_cycles)
        self.auto_reset = bool(auto_reset)
        self.num_workers = int(num_workers)
        self.agents = ("agent_0",)
        self.num_agents = 1

        factory = env_factory or (
            lambda env_seed: make_dmc_pixel_env(
                env_id,
                seed=env_seed,
                height=image_size,
                width=image_size,
                camera_id=camera_id,
            )
        )
        self._envs = [factory(seed + index) for index in range(self.num_envs)]
        self._executor = (
            ThreadPoolExecutor(max_workers=min(self.num_workers, self.num_envs))
            if self.num_workers > 1
            else None
        )

        first_env = self._envs[0]
        self.observation_shape = _pixel_observation_shape(first_env.observation_spec())
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

        self.environment_metadata = {
            "environment_backend": "dm_control",
            "observation_mode": "pixels",
            "dmc_domain": domain_name,
            "dmc_task": task_name,
            "image_height": self.observation_shape[0],
            "image_width": self.observation_shape[1],
            "camera_id": int(camera_id),
        }
        self._episode_returns = np.zeros((self.num_envs, 1), dtype=np.float32)
        self._episode_lengths = np.zeros((self.num_envs,), dtype=np.int32)

    def reset(self) -> np.ndarray:
        self._episode_returns[:] = 0.0
        self._episode_lengths[:] = 0
        if self._executor is None:
            timesteps = [env.reset() for env in self._envs]
        else:
            timesteps = list(self._executor.map(lambda env: env.reset(), self._envs))
        observations = [_normalize_pixels(step.observation) for step in timesteps]
        return np.asarray(observations, dtype=np.float32)[:, None, ...]

    def step(self, actions: np.ndarray) -> VectorStep:
        action_batch = np.asarray(actions, dtype=np.float32).reshape(
            (self.num_envs, self.action_dim)
        )
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

        observations = []
        rewards = np.zeros((self.num_envs, 1), dtype=np.float32)
        dones = np.zeros((self.num_envs, 1), dtype=np.float32)
        is_last = np.zeros((self.num_envs, 1), dtype=np.float32)
        is_terminal = np.zeros((self.num_envs, 1), dtype=np.float32)
        completed_returns: list[tuple[float, ...]] = []
        completed_lengths: list[int] = []
        infos: list[dict[str, Any]] = []

        for env_index, (env, timestep) in enumerate(
            zip(self._envs, timesteps, strict=True)
        ):
            reward = 0.0 if timestep.reward is None else float(timestep.reward)
            self._episode_returns[env_index, 0] += reward
            self._episode_lengths[env_index] += 1
            last, terminal = dmc_boundary_flags(
                timestep,
                max_cycles_reached=(
                    self._episode_lengths[env_index] >= self.max_cycles
                ),
            )
            rewards[env_index, 0] = reward
            dones[env_index, 0] = float(last)
            is_last[env_index, 0] = float(last)
            is_terminal[env_index, 0] = float(terminal)

            if last:
                completed_returns.append((float(self._episode_returns[env_index, 0]),))
                completed_lengths.append(int(self._episode_lengths[env_index]))
                infos.append(
                    {
                        "env_index": int(env_index),
                        "terminated": terminal,
                        "truncated": not terminal,
                        "agent_infos": {},
                    }
                )
                if self.auto_reset:
                    timestep = env.reset()
                self._episode_returns[env_index] = 0.0
                self._episode_lengths[env_index] = 0

            observations.append(_normalize_pixels(timestep.observation))

        return VectorStep(
            observations=np.asarray(observations, dtype=np.float32)[:, None, ...],
            rewards=rewards,
            dones=dones,
            completed_returns=tuple(completed_returns),
            completed_lengths=tuple(completed_lengths),
            step_infos=tuple({} for _ in range(self.num_envs)),
            infos=tuple(infos),
            is_last=is_last,
            is_terminal=is_terminal,
        )

    def sample_actions(self, rng: np.random.Generator) -> np.ndarray:
        actions = rng.uniform(
            low=self.action_low,
            high=self.action_high,
            size=(self.num_envs, self.action_dim),
        ).astype(np.float32)
        return actions[:, None, :]

    def close(self) -> None:
        for env in self._envs:
            close = getattr(env, "close", None)
            if close is not None:
                close()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


def is_dmc_pixel_substrate(substrate: str) -> bool:
    return substrate.startswith("dmc-pixels:")


def dmc_pixel_env_name(substrate: str) -> str:
    if not is_dmc_pixel_substrate(substrate):
        raise ValueError(f"not a DMC pixel substrate: {substrate!r}")
    env_id = substrate.split(":", 1)[1]
    _split_env_id(env_id)
    return env_id


def _split_env_id(env_id: str) -> tuple[str, str]:
    if "/" not in env_id:
        raise ValueError(
            "DMC pixel substrates must be formatted as 'dmc-pixels:<domain>/<task>'"
        )
    domain_name, task_name = env_id.split("/", 1)
    if not domain_name or not task_name:
        raise ValueError(
            "DMC pixel substrates must be formatted as 'dmc-pixels:<domain>/<task>'"
        )
    return domain_name, task_name


def _pixel_observation_shape(observation_spec: Any) -> tuple[int, int, int]:
    if not isinstance(observation_spec, Mapping) or "pixels" not in observation_spec:
        raise ValueError("DMC pixel observations must contain a 'pixels' entry")
    shape = tuple(int(dim) for dim in observation_spec["pixels"].shape)
    if len(shape) != 3 or shape[-1] != 3:
        raise ValueError(f"expected HWC RGB pixel observations, got {shape}")
    return shape


def _normalize_pixels(observation: Any) -> np.ndarray:
    if not isinstance(observation, Mapping) or "pixels" not in observation:
        raise ValueError("DMC pixel observations must contain a 'pixels' entry")
    pixels = np.asarray(observation["pixels"])
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB pixel observations, got {pixels.shape}")
    if np.issubdtype(pixels.dtype, np.integer):
        return pixels.astype(np.float32) / np.float32(255.0)
    normalized = pixels.astype(np.float32)
    if normalized.size and float(np.max(normalized)) > 1.0:
        normalized = normalized / np.float32(255.0)
    return normalized
