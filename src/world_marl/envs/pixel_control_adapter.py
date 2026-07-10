"""Small pixel-observation continuous-control adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from world_marl.envs.meltingpot_adapter import VectorStep


class PixelPointMassAdapter:
    """DMC-like point-mass control task that returns HWC pixels."""

    def __init__(
        self,
        env_id: str = "pointmass",
        *,
        num_envs: int = 1,
        max_cycles: int = 100,
        seed: int = 0,
        auto_reset: bool = True,
        image_size: int = 16,
    ) -> None:
        if env_id != "pointmass":
            raise ValueError(
                "pixel substrates currently support only 'pixels:pointmass'"
            )
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")
        if image_size < 4:
            raise ValueError("image_size must be >= 4")

        self.env_id = env_id
        self.substrate = f"pixels:{env_id}"
        self.num_envs = int(num_envs)
        self.max_cycles = int(max_cycles)
        self.auto_reset = bool(auto_reset)
        self.image_size = int(image_size)
        self.agents = ("agent_0",)
        self.num_agents = 1

        self.observation_shape = (self.image_size, self.image_size, 3)
        self.raw_observation_shape = self.observation_shape
        self.observation_size = None
        self.include_observation_scalars = False
        self.scalar_observation_keys: tuple[str, ...] = ()
        self.append_agent_id = False

        self.action_shape = (2,)
        self.action_dim = 2
        self.action_low = -np.ones((self.action_dim,), dtype=np.float32)
        self.action_high = np.ones((self.action_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._target = np.array([0.65, 0.65], dtype=np.float32)
        self._episode_returns = np.zeros((self.num_envs, 1), dtype=np.float32)
        self._episode_lengths = np.zeros((self.num_envs,), dtype=np.int32)
        axis = np.linspace(-1.0, 1.0, self.image_size, dtype=np.float32)
        self._grid_x, self._grid_y = np.meshgrid(axis, axis)
        self._positions = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.reset()

    def reset(self) -> np.ndarray:
        self._episode_returns[:] = 0.0
        self._episode_lengths[:] = 0
        self._positions = self._rng.uniform(
            low=-0.75,
            high=-0.25,
            size=(self.num_envs, 2),
        ).astype(np.float32)
        return self._observations()

    def step(self, actions: np.ndarray) -> VectorStep:
        action_batch = np.asarray(actions, dtype=np.float32).reshape(
            (self.num_envs, self.action_dim)
        )
        action_batch = np.clip(action_batch, self.action_low, self.action_high)
        self._positions = np.clip(
            self._positions + 0.18 * action_batch,
            -1.0,
            1.0,
        ).astype(np.float32)

        distance = np.linalg.norm(self._positions - self._target[None, :], axis=-1)
        rewards = (1.0 - distance).astype(np.float32).reshape((self.num_envs, 1))
        self._episode_returns += rewards
        self._episode_lengths += 1
        done_mask = np.logical_or(
            distance < 0.12, self._episode_lengths >= self.max_cycles
        )

        completed_returns: list[tuple[float, ...]] = []
        completed_lengths: list[int] = []
        infos: list[dict[str, Any]] = []
        for env_index in np.flatnonzero(done_mask):
            completed_returns.append((float(self._episode_returns[env_index, 0]),))
            completed_lengths.append(int(self._episode_lengths[env_index]))
            infos.append(
                {
                    "env_index": int(env_index),
                    "terminated": bool(distance[env_index] < 0.12),
                    "truncated": bool(
                        self._episode_lengths[env_index] >= self.max_cycles
                    ),
                    "agent_infos": {},
                }
            )

        if self.auto_reset and bool(np.any(done_mask)):
            reset_positions = self._rng.uniform(
                low=-0.75,
                high=-0.25,
                size=(self.num_envs, 2),
            ).astype(np.float32)
            self._positions[done_mask] = reset_positions[done_mask]
            self._episode_returns[done_mask] = 0.0
            self._episode_lengths[done_mask] = 0

        return VectorStep(
            observations=self._observations(),
            rewards=rewards,
            dones=done_mask.astype(np.float32).reshape((self.num_envs, 1)),
            completed_returns=tuple(completed_returns),
            completed_lengths=tuple(completed_lengths),
            step_infos=tuple({} for _ in range(self.num_envs)),
            infos=tuple(infos),
        )

    def sample_actions(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(
            low=self.action_low,
            high=self.action_high,
            size=(self.num_envs, self.action_dim),
        ).astype(np.float32)[:, None, :]

    def close(self) -> None:
        return None

    def _observations(self) -> np.ndarray:
        frames = np.empty(
            (self.num_envs, self.image_size, self.image_size, 3),
            dtype=np.float32,
        )
        target_blob = self._blob(self._target)
        for env_index, position in enumerate(self._positions):
            agent_blob = self._blob(position)
            frame = np.zeros(self.observation_shape, dtype=np.float32)
            frame[..., 0] = agent_blob
            frame[..., 1] = target_blob
            frame[..., 2] = 0.15 + 0.35 * np.maximum(agent_blob, target_blob)
            frames[env_index] = np.clip(frame, 0.0, 1.0)
        return frames[:, None, ...]

    def _blob(self, position: np.ndarray) -> np.ndarray:
        dx = self._grid_x - np.float32(position[0])
        dy = self._grid_y - np.float32(position[1])
        return np.exp(-18.0 * (dx * dx + dy * dy)).astype(np.float32)


def is_pixel_substrate(substrate: str) -> bool:
    return substrate.startswith("pixels:")


def pixel_env_name(substrate: str) -> str:
    if not is_pixel_substrate(substrate):
        raise ValueError(f"not a pixel substrate: {substrate!r}")
    env_id = substrate.split(":", 1)[1]
    if not env_id:
        raise ValueError("pixel substrates must be formatted as 'pixels:<env_name>'")
    if env_id != "pointmass":
        raise ValueError("pixel substrates currently support only 'pixels:pointmass'")
    return env_id
