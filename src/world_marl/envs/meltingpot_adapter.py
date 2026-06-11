"""Vector adapter for Melting Pot substrates exposed through Shimmy.

Melting Pot/dmlab2d is not a native JAX environment. This adapter keeps
environment stepping in Python and converts the PettingZoo-style Shimmy API into
stable numpy batches for JAX policy inference and PPO updates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


EnvFactory = Callable[[], Any]


@dataclass(frozen=True)
class VectorStep:
  """Result from one vectorized environment step."""

  observations: np.ndarray
  rewards: np.ndarray
  dones: np.ndarray
  completed_returns: tuple[tuple[float, ...], ...]
  completed_lengths: tuple[int, ...]
  infos: tuple[dict[str, Any], ...]


def make_meltingpot_env(substrate: str, max_cycles: int = 1000):
  """Build a single Shimmy-wrapped Melting Pot substrate."""
  from shimmy import MeltingPotCompatibilityV0

  return MeltingPotCompatibilityV0(
    substrate_name=substrate,
    max_cycles=max_cycles,
    render_mode=None,
  )


class MeltingPotVectorAdapter:
  """Small synchronous vector env for homogeneous Melting Pot substrates."""

  def __init__(
    self,
    substrate: str = "coins",
    num_envs: int = 1,
    max_cycles: int = 1000,
    observation_size: int | tuple[int, int] | None = None,
    env_factory: EnvFactory | None = None,
    auto_reset: bool = True,
  ) -> None:
    if num_envs < 1:
      raise ValueError("num_envs must be >= 1")

    self.substrate = substrate
    self.num_envs = num_envs
    self.max_cycles = max_cycles
    self.observation_size = _normalize_observation_size(observation_size)
    self.auto_reset = auto_reset
    self._env_factory = env_factory or (
      lambda: make_meltingpot_env(substrate, max_cycles=max_cycles)
    )
    self._envs = [self._env_factory() for _ in range(num_envs)]

    first_env = self._envs[0]
    self.agents = tuple(first_env.possible_agents)
    self.num_agents = len(self.agents)
    if self.num_agents < 1:
      raise ValueError("expected at least one agent")

    self.action_dim = self._get_action_dim(first_env, self.agents[0])
    self.raw_observation_shape = self._get_rgb_shape(first_env, self.agents[0])
    self.observation_shape = self._downsampled_shape(self.raw_observation_shape)
    for env in self._envs:
      if tuple(env.possible_agents) != self.agents:
        raise ValueError("all vectorized envs must expose identical agents")
      for agent in self.agents:
        if self._get_action_dim(env, agent) != self.action_dim:
          raise ValueError("all agents must share the same discrete action dim")
        if self._get_rgb_shape(env, agent) != self.raw_observation_shape:
          raise ValueError("all agents must share the same RGB observation shape")

    self._episode_returns = np.zeros(
      (self.num_envs, self.num_agents), dtype=np.float32
    )
    self._episode_lengths = np.zeros((self.num_envs,), dtype=np.int32)
    self._last_observations: np.ndarray | None = None

  def reset(self) -> np.ndarray:
    """Reset every env and return observations shaped [env, agent, H, W, C]."""
    observations = []
    for env_index, env in enumerate(self._envs):
      obs, _ = env.reset()
      self._episode_returns[env_index] = 0.0
      self._episode_lengths[env_index] = 0
      observations.append(self._stack_observations(obs))
    self._last_observations = np.stack(observations, axis=0)
    return self._last_observations

  def step(self, actions: np.ndarray) -> VectorStep:
    """Step every env with integer actions shaped [env, agent]."""
    actions = np.asarray(actions)
    expected_shape = (self.num_envs, self.num_agents)
    if actions.shape != expected_shape:
      raise ValueError(f"actions must have shape {expected_shape}, got {actions.shape}")

    next_observations: list[np.ndarray] = []
    rewards = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
    dones = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
    completed_returns: list[tuple[float, ...]] = []
    completed_lengths: list[int] = []
    infos: list[dict[str, Any]] = []

    for env_index, env in enumerate(self._envs):
      action_dict = {
        agent: int(actions[env_index, agent_index])
        for agent_index, agent in enumerate(self.agents)
      }
      obs, reward_dict, terminations, truncations, info_dict = env.step(action_dict)

      reward_row = np.array(
        [reward_dict.get(agent, 0.0) for agent in self.agents], dtype=np.float32
      )
      rewards[env_index] = reward_row
      self._episode_returns[env_index] += reward_row
      self._episode_lengths[env_index] += 1

      done = (
        not getattr(env, "agents", None)
        or any(bool(terminations.get(agent, False)) for agent in self.agents)
        or any(bool(truncations.get(agent, False)) for agent in self.agents)
      )
      if done:
        dones[env_index] = 1.0
        completed_returns.append(tuple(float(x) for x in self._episode_returns[env_index]))
        completed_lengths.append(int(self._episode_lengths[env_index]))
        infos.append(
          {
            "env_index": env_index,
            "terminated": any(
              bool(terminations.get(agent, False)) for agent in self.agents
            ),
            "truncated": any(bool(truncations.get(agent, False)) for agent in self.agents),
            "agent_infos": info_dict,
          }
        )
        if self.auto_reset:
          obs, _ = env.reset()
          self._episode_returns[env_index] = 0.0
          self._episode_lengths[env_index] = 0

      next_observations.append(self._stack_observations(obs))

    observations = np.stack(next_observations, axis=0)
    self._last_observations = observations
    return VectorStep(
      observations=observations,
      rewards=rewards,
      dones=dones,
      completed_returns=tuple(completed_returns),
      completed_lengths=tuple(completed_lengths),
      infos=tuple(infos),
    )

  def sample_actions(self, rng: np.random.Generator) -> np.ndarray:
    """Sample random discrete actions for every env and agent."""
    return rng.integers(
      low=0,
      high=self.action_dim,
      size=(self.num_envs, self.num_agents),
      dtype=np.int32,
    )

  def close(self) -> None:
    for env in self._envs:
      env.close()

  def _stack_observations(self, observations: dict[str, Any]) -> np.ndarray:
    rows = [self._extract_rgb(observations[agent]) for agent in self.agents]
    return np.stack(rows, axis=0)

  def _extract_rgb(self, agent_observation: Any) -> np.ndarray:
    if isinstance(agent_observation, dict):
      if "RGB" not in agent_observation:
        raise KeyError("expected Melting Pot observation key 'RGB'")
      rgb = agent_observation["RGB"]
    else:
      rgb = agent_observation
    rgb_array = np.asarray(rgb, dtype=np.float32)
    if rgb_array.ndim != 3:
      raise ValueError(f"expected RGB observation with 3 dims, got {rgb_array.shape}")
    if rgb_array.max(initial=0.0) > 1.0:
      rgb_array = rgb_array / 255.0
    return self._resize_rgb(rgb_array)

  def _resize_rgb(self, rgb_array: np.ndarray) -> np.ndarray:
    if self.observation_size is None:
      return rgb_array
    target_height, target_width = self.observation_size
    height, width = rgb_array.shape[:2]
    row_indices = np.linspace(0, height - 1, target_height).astype(np.int32)
    col_indices = np.linspace(0, width - 1, target_width).astype(np.int32)
    return rgb_array[row_indices][:, col_indices]

  def _downsampled_shape(self, raw_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if self.observation_size is None:
      return raw_shape
    return (self.observation_size[0], self.observation_size[1], raw_shape[2])

  @staticmethod
  def _get_action_dim(env: Any, agent: str) -> int:
    action_space = env.action_space(agent)
    if not hasattr(action_space, "n"):
      raise TypeError("only homogeneous discrete action spaces are supported")
    return int(action_space.n)

  @staticmethod
  def _get_rgb_shape(env: Any, agent: str) -> tuple[int, int, int]:
    observation_space = env.observation_space(agent)
    if hasattr(observation_space, "spaces"):
      rgb_space = observation_space.spaces.get("RGB")
      if rgb_space is None:
        raise KeyError("expected observation space key 'RGB'")
      shape = rgb_space.shape
    else:
      shape = observation_space.shape
    if shape is None or len(shape) != 3:
      raise ValueError(f"expected RGB observation shape [H, W, C], got {shape}")
    return tuple(int(dim) for dim in shape)


def flatten_agent_batch(observations: np.ndarray) -> np.ndarray:
  """Flatten [env, agent, ...] observations to [env * agent, ...]."""
  if observations.ndim < 3:
    raise ValueError("expected observations shaped [env, agent, ...]")
  return observations.reshape((-1, *observations.shape[2:]))


def unflatten_agent_actions(
  actions: Sequence[int] | np.ndarray,
  *,
  num_envs: int,
  num_agents: int,
) -> np.ndarray:
  """Restore flat actor actions to [env, agent]."""
  return np.asarray(actions, dtype=np.int32).reshape((num_envs, num_agents))


def _normalize_observation_size(
  observation_size: int | tuple[int, int] | None,
) -> tuple[int, int] | None:
  if observation_size is None:
    return None
  if isinstance(observation_size, int):
    if observation_size < 1:
      raise ValueError("observation_size must be positive")
    return (observation_size, observation_size)
  if len(observation_size) != 2:
    raise ValueError("observation_size tuple must be (height, width)")
  height, width = observation_size
  if height < 1 or width < 1:
    raise ValueError("observation_size dimensions must be positive")
  return (int(height), int(width))
