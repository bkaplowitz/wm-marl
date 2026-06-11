from __future__ import annotations

import numpy as np
import pytest


class DummyDiscrete:
  def __init__(self, n: int) -> None:
    self.n = n


class DummyBox:
  def __init__(self, shape: tuple[int, ...]) -> None:
    self.shape = shape


class DummyDictSpace:
  def __init__(self, shape: tuple[int, ...]) -> None:
    self.spaces = {
      "RGB": DummyBox(shape),
      "COLLECTIVE_REWARD": DummyBox(()),
      "MISMATCHED_COIN_COLLECTED_BY_PARTNER": DummyBox(()),
    }


class DummyParallelEnv:
  possible_agents = ["player_0", "player_1"]

  def __init__(self, horizon: int = 3, image_shape: tuple[int, int, int] = (8, 8, 3)):
    self.horizon = horizon
    self.image_shape = image_shape
    self.agents = list(self.possible_agents)
    self.steps = 0

  def observation_space(self, agent: str):
    del agent
    return DummyDictSpace(self.image_shape)

  def action_space(self, agent: str):
    del agent
    return DummyDiscrete(3)

  def reset(self):
    self.steps = 0
    self.agents = list(self.possible_agents)
    return self._obs(), {agent: {} for agent in self.possible_agents}

  def step(self, actions: dict[str, int]):
    self.steps += 1
    rewards = {
      agent: 1.0 if int(actions[agent]) == 1 else 0.0
      for agent in self.possible_agents
    }
    done = self.steps >= self.horizon
    terminations = {agent: done for agent in self.possible_agents}
    truncations = {agent: False for agent in self.possible_agents}
    infos = {agent: {} for agent in self.possible_agents}
    if done:
      self.agents = []
    return self._obs(), rewards, terminations, truncations, infos

  def close(self):
    pass

  def _obs(self):
    value = self.steps * 10
    rgb = np.full(self.image_shape, value, dtype=np.uint8)
    return {
      agent: {
        "RGB": rgb,
        "COLLECTIVE_REWARD": np.asarray(self.steps, dtype=np.float64),
        "MISMATCHED_COIN_COLLECTED_BY_PARTNER": np.asarray(
          agent_index,
          dtype=np.float64,
        ),
      }
      for agent_index, agent in enumerate(self.possible_agents)
    }


@pytest.fixture
def dummy_env_factory():
  return lambda: DummyParallelEnv()
