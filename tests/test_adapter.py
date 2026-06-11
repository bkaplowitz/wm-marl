from __future__ import annotations

import numpy as np

from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter


def test_adapter_shapes_and_auto_reset(dummy_env_factory):
  adapter = MeltingPotVectorAdapter(
    num_envs=2,
    env_factory=dummy_env_factory,
    observation_size=4,
  )
  try:
    observations = adapter.reset()
    assert observations.shape == (2, 2, 4, 4, 3)
    assert adapter.raw_observation_shape == (8, 8, 3)
    assert adapter.observation_shape == (4, 4, 3)
    assert observations.dtype == np.float32
    assert observations.max() <= 1.0
    assert adapter.action_dim == 3

    actions = np.ones((2, 2), dtype=np.int32)
    for _ in range(3):
      step = adapter.step(actions)
    assert step.observations.shape == (2, 2, 4, 4, 3)
    assert step.rewards.shape == (2, 2)
    assert step.dones.shape == (2, 2)
    assert len(step.completed_returns) == 2
    assert step.completed_returns[0] == (3.0, 3.0)
  finally:
    adapter.close()
