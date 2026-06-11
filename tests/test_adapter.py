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


def test_adapter_can_append_agent_id_channels(dummy_env_factory):
  adapter = MeltingPotVectorAdapter(
    num_envs=1,
    env_factory=dummy_env_factory,
    observation_size=4,
    append_agent_id=True,
  )
  try:
    observations = adapter.reset()
    assert observations.shape == (1, 2, 4, 4, 5)
    np.testing.assert_allclose(observations[0, 0, :, :, 3], 1.0)
    np.testing.assert_allclose(observations[0, 0, :, :, 4], 0.0)
    np.testing.assert_allclose(observations[0, 1, :, :, 3], 0.0)
    np.testing.assert_allclose(observations[0, 1, :, :, 4], 1.0)
  finally:
    adapter.close()


def test_adapter_can_append_scalar_observation_channels(dummy_env_factory):
  adapter = MeltingPotVectorAdapter(
    num_envs=1,
    env_factory=dummy_env_factory,
    observation_size=4,
    include_observation_scalars=True,
    append_agent_id=True,
  )
  try:
    observations = adapter.reset()
    assert adapter.scalar_observation_keys == (
      "COLLECTIVE_REWARD",
      "MISMATCHED_COIN_COLLECTED_BY_PARTNER",
    )
    assert adapter.observation_shape == (4, 4, 7)
    assert observations.shape == (1, 2, 4, 4, 7)

    np.testing.assert_allclose(observations[0, 0, :, :, 3], 0.0)
    np.testing.assert_allclose(observations[0, 0, :, :, 4], 0.0)
    np.testing.assert_allclose(observations[0, 1, :, :, 3], 0.0)
    np.testing.assert_allclose(observations[0, 1, :, :, 4], 1.0)

    np.testing.assert_allclose(observations[0, 0, :, :, 5], 1.0)
    np.testing.assert_allclose(observations[0, 0, :, :, 6], 0.0)
    np.testing.assert_allclose(observations[0, 1, :, :, 5], 0.0)
    np.testing.assert_allclose(observations[0, 1, :, :, 6], 1.0)
  finally:
    adapter.close()
