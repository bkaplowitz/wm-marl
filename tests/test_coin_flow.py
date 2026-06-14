from __future__ import annotations

import jax
import numpy as np

from world_marl.coin_flow import (
  collect_random_joint_actions,
  decode_joint_actions,
  fit_joint_action_gmm,
  flow_joint_action_policy,
  normalize_joint_actions,
  sample_flow_points,
  train_flow_for_gmm,
)
from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter


def test_joint_action_gmm_roundtrip_and_collection(dummy_env_factory):
  adapter = MeltingPotVectorAdapter(num_envs=2, env_factory=dummy_env_factory)
  try:
    dataset = collect_random_joint_actions(
      adapter,
      np.random.default_rng(0),
      rollout_steps=5,
    )
  finally:
    adapter.close()

  assert dataset.joint_actions.shape == (10, 2)
  assert dataset.rewards.shape == (10, 2)
  assert dataset.action_dim == 3

  normalized = normalize_joint_actions(
    np.asarray([[0, 1], [2, 2]], dtype=np.int32),
    action_dim=3,
  )
  np.testing.assert_allclose(normalized, np.asarray([[-1.0, 0.0], [1.0, 1.0]]))
  decoded = decode_joint_actions(normalized, action_dim=3)
  np.testing.assert_array_equal(decoded, np.asarray([[0, 1], [2, 2]]))

  fitted = fit_joint_action_gmm(
    dataset.joint_actions,
    action_dim=dataset.action_dim,
    std=0.2,
  )
  assert fitted.gmm.means.shape[1] == 2
  assert fitted.action_pairs.shape[1] == 2
  np.testing.assert_allclose(np.asarray(fitted.gmm.weights).sum(), 1.0, atol=1e-6)


def test_flow_training_samples_joint_actions():
  joint_actions = np.asarray(
    [[0, 0], [0, 0], [2, 2], [2, 2], [2, 2]],
    dtype=np.int32,
  )
  fitted = fit_joint_action_gmm(joint_actions, action_dim=3, std=0.2)
  state, losses = train_flow_for_gmm(
    jax.random.PRNGKey(0),
    fitted.gmm,
    train_steps=2,
    batch_size=8,
    learning_rate=1e-3,
    hidden_dims=(8,),
  )

  assert len(losses) == 2
  points = sample_flow_points(
    state,
    jax.random.PRNGKey(1),
    num_samples=4,
    integration_steps=4,
  )
  assert points.shape == (4, 2)

  policy = flow_joint_action_policy(
    state,
    num_envs=2,
    action_dim=3,
    seed=2,
    integration_steps=4,
  )
  actions = policy(np.zeros((2, 2, 4, 4, 3), dtype=np.float32))
  assert actions.shape == (2, 2)
  assert actions.min() >= 0
  assert actions.max() < 3
