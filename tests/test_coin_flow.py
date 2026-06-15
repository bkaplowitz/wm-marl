from __future__ import annotations

import jax
import numpy as np

from world_marl.coin_flow import (
  compare_joint_action_distributions,
  collect_policy_joint_actions,
  collect_random_joint_actions,
  decode_joint_actions,
  fit_joint_action_gmm,
  flow_joint_action_policy,
  joint_action_counts,
  joint_action_probabilities,
  normalize_joint_actions,
  sample_flow_points,
  split_joint_actions,
  summarize_joint_action_distribution,
  train_flow_for_gmm,
  uniform_joint_actions,
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


def test_policy_joint_action_collection_uses_policy_actions(dummy_env_factory):
  adapter = MeltingPotVectorAdapter(num_envs=2, env_factory=dummy_env_factory)
  try:
    dataset = collect_policy_joint_actions(
      adapter,
      lambda observations: np.ones(
        (observations.shape[0], observations.shape[1]),
        dtype=np.int32,
      ),
      rollout_steps=4,
    )
  finally:
    adapter.close()

  assert dataset.joint_actions.shape == (8, 2)
  np.testing.assert_array_equal(dataset.joint_actions, np.ones((8, 2), dtype=np.int32))
  np.testing.assert_allclose(dataset.rewards, np.ones((8, 2), dtype=np.float32))


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


def test_joint_action_distribution_metrics_identify_better_match():
  reference = np.asarray(
    [[0, 0], [0, 0], [0, 0], [1, 1]],
    dtype=np.int32,
  )
  good_candidate = np.asarray(
    [[0, 0], [0, 0], [0, 0], [1, 1]],
    dtype=np.int32,
  )
  bad_candidate = np.asarray(
    [[2, 2], [2, 2], [2, 2], [1, 1]],
    dtype=np.int32,
  )

  counts = joint_action_counts(reference, action_dim=3)
  assert counts[0, 0] == 3
  assert counts[1, 1] == 1
  probabilities = joint_action_probabilities(reference, action_dim=3)
  np.testing.assert_allclose(probabilities.sum(), 1.0)

  good = compare_joint_action_distributions(
    reference,
    good_candidate,
    action_dim=3,
    top_k=2,
  )
  bad = compare_joint_action_distributions(
    reference,
    bad_candidate,
    action_dim=3,
    top_k=2,
  )
  assert good["js_divergence"] < bad["js_divergence"]
  assert good["total_variation"] < bad["total_variation"]
  assert good["mode_matches"]
  assert not bad["mode_matches"]

  summary = summarize_joint_action_distribution(reference, action_dim=3, top_k=2)
  assert summary["top_pairs"][0]["action_pair"] == [0, 0]


def test_joint_action_split_and_uniform_sampler():
  actions = np.asarray(
    [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]],
    dtype=np.int32,
  )
  train_actions, validation_actions = split_joint_actions(
    actions,
    validation_fraction=0.4,
    seed=0,
  )
  assert train_actions.shape == (3, 2)
  assert validation_actions.shape == (2, 2)

  uniform = uniform_joint_actions(
    np.random.default_rng(0),
    num_samples=10,
    action_dim=3,
  )
  assert uniform.shape == (10, 2)
  assert uniform.min() >= 0
  assert uniform.max() < 3
