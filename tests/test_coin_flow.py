from __future__ import annotations

import jax
import numpy as np

from world_marl.coin_flow import (
  action_prediction_metrics,
  classifier_joint_action_policy,
  collect_policy_state_actions,
  compare_joint_action_distributions,
  collect_policy_joint_actions,
  collect_random_joint_actions,
  collect_random_state_actions,
  conditional_flow_joint_action_policy,
  fit_feature_normalizer,
  decode_joint_actions,
  fit_joint_action_gmm,
  flow_joint_action_policy,
  joint_action_counts,
  joint_action_probabilities,
  normalize_joint_actions,
  predict_action_logits,
  sample_conditional_action_flow_points,
  sample_flow_points,
  sampled_action_prediction_metrics,
  split_joint_actions,
  split_state_action_dataset,
  summarize_joint_action_distribution,
  train_action_classifier,
  train_conditional_action_flow,
  train_flow_for_gmm,
  uniform_joint_actions,
)
from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter


def test_joint_action_gmm_roundtrip_and_collection():
  adapter = JaxMARLCoinGameVectorAdapter(num_envs=2, max_cycles=5, seed=0)
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
  assert dataset.action_dim == 5

  normalized = normalize_joint_actions(
    np.asarray([[0, 2], [4, 4]], dtype=np.int32),
    action_dim=5,
  )
  np.testing.assert_allclose(normalized, np.asarray([[-1.0, 0.0], [1.0, 1.0]]))
  decoded = decode_joint_actions(normalized, action_dim=5)
  np.testing.assert_array_equal(decoded, np.asarray([[0, 2], [4, 4]]))

  fitted = fit_joint_action_gmm(
    dataset.joint_actions,
    action_dim=dataset.action_dim,
    std=0.2,
  )
  assert fitted.gmm.means.shape[1] == 2
  assert fitted.action_pairs.shape[1] == 2
  np.testing.assert_allclose(np.asarray(fitted.gmm.weights).sum(), 1.0, atol=1e-6)


def test_policy_joint_action_collection_uses_policy_actions():
  adapter = JaxMARLCoinGameVectorAdapter(num_envs=2, max_cycles=5, seed=0)
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
  assert dataset.rewards.shape == (8, 2)
  assert np.isfinite(dataset.rewards).all()


def test_state_action_collection_and_split():
  adapter = JaxMARLCoinGameVectorAdapter(num_envs=2, max_cycles=5, seed=0)
  try:
    random_dataset = collect_random_state_actions(
      adapter,
      np.random.default_rng(0),
      rollout_steps=3,
    )
  finally:
    adapter.close()

  assert random_dataset.state_features.shape == (6, 72)
  assert random_dataset.joint_actions.shape == (6, 2)
  assert random_dataset.rewards.shape == (6, 2)

  train_features, train_actions, heldout_features, heldout_actions = (
    split_state_action_dataset(
      random_dataset,
      validation_fraction=0.5,
      seed=0,
    )
  )
  assert train_features.shape == (3, 72)
  assert train_actions.shape == (3, 2)
  assert heldout_features.shape == (3, 72)
  assert heldout_actions.shape == (3, 2)

  normalizer = fit_feature_normalizer(train_features)
  normalized = normalizer.transform(train_features)
  assert normalized.shape == train_features.shape
  assert np.isfinite(normalized).all()


def test_policy_state_action_collection_uses_policy_actions():
  adapter = JaxMARLCoinGameVectorAdapter(num_envs=2, max_cycles=5, seed=0)
  try:
    dataset = collect_policy_state_actions(
      adapter,
      lambda observations: np.full(
        (observations.shape[0], observations.shape[1]),
        2,
        dtype=np.int32,
      ),
      rollout_steps=3,
    )
  finally:
    adapter.close()

  assert dataset.state_features.shape == (6, 72)
  np.testing.assert_array_equal(dataset.joint_actions, np.full((6, 2), 2))


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


def test_action_classifier_learns_synthetic_state_action_mapping():
  features = np.asarray(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 16,
    dtype=np.float32,
  )
  actions = np.asarray(
    [[0, 0], [1, 1], [1, 0], [0, 1]] * 16,
    dtype=np.int32,
  )
  normalizer = fit_feature_normalizer(features)
  features = normalizer.transform(features)

  initial_state, losses = train_action_classifier(
    jax.random.PRNGKey(0),
    features,
    actions,
    action_dim=2,
    num_agents=2,
    train_steps=60,
    batch_size=16,
    learning_rate=5e-2,
    hidden_dims=(16,),
  )
  del losses
  logits = predict_action_logits(initial_state, features)
  metrics = action_prediction_metrics(
    logits=logits,
    reference_actions=actions,
    train_actions=actions,
    action_dim=2,
  )

  assert metrics["cross_entropy"] < metrics["marginal_cross_entropy"]
  assert metrics["per_agent_accuracy"] > 0.95
  assert metrics["joint_accuracy"] > 0.9


def test_conditional_flow_samples_and_policy_have_expected_shapes():
  features = np.asarray(
    [
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 1.0, 1.0, 1.0],
      [1.0, 0.0, 0.0, 0.0],
      [1.0, 1.0, 0.0, 1.0],
    ]
    * 4,
    dtype=np.float32,
  )
  actions = np.asarray(
    [[0, 0], [1, 1], [1, 0], [0, 1]] * 4,
    dtype=np.int32,
  )
  normalizer = fit_feature_normalizer(features)
  normalized_features = normalizer.transform(features)
  state, losses = train_conditional_action_flow(
    jax.random.PRNGKey(0),
    normalized_features,
    actions,
    action_dim=2,
    train_steps=2,
    batch_size=8,
    learning_rate=1e-3,
    hidden_dims=(8,),
  )

  assert len(losses) == 2
  points = sample_conditional_action_flow_points(
    state,
    jax.random.PRNGKey(1),
    normalized_features[:5],
    integration_steps=4,
  )
  assert points.shape == (5, 2)
  sampled = decode_joint_actions(np.asarray(points), action_dim=2)
  metrics = sampled_action_prediction_metrics(
    sampled_actions=sampled,
    reference_actions=actions[:5],
    train_actions=actions,
    action_dim=2,
  )
  assert "distribution_vs_heldout" in metrics

  policy = conditional_flow_joint_action_policy(
    state,
    normalizer,
    action_dim=2,
    seed=2,
    integration_steps=4,
  )
  observations = features[:6].reshape((6, 2, 2))
  policy_actions = policy(observations)
  assert policy_actions.shape == (6, 2)
  assert policy_actions.min() >= 0
  assert policy_actions.max() < 2


def test_classifier_policy_shape():
  features = np.asarray(
    [
      [0.0, 0.0, 1.0, 0.0],
      [1.0, 1.0, 0.0, 1.0],
      [0.0, 1.0, 1.0, 1.0],
      [1.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
  )
  actions = np.asarray([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.int32)
  normalizer = fit_feature_normalizer(features)
  normalized = normalizer.transform(features)
  state, _ = train_action_classifier(
    jax.random.PRNGKey(0),
    normalized,
    actions,
    action_dim=2,
    num_agents=2,
    train_steps=2,
    batch_size=4,
    learning_rate=1e-2,
    hidden_dims=(8,),
  )
  policy = classifier_joint_action_policy(state, normalizer)
  observations = features.reshape((4, 2, 2))
  policy_actions = policy(observations)
  assert policy_actions.shape == (4, 2)
  assert policy_actions.min() >= 0
  assert policy_actions.max() < 2


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
