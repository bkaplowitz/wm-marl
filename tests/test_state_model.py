from __future__ import annotations

import jax
import numpy as np

from world_marl.checkpointing import load_params, save_checkpoint
from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter
from world_marl.state_model import (
  StateRepresentationConfig,
  WorldModelConfig,
  collect_transition_dataset,
  create_world_model_train_state,
  embed_observations,
  evaluate_state_fit,
  predict_world_model,
  prepare_transition_data,
  split_prepared_data,
  summarize_validation_criteria,
  train_world_model,
  transition_sampling_probabilities,
)


def test_state_embedding_shape_and_values():
  observations = np.zeros((1, 2, 4, 4, 1), dtype=np.float32)
  observations[0, 0, :, :, 0] = np.arange(16, dtype=np.float32).reshape(4, 4)
  observations[0, 1, :, :, 0] = 10.0

  features = embed_observations(
    observations,
    StateRepresentationConfig(pool_size=2, include_channel_stats=False),
  )

  assert features.shape == (1, 8)
  np.testing.assert_allclose(
    features[0, :4],
    np.asarray([2.5, 4.5, 10.5, 12.5], dtype=np.float32),
  )
  np.testing.assert_allclose(features[0, 4:], np.full((4,), 10.0, dtype=np.float32))


def test_transition_collection_and_preparation(dummy_env_factory):
  adapter = MeltingPotVectorAdapter(num_envs=2, env_factory=dummy_env_factory)
  try:
    dataset = collect_transition_dataset(
      adapter,
      np.random.default_rng(0),
      rollout_steps=4,
      policy_fn=lambda observations: np.ones(
        (observations.shape[0], observations.shape[1]),
        dtype=np.int32,
      ),
    )
  finally:
    adapter.close()

  assert dataset.obs.shape == (8, 2, 8, 8, 3)
  assert dataset.next_obs.shape == dataset.obs.shape
  assert dataset.actions.shape == (8, 2)
  np.testing.assert_array_equal(dataset.actions, np.ones((8, 2), dtype=np.int32))
  np.testing.assert_allclose(dataset.rewards, np.ones((8, 2), dtype=np.float32))

  prepared = prepare_transition_data(
    dataset,
    StateRepresentationConfig(pool_size=2, include_channel_stats=True),
  )
  assert prepared.state_features.shape[0] == 8
  assert prepared.next_state_features.shape == prepared.state_features.shape
  assert prepared.feature_dim == 2 * ((2 * 2 * 3) + (4 * 3))


def test_state_world_model_training_metrics_and_reload(tmp_path, dummy_env_factory):
  adapter = MeltingPotVectorAdapter(num_envs=2, env_factory=dummy_env_factory)
  try:
    dataset = collect_transition_dataset(
      adapter,
      np.random.default_rng(0),
      rollout_steps=8,
    )
  finally:
    adapter.close()

  prepared = prepare_transition_data(
    dataset,
    StateRepresentationConfig(pool_size=2, include_channel_stats=True),
  )
  train_data, validation_data = split_prepared_data(
    prepared,
    validation_fraction=0.25,
    seed=0,
  )
  config = WorldModelConfig(
    hidden_dims=(16,),
    learning_rate=1e-3,
    batch_size=4,
    train_steps=4,
  )
  state, rows = train_world_model(
    jax.random.PRNGKey(0),
    train_data,
    config=config,
  )
  assert len(rows) == 4
  assert np.isfinite([row["loss"] for row in rows]).all()

  predictions = predict_world_model(state, validation_data)
  assert predictions.next_state_features.shape == validation_data.next_state_features.shape
  assert predictions.rewards.shape == validation_data.rewards.shape
  assert predictions.reward_event_logits.shape == validation_data.rewards.shape
  metrics = evaluate_state_fit(train_data, validation_data, predictions, seed=0)
  assert metrics["next_state"]["model_mse"] >= 0.0
  assert metrics["delta_state"]["model_mse"] >= 0.0
  assert metrics["changed_features"]["feature_count"] >= 1
  assert "delta_model_beats_zero" in metrics["changed_features"]
  assert metrics["reward"]["model_mse"] >= 0.0
  assert metrics["reward_event"]["model_bce"] >= 0.0
  assert "best_f1" in metrics["reward_event"]
  assert metrics["policy"]["model_cross_entropy"] > 0.0
  passed, criteria = summarize_validation_criteria(
    metrics,
    finite_losses=True,
    reload_passed=True,
  )
  assert isinstance(passed, bool)
  assert "transition_model_has_signal" in criteria
  assert "reward_model_has_signal" in criteria

  save_checkpoint(tmp_path / "checkpoint", state, metadata={"kind": "test"})
  reload_state = create_world_model_train_state(
    jax.random.PRNGKey(1),
    feature_dim=validation_data.feature_dim,
    num_agents=validation_data.num_agents,
    action_dim=validation_data.action_dim,
    config=config,
  )
  reload_params = load_params(
    tmp_path / "checkpoint" / "checkpoint.msgpack",
    reload_state.params,
  )
  reload_state = reload_state.replace(params=reload_params)
  reload_predictions = predict_world_model(reload_state, validation_data)
  np.testing.assert_allclose(
    reload_predictions.next_state_features,
    predictions.next_state_features,
    atol=0.0,
  )


def test_transition_sampler_emphasizes_reward_events(dummy_env_factory):
  adapter = MeltingPotVectorAdapter(num_envs=2, env_factory=dummy_env_factory)
  try:
    dataset = collect_transition_dataset(
      adapter,
      np.random.default_rng(0),
      rollout_steps=8,
    )
  finally:
    adapter.close()

  prepared = prepare_transition_data(
    dataset,
    StateRepresentationConfig(pool_size=2, include_channel_stats=True),
  )
  prepared.rewards[:2] = 1.0
  prepared.rewards[2:] = 0.0
  changed_mask = np.ones(prepared.feature_dim, dtype=bool)
  probabilities = transition_sampling_probabilities(
    prepared,
    changed_mask=changed_mask,
    reward_event_epsilon=1e-6,
    reward_oversample_factor=8.0,
    delta_oversample_factor=0.0,
  )

  assert np.isclose(probabilities.sum(), 1.0)
  assert probabilities[:2].mean() > probabilities[2:].mean()


def test_state_model_cli_smoke(monkeypatch, tmp_path, dummy_env_factory):
  from world_marl.scripts import validate_state_model

  def make_dummy_adapter(args):
    return MeltingPotVectorAdapter(
      num_envs=args.num_envs,
      env_factory=dummy_env_factory,
    )

  monkeypatch.setattr(validate_state_model, "make_adapter", make_dummy_adapter)
  monkeypatch.setattr(
    "sys.argv",
    [
      "world-marl-validate-state-model",
      "--num-envs",
      "1",
      "--collect-steps",
      "5",
      "--train-steps",
      "2",
      "--batch-size",
      "2",
      "--hidden-dims",
      "8",
      "--pool-size",
      "2",
      "--recovery-examples",
      "2",
      "--out-dir",
      str(tmp_path),
      "--quiet",
    ],
  )
  validate_state_model.main()

  run_dirs = list(tmp_path.glob("state_model_*"))
  assert len(run_dirs) == 1
  assert (run_dirs[0] / "prediction_metrics.json").exists()
  assert (run_dirs[0] / "prediction_dashboard.png").exists()
  assert (run_dirs[0] / "state_recoveries.png").exists()
  assert (run_dirs[0] / "checkpoint" / "checkpoint.msgpack").exists()
