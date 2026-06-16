"""State-representation fit validation for Melting Pot rollouts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter


PolicyFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class TransitionDataset:
  """Full multi-agent transitions collected from a vectorized environment."""

  obs: np.ndarray
  actions: np.ndarray
  rewards: np.ndarray
  dones: np.ndarray
  next_obs: np.ndarray
  action_dim: int
  num_agents: int
  num_envs: int
  rollout_steps: int
  completed_returns: tuple[tuple[float, ...], ...]
  completed_lengths: tuple[int, ...]

  @property
  def num_transitions(self) -> int:
    return int(self.actions.shape[0])

  @property
  def observation_shape(self) -> tuple[int, ...]:
    return tuple(int(dim) for dim in self.obs.shape[2:])

  def to_metadata(self) -> dict[str, Any]:
    return {
      "num_transitions": self.num_transitions,
      "action_dim": self.action_dim,
      "num_agents": self.num_agents,
      "num_envs": self.num_envs,
      "rollout_steps": self.rollout_steps,
      "observation_shape": self.observation_shape,
      "mean_reward_per_agent": self.rewards.mean(axis=0).tolist(),
      "done_fraction_per_agent": self.dones.mean(axis=0).tolist(),
      "completed_episodes": len(self.completed_returns),
      "completed_return_mean": (
        float(np.asarray(self.completed_returns, dtype=np.float32).mean())
        if self.completed_returns
        else None
      ),
      "completed_lengths": list(self.completed_lengths),
    }


@dataclass(frozen=True)
class StateRepresentationConfig:
  """Deterministic, inspectable observation embedding settings."""

  pool_size: int = 4
  include_channel_stats: bool = True


@dataclass(frozen=True)
class FeatureNormalizer:
  """Mean/std feature normalization used by the supervised model."""

  mean: np.ndarray
  std: np.ndarray

  def normalize(self, values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

  def denormalize(self, values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) * self.std + self.mean

  def to_metadata(self) -> dict[str, Any]:
    return {
      "mean_shape": list(self.mean.shape),
      "std_shape": list(self.std.shape),
      "mean_preview": self.mean[: min(10, self.mean.shape[0])].tolist(),
      "std_preview": self.std[: min(10, self.std.shape[0])].tolist(),
    }


@dataclass(frozen=True)
class PreparedTransitionData:
  """Transition tensors after deterministic state embedding."""

  state_features: np.ndarray
  next_state_features: np.ndarray
  actions: np.ndarray
  rewards: np.ndarray
  dones: np.ndarray
  obs: np.ndarray
  next_obs: np.ndarray
  action_dim: int
  num_agents: int
  normalizer: FeatureNormalizer
  representation_config: StateRepresentationConfig

  @property
  def num_transitions(self) -> int:
    return int(self.actions.shape[0])

  @property
  def feature_dim(self) -> int:
    return int(self.state_features.shape[1])


@dataclass(frozen=True)
class WorldModelConfig:
  """Training configuration for the first supervised world-model fit."""

  hidden_dims: tuple[int, ...] = (256, 256)
  learning_rate: float = 1e-3
  batch_size: int = 256
  train_steps: int = 1000
  next_loss_weight: float = 1.0
  reward_loss_weight: float = 1.0
  done_loss_weight: float = 0.1
  policy_loss_weight: float = 0.1


@dataclass(frozen=True)
class WorldModelPredictions:
  """Raw-space predictions from the supervised state-fit model."""

  next_state_features: np.ndarray
  rewards: np.ndarray
  done_logits: np.ndarray
  policy_logits: np.ndarray


class StateFitWorldModel(nn.Module):
  """Small MLP that predicts next features, rewards, dones, and behavior policy."""

  feature_dim: int
  num_agents: int
  action_dim: int
  hidden_dims: tuple[int, ...] = (256, 256)

  @nn.compact
  def __call__(self, state_features: jax.Array, actions: jax.Array) -> dict[str, jax.Array]:
    action_features = jax.nn.one_hot(
      actions.astype(jnp.int32),
      self.action_dim,
    ).reshape((state_features.shape[0], -1))
    transition_input = jnp.concatenate([state_features, action_features], axis=-1)
    transition_hidden = transition_input
    for hidden_dim in self.hidden_dims:
      transition_hidden = nn.Dense(hidden_dim)(transition_hidden)
      transition_hidden = nn.silu(transition_hidden)

    policy_hidden = state_features
    for hidden_dim in self.hidden_dims:
      policy_hidden = nn.Dense(hidden_dim)(policy_hidden)
      policy_hidden = nn.silu(policy_hidden)

    return {
      "next_state_features": nn.Dense(self.feature_dim)(transition_hidden),
      "rewards": nn.Dense(self.num_agents)(transition_hidden),
      "done_logits": nn.Dense(self.num_agents)(transition_hidden),
      "policy_logits": nn.Dense(self.num_agents * self.action_dim)(policy_hidden).reshape(
        (state_features.shape[0], self.num_agents, self.action_dim)
      ),
    }


def collect_transition_dataset(
  adapter: MeltingPotVectorAdapter,
  rng: np.random.Generator,
  *,
  rollout_steps: int,
  policy_fn: PolicyFn | None = None,
  progress_callback: Callable[[int], None] | None = None,
) -> TransitionDataset:
  """Collect real multi-agent transitions from the adapter."""
  if rollout_steps < 1:
    raise ValueError("rollout_steps must be >= 1")

  obs_rows = []
  action_rows = []
  reward_rows = []
  done_rows = []
  next_obs_rows = []
  completed_returns: list[tuple[float, ...]] = []
  completed_lengths: list[int] = []

  observations = adapter.reset()
  for step_index in range(rollout_steps):
    if policy_fn is None:
      actions = adapter.sample_actions(rng)
    else:
      actions = np.asarray(policy_fn(observations), dtype=np.int32)
    expected_shape = (adapter.num_envs, adapter.num_agents)
    if actions.shape != expected_shape:
      raise ValueError(f"actions must have shape {expected_shape}, got {actions.shape}")

    step = adapter.step(actions)
    obs_rows.append(observations.copy())
    action_rows.append(actions.copy())
    reward_rows.append(step.rewards.copy())
    done_rows.append(step.dones.copy())
    next_obs_rows.append(step.observations.copy())
    completed_returns.extend(step.completed_returns)
    completed_lengths.extend(step.completed_lengths)
    observations = step.observations
    if progress_callback is not None:
      progress_callback(step_index + 1)

  obs = np.concatenate(obs_rows, axis=0).astype(np.float32)
  actions = np.concatenate(action_rows, axis=0).astype(np.int32)
  rewards = np.concatenate(reward_rows, axis=0).astype(np.float32)
  dones = np.concatenate(done_rows, axis=0).astype(np.float32)
  next_obs = np.concatenate(next_obs_rows, axis=0).astype(np.float32)
  return TransitionDataset(
    obs=obs,
    actions=actions,
    rewards=rewards,
    dones=dones,
    next_obs=next_obs,
    action_dim=adapter.action_dim,
    num_agents=adapter.num_agents,
    num_envs=adapter.num_envs,
    rollout_steps=rollout_steps,
    completed_returns=tuple(completed_returns),
    completed_lengths=tuple(completed_lengths),
  )


def embed_observations(
  observations: np.ndarray,
  config: StateRepresentationConfig,
) -> np.ndarray:
  """Embed observations into deterministic joint-state feature vectors."""
  observations = np.asarray(observations, dtype=np.float32)
  if observations.ndim != 5:
    raise ValueError(
      "expected observations shaped [transition, agent, height, width, channel], "
      f"got {observations.shape}"
    )
  if config.pool_size < 1:
    raise ValueError("pool_size must be >= 1")

  num_transitions, num_agents, height, width, channels = observations.shape
  flat = observations.reshape((num_transitions * num_agents, height, width, channels))
  pooled = average_pool_observations(flat, pool_size=config.pool_size).reshape(
    (num_transitions, num_agents, -1)
  )
  features = [pooled]
  if config.include_channel_stats:
    stats = np.concatenate(
      [
        flat.mean(axis=(1, 2)),
        flat.std(axis=(1, 2)),
        flat.min(axis=(1, 2)),
        flat.max(axis=(1, 2)),
      ],
      axis=-1,
    ).reshape((num_transitions, num_agents, -1))
    features.append(stats)
  per_agent_features = np.concatenate(features, axis=-1)
  return per_agent_features.reshape((num_transitions, -1)).astype(np.float32)


def average_pool_observations(observations: np.ndarray, *, pool_size: int) -> np.ndarray:
  """Average-pool `[N, H, W, C]` observations into `[N, pool, pool, C]`."""
  observations = np.asarray(observations, dtype=np.float32)
  if observations.ndim != 4:
    raise ValueError(f"expected observations shaped [N, H, W, C], got {observations.shape}")
  height, width = observations.shape[1:3]
  row_edges = _pool_edges(height, pool_size)
  col_edges = _pool_edges(width, pool_size)
  pooled = np.zeros(
    (observations.shape[0], pool_size, pool_size, observations.shape[-1]),
    dtype=np.float32,
  )
  for row in range(pool_size):
    for col in range(pool_size):
      pooled[:, row, col] = observations[
        :,
        row_edges[row] : row_edges[row + 1],
        col_edges[col] : col_edges[col + 1],
      ].mean(axis=(1, 2))
  return pooled


def prepare_transition_data(
  dataset: TransitionDataset,
  representation_config: StateRepresentationConfig,
) -> PreparedTransitionData:
  """Embed observations and build a feature normalizer."""
  state_features = embed_observations(dataset.obs, representation_config)
  next_state_features = embed_observations(dataset.next_obs, representation_config)
  mean = state_features.mean(axis=0).astype(np.float32)
  std = state_features.std(axis=0).astype(np.float32)
  std = np.maximum(std, 1e-6)
  return PreparedTransitionData(
    state_features=state_features,
    next_state_features=next_state_features,
    actions=dataset.actions,
    rewards=dataset.rewards,
    dones=dataset.dones,
    obs=dataset.obs,
    next_obs=dataset.next_obs,
    action_dim=dataset.action_dim,
    num_agents=dataset.num_agents,
    normalizer=FeatureNormalizer(mean=mean, std=std),
    representation_config=representation_config,
  )


def split_prepared_data(
  data: PreparedTransitionData,
  *,
  validation_fraction: float,
  seed: int,
) -> tuple[PreparedTransitionData, PreparedTransitionData]:
  """Shuffle transitions into train and heldout splits."""
  if not 0.0 < validation_fraction < 1.0:
    raise ValueError("validation_fraction must be between 0 and 1")
  if data.num_transitions < 2:
    raise ValueError("at least two transitions are required")
  validation_size = int(round(data.num_transitions * validation_fraction))
  validation_size = min(max(1, validation_size), data.num_transitions - 1)
  indices = np.random.default_rng(seed).permutation(data.num_transitions)
  validation_indices = indices[:validation_size]
  train_indices = indices[validation_size:]
  return (
    _take_prepared_data(data, train_indices),
    _take_prepared_data(data, validation_indices),
  )


def create_world_model_train_state(
  rng: jax.Array,
  *,
  feature_dim: int,
  num_agents: int,
  action_dim: int,
  config: WorldModelConfig,
) -> TrainState:
  """Initialize the supervised world-model TrainState."""
  model = StateFitWorldModel(
    feature_dim=feature_dim,
    num_agents=num_agents,
    action_dim=action_dim,
    hidden_dims=config.hidden_dims,
  )
  params = model.init(
    rng,
    jnp.zeros((1, feature_dim), dtype=jnp.float32),
    jnp.zeros((1, num_agents), dtype=jnp.int32),
  )["params"]
  return TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.adam(config.learning_rate),
  )


def train_world_model(
  rng: jax.Array,
  train_data: PreparedTransitionData,
  *,
  config: WorldModelConfig,
  progress_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[TrainState, list[dict[str, float]]]:
  """Train the supervised state-fit world model."""
  if config.train_steps < 1:
    raise ValueError("train_steps must be >= 1")
  if config.batch_size < 1:
    raise ValueError("batch_size must be >= 1")

  normalized_state = jnp.asarray(
    train_data.normalizer.normalize(train_data.state_features),
    dtype=jnp.float32,
  )
  normalized_next = jnp.asarray(
    train_data.normalizer.normalize(train_data.next_state_features),
    dtype=jnp.float32,
  )
  actions = jnp.asarray(train_data.actions, dtype=jnp.int32)
  rewards = jnp.asarray(train_data.rewards, dtype=jnp.float32)
  dones = jnp.asarray(train_data.dones, dtype=jnp.float32)
  train_size = int(train_data.num_transitions)

  rng, init_key = jax.random.split(rng)
  state = create_world_model_train_state(
    init_key,
    feature_dim=train_data.feature_dim,
    num_agents=train_data.num_agents,
    action_dim=train_data.action_dim,
    config=config,
  )

  @jax.jit
  def train_step(state: TrainState, step_key: jax.Array):
    batch_indices = jax.random.randint(
      step_key,
      (config.batch_size,),
      minval=0,
      maxval=train_size,
    )

    batch = {
      "state": normalized_state[batch_indices],
      "next": normalized_next[batch_indices],
      "actions": actions[batch_indices],
      "rewards": rewards[batch_indices],
      "dones": dones[batch_indices],
    }

    def loss_fn(params):
      predictions = state.apply_fn(
        {"params": params},
        batch["state"],
        batch["actions"],
      )
      losses = world_model_losses(
        predictions,
        batch,
        action_dim=train_data.action_dim,
      )
      total = (
        config.next_loss_weight * losses["next_mse"]
        + config.reward_loss_weight * losses["reward_mse"]
        + config.done_loss_weight * losses["done_bce"]
        + config.policy_loss_weight * losses["policy_ce"]
      )
      return total, {**losses, "loss": total}

    (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    del loss
    return state.apply_gradients(grads=grads), losses

  rows: list[dict[str, float]] = []
  for step_index in range(config.train_steps):
    rng, step_key = jax.random.split(rng)
    state, losses = train_step(state, step_key)
    row = {key: float(value) for key, value in losses.items()}
    rows.append(row)
    if progress_callback is not None:
      progress_callback(step_index + 1, row)
  jax.block_until_ready(state.params)
  return state, rows


def world_model_losses(
  predictions: dict[str, jax.Array],
  batch: dict[str, jax.Array],
  *,
  action_dim: int,
) -> dict[str, jax.Array]:
  """Compute supervised component losses."""
  next_mse = jnp.mean(jnp.square(predictions["next_state_features"] - batch["next"]))
  reward_mse = jnp.mean(jnp.square(predictions["rewards"] - batch["rewards"]))
  done_bce = jnp.mean(
    optax.sigmoid_binary_cross_entropy(
      predictions["done_logits"],
      batch["dones"],
    )
  )
  action_targets = jax.nn.one_hot(batch["actions"], action_dim)
  policy_ce = -jnp.mean(
    jnp.sum(jax.nn.log_softmax(predictions["policy_logits"], axis=-1) * action_targets, axis=-1)
  )
  return {
    "next_mse": next_mse,
    "reward_mse": reward_mse,
    "done_bce": done_bce,
    "policy_ce": policy_ce,
  }


def predict_world_model(
  train_state: TrainState,
  data: PreparedTransitionData,
) -> WorldModelPredictions:
  """Run model predictions on a prepared dataset and return raw feature space."""
  normalized_state = data.normalizer.normalize(data.state_features)
  predictions = train_state.apply_fn(
    {"params": train_state.params},
    jnp.asarray(normalized_state, dtype=jnp.float32),
    jnp.asarray(data.actions, dtype=jnp.int32),
  )
  next_normalized = np.asarray(predictions["next_state_features"], dtype=np.float32)
  return WorldModelPredictions(
    next_state_features=data.normalizer.denormalize(next_normalized),
    rewards=np.asarray(predictions["rewards"], dtype=np.float32),
    done_logits=np.asarray(predictions["done_logits"], dtype=np.float32),
    policy_logits=np.asarray(predictions["policy_logits"], dtype=np.float32),
  )


def evaluate_state_fit(
  train_data: PreparedTransitionData,
  validation_data: PreparedTransitionData,
  predictions: WorldModelPredictions,
  *,
  seed: int = 0,
) -> dict[str, Any]:
  """Compare model predictions against simple state/reward/policy baselines."""
  mean_next = np.repeat(
    train_data.next_state_features.mean(axis=0, keepdims=True),
    validation_data.num_transitions,
    axis=0,
  )
  persistence = validation_data.state_features
  model_next = predictions.next_state_features
  true_next = validation_data.next_state_features

  reward_mean = np.repeat(
    train_data.rewards.mean(axis=0, keepdims=True),
    validation_data.num_transitions,
    axis=0,
  )
  done_prob = np.clip(train_data.dones.mean(axis=0), 1e-6, 1.0 - 1e-6)
  no_done = np.zeros_like(validation_data.dones)
  policy_marginals = action_marginals(
    train_data.actions,
    action_dim=train_data.action_dim,
    num_agents=train_data.num_agents,
  )

  policy_probs = softmax_np(predictions.policy_logits, axis=-1)
  policy_actions = np.argmax(policy_probs, axis=-1)
  marginal_actions = np.argmax(policy_marginals, axis=-1)
  marginal_action_rows = np.repeat(
    marginal_actions.reshape(1, -1),
    validation_data.num_transitions,
    axis=0,
  )

  metrics = {
    "next_state": {
      "model_mse": mse(model_next, true_next),
      "mean_baseline_mse": mse(mean_next, true_next),
      "persistence_baseline_mse": mse(persistence, true_next),
      "model_beats_mean": bool(mse(model_next, true_next) < mse(mean_next, true_next)),
      "model_beats_persistence": bool(
        mse(model_next, true_next) < mse(persistence, true_next)
      ),
    },
    "state_distribution": {
      "model": distribution_metrics(true_next, model_next, seed=seed),
      "mean_baseline": distribution_metrics(true_next, mean_next, seed=seed),
      "persistence_baseline": distribution_metrics(true_next, persistence, seed=seed),
    },
    "reward": {
      "model_mse": mse(predictions.rewards, validation_data.rewards),
      "mean_baseline_mse": mse(reward_mean, validation_data.rewards),
      "model_beats_mean": bool(
        mse(predictions.rewards, validation_data.rewards)
        < mse(reward_mean, validation_data.rewards)
      ),
    },
    "done": {
      "model_bce": binary_cross_entropy_from_logits(
        predictions.done_logits,
        validation_data.dones,
      ),
      "mean_prob_bce": binary_cross_entropy_np(
        np.repeat(done_prob.reshape(1, -1), validation_data.num_transitions, axis=0),
        validation_data.dones,
      ),
      "no_done_accuracy": float((no_done == validation_data.dones).mean()),
      "model_accuracy": float(
        ((predictions.done_logits > 0.0).astype(np.float32) == validation_data.dones).mean()
      ),
    },
    "policy": {
      "model_cross_entropy": categorical_cross_entropy(
        policy_probs,
        validation_data.actions,
      ),
      "marginal_cross_entropy": categorical_cross_entropy(
        np.repeat(
          policy_marginals.reshape(1, *policy_marginals.shape),
          validation_data.num_transitions,
          axis=0,
        ),
        validation_data.actions,
      ),
      "model_accuracy": float((policy_actions == validation_data.actions).mean()),
      "marginal_mode_accuracy": float(
        (marginal_action_rows == validation_data.actions).mean()
      ),
      "model_beats_marginal_ce": bool(
        categorical_cross_entropy(policy_probs, validation_data.actions)
        < categorical_cross_entropy(
          np.repeat(
            policy_marginals.reshape(1, *policy_marginals.shape),
            validation_data.num_transitions,
            axis=0,
          ),
          validation_data.actions,
        )
      ),
    },
  }
  return metrics


def state_recovery_examples(
  validation_data: PreparedTransitionData,
  predictions: WorldModelPredictions,
  *,
  num_examples: int = 6,
  seed: int = 0,
) -> list[dict[str, Any]]:
  """Find nearest heldout states to predicted next-state features."""
  rng = np.random.default_rng(seed)
  count = min(num_examples, validation_data.num_transitions)
  indices = rng.choice(validation_data.num_transitions, size=count, replace=False)
  distances = pairwise_squared_distances(
    predictions.next_state_features[indices],
    validation_data.next_state_features,
  )
  nearest = np.argmin(distances, axis=1)
  return [
    {
      "index": int(index),
      "nearest_index": int(nearest_index),
      "nearest_feature_l2": float(np.sqrt(distances[row, nearest_index])),
      "true_reward": validation_data.rewards[index].tolist(),
      "predicted_reward": predictions.rewards[index].tolist(),
      "actions": validation_data.actions[index].astype(int).tolist(),
    }
    for row, (index, nearest_index) in enumerate(zip(indices, nearest, strict=True))
  ]


def plot_prediction_dashboard(
  output_path: str,
  train_data: PreparedTransitionData,
  validation_data: PreparedTransitionData,
  predictions: WorldModelPredictions,
  metrics: dict[str, Any],
  *,
  seed: int = 0,
) -> None:
  """Write a compact dashboard for state, reward, and policy fit quality."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  rng = np.random.default_rng(seed)
  count = min(500, validation_data.num_transitions)
  indices = rng.choice(validation_data.num_transitions, size=count, replace=False)
  projected = pca_project(
    {
      "heldout next": validation_data.next_state_features[indices],
      "model pred": predictions.next_state_features[indices],
      "persistence": validation_data.state_features[indices],
      "train": train_data.next_state_features[
        rng.choice(train_data.num_transitions, size=min(count, train_data.num_transitions), replace=False)
      ],
    }
  )

  fig = plt.figure(figsize=(14, 9))
  grid = fig.add_gridspec(2, 2)
  ax = fig.add_subplot(grid[0, 0])
  for label, points in projected.items():
    ax.scatter(points[:, 0], points[:, 1], s=12, alpha=0.55, label=label)
  ax.set_title("PCA of state representation")
  ax.set_xlabel("PC1")
  ax.set_ylabel("PC2")
  ax.grid(True, alpha=0.25)
  ax.legend(fontsize=8)

  ax = fig.add_subplot(grid[0, 1])
  labels = ["model", "mean", "persistence"]
  values = [
    metrics["next_state"]["model_mse"],
    metrics["next_state"]["mean_baseline_mse"],
    metrics["next_state"]["persistence_baseline_mse"],
  ]
  ax.bar(labels, values)
  ax.set_title("Next-state feature MSE")
  ax.grid(True, axis="y", alpha=0.25)

  ax = fig.add_subplot(grid[1, 0])
  reward_true = validation_data.rewards.reshape(-1)
  reward_pred = predictions.rewards.reshape(-1)
  ax.scatter(reward_true, reward_pred, s=14, alpha=0.55)
  low = min(float(reward_true.min(initial=0.0)), float(reward_pred.min(initial=0.0)))
  high = max(float(reward_true.max(initial=1.0)), float(reward_pred.max(initial=1.0)))
  ax.plot([low, high], [low, high], color="black", linewidth=1)
  ax.set_title("Reward prediction")
  ax.set_xlabel("true reward")
  ax.set_ylabel("predicted reward")
  ax.grid(True, alpha=0.25)

  ax = fig.add_subplot(grid[1, 1])
  labels = ["reward MSE", "policy CE", "done BCE"]
  model_values = [
    metrics["reward"]["model_mse"],
    metrics["policy"]["model_cross_entropy"],
    metrics["done"]["model_bce"],
  ]
  baseline_values = [
    metrics["reward"]["mean_baseline_mse"],
    metrics["policy"]["marginal_cross_entropy"],
    metrics["done"]["mean_prob_bce"],
  ]
  positions = np.arange(len(labels))
  width = 0.38
  ax.bar(positions - width / 2, model_values, width, label="model")
  ax.bar(positions + width / 2, baseline_values, width, label="baseline")
  ax.set_xticks(positions, labels)
  ax.set_title("Auxiliary recovery heads")
  ax.grid(True, axis="y", alpha=0.25)
  ax.legend(fontsize=8)

  fig.suptitle("State-Representation World-Model Fit Validation", fontsize=14)
  fig.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)


def plot_state_recoveries(
  output_path: str,
  validation_data: PreparedTransitionData,
  predictions: WorldModelPredictions,
  *,
  num_examples: int = 6,
  seed: int = 0,
) -> list[dict[str, Any]]:
  """Plot current, true next, and nearest recovered next observation."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  examples = state_recovery_examples(
    validation_data,
    predictions,
    num_examples=num_examples,
    seed=seed,
  )
  if not examples:
    return []

  fig, axes = plt.subplots(len(examples), 3, figsize=(8, 2.4 * len(examples)))
  if len(examples) == 1:
    axes = np.asarray([axes])
  for row, example in enumerate(examples):
    index = example["index"]
    nearest_index = example["nearest_index"]
    panels = [
      ("current", validation_data.obs[index, 0]),
      ("true next", validation_data.next_obs[index, 0]),
      ("nearest recovered", validation_data.next_obs[nearest_index, 0]),
    ]
    for col, (title, obs) in enumerate(panels):
      axes[row, col].imshow(to_rgb_image(obs))
      axes[row, col].set_title(title if row == 0 else "")
      axes[row, col].axis("off")
    axes[row, 0].set_ylabel(
      f"a={example['actions']}\nL2={example['nearest_feature_l2']:.3g}",
      rotation=0,
      ha="right",
      va="center",
      fontsize=8,
    )
  fig.suptitle("Predicted State Features Recovered by Nearest Heldout Frames")
  fig.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)
  return examples


def softmax_np(values: np.ndarray, axis: int = -1) -> np.ndarray:
  shifted = values - values.max(axis=axis, keepdims=True)
  exp_values = np.exp(shifted)
  return exp_values / exp_values.sum(axis=axis, keepdims=True)


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
  return float(np.mean(np.square(np.asarray(prediction) - np.asarray(target))))


def binary_cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
  logits = np.asarray(logits, dtype=np.float64)
  targets = np.asarray(targets, dtype=np.float64)
  return float(np.mean(np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))))


def binary_cross_entropy_np(probs: np.ndarray, targets: np.ndarray) -> float:
  probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-8, 1.0 - 1e-8)
  targets = np.asarray(targets, dtype=np.float64)
  return float(-np.mean(targets * np.log(probs) + (1.0 - targets) * np.log(1.0 - probs)))


def categorical_cross_entropy(probs: np.ndarray, actions: np.ndarray) -> float:
  probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-8, 1.0)
  actions = np.asarray(actions, dtype=np.int32)
  rows = np.arange(actions.shape[0])[:, None]
  agents = np.arange(actions.shape[1])[None, :]
  selected = probs[rows, agents, actions]
  return float(-np.mean(np.log(selected)))


def action_marginals(actions: np.ndarray, *, action_dim: int, num_agents: int) -> np.ndarray:
  counts = np.zeros((num_agents, action_dim), dtype=np.float64)
  for agent_index in range(num_agents):
    np.add.at(counts[agent_index], actions[:, agent_index], 1.0)
  counts += 1e-6
  return counts / counts.sum(axis=-1, keepdims=True)


def distribution_metrics(reference: np.ndarray, candidate: np.ndarray, *, seed: int) -> dict[str, float]:
  reference = np.asarray(reference, dtype=np.float32)
  candidate = np.asarray(candidate, dtype=np.float32)
  return {
    "mean_abs_error": float(np.abs(reference.mean(axis=0) - candidate.mean(axis=0)).mean()),
    "std_abs_error": float(np.abs(reference.std(axis=0) - candidate.std(axis=0)).mean()),
    "sliced_wasserstein": sliced_wasserstein(reference, candidate, seed=seed),
  }


def sliced_wasserstein(
  reference: np.ndarray,
  candidate: np.ndarray,
  *,
  seed: int,
  num_projections: int = 64,
) -> float:
  reference = np.asarray(reference, dtype=np.float32)
  candidate = np.asarray(candidate, dtype=np.float32)
  count = min(reference.shape[0], candidate.shape[0])
  if count < 1:
    raise ValueError("at least one sample is required")
  rng = np.random.default_rng(seed)
  if reference.shape[0] != count:
    reference = reference[rng.choice(reference.shape[0], size=count, replace=False)]
  if candidate.shape[0] != count:
    candidate = candidate[rng.choice(candidate.shape[0], size=count, replace=False)]
  projections = rng.normal(size=(reference.shape[1], num_projections)).astype(np.float32)
  projections /= np.linalg.norm(projections, axis=0, keepdims=True) + 1e-8
  ref_proj = np.sort(reference @ projections, axis=0)
  cand_proj = np.sort(candidate @ projections, axis=0)
  return float(np.mean(np.abs(ref_proj - cand_proj)))


def pairwise_squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
  left = np.asarray(left, dtype=np.float32)
  right = np.asarray(right, dtype=np.float32)
  left_norm = np.sum(np.square(left), axis=1, keepdims=True)
  right_norm = np.sum(np.square(right), axis=1, keepdims=True).T
  distances = left_norm + right_norm - 2.0 * left @ right.T
  return np.maximum(distances, 0.0)


def pca_project(groups: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  """Project grouped features to 2D using a shared SVD basis."""
  labels = list(groups)
  sizes = [groups[label].shape[0] for label in labels]
  values = np.concatenate([groups[label] for label in labels], axis=0).astype(np.float32)
  centered = values - values.mean(axis=0, keepdims=True)
  _, _, vt = np.linalg.svd(centered, full_matrices=False)
  components = vt[:2].T
  projected = centered @ components
  result = {}
  start = 0
  for label, size in zip(labels, sizes, strict=True):
    result[label] = projected[start : start + size]
    start += size
  return result


def to_rgb_image(observation: np.ndarray) -> np.ndarray:
  image = np.asarray(observation, dtype=np.float32)[..., :3]
  if image.max(initial=0.0) > 1.0:
    image = image / 255.0
  return np.clip(image, 0.0, 1.0)


def _take_prepared_data(
  data: PreparedTransitionData,
  indices: np.ndarray,
) -> PreparedTransitionData:
  indices = np.asarray(indices, dtype=np.int32)
  return PreparedTransitionData(
    state_features=data.state_features[indices],
    next_state_features=data.next_state_features[indices],
    actions=data.actions[indices],
    rewards=data.rewards[indices],
    dones=data.dones[indices],
    obs=data.obs[indices],
    next_obs=data.next_obs[indices],
    action_dim=data.action_dim,
    num_agents=data.num_agents,
    normalizer=data.normalizer,
    representation_config=data.representation_config,
  )


def _pool_edges(size: int, pool_size: int) -> np.ndarray:
  edges = np.linspace(0, size, pool_size + 1).round().astype(np.int32)
  edges[0] = 0
  edges[-1] = size
  for index in range(1, len(edges)):
    if edges[index] <= edges[index - 1]:
      edges[index] = edges[index - 1] + 1
  if edges[-1] > size:
    raise ValueError("pool_size cannot exceed observation size")
  return edges
