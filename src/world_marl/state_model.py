"""One-step state-representation validation for Melting Pot rollouts.

1. Does `z_t, joint_action_t` predict meaningful state change in `z_{t+1}`?
2. Does `z_t` recover the behavior policy better than action marginals?
3. Is sparse reward/event information present in the representation?

The pass criterion is based on transition and behavior-policy recovery. Reward
and done recovery are reported as diagnostic signals because sparse rewards can
be informative on event-only metrics even when full-sequence MSE is dominated by
zeros.
"""

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
  """Training configuration for one-step representation-fit validation."""

  hidden_dims: tuple[int, ...] = (256, 256)
  learning_rate: float = 1e-3
  batch_size: int = 256
  train_steps: int = 1000
  next_loss_weight: float = 1.0
  delta_loss_weight: float = 0.5
  changed_loss_weight: float = 1.0
  reward_loss_weight: float = 1.0
  reward_event_loss_weight: float = 0.25
  done_loss_weight: float = 0.1
  policy_loss_weight: float = 0.1
  max_grad_norm: float = 1.0
  reward_oversample_factor: float = 8.0
  delta_oversample_factor: float = 2.0
  changed_feature_fraction: float = 0.25
  reward_event_epsilon: float = 1e-6


@dataclass(frozen=True)
class WorldModelPredictions:
  """Raw-space predictions from the supervised state-fit model."""

  next_state_features: np.ndarray
  rewards: np.ndarray
  reward_event_logits: np.ndarray
  done_logits: np.ndarray
  policy_logits: np.ndarray


class StateFitWorldModel(nn.Module):
  """Small supervised model for representation, reward, and policy recovery.

  The transition branch is residual by construction. At initialization it is a
  persistence model, so improvements over zero-delta/persistence baselines are a
  meaningful signal that training learned environment dynamics rather than only
  reconstructing static background features.
  """

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

    delta_features = nn.Dense(
      self.feature_dim,
      kernel_init=nn.initializers.zeros,
      bias_init=nn.initializers.zeros,
      name="state_delta",
    )(transition_hidden)

    return {
      "next_state_features": state_features + delta_features,
      "rewards": nn.Dense(self.num_agents)(transition_hidden),
      "reward_event_logits": nn.Dense(self.num_agents)(transition_hidden),
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
  normalizer = fit_feature_normalizer(state_features, next_state_features)
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
    normalizer=normalizer,
    representation_config=representation_config,
  )


def fit_feature_normalizer(*feature_arrays: np.ndarray) -> FeatureNormalizer:
  """Fit a feature normalizer over current and next state features."""
  if not feature_arrays:
    raise ValueError("at least one feature array is required")
  values = np.concatenate(
    [np.asarray(array, dtype=np.float32) for array in feature_arrays],
    axis=0,
  )
  mean = values.mean(axis=0).astype(np.float32)
  std = np.maximum(values.std(axis=0).astype(np.float32), 1e-6)
  return FeatureNormalizer(mean=mean, std=std)


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
    tx=optax.chain(
      optax.clip_by_global_norm(config.max_grad_norm),
      optax.adam(config.learning_rate),
    ),
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
  reward_events = jnp.asarray(
    np.abs(train_data.rewards) > config.reward_event_epsilon,
    dtype=jnp.float32,
  )
  dones = jnp.asarray(train_data.dones, dtype=jnp.float32)
  changed_mask_np = changed_feature_mask(
    train_data.next_state_features - train_data.state_features,
    top_fraction=config.changed_feature_fraction,
  )
  changed_mask = jnp.asarray(changed_mask_np)
  sampling_probs = jnp.asarray(
    transition_sampling_probabilities(
      train_data,
      changed_mask=changed_mask_np,
      reward_event_epsilon=config.reward_event_epsilon,
      reward_oversample_factor=config.reward_oversample_factor,
      delta_oversample_factor=config.delta_oversample_factor,
    ),
    dtype=jnp.float32,
  )
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
    batch_indices = jax.random.choice(
      step_key,
      train_size,
      shape=(config.batch_size,),
      replace=True,
      p=sampling_probs,
    )

    batch = {
      "state": normalized_state[batch_indices],
      "next": normalized_next[batch_indices],
      "actions": actions[batch_indices],
      "rewards": rewards[batch_indices],
      "reward_events": reward_events[batch_indices],
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
        changed_mask=changed_mask,
      )
      total = (
        config.next_loss_weight * losses["next_mse"]
        + config.delta_loss_weight * losses["delta_mse"]
        + config.changed_loss_weight * losses["changed_delta_mse"]
        + config.reward_loss_weight * losses["reward_mse"]
        + config.reward_event_loss_weight * losses["reward_event_bce"]
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
  changed_mask: jax.Array | None = None,
) -> dict[str, jax.Array]:
  """Compute supervised component losses."""
  next_mse = jnp.mean(jnp.square(predictions["next_state_features"] - batch["next"]))
  predicted_delta = predictions["next_state_features"] - batch["state"]
  target_delta = batch["next"] - batch["state"]
  delta_mse = jnp.mean(jnp.square(predicted_delta - target_delta))
  if changed_mask is None:
    changed_delta_mse = delta_mse
  else:
    mask = changed_mask.astype(jnp.float32).reshape((1, -1))
    squared_error = jnp.square(predicted_delta - target_delta)
    changed_delta_mse = jnp.sum(squared_error * mask) / (
      squared_error.shape[0] * jnp.maximum(jnp.sum(mask), 1.0)
    )
  reward_mse = jnp.mean(jnp.square(predictions["rewards"] - batch["rewards"]))
  reward_event_bce = jnp.mean(
    optax.sigmoid_binary_cross_entropy(
      predictions["reward_event_logits"],
      batch["reward_events"],
    )
  )
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
    "delta_mse": delta_mse,
    "changed_delta_mse": changed_delta_mse,
    "reward_mse": reward_mse,
    "reward_event_bce": reward_event_bce,
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
    reward_event_logits=np.asarray(predictions["reward_event_logits"], dtype=np.float32),
    done_logits=np.asarray(predictions["done_logits"], dtype=np.float32),
    policy_logits=np.asarray(predictions["policy_logits"], dtype=np.float32),
  )


def evaluate_state_fit(
  train_data: PreparedTransitionData,
  validation_data: PreparedTransitionData,
  predictions: WorldModelPredictions,
  *,
  seed: int = 0,
  changed_feature_fraction: float = 0.25,
  reward_event_epsilon: float = 1e-6,
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
  true_delta = true_next - validation_data.state_features
  model_delta = model_next - validation_data.state_features
  train_delta = train_data.next_state_features - train_data.state_features
  mean_delta = np.repeat(
    train_delta.mean(axis=0, keepdims=True),
    validation_data.num_transitions,
    axis=0,
  )
  zero_delta = np.zeros_like(true_delta)
  changed_mask = changed_feature_mask(train_delta, top_fraction=changed_feature_fraction)

  reward_mean = np.repeat(
    train_data.rewards.mean(axis=0, keepdims=True),
    validation_data.num_transitions,
    axis=0,
  )
  reward_zero = np.zeros_like(validation_data.rewards)
  train_reward_events = (np.abs(train_data.rewards) > reward_event_epsilon).astype(np.float32)
  validation_reward_events = (np.abs(validation_data.rewards) > reward_event_epsilon).astype(
    np.float32
  )
  reward_event_prior = np.clip(train_reward_events.mean(axis=0), 1e-6, 1.0 - 1e-6)
  reward_event_prior_rows = np.repeat(
    reward_event_prior.reshape(1, -1),
    validation_data.num_transitions,
    axis=0,
  )
  reward_event_probs = sigmoid_np(predictions.reward_event_logits)
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
    "delta_state": {
      "model_mse": mse(model_delta, true_delta),
      "zero_delta_baseline_mse": mse(zero_delta, true_delta),
      "mean_delta_baseline_mse": mse(mean_delta, true_delta),
      "model_beats_zero_delta": bool(mse(model_delta, true_delta) < mse(zero_delta, true_delta)),
      "model_beats_mean_delta": bool(mse(model_delta, true_delta) < mse(mean_delta, true_delta)),
      "true_delta_abs_mean": float(np.abs(true_delta).mean()),
      "true_delta_abs_max": float(np.abs(true_delta).max(initial=0.0)),
    },
    "changed_features": changed_feature_metrics(
      true_next=true_next,
      model_next=model_next,
      mean_next=mean_next,
      persistence=persistence,
      true_delta=true_delta,
      model_delta=model_delta,
      mean_delta=mean_delta,
      zero_delta=zero_delta,
      mask=changed_mask,
    ),
    "state_distribution": {
      "model": distribution_metrics(true_next, model_next, seed=seed),
      "mean_baseline": distribution_metrics(true_next, mean_next, seed=seed),
      "persistence_baseline": distribution_metrics(true_next, persistence, seed=seed),
    },
    "delta_distribution": {
      "model": distribution_metrics(true_delta, model_delta, seed=seed),
      "zero_delta_baseline": distribution_metrics(true_delta, zero_delta, seed=seed),
      "mean_delta_baseline": distribution_metrics(true_delta, mean_delta, seed=seed),
    },
    "reward": {
      "model_mse": mse(predictions.rewards, validation_data.rewards),
      "mean_baseline_mse": mse(reward_mean, validation_data.rewards),
      "zero_baseline_mse": mse(reward_zero, validation_data.rewards),
      "model_beats_mean": bool(
        mse(predictions.rewards, validation_data.rewards)
        < mse(reward_mean, validation_data.rewards)
      ),
      "model_beats_zero": bool(
        mse(predictions.rewards, validation_data.rewards)
        < mse(reward_zero, validation_data.rewards)
      ),
      "event_model_mse": masked_mse(
        predictions.rewards,
        validation_data.rewards,
        validation_reward_events.astype(bool),
      ),
      "event_mean_baseline_mse": masked_mse(
        reward_mean,
        validation_data.rewards,
        validation_reward_events.astype(bool),
      ),
      "event_zero_baseline_mse": masked_mse(
        reward_zero,
        validation_data.rewards,
        validation_reward_events.astype(bool),
      ),
      "event_count": int(validation_reward_events.sum()),
      "event_fraction": float(validation_reward_events.mean()),
    },
    "reward_event": {
      "model_bce": binary_cross_entropy_np(reward_event_probs, validation_reward_events),
      "prior_bce": binary_cross_entropy_np(reward_event_prior_rows, validation_reward_events),
      "no_event_accuracy": float((validation_reward_events == 0.0).mean()),
      "model_beats_prior_bce": bool(
        binary_cross_entropy_np(reward_event_probs, validation_reward_events)
        < binary_cross_entropy_np(reward_event_prior_rows, validation_reward_events)
      ),
      **binary_event_metrics(reward_event_probs, validation_reward_events),
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


def summarize_validation_criteria(
  metrics: dict[str, Any],
  *,
  finite_losses: bool,
  reload_passed: bool,
) -> tuple[bool, dict[str, Any]]:
  """Convert detailed metrics into the milestone pass/fail criteria.

  The current rung is a representation-fit validator. A passing run must show
  reusable training artifacts, state-transition signal, and behavior-policy
  signal. Reward/event recovery is summarized separately so sparse rewards do
  not turn the main milestone into a reward-calibration benchmark.
  """
  transition_model_has_signal = bool(
    metrics["next_state"]["model_beats_persistence"]
    or metrics["delta_state"]["model_beats_zero_delta"]
    or metrics["changed_features"]["delta_model_beats_zero"]
  )
  policy_model_has_signal = bool(metrics["policy"]["model_beats_marginal_ce"])
  event_reward_beats_mean = optional_less(
    metrics["reward"]["event_model_mse"],
    metrics["reward"]["event_mean_baseline_mse"],
  )
  event_reward_beats_zero = optional_less(
    metrics["reward"]["event_model_mse"],
    metrics["reward"]["event_zero_baseline_mse"],
  )
  reward_value_has_signal = bool(
    metrics["reward"]["model_beats_mean"]
    or metrics["reward"]["model_beats_zero"]
    or event_reward_beats_mean
    or event_reward_beats_zero
  )
  reward_event_has_signal = bool(
    metrics["reward_event"]["model_beats_prior_bce"]
    or metrics["reward_event"]["best_f1"] > 0.0
  )
  reward_model_has_signal = bool(reward_value_has_signal or reward_event_has_signal)
  criteria = {
    "finite_losses": bool(finite_losses),
    "reload_passed": bool(reload_passed),
    "transition_model_has_signal": transition_model_has_signal,
    "policy_model_has_signal": policy_model_has_signal,
    "next_state_model_beats_mean": metrics["next_state"]["model_beats_mean"],
    "next_state_model_beats_persistence": metrics["next_state"]["model_beats_persistence"],
    "delta_model_beats_zero": metrics["delta_state"]["model_beats_zero_delta"],
    "delta_model_beats_mean_delta": metrics["delta_state"]["model_beats_mean_delta"],
    "changed_feature_delta_model_beats_zero": metrics["changed_features"][
      "delta_model_beats_zero"
    ],
    "reward_model_beats_mean": metrics["reward"]["model_beats_mean"],
    "reward_model_beats_zero": metrics["reward"]["model_beats_zero"],
    "reward_event_value_beats_mean": event_reward_beats_mean,
    "reward_event_value_beats_zero": event_reward_beats_zero,
    "reward_event_model_beats_prior_bce": metrics["reward_event"][
      "model_beats_prior_bce"
    ],
    "reward_event_best_f1_positive": bool(metrics["reward_event"]["best_f1"] > 0.0),
    "reward_model_has_signal": reward_model_has_signal,
    "policy_model_beats_marginal_ce": metrics["policy"]["model_beats_marginal_ce"],
  }
  passed = bool(
    finite_losses
    and reload_passed
    and transition_model_has_signal
    and policy_model_has_signal
  )
  return passed, criteria


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
      "predicted_reward_event_probability": sigmoid_np(
        predictions.reward_event_logits[index]
      ).tolist(),
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
  labels = ["next\nmodel", "next\nmean", "next\npersist", "delta\nmodel", "delta\nzero"]
  values = [
    metrics["next_state"]["model_mse"],
    metrics["next_state"]["mean_baseline_mse"],
    metrics["next_state"]["persistence_baseline_mse"],
    metrics["delta_state"]["model_mse"],
    metrics["delta_state"]["zero_delta_baseline_mse"],
  ]
  ax.bar(labels, values)
  ax.set_title("Transition feature MSE")
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
  labels = ["reward MSE", "event BCE", "policy CE", "done BCE"]
  model_values = [
    metrics["reward"]["model_mse"],
    metrics["reward_event"]["model_bce"],
    metrics["policy"]["model_cross_entropy"],
    metrics["done"]["model_bce"],
  ]
  baseline_values = [
    metrics["reward"]["mean_baseline_mse"],
    metrics["reward_event"]["prior_bce"],
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


def optional_less(left: float | None, right: float | None) -> bool:
  return left is not None and right is not None and left < right


def masked_mse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
  prediction = np.asarray(prediction, dtype=np.float32)
  target = np.asarray(target, dtype=np.float32)
  mask = np.asarray(mask, dtype=bool)
  if not bool(mask.any()):
    return None
  return float(np.mean(np.square(prediction[mask] - target[mask])))


def sigmoid_np(values: np.ndarray) -> np.ndarray:
  values = np.asarray(values, dtype=np.float64)
  return 1.0 / (1.0 + np.exp(-values))


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


def transition_sampling_probabilities(
  data: PreparedTransitionData,
  *,
  changed_mask: np.ndarray,
  reward_event_epsilon: float,
  reward_oversample_factor: float,
  delta_oversample_factor: float,
) -> np.ndarray:
  """Build minibatch sampling probabilities that emphasize rare reward/state events."""
  reward_events = np.any(np.abs(data.rewards) > reward_event_epsilon, axis=1)
  delta = np.abs(data.next_state_features - data.state_features)
  mask = np.asarray(changed_mask, dtype=bool)
  if mask.ndim != 1 or mask.shape[0] != data.feature_dim:
    raise ValueError("changed_mask has incompatible shape")
  delta_score = delta[:, mask].mean(axis=1)
  delta_threshold = max(1e-6, float(np.quantile(delta_score, 0.75)))
  delta_events = delta_score > delta_threshold
  weights = np.ones(data.num_transitions, dtype=np.float64)
  weights += max(0.0, reward_oversample_factor) * reward_events.astype(np.float64)
  weights += max(0.0, delta_oversample_factor) * delta_events.astype(np.float64)
  weights_sum = float(weights.sum())
  if not np.isfinite(weights_sum) or weights_sum <= 0.0:
    return np.full(data.num_transitions, 1.0 / data.num_transitions, dtype=np.float32)
  return (weights / weights_sum).astype(np.float32)


def binary_event_metrics(
  probabilities: np.ndarray,
  targets: np.ndarray,
  *,
  threshold: float = 0.5,
) -> dict[str, Any]:
  """Summarize rare binary event prediction without hiding behind accuracy."""
  probabilities = np.asarray(probabilities, dtype=np.float64)
  targets = np.asarray(targets, dtype=bool)
  predictions = probabilities >= threshold
  precision, recall, f1 = precision_recall_f1(predictions, targets)
  best = {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1}
  for candidate_threshold in np.linspace(0.05, 0.95, 19):
    candidate_predictions = probabilities >= candidate_threshold
    candidate_precision, candidate_recall, candidate_f1 = precision_recall_f1(
      candidate_predictions,
      targets,
    )
    if candidate_f1 > best["f1"]:
      best = {
        "threshold": float(candidate_threshold),
        "precision": candidate_precision,
        "recall": candidate_recall,
        "f1": candidate_f1,
      }
  return {
    "event_count": int(targets.sum()),
    "event_fraction": float(targets.mean()),
    "model_positive_fraction": float(predictions.mean()),
    "model_accuracy": float((predictions == targets).mean()),
    "model_precision": precision,
    "model_recall": recall,
    "model_f1": f1,
    "best_threshold": best["threshold"],
    "best_precision": best["precision"],
    "best_recall": best["recall"],
    "best_f1": best["f1"],
  }


def precision_recall_f1(predictions: np.ndarray, targets: np.ndarray) -> tuple[float, float, float]:
  predictions = np.asarray(predictions, dtype=bool)
  targets = np.asarray(targets, dtype=bool)
  true_positive = float(np.logical_and(predictions, targets).sum())
  false_positive = float(np.logical_and(predictions, ~targets).sum())
  false_negative = float(np.logical_and(~predictions, targets).sum())
  precision = true_positive / (true_positive + false_positive + 1e-12)
  recall = true_positive / (true_positive + false_negative + 1e-12)
  f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
  return float(precision), float(recall), float(f1)


def changed_feature_mask(
  train_delta: np.ndarray,
  *,
  top_fraction: float = 0.25,
  min_std: float = 1e-6,
) -> np.ndarray:
  """Select feature dimensions that change most in the training transitions."""
  if not 0.0 < top_fraction <= 1.0:
    raise ValueError("top_fraction must be in (0, 1]")
  delta_std = np.asarray(train_delta, dtype=np.float32).std(axis=0)
  active = delta_std > min_std
  if not bool(active.any()):
    return np.ones_like(delta_std, dtype=bool)
  active_indices = np.flatnonzero(active)
  keep = max(1, int(round(active_indices.shape[0] * top_fraction)))
  ranked_active = active_indices[np.argsort(delta_std[active_indices])[::-1]]
  mask = np.zeros_like(delta_std, dtype=bool)
  mask[ranked_active[:keep]] = True
  return mask


def changed_feature_metrics(
  *,
  true_next: np.ndarray,
  model_next: np.ndarray,
  mean_next: np.ndarray,
  persistence: np.ndarray,
  true_delta: np.ndarray,
  model_delta: np.ndarray,
  mean_delta: np.ndarray,
  zero_delta: np.ndarray,
  mask: np.ndarray,
) -> dict[str, Any]:
  """Evaluate transition prediction only on the most-changing feature dimensions."""
  mask = np.asarray(mask, dtype=bool)
  if mask.ndim != 1 or mask.shape[0] != true_next.shape[1]:
    raise ValueError("changed-feature mask has incompatible shape")
  next_model_mse = mse(model_next[:, mask], true_next[:, mask])
  next_mean_mse = mse(mean_next[:, mask], true_next[:, mask])
  next_persistence_mse = mse(persistence[:, mask], true_next[:, mask])
  delta_model_mse = mse(model_delta[:, mask], true_delta[:, mask])
  delta_zero_mse = mse(zero_delta[:, mask], true_delta[:, mask])
  delta_mean_mse = mse(mean_delta[:, mask], true_delta[:, mask])
  return {
    "feature_count": int(mask.sum()),
    "feature_fraction": float(mask.mean()),
    "next_model_mse": next_model_mse,
    "next_mean_baseline_mse": next_mean_mse,
    "next_persistence_baseline_mse": next_persistence_mse,
    "next_model_beats_mean": bool(next_model_mse < next_mean_mse),
    "next_model_beats_persistence": bool(next_model_mse < next_persistence_mse),
    "delta_model_mse": delta_model_mse,
    "delta_zero_baseline_mse": delta_zero_mse,
    "delta_mean_baseline_mse": delta_mean_mse,
    "delta_model_beats_zero": bool(delta_model_mse < delta_zero_mse),
    "delta_model_beats_mean": bool(delta_model_mse < delta_mean_mse),
  }


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
