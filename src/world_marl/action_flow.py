"""State-conditioned action imitation utilities for JaxMARL CoinGame."""

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

from flow_matching.models import MLPVectorField
from flow_matching.simulate import sample_conditioned_flow
from flow_matching.train import conditioned_train_step, create_conditioned_train_state
from world_marl.coin_flow import (
  _validate_joint_actions,
  compare_joint_action_distributions,
  decode_joint_actions,
  normalize_joint_actions,
  summarize_joint_action_distribution,
)


@dataclass(frozen=True)
class StateActionDataset:
  """State-conditioned action samples from a two-agent vector environment."""

  state_features: np.ndarray
  joint_actions: np.ndarray
  rewards: np.ndarray
  action_dim: int
  num_agents: int
  num_envs: int
  rollout_steps: int
  observation_shape: tuple[int, ...]
  completed_returns: tuple[tuple[float, ...], ...]
  completed_lengths: tuple[int, ...]

  def to_metadata(self) -> dict[str, Any]:
    return {
      "num_samples": int(self.joint_actions.shape[0]),
      "state_feature_dim": int(self.state_features.shape[1]),
      "action_dim": self.action_dim,
      "num_agents": self.num_agents,
      "num_envs": self.num_envs,
      "rollout_steps": self.rollout_steps,
      "observation_shape": list(self.observation_shape),
      "mean_reward_per_agent": self.rewards.mean(axis=0).tolist(),
      "completed_episodes": len(self.completed_returns),
      "completed_return_mean": (
        float(np.asarray(self.completed_returns, dtype=np.float32).mean())
        if self.completed_returns
        else None
      ),
      "completed_lengths": list(self.completed_lengths),
    }


@dataclass(frozen=True)
class FeatureNormalizer:
  """Mean/std normalizer fit on train states and reused at evaluation time."""

  mean: np.ndarray
  std: np.ndarray

  def transform(self, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    return (features - self.mean) / self.std

  def to_metadata(self) -> dict[str, Any]:
    return {
      "mean": self.mean.astype(float).tolist(),
      "std": self.std.astype(float).tolist(),
    }


class ConditionalActionClassifier(nn.Module):
  """Categorical sanity baseline for p(joint_action | state)."""

  action_dim: int
  num_agents: int = 2
  hidden_dims: tuple[int, ...] = (128, 128)

  @nn.compact
  def __call__(self, state_features: jax.Array) -> jax.Array:
    x = state_features
    for hidden_dim in self.hidden_dims:
      x = nn.Dense(hidden_dim)(x)
      x = nn.silu(x)
    logits = nn.Dense(self.num_agents * self.action_dim)(x)
    return logits.reshape((state_features.shape[0], self.num_agents, self.action_dim))


def flatten_joint_observations(observations: np.ndarray) -> np.ndarray:
  """Flatten batched multi-agent observations into one joint state vector."""
  observations = np.asarray(observations, dtype=np.float32)
  if observations.ndim < 3:
    raise ValueError("expected observations shaped [env, agent, ...]")
  return observations.reshape((observations.shape[0], -1))


def split_state_action_dataset(
  dataset: StateActionDataset,
  *,
  validation_fraction: float,
  seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Shuffle a state-action dataset into train and validation arrays."""
  if not 0.0 < validation_fraction < 1.0:
    raise ValueError("validation_fraction must be between 0 and 1")
  features = np.asarray(dataset.state_features, dtype=np.float32)
  actions = _validate_joint_actions(dataset.joint_actions, action_dim=dataset.action_dim)
  if features.ndim != 2:
    raise ValueError(f"expected state features shaped [N, D], got {features.shape}")
  if features.shape[0] != actions.shape[0]:
    raise ValueError("state/action sample counts do not match")
  if actions.shape[0] < 2:
    raise ValueError("at least two samples are required to split")
  validation_size = int(round(actions.shape[0] * validation_fraction))
  validation_size = min(max(1, validation_size), actions.shape[0] - 1)
  indices = np.random.default_rng(seed).permutation(actions.shape[0])
  validation_indices = indices[:validation_size]
  train_indices = indices[validation_size:]
  return (
    features[train_indices],
    actions[train_indices],
    features[validation_indices],
    actions[validation_indices],
  )


def fit_feature_normalizer(features: np.ndarray, *, epsilon: float = 1e-6) -> FeatureNormalizer:
  """Fit a numerically stable mean/std normalizer."""
  features = np.asarray(features, dtype=np.float32)
  if features.ndim != 2:
    raise ValueError(f"expected features shaped [N, D], got {features.shape}")
  if epsilon <= 0.0:
    raise ValueError("epsilon must be positive")
  mean = features.mean(axis=0, keepdims=True)
  std = features.std(axis=0, keepdims=True)
  std = np.where(std < epsilon, 1.0, std)
  return FeatureNormalizer(mean=mean.astype(np.float32), std=std.astype(np.float32))

def collect_random_state_actions(
  adapter,
  rng: np.random.Generator,
  *,
  rollout_steps: int,
  progress_callback: Callable[[int], None] | None = None,
) -> StateActionDataset:
  """Collect ``(state_t, joint_action_t)`` samples from a random policy."""
  if adapter.num_agents != 2:
    raise ValueError("conditional action milestone currently expects exactly 2 agents")
  if rollout_steps < 1:
    raise ValueError("rollout_steps must be >= 1")

  feature_rows = []
  joint_action_rows = []
  reward_rows = []
  completed_returns: list[tuple[float, ...]] = []
  completed_lengths: list[int] = []
  observations = adapter.reset()
  observation_shape = tuple(int(dim) for dim in observations.shape[2:])
  for step_index in range(rollout_steps):
    actions = adapter.sample_actions(rng)
    feature_rows.append(flatten_joint_observations(observations))
    joint_action_rows.append(actions.copy())
    step = adapter.step(actions)
    observations = step.observations
    reward_rows.append(step.rewards.copy())
    completed_returns.extend(step.completed_returns)
    completed_lengths.extend(step.completed_lengths)
    if progress_callback is not None:
      progress_callback(step_index + 1)

  return StateActionDataset(
    state_features=np.concatenate(feature_rows, axis=0).astype(np.float32),
    joint_actions=np.concatenate(joint_action_rows, axis=0).astype(np.int32),
    rewards=np.concatenate(reward_rows, axis=0).astype(np.float32),
    action_dim=adapter.action_dim,
    num_agents=adapter.num_agents,
    num_envs=adapter.num_envs,
    rollout_steps=rollout_steps,
    observation_shape=observation_shape,
    completed_returns=tuple(completed_returns),
    completed_lengths=tuple(completed_lengths),
  )


def collect_policy_state_actions(
  adapter,
  policy_fn: Callable[[np.ndarray], np.ndarray],
  *,
  rollout_steps: int,
  progress_callback: Callable[[int], None] | None = None,
) -> StateActionDataset:
  """Collect ``(state_t, joint_action_t)`` samples from a source policy."""
  if adapter.num_agents != 2:
    raise ValueError("conditional action milestone currently expects exactly 2 agents")
  if rollout_steps < 1:
    raise ValueError("rollout_steps must be >= 1")

  feature_rows = []
  joint_action_rows = []
  reward_rows = []
  completed_returns: list[tuple[float, ...]] = []
  completed_lengths: list[int] = []
  observations = adapter.reset()
  observation_shape = tuple(int(dim) for dim in observations.shape[2:])
  for step_index in range(rollout_steps):
    actions = np.asarray(policy_fn(observations), dtype=np.int32)
    expected_shape = (adapter.num_envs, adapter.num_agents)
    if actions.shape != expected_shape:
      raise ValueError(f"policy actions must have shape {expected_shape}, got {actions.shape}")
    feature_rows.append(flatten_joint_observations(observations))
    joint_action_rows.append(actions.copy())
    step = adapter.step(actions)
    observations = step.observations
    reward_rows.append(step.rewards.copy())
    completed_returns.extend(step.completed_returns)
    completed_lengths.extend(step.completed_lengths)
    if progress_callback is not None:
      progress_callback(step_index + 1)

  return StateActionDataset(
    state_features=np.concatenate(feature_rows, axis=0).astype(np.float32),
    joint_actions=np.concatenate(joint_action_rows, axis=0).astype(np.int32),
    rewards=np.concatenate(reward_rows, axis=0).astype(np.float32),
    action_dim=adapter.action_dim,
    num_agents=adapter.num_agents,
    num_envs=adapter.num_envs,
    rollout_steps=rollout_steps,
    observation_shape=observation_shape,
    completed_returns=tuple(completed_returns),
    completed_lengths=tuple(completed_lengths),
  )

def create_action_classifier_train_state(
  rng: jax.Array,
  *,
  feature_dim: int,
  action_dim: int,
  num_agents: int,
  hidden_dims: tuple[int, ...],
  learning_rate: float,
) -> TrainState:
  """Initialize the categorical p(action | state) baseline."""
  model = ConditionalActionClassifier(
    action_dim=action_dim,
    num_agents=num_agents,
    hidden_dims=hidden_dims,
  )
  rng, init_key = jax.random.split(rng)
  params = model.init(init_key, jnp.zeros((1, feature_dim), dtype=jnp.float32))["params"]
  return TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.adam(learning_rate),
  )


def action_classifier_loss(
  params: Any,
  apply_fn: Any,
  state_features: jax.Array,
  joint_actions: jax.Array,
) -> jax.Array:
  """Mean per-agent categorical cross entropy."""
  logits = apply_fn({"params": params}, state_features)
  return optax.softmax_cross_entropy_with_integer_labels(
    logits.reshape((-1, logits.shape[-1])),
    joint_actions.reshape((-1,)),
  ).mean()


@jax.jit
def action_classifier_train_step(
  state: TrainState,
  state_features: jax.Array,
  joint_actions: jax.Array,
) -> tuple[TrainState, jax.Array]:
  """Run one categorical imitation update."""
  loss, grads = jax.value_and_grad(action_classifier_loss)(
    state.params,
    state.apply_fn,
    state_features,
    joint_actions,
  )
  return state.apply_gradients(grads=grads), loss


def train_action_classifier(
  rng: jax.Array,
  state_features: np.ndarray,
  joint_actions: np.ndarray,
  *,
  action_dim: int,
  num_agents: int,
  train_steps: int,
  batch_size: int,
  learning_rate: float,
  hidden_dims: tuple[int, ...] = (128, 128),
  progress_callback: Callable[[int, float], None] | None = None,
) -> tuple[TrainState, list[float]]:
  """Train a categorical state-conditioned action model."""
  if train_steps < 1:
    raise ValueError("train_steps must be >= 1")
  if batch_size < 1:
    raise ValueError("batch_size must be >= 1")
  features = np.asarray(state_features, dtype=np.float32)
  actions = _validate_joint_actions(joint_actions, action_dim=action_dim)
  if features.ndim != 2:
    raise ValueError(f"expected features shaped [N, D], got {features.shape}")
  if features.shape[0] != actions.shape[0]:
    raise ValueError("state/action sample counts do not match")

  state = create_action_classifier_train_state(
    rng,
    feature_dim=features.shape[1],
    action_dim=action_dim,
    num_agents=num_agents,
    hidden_dims=hidden_dims,
    learning_rate=learning_rate,
  )
  np_rng = np.random.default_rng(int(jax.random.randint(rng, (), 0, 2**31 - 1)))
  losses: list[float] = []
  for step_index in range(train_steps):
    batch_indices = np_rng.integers(0, features.shape[0], size=batch_size)
    state, loss = action_classifier_train_step(
      state,
      jnp.asarray(features[batch_indices]),
      jnp.asarray(actions[batch_indices]),
    )
    loss_value = float(loss)
    losses.append(loss_value)
    if progress_callback is not None:
      progress_callback(step_index + 1, loss_value)
  jax.block_until_ready(state.params)
  return state, losses


def predict_action_logits(
  train_state: TrainState,
  state_features: np.ndarray,
) -> np.ndarray:
  """Run the categorical action model on numpy state features."""
  logits = train_state.apply_fn(
    {"params": train_state.params},
    jnp.asarray(state_features, dtype=jnp.float32),
  )
  return np.asarray(logits, dtype=np.float32)


def action_prediction_metrics(
  *,
  logits: np.ndarray,
  reference_actions: np.ndarray,
  train_actions: np.ndarray,
  action_dim: int,
) -> dict[str, Any]:
  """Evaluate categorical predictions against heldout source actions."""
  logits = np.asarray(logits, dtype=np.float32)
  actions = _validate_joint_actions(reference_actions, action_dim=action_dim)
  train_actions = _validate_joint_actions(train_actions, action_dim=action_dim)
  if logits.shape != (actions.shape[0], actions.shape[1], action_dim):
    raise ValueError(
      "expected logits shaped "
      f"{(actions.shape[0], actions.shape[1], action_dim)}, got {logits.shape}"
    )
  shifted = logits - logits.max(axis=-1, keepdims=True)
  log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
  selected = np.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(-1)
  predictions = logits.argmax(axis=-1).astype(np.int32)

  marginal_counts = np.zeros((actions.shape[1], action_dim), dtype=np.float64)
  for agent_index in range(actions.shape[1]):
    marginal_counts[agent_index] = np.bincount(
      train_actions[:, agent_index],
      minlength=action_dim,
    )
  marginal_probs = (marginal_counts + 1.0) / (
    marginal_counts.sum(axis=1, keepdims=True) + action_dim
  )
  marginal_modes = marginal_probs.argmax(axis=1).astype(np.int32)
  marginal_predictions = np.broadcast_to(marginal_modes[None, :], actions.shape)
  marginal_selected = np.take_along_axis(
    np.log(marginal_probs)[None, ...],
    actions[..., None],
    axis=-1,
  ).squeeze(-1)

  return {
    "cross_entropy": float(-selected.mean()),
    "random_cross_entropy": float(np.log(action_dim)),
    "marginal_cross_entropy": float(-marginal_selected.mean()),
    "model_beats_marginal_ce": bool(-selected.mean() < -marginal_selected.mean()),
    "per_agent_accuracy": float((predictions == actions).mean()),
    "joint_accuracy": float(np.all(predictions == actions, axis=1).mean()),
    "marginal_per_agent_accuracy": float((marginal_predictions == actions).mean()),
    "marginal_joint_accuracy": float(
      np.all(marginal_predictions == actions, axis=1).mean()
    ),
    "predicted_action_distribution": summarize_joint_action_distribution(
      predictions,
      action_dim,
    ),
  }


def train_conditional_action_flow(
  rng: jax.Array,
  state_features: np.ndarray,
  joint_actions: np.ndarray,
  *,
  action_dim: int,
  train_steps: int,
  batch_size: int,
  learning_rate: float,
  hidden_dims: tuple[int, ...] = (64, 64, 64, 64),
  progress_callback: Callable[[int, float], None] | None = None,
) -> tuple[TrainState, list[float]]:
  """Train conditional flow matching for p(joint_action | state)."""
  if train_steps < 1:
    raise ValueError("train_steps must be >= 1")
  if batch_size < 1:
    raise ValueError("batch_size must be >= 1")
  features = np.asarray(state_features, dtype=np.float32)
  actions = _validate_joint_actions(joint_actions, action_dim=action_dim)
  if features.ndim != 2:
    raise ValueError(f"expected features shaped [N, D], got {features.shape}")
  if features.shape[0] != actions.shape[0]:
    raise ValueError("state/action sample counts do not match")

  x1 = normalize_joint_actions(actions, action_dim).astype(np.float32)
  state = create_conditioned_train_state(
    rng,
    MLPVectorField(hidden_dims=hidden_dims),
    learning_rate=learning_rate,
    dim=x1.shape[1],
    cond_dim=features.shape[1],
  )
  np_rng = np.random.default_rng(int(jax.random.randint(rng, (), 0, 2**31 - 1)))
  losses: list[float] = []
  for step_index in range(train_steps):
    batch_indices = np_rng.integers(0, features.shape[0], size=batch_size)
    rng, step_key = jax.random.split(rng)
    state, loss = conditioned_train_step(
      state,
      step_key,
      jnp.asarray(x1[batch_indices]),
      jnp.asarray(features[batch_indices]),
    )
    loss_value = float(loss)
    losses.append(loss_value)
    if progress_callback is not None:
      progress_callback(step_index + 1, loss_value)
  jax.block_until_ready(state.params)
  return state, losses


def sample_conditional_action_flow_points(
  train_state: TrainState,
  rng: jax.Array,
  state_features: np.ndarray,
  *,
  integration_steps: int = 64,
) -> jax.Array:
  """Sample normalized joint-action points conditioned on state features."""
  if integration_steps < 1:
    raise ValueError("integration_steps must be >= 1")
  features = np.asarray(state_features, dtype=np.float32)
  if features.ndim != 2:
    raise ValueError(f"expected features shaped [N, D], got {features.shape}")
  return sample_conditioned_flow(
    train_state.apply_fn,
    train_state.params,
    rng,
    jnp.asarray(features),
    dim=2,
    steps=integration_steps,
  )


def sampled_action_prediction_metrics(
  *,
  sampled_actions: np.ndarray,
  reference_actions: np.ndarray,
  train_actions: np.ndarray,
  action_dim: int,
  top_k: int = 5,
) -> dict[str, Any]:
  """Evaluate sampled joint actions against heldout source actions."""
  sampled = _validate_joint_actions(sampled_actions, action_dim=action_dim)
  reference = _validate_joint_actions(reference_actions, action_dim=action_dim)
  train_actions = _validate_joint_actions(train_actions, action_dim=action_dim)
  if sampled.shape != reference.shape:
    raise ValueError(f"sampled actions shape {sampled.shape} != {reference.shape}")

  marginal_counts = np.zeros((reference.shape[1], action_dim), dtype=np.float64)
  for agent_index in range(reference.shape[1]):
    marginal_counts[agent_index] = np.bincount(
      train_actions[:, agent_index],
      minlength=action_dim,
    )
  marginal_modes = marginal_counts.argmax(axis=1).astype(np.int32)
  marginal_predictions = np.broadcast_to(marginal_modes[None, :], reference.shape)

  distribution_metrics = compare_joint_action_distributions(
    reference,
    sampled,
    action_dim=action_dim,
    top_k=top_k,
  )
  return {
    "per_agent_accuracy": float((sampled == reference).mean()),
    "joint_accuracy": float(np.all(sampled == reference, axis=1).mean()),
    "marginal_per_agent_accuracy": float(
      (marginal_predictions == reference).mean()
    ),
    "marginal_joint_accuracy": float(
      np.all(marginal_predictions == reference, axis=1).mean()
    ),
    "beats_marginal_joint_accuracy": bool(
      np.all(sampled == reference, axis=1).mean()
      > np.all(marginal_predictions == reference, axis=1).mean()
    ),
    "distribution_vs_heldout": distribution_metrics,
    "sampled_action_distribution": summarize_joint_action_distribution(
      sampled,
      action_dim,
      top_k=top_k,
    ),
  }

def conditional_flow_joint_action_policy(
  train_state: TrainState,
  normalizer: FeatureNormalizer,
  *,
  action_dim: int,
  seed: int,
  integration_steps: int = 64,
):
  """Create an env policy by sampling p(joint_action | current state)."""
  key = jax.random.PRNGKey(seed)
  sample_fn = jax.jit(
    lambda params, sample_key, cond: sample_conditioned_flow(
      train_state.apply_fn,
      params,
      sample_key,
      cond,
      dim=2,
      steps=integration_steps,
    )
  )

  def act(observations: np.ndarray) -> np.ndarray:
    nonlocal key
    features = normalizer.transform(flatten_joint_observations(observations))
    key, sample_key = jax.random.split(key)
    points = np.asarray(
      sample_fn(train_state.params, sample_key, jnp.asarray(features)),
      dtype=np.float32,
    )
    return decode_joint_actions(points, action_dim)

  return act


def classifier_joint_action_policy(
  train_state: TrainState,
  normalizer: FeatureNormalizer,
  *,
  deterministic: bool = True,
  seed: int = 0,
):
  """Create an env policy from the categorical p(action | state) baseline."""
  key = jax.random.PRNGKey(seed)
  infer_fn = jax.jit(
    lambda params, action_key, cond: _classifier_select_actions(
      train_state.apply_fn,
      params,
      action_key,
      cond,
      deterministic=deterministic,
    )
  )

  def act(observations: np.ndarray) -> np.ndarray:
    nonlocal key
    features = normalizer.transform(flatten_joint_observations(observations))
    key, action_key = jax.random.split(key)
    return np.asarray(
      infer_fn(train_state.params, action_key, jnp.asarray(features)),
      dtype=np.int32,
    )

  return act


def _classifier_select_actions(
  apply_fn: Any,
  params: Any,
  rng: jax.Array,
  state_features: jax.Array,
  *,
  deterministic: bool,
) -> jax.Array:
  logits = apply_fn({"params": params}, state_features)
  if deterministic:
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)
  return jax.random.categorical(rng, logits, axis=-1).astype(jnp.int32)
