"""Flow-matching utilities for two-agent JaxMARL CoinGame actions."""

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

from flow_matching.distributions import GaussianMixture2D
from flow_matching.models import MLPVectorField
from flow_matching.simulate import euler_integrate, sample_conditioned_flow
from flow_matching.train import (
  conditioned_train_step,
  create_conditioned_train_state,
  create_train_state,
  train_step,
)

@dataclass(frozen=True)
class JointActionDataset:
  """Joint-action samples collected from a two-agent vector environment."""

  joint_actions: np.ndarray
  rewards: np.ndarray
  action_dim: int
  num_agents: int
  num_envs: int
  rollout_steps: int
  completed_returns: tuple[tuple[float, ...], ...]
  completed_lengths: tuple[int, ...]

  def to_metadata(self) -> dict[str, Any]:
    return {
      "num_samples": int(self.joint_actions.shape[0]),
      "action_dim": self.action_dim,
      "num_agents": self.num_agents,
      "num_envs": self.num_envs,
      "rollout_steps": self.rollout_steps,
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


@dataclass(frozen=True)
class JointActionGMM:
  """A GMM target plus the discrete action pairs it was fitted from."""

  gmm: GaussianMixture2D
  action_pairs: np.ndarray
  counts: np.ndarray

  def to_metadata(self) -> dict[str, Any]:
    return {
      "num_components": int(self.action_pairs.shape[0]),
      "std": float(self.gmm.std),
      "action_pairs": self.action_pairs.astype(int).tolist(),
      "counts": self.counts.astype(int).tolist(),
      "weights": np.asarray(self.gmm.weights).astype(float).tolist(),
      "means": np.asarray(self.gmm.means).astype(float).tolist(),
    }


def split_joint_actions(
  joint_actions: np.ndarray,
  *,
  validation_fraction: float,
  seed: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Shuffle joint actions into train and validation splits."""
  if not 0.0 < validation_fraction < 1.0:
    raise ValueError("validation_fraction must be between 0 and 1")
  actions = _validate_joint_actions(joint_actions, action_dim=None)
  if actions.shape[0] < 2:
    raise ValueError("at least two joint actions are required to split")
  validation_size = int(round(actions.shape[0] * validation_fraction))
  validation_size = min(max(1, validation_size), actions.shape[0] - 1)
  indices = np.random.default_rng(seed).permutation(actions.shape[0])
  validation_indices = indices[:validation_size]
  train_indices = indices[validation_size:]
  return actions[train_indices], actions[validation_indices]


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


def joint_action_counts(joint_actions: np.ndarray, action_dim: int) -> np.ndarray:
  """Count every discrete two-agent joint action in an action_dim x action_dim grid."""
  actions = _validate_joint_actions(joint_actions, action_dim=action_dim)
  counts = np.zeros((action_dim, action_dim), dtype=np.int64)
  np.add.at(counts, (actions[:, 0], actions[:, 1]), 1)
  return counts


def joint_action_probabilities(
  joint_actions: np.ndarray,
  action_dim: int,
  *,
  smoothing: float = 0.0,
) -> np.ndarray:
  """Return a probability grid over two-agent joint actions."""
  if smoothing < 0.0:
    raise ValueError("smoothing must be non-negative")
  counts = joint_action_counts(joint_actions, action_dim).astype(np.float64)
  counts = counts + smoothing
  total = counts.sum()
  if total <= 0.0:
    raise ValueError("at least one joint action is required")
  return counts / total


def summarize_joint_action_distribution(
  joint_actions: np.ndarray,
  action_dim: int,
  *,
  top_k: int = 10,
) -> dict[str, Any]:
  """Summarize an empirical joint-action distribution for JSON artifacts."""
  if top_k < 1:
    raise ValueError("top_k must be >= 1")
  counts = joint_action_counts(joint_actions, action_dim)
  probabilities = joint_action_probabilities(joint_actions, action_dim)
  flat_order = np.argsort(probabilities.reshape(-1))[::-1]
  rows = []
  for flat_index in flat_order[: min(top_k, action_dim * action_dim)]:
    first, second = np.unravel_index(flat_index, probabilities.shape)
    rows.append(
      {
        "action_pair": [int(first), int(second)],
        "count": int(counts[first, second]),
        "probability": float(probabilities[first, second]),
      }
    )
  return {
    "num_samples": int(np.asarray(joint_actions).shape[0]),
    "action_dim": int(action_dim),
    "counts": counts.astype(int).tolist(),
    "probabilities": probabilities.astype(float).tolist(),
    "top_pairs": rows,
  }


def compare_joint_action_distributions(
  reference_actions: np.ndarray,
  candidate_actions: np.ndarray,
  *,
  action_dim: int,
  top_k: int = 5,
  epsilon: float = 1e-8,
) -> dict[str, Any]:
  """Compare a candidate joint-action distribution against a reference."""
  if top_k < 1:
    raise ValueError("top_k must be >= 1")
  if epsilon <= 0.0:
    raise ValueError("epsilon must be positive")
  reference_probs = joint_action_probabilities(reference_actions, action_dim)
  candidate_probs = joint_action_probabilities(candidate_actions, action_dim)
  reference_smooth = joint_action_probabilities(
    reference_actions,
    action_dim,
    smoothing=epsilon,
  )
  candidate_smooth = joint_action_probabilities(
    candidate_actions,
    action_dim,
    smoothing=epsilon,
  )
  mixture = 0.5 * (reference_smooth + candidate_smooth)

  total_variation = 0.5 * np.abs(reference_probs - candidate_probs).sum()
  kl_reference_candidate = np.sum(
    reference_smooth * np.log(reference_smooth / candidate_smooth)
  )
  kl_candidate_reference = np.sum(
    candidate_smooth * np.log(candidate_smooth / reference_smooth)
  )
  js_divergence = 0.5 * np.sum(reference_smooth * np.log(reference_smooth / mixture))
  js_divergence += 0.5 * np.sum(candidate_smooth * np.log(candidate_smooth / mixture))

  flat_reference = reference_probs.reshape(-1)
  flat_candidate = candidate_probs.reshape(-1)
  k = min(top_k, action_dim * action_dim)
  reference_top = set(np.argsort(flat_reference)[-k:])
  candidate_top = set(np.argsort(flat_candidate)[-k:])
  support_reference = reference_probs > 0.0
  support_candidate = candidate_probs > 0.0

  return {
    "reference_samples": int(np.asarray(reference_actions).shape[0]),
    "candidate_samples": int(np.asarray(candidate_actions).shape[0]),
    "total_variation": float(total_variation),
    "js_divergence": float(js_divergence),
    "kl_reference_to_candidate": float(kl_reference_candidate),
    "kl_candidate_to_reference": float(kl_candidate_reference),
    "mean_abs_frequency_error": float(np.abs(reference_probs - candidate_probs).mean()),
    "max_abs_frequency_error": float(np.abs(reference_probs - candidate_probs).max()),
    "top_k": int(k),
    "top_k_overlap": int(len(reference_top & candidate_top)),
    "top_k_overlap_fraction": float(len(reference_top & candidate_top) / k),
    "mode_matches": bool(np.argmax(flat_reference) == np.argmax(flat_candidate)),
    "support_recall_mass": float(reference_probs[support_candidate].sum()),
    "support_precision_mass": float(candidate_probs[support_reference].sum()),
  }


def uniform_joint_actions(
  rng: np.random.Generator,
  *,
  num_samples: int,
  action_dim: int,
) -> np.ndarray:
  """Sample independent uniformly random two-agent joint actions."""
  if num_samples < 1:
    raise ValueError("num_samples must be >= 1")
  if action_dim < 2:
    raise ValueError("action_dim must be >= 2")
  return rng.integers(
    low=0,
    high=action_dim,
    size=(num_samples, 2),
    dtype=np.int32,
  )


def collect_random_joint_actions(
  adapter,
  rng: np.random.Generator,
  *,
  rollout_steps: int,
  progress_callback: Callable[[int], None] | None = None,
) -> JointActionDataset:
  """Collect random joint-action samples from a live two-agent vector adapter."""
  if adapter.num_agents != 2:
    raise ValueError("flow/GMM joint-action demo currently expects exactly 2 agents")
  if rollout_steps < 1:
    raise ValueError("rollout_steps must be >= 1")

  joint_action_rows = []
  reward_rows = []
  completed_returns: list[tuple[float, ...]] = []
  completed_lengths: list[int] = []
  adapter.reset()
  for step_index in range(rollout_steps):
    actions = adapter.sample_actions(rng)
    step = adapter.step(actions)
    joint_action_rows.append(actions.copy())
    reward_rows.append(step.rewards.copy())
    completed_returns.extend(step.completed_returns)
    completed_lengths.extend(step.completed_lengths)
    if progress_callback is not None:
      progress_callback(step_index + 1)

  joint_actions = np.concatenate(joint_action_rows, axis=0).astype(np.int32)
  rewards = np.concatenate(reward_rows, axis=0).astype(np.float32)
  return JointActionDataset(
    joint_actions=joint_actions,
    rewards=rewards,
    action_dim=adapter.action_dim,
    num_agents=adapter.num_agents,
    num_envs=adapter.num_envs,
    rollout_steps=rollout_steps,
    completed_returns=tuple(completed_returns),
    completed_lengths=tuple(completed_lengths),
  )


def collect_policy_joint_actions(
  adapter,
  policy_fn: Callable[[np.ndarray], np.ndarray],
  *,
  rollout_steps: int,
  progress_callback: Callable[[int], None] | None = None,
) -> JointActionDataset:
  """Collect joint-action samples from a policy in a live two-agent vector adapter."""
  if adapter.num_agents != 2:
    raise ValueError("flow/GMM joint-action demo currently expects exactly 2 agents")
  if rollout_steps < 1:
    raise ValueError("rollout_steps must be >= 1")

  joint_action_rows = []
  reward_rows = []
  completed_returns: list[tuple[float, ...]] = []
  completed_lengths: list[int] = []
  observations = adapter.reset()
  for step_index in range(rollout_steps):
    actions = np.asarray(policy_fn(observations), dtype=np.int32)
    expected_shape = (adapter.num_envs, adapter.num_agents)
    if actions.shape != expected_shape:
      raise ValueError(f"policy actions must have shape {expected_shape}, got {actions.shape}")
    step = adapter.step(actions)
    observations = step.observations
    joint_action_rows.append(actions.copy())
    reward_rows.append(step.rewards.copy())
    completed_returns.extend(step.completed_returns)
    completed_lengths.extend(step.completed_lengths)
    if progress_callback is not None:
      progress_callback(step_index + 1)

  joint_actions = np.concatenate(joint_action_rows, axis=0).astype(np.int32)
  rewards = np.concatenate(reward_rows, axis=0).astype(np.float32)
  return JointActionDataset(
    joint_actions=joint_actions,
    rewards=rewards,
    action_dim=adapter.action_dim,
    num_agents=adapter.num_agents,
    num_envs=adapter.num_envs,
    rollout_steps=rollout_steps,
    completed_returns=tuple(completed_returns),
    completed_lengths=tuple(completed_lengths),
  )


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


def normalize_joint_actions(joint_actions: np.ndarray, action_dim: int) -> np.ndarray:
  """Map integer action pairs from ``[0, action_dim)`` to ``[-1, 1]``."""
  if action_dim < 2:
    raise ValueError("action_dim must be >= 2")
  actions = np.asarray(joint_actions, dtype=np.float32)
  if actions.ndim != 2 or actions.shape[1] != 2:
    raise ValueError(f"expected joint actions shaped [N, 2], got {actions.shape}")
  return (actions / float(action_dim - 1)) * 2.0 - 1.0


def decode_joint_actions(points: np.ndarray, action_dim: int) -> np.ndarray:
  """Map generated 2D flow samples back to clipped integer action pairs."""
  if action_dim < 2:
    raise ValueError("action_dim must be >= 2")
  points = np.asarray(points, dtype=np.float32)
  if points.ndim != 2 or points.shape[1] != 2:
    raise ValueError(f"expected points shaped [N, 2], got {points.shape}")
  actions = np.rint((points + 1.0) * 0.5 * float(action_dim - 1))
  return np.clip(actions, 0, action_dim - 1).astype(np.int32)


def fit_joint_action_gmm(
  joint_actions: np.ndarray,
  *,
  action_dim: int,
  std: float = 0.10,
  max_components: int | None = None,
) -> JointActionGMM:
  """Fit an empirical 2D GMM over normalized two-agent action pairs."""
  if std <= 0.0:
    raise ValueError("std must be positive")
  actions = _validate_joint_actions(joint_actions, action_dim=action_dim)
  if actions.size == 0:
    raise ValueError("at least one joint action is required")

  action_pairs, counts = np.unique(actions, axis=0, return_counts=True)
  order = np.argsort(counts)[::-1]
  if max_components is not None:
    if max_components < 1:
      raise ValueError("max_components must be >= 1")
    order = order[:max_components]
  action_pairs = action_pairs[order]
  counts = counts[order]
  weights = counts.astype(np.float32) / float(counts.sum())
  means = normalize_joint_actions(action_pairs, action_dim)
  gmm = GaussianMixture2D(
    means=jnp.asarray(means, dtype=jnp.float32),
    std=float(std),
    weights=jnp.asarray(weights, dtype=jnp.float32),
  )
  return JointActionGMM(gmm=gmm, action_pairs=action_pairs, counts=counts)


def train_flow_for_gmm(
  rng: jax.Array,
  gmm: GaussianMixture2D,
  *,
  train_steps: int,
  batch_size: int,
  learning_rate: float,
  hidden_dims: tuple[int, ...] = (64, 64, 64, 64),
  progress_callback: Callable[[int, float], None] | None = None,
) -> tuple[TrainState, list[float]]:
  """Train the existing JAX flow-matching MLP on a joint-action GMM."""
  if train_steps < 1:
    raise ValueError("train_steps must be >= 1")
  if batch_size < 1:
    raise ValueError("batch_size must be >= 1")

  rng, init_key = jax.random.split(rng)
  state = create_train_state(
    init_key,
    MLPVectorField(hidden_dims=hidden_dims),
    learning_rate=learning_rate,
    dim=gmm.dim,
  )
  losses: list[float] = []
  for step_index in range(train_steps):
    rng, step_key = jax.random.split(rng)
    state, loss = train_step(state, step_key, gmm, batch_size)
    loss_value = float(loss)
    losses.append(loss_value)
    if progress_callback is not None:
      progress_callback(step_index + 1, loss_value)
  jax.block_until_ready(state.params)
  return state, losses


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


def sample_flow_points(
  train_state: TrainState,
  rng: jax.Array,
  *,
  num_samples: int,
  integration_steps: int = 64,
) -> jax.Array:
  """Generate 2D samples by integrating the learned vector field from noise."""
  if num_samples < 1:
    raise ValueError("num_samples must be >= 1")
  if integration_steps < 1:
    raise ValueError("integration_steps must be >= 1")

  x0 = jax.random.normal(rng, shape=(num_samples, 2))
  ts = jnp.linspace(0.0, 1.0, integration_steps + 1)

  def drift_fn(x: jax.Array, t: jax.Array) -> jax.Array:
    t_batch = jnp.full((x.shape[0], 1), t)
    return train_state.apply_fn({"params": train_state.params}, x, t_batch)

  return euler_integrate(drift_fn, x0, ts)[-1]


def flow_joint_action_policy(
  train_state: TrainState,
  *,
  num_envs: int,
  action_dim: int,
  seed: int,
  integration_steps: int = 64,
):
  """Create an evaluation policy that samples joint actions from a flow model."""
  key = jax.random.PRNGKey(seed)
  sample_fn = jax.jit(
    lambda state, sample_key: sample_flow_points(
      state,
      sample_key,
      num_samples=num_envs,
      integration_steps=integration_steps,
    )
  )

  def act(observations: np.ndarray) -> np.ndarray:
    nonlocal key
    del observations
    key, sample_key = jax.random.split(key)
    points = np.asarray(sample_fn(train_state, sample_key), dtype=np.float32)
    return decode_joint_actions(points, action_dim)

  return act


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


def _validate_joint_actions(
  joint_actions: np.ndarray,
  *,
  action_dim: int | None,
) -> np.ndarray:
  actions = np.asarray(joint_actions, dtype=np.int32)
  if actions.ndim != 2 or actions.shape[1] != 2:
    raise ValueError(f"expected joint actions shaped [N, 2], got {actions.shape}")
  if actions.size == 0:
    raise ValueError("at least one joint action is required")
  if action_dim is not None:
    if action_dim < 2:
      raise ValueError("action_dim must be >= 2")
    if actions.min() < 0 or actions.max() >= action_dim:
      raise ValueError("joint actions contain values outside the action space")
  return actions
