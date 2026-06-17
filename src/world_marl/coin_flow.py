"""Flow-matching utilities for two-agent JaxMARL CoinGame actions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from flow_matching.distributions import GaussianMixture2D
from flow_matching.models import MLPVectorField
from flow_matching.simulate import euler_integrate
from flow_matching.train import create_train_state, train_step

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
