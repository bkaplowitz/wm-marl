"""Shared joint-action helpers for JaxMARL CoinGame flow experiments."""

from __future__ import annotations

from typing import Any

import numpy as np


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


def joint_action_counts(joint_actions: np.ndarray, action_dim: int) -> np.ndarray:
  """Count every two-agent joint action in an ``action_dim x action_dim`` grid."""
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
