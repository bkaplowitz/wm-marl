"""Training utilities for Joint-Embedding Counterfactual Credit (JECC).

JECC learns from grouped replay tensors shaped ``[B, T, A, ...]`` and assigns
credit on synchronized imagined tensors with the same explicit agent axis. The
functions here are intentionally model agnostic: neural modules live in
``dreamarl.models.jecc`` while this module owns temporal alignment, outcome
targets, and policy-centred all-action interventions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp


f32 = jnp.float32
sg = jax.lax.stop_gradient


class OutcomeWindows(NamedTuple):
    """Multi-horizon factual outcome targets aligned to replay states."""

    tokens: tuple[jax.Array, ...]
    masks: tuple[jax.Array, ...]
    returns: jax.Array
    valid: jax.Array


class CounterfactualOutcomeMap(NamedTuple):
    """Policy-centred result of intervening on every focal action."""

    utilities: jax.Array
    probabilities: jax.Array
    factual_utility: jax.Array
    expected_utility: jax.Array
    factual_embedding: jax.Array
    expected_embedding: jax.Array


def build_outcome_windows(
    rewards,
    active,
    is_first,
    is_last,
    is_terminal,
    *,
    horizons: Sequence[int] = (5, 15, 32),
    gamma: float,
) -> OutcomeWindows:
    """Build generic future-outcome tokens beginning exactly at ``t + 1``.

    Args:
      rewards: Per-agent training rewards ``[B, T, A]``.
      active: Active-agent mask ``[B, T, A]``.
      is_first: Episode-reset indicators ``[B, T]`` or ``[B, T, A]``.
      is_last: Episode-boundary indicators with the same leading axes.
      is_terminal: Terminal indicators with the same leading axes.
      horizons: Prediction horizons. Each horizon uses the same source states.
      gamma: Discount used for the team-mean return target.

    A target is valid when its complete horizon is present in replay or when an
    observed episode boundary completes it early. Boundary transitions are
    included, while later transitions from the next episode are always masked.
    Incomplete replay tails without a boundary are never treated as targets.

    Returns:
      ``tokens[k]`` shaped ``[B,T,A,H_k,4]`` containing symlog team-mean
      reward, symlog focal reward, active-team reward disagreement, and
      continuation. ``masks[k]`` marks included future steps. ``returns`` and
      ``valid`` are shaped ``[B,T,A,K]``.
    """

    rewards = f32(rewards)
    active = jnp.asarray(active, bool)
    if rewards.ndim != 3 or active.shape != rewards.shape:
        raise ValueError(
            "outcome windows require rewards and active shaped [B,T,A], got "
            f"{rewards.shape} and {active.shape}"
        )
    horizons = tuple(int(horizon) for horizon in horizons)
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError(f"outcome horizons must be positive, got {horizons}")
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError(f"outcome discount must lie in [0, 1], got {gamma}")

    first = _environment_flag(is_first, rewards.shape[:2], "is_first")
    last = _environment_flag(is_last, rewards.shape[:2], "is_last")
    terminal = _environment_flag(is_terminal, rewards.shape[:2], "is_terminal")
    batch, length, agents = rewards.shape
    source_active = active[..., None]

    windows = []
    masks = []
    returns = []
    validities = []
    for horizon in horizons:
        offsets = jnp.arange(1, horizon + 1, dtype=jnp.int32)
        indices = jnp.arange(length, dtype=jnp.int32)[:, None] + offsets[None]
        available = indices < length
        safe_indices = jnp.minimum(indices, max(length - 1, 0))

        future_rewards = rewards[:, safe_indices, :]
        future_active = active[:, safe_indices, :]
        future_first = first[:, safe_indices]
        future_last = last[:, safe_indices]
        future_terminal = terminal[:, safe_indices]

        available_bt = available[None]
        first_before = jnp.concatenate(
            [
                jnp.zeros((batch, length, 1), bool),
                jnp.cumsum(future_first[..., :-1], axis=-1) > 0,
            ],
            axis=-1,
        )
        last_before = jnp.concatenate(
            [
                jnp.zeros((batch, length, 1), bool),
                jnp.cumsum(future_last[..., :-1], axis=-1) > 0,
            ],
            axis=-1,
        )
        step_mask = available_bt & ~first_before & ~last_before & ~future_first
        boundary_seen = ((future_last | future_terminal) & step_mask).any(axis=-1)
        complete = step_mask[..., -1]
        source_valid = ~last & (complete | boundary_seen)

        focal_step_mask = step_mask[:, :, None, :] & source_active
        future_active_f = future_active.astype(f32)
        active_count = future_active_f.sum(axis=-1)
        safe_count = jnp.maximum(active_count, 1.0)
        team_mean = (future_rewards * future_active_f).sum(axis=-1) / safe_count
        centered = future_rewards - team_mean[..., None]
        disagreement = jnp.sqrt(
            jnp.maximum(
                (jnp.square(centered) * future_active_f).sum(axis=-1) / safe_count,
                0.0,
            )
        )
        focal_reward = jnp.swapaxes(
            jnp.where(future_active, future_rewards, 0.0), -1, -2
        )
        team_mean = jnp.broadcast_to(
            team_mean[:, :, None, :], (batch, length, agents, horizon)
        )
        disagreement = jnp.broadcast_to(
            disagreement[:, :, None, :], (batch, length, agents, horizon)
        )
        continuation = jnp.broadcast_to(
            (~future_terminal)[:, :, None, :],
            (batch, length, agents, horizon),
        ).astype(f32)
        outcome = jnp.stack(
            [
                _symlog(team_mean),
                _symlog(focal_reward),
                disagreement,
                continuation,
            ],
            axis=-1,
        )
        outcome = jnp.where(focal_step_mask[..., None], outcome, 0.0)

        discounts = f32(gamma) ** jnp.arange(horizon, dtype=f32)
        horizon_return = (team_mean * focal_step_mask.astype(f32) * discounts).sum(
            axis=-1
        )
        horizon_valid = source_valid[:, :, None] & active

        windows.append(outcome)
        masks.append(focal_step_mask)
        returns.append(horizon_return)
        validities.append(horizon_valid)

    return OutcomeWindows(
        tuple(windows),
        tuple(masks),
        jnp.stack(returns, axis=-1),
        jnp.stack(validities, axis=-1),
    )


def factual_jecc_losses(
    outcome_prediction,
    outcome_target,
    utility_loss,
    predicted_utility_loss,
    outcome_valid,
    outer_valid=None,
):
    """Assemble factual JECC losses without choosing their relative weights.

    Predictions and targets use ``[B,T,A,K,D]`` for outcomes. Utility losses
    and ``outcome_valid`` use ``[B,T,A,K]``. Returned values are ``[B,T,A]``
    and normalized so their ordinary mean equals the corresponding
    valid-sample mean. The caller owns the scientific loss weights.
    """

    outcome_distance, outcome_cosine = cosine_distance(
        outcome_prediction, outcome_target
    )
    outcome_valid = jnp.asarray(outcome_valid, bool)
    utility_loss = f32(utility_loss)
    predicted_utility_loss = f32(predicted_utility_loss)
    if outer_valid is None:
        outer_valid = jnp.ones(outcome_valid.shape[:3], bool)
    outer_valid = jnp.asarray(outer_valid, bool)

    losses = {
        "outcome": _normalized_per_transition(
            outcome_distance, outcome_valid, outer_valid
        ),
        "utility": _normalized_per_transition(utility_loss, outcome_valid, outer_valid),
        "predicted_utility": _normalized_per_transition(
            predicted_utility_loss, outcome_valid, outer_valid
        ),
    }
    metrics = {
        "outcome_cosine": masked_mean(outcome_cosine, outcome_valid),
        "outcome_valid_fraction": outcome_valid.astype(f32).mean(),
        "utility_loss": masked_mean(utility_loss, outcome_valid),
        "predicted_utility_loss": masked_mean(predicted_utility_loss, outcome_valid),
    }
    return losses, metrics


def all_action_outcome_map(
    factual_actions,
    policy_probabilities,
    predict_outcomes: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    *,
    action_mask=None,
) -> CounterfactualOutcomeMap:
    """Evaluate every focal action while holding sampled peer actions fixed.

    Args:
      factual_actions: Sampled joint action ``[..., A]``.
      policy_probabilities: Focal policy probabilities ``[..., A, C]``.
      predict_outcomes: Callback receiving interventions ``[..., C, A, A]``.
        The first added axis enumerates every action, the next enumerates focal
        queries, and the final axis is the intervened joint action. It returns
        outcome embeddings ``[...,C,A,K,D]`` and horizon utilities
        ``[...,C,A,K]``.
      action_mask: Optional observed legal-action mask ``[...,A,C]``. Replay
        uses the exact mask; imagination omits it and retains the policy's soft
        predicted availability probabilities.

    The utility for one action is the equal mean over configured horizons.
    Embedding expectations retain their horizon axis for representation-level
    credit diagnostics. Only the compact scalar all-action map is materialized.
    """

    factual_actions = jnp.asarray(factual_actions, jnp.int32)
    probabilities = normalized_action_probabilities(policy_probabilities, action_mask)
    if probabilities.shape[:-1] != factual_actions.shape:
        raise ValueError(
            "policy probabilities must match factual actions and add a class "
            f"axis, got {probabilities.shape} and {factual_actions.shape}"
        )
    agents = factual_actions.shape[-1]
    action_count = probabilities.shape[-1]
    identity = jnp.eye(agents, dtype=bool)
    base = jnp.broadcast_to(
        factual_actions[..., None, :], (*factual_actions.shape, agents)
    )

    alternatives = jnp.arange(action_count, dtype=jnp.int32).reshape(
        (action_count, 1, 1)
    )
    interventions = jnp.where(identity, alternatives, base[..., None, :, :])
    embeddings, horizon_utilities = predict_outcomes(interventions)
    embeddings = f32(embeddings)
    horizon_utilities = f32(horizon_utilities)
    expected_embedding_shape = (*factual_actions.shape[:-1], action_count, agents)
    if horizon_utilities.shape[:-1] != expected_embedding_shape:
        raise ValueError(
            "counterfactual utility callback must return [...,C,A,K], got "
            f"{horizon_utilities.shape} for interventions {interventions.shape}"
        )
    if embeddings.shape[:-2] != expected_embedding_shape:
        raise ValueError(
            "counterfactual embedding callback must return [...,C,A,K,D], got "
            f"{embeddings.shape} for interventions {interventions.shape}"
        )

    # Public maps are focal-major [..., A, C, ...], matching the policy tensor.
    utilities = jnp.swapaxes(horizon_utilities.mean(axis=-1), -1, -2)
    embeddings = jnp.swapaxes(embeddings, -4, -3)
    selected = jax.nn.one_hot(factual_actions, action_count, dtype=f32)
    factual_utility = (selected * utilities).sum(axis=-1)
    expected_utility = (probabilities * utilities).sum(axis=-1)
    factual_embedding = (selected[..., None, None] * embeddings).sum(axis=-3)
    expected_embedding = (probabilities[..., None, None] * embeddings).sum(axis=-3)
    return CounterfactualOutcomeMap(
        utilities,
        probabilities,
        factual_utility,
        expected_utility,
        factual_embedding,
        expected_embedding,
    )


def jecc_advantage(
    base_advantage,
    counterfactual: CounterfactualOutcomeMap,
    return_scale,
    alpha,
):
    """Blend normalized B0 and JECC credit as a score-function signal."""

    raw = counterfactual.factual_utility - counterfactual.expected_utility
    normalized = raw / jnp.maximum(f32(return_scale), 1e-8)
    normalized = sg(normalized)
    alpha = f32(alpha)
    blended = (1.0 - alpha) * f32(base_advantage) + alpha * normalized
    contribution = sg(
        counterfactual.factual_embedding - counterfactual.expected_embedding
    )
    return blended, normalized, contribution


def jecc_metrics(
    counterfactual: CounterfactualOutcomeMap,
    normalized_advantage,
    contribution,
    *,
    valid=None,
):
    """Return concise credit diagnostics used to interpret actor behavior."""

    if valid is None:
        valid = jnp.ones_like(normalized_advantage, bool)
    valid = jnp.asarray(valid, bool)
    q_spread = counterfactual.utilities.std(axis=-1)
    contribution_norm = jnp.linalg.norm(f32(contribution), axis=-1).mean(axis=-1)
    return {
        "advantage_mean": masked_mean(normalized_advantage, valid),
        "advantage_abs": masked_mean(jnp.abs(normalized_advantage), valid),
        "advantage_std": masked_std(normalized_advantage, valid),
        "action_value_spread": masked_mean(q_spread, valid),
        "contribution_norm": masked_mean(contribution_norm, valid),
    }


def normalized_action_probabilities(probabilities, action_mask=None):
    """Apply an exact observed mask or preserve soft imagined probabilities."""

    probabilities = f32(probabilities)
    if action_mask is not None:
        probabilities = jnp.where(jnp.asarray(action_mask, bool), probabilities, 0.0)
    denominator = probabilities.sum(axis=-1, keepdims=True)
    fallback = jax.nn.one_hot(
        jnp.argmax(probabilities, axis=-1), probabilities.shape[-1], dtype=f32
    )
    return jnp.where(
        denominator > 0.0,
        probabilities / jnp.maximum(denominator, 1e-8),
        fallback,
    )


def cosine_distance(prediction, target):
    """Return per-item cosine distance and similarity with a stopped target."""

    prediction = f32(prediction)
    target = sg(f32(target))
    prediction /= jnp.maximum(jnp.linalg.norm(prediction, axis=-1, keepdims=True), 1e-8)
    target /= jnp.maximum(jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-8)
    cosine = (prediction * target).sum(axis=-1)
    return 1.0 - cosine, cosine


def masked_cosine_loss(prediction, target, valid):
    """Return valid-sample cosine loss and similarity."""

    distance, cosine = cosine_distance(prediction, target)
    return masked_mean(distance, valid), masked_mean(cosine, valid)


def masked_mean(value, valid):
    """Average a tensor over an explicitly broadcastable validity mask."""

    value = f32(value)
    valid = jnp.broadcast_to(jnp.asarray(valid, bool), value.shape)
    weight = valid.astype(f32)
    return (value * weight).sum() / jnp.maximum(weight.sum(), 1.0)


def masked_std(value, valid):
    """Standard deviation over explicitly valid entries."""

    mean = masked_mean(value, valid)
    return jnp.sqrt(masked_mean(jnp.square(f32(value) - mean), valid))


def _normalized_per_transition(value, valid, outer_valid):
    value = f32(value)
    valid = jnp.asarray(valid, bool)
    if value.shape != valid.shape:
        raise ValueError(
            f"loss and validity shapes must match, got {value.shape} and {valid.shape}"
        )
    if outer_valid.shape != value.shape[:3]:
        raise ValueError(
            "outer validity must match [B,T,A], got "
            f"{outer_valid.shape} for {value.shape}"
        )
    outer_count = outer_valid.astype(f32).sum()
    if value.ndim == 3:
        scale = outer_count / jnp.maximum(valid.astype(f32).sum(), 1.0)
        return value * valid.astype(f32) * scale
    if value.ndim != 4:
        raise ValueError(f"JECC losses must be [B,T,A] or [B,T,A,K], got {value.shape}")
    weight = valid.astype(f32)
    scale = outer_count / jnp.maximum(weight.sum(), 1.0)
    return (value * weight).sum(axis=-1) * scale


def _environment_flag(value, shape, name):
    value = jnp.asarray(value, bool)
    if value.shape == shape:
        return value
    if value.ndim == 3 and value.shape[:2] == shape:
        return value.any(axis=-1)
    raise ValueError(f"{name} must be [B,T] or [B,T,A], got {value.shape}")


def _symlog(value):
    value = f32(value)
    return jnp.sign(value) * jnp.log1p(jnp.abs(value))


__all__ = [
    "CounterfactualOutcomeMap",
    "OutcomeWindows",
    "all_action_outcome_map",
    "build_outcome_windows",
    "cosine_distance",
    "factual_jecc_losses",
    "jecc_advantage",
    "jecc_metrics",
    "masked_cosine_loss",
    "masked_mean",
    "masked_std",
    "normalized_action_probabilities",
]
