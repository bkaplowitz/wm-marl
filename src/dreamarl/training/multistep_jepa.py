"""Alignment and objective helpers for direct multi-step JEPA prediction.

The helpers in this module are deliberately independent from the recurrent
world model.  A factual root is paired with its own replay action prefix and a
future EMA target; no predicted state is ever fed into another prediction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp


f32 = jnp.float32
sg = jax.lax.stop_gradient


def aligned_action_windows(
    actions,
    action_mask,
    present,
    controllable_alive,
    is_first,
    *,
    action_low: int,
    max_horizon: int,
):
    """Build identity-preserving own-action windows rooted on factual states.

    Args:
      actions: Actions for transitions ``t -> t + 1``, shaped ``[B,T-1,A]``.
      action_mask: Availability at each transition source, ``[B,T-1,A,C]``.
      present: Factual roster mask, shaped ``[B,T,A]``.
      controllable_alive: Factual controllability mask, shaped ``[B,T,A]``.
      is_first: Environment reset markers on observations, shaped ``[B,T]``.

    Returns:
      Action windows ``[B,T-K,A,K]`` and a callable-free dictionary containing
      one supervision mask ``[B,T-K,A]`` for every horizon ``1..K``.  A target
      remains valid when every source action in its prefix is legal and taken
      by a present controllable agent, no reset is crossed, and the target
      agent slot is present.  Death exactly at the target is retained.
    """

    actions = jnp.asarray(actions)
    action_mask = jnp.asarray(action_mask, bool)
    present = jnp.asarray(present, bool)
    controllable_alive = jnp.asarray(controllable_alive, bool)
    is_first = jnp.asarray(is_first, bool)
    if max_horizon < 1:
        raise ValueError("multi-step JEPA max_horizon must be positive")
    if actions.ndim != 3:
        raise ValueError(f"actions must be [B,T-1,A], got {actions.shape}")
    batch, transitions, agents = actions.shape
    length = transitions + 1
    if action_mask.ndim != 4 or action_mask.shape[:3] != actions.shape:
        raise ValueError(
            "action masks must be [B,T-1,A,C] and align with actions, got "
            f"{action_mask.shape} versus {actions.shape}"
        )
    if present.shape != (batch, length, agents):
        raise ValueError(
            f"present must be {(batch, length, agents)}, got {present.shape}"
        )
    if controllable_alive.shape != present.shape:
        raise ValueError("controllable_alive must align with present")
    if is_first.shape != (batch, length):
        raise ValueError(f"is_first must be {(batch, length)}, got {is_first.shape}")
    roots = length - max_horizon
    if roots < 1:
        raise ValueError(
            "multi-step JEPA needs sequence length greater than max_horizon, "
            f"got T={length}, K={max_horizon}"
        )

    windows = jnp.stack(
        [actions[:, offset : offset + roots] for offset in range(max_horizon)],
        axis=-1,
    )
    return windows, action_window_validity(
        windows,
        action_mask,
        present,
        controllable_alive,
        is_first,
        action_low=action_low,
    )


def action_window_validity(
    action_windows,
    action_mask,
    present,
    controllable_alive,
    is_first,
    *,
    action_low: int,
):
    """Validate candidate windows against the focal agent's factual path."""

    action_windows = jnp.asarray(action_windows)
    action_mask = jnp.asarray(action_mask, bool)
    present = jnp.asarray(present, bool)
    controllable_alive = jnp.asarray(controllable_alive, bool)
    is_first = jnp.asarray(is_first, bool)
    if action_windows.ndim != 4:
        raise ValueError(
            f"action windows must be [B,R,A,K], got {action_windows.shape}"
        )
    batch, roots, agents, max_horizon = action_windows.shape
    length = present.shape[1]
    if length != roots + max_horizon:
        raise ValueError(
            "action windows do not cover the factual target range: "
            f"R={roots}, K={max_horizon}, T={length}"
        )
    if action_mask.shape[:3] != (batch, length - 1, agents):
        raise ValueError("candidate action masks do not align with factual paths")
    if present.shape != (batch, length, agents):
        raise ValueError("candidate roster mask does not align with action windows")
    if controllable_alive.shape != present.shape:
        raise ValueError("candidate controllability does not align with roster")
    if is_first.shape != (batch, length):
        raise ValueError("candidate reset mask does not align with action windows")

    source_masks = jnp.stack(
        [
            action_mask[:, offset : offset + roots]
            for offset in range(max_horizon)
        ],
        axis=-2,
    )
    classes = action_mask.shape[-1]
    targets = action_windows.astype(jnp.int32) - int(action_low)
    in_range = (targets >= 0) & (targets < classes)
    safe_targets = jnp.clip(targets, 0, classes - 1)
    legal = jnp.take_along_axis(
        source_masks, safe_targets[..., None], axis=-1
    )[..., 0]
    source_present = jnp.stack(
        [present[:, offset : offset + roots] for offset in range(max_horizon)],
        axis=-1,
    )
    source_alive = jnp.stack(
        [
            controllable_alive[:, offset : offset + roots]
            for offset in range(max_horizon)
        ],
        axis=-1,
    )
    no_reset = jnp.stack(
        [
            ~is_first[:, offset + 1 : offset + roots + 1]
            for offset in range(max_horizon)
        ],
        axis=-1,
    )[:, :, None, :]
    step_valid = (
        source_present & source_alive & in_range & legal & no_reset
    )

    horizon_valid = {}
    for horizon in range(1, max_horizon + 1):
        target_present = present[:, horizon : horizon + roots]
        horizon_valid[horizon] = (
            step_valid[..., :horizon].all(axis=-1) & target_present
        )
    return horizon_valid


def same_focal_legal_action_interventions(
    action_windows,
    action_mask,
    *,
    action_low: int,
    horizons: Sequence[int],
):
    """Change one legal focal action without borrowing a teammate's action.

    For each horizon ``h > 1``, the intervention replaces only the last action
    used by that horizon (position ``h - 1``) with the first cyclically ordered
    legal alternative under the same focal slot's factual availability mask.
    H1 has no tail intervention because its action is already encoded in the
    live joint root. Returned windows and distinct masks are keyed by horizon.
    """

    action_windows = jnp.asarray(action_windows)
    if action_windows.ndim != 4:
        raise ValueError(
            f"action windows must be [B,R,A,K], got {action_windows.shape}"
        )
    batch, roots, agents, max_horizon = action_windows.shape
    if action_mask.ndim != 4 or action_mask.shape[:3] != (
        batch,
        roots + max_horizon - 1,
        agents,
    ):
        raise ValueError(
            "action masks must align with intervention roots and horizon, got "
            f"{action_mask.shape} for {action_windows.shape}"
        )
    classes = action_mask.shape[-1]
    if classes < 2:
        raise ValueError("legal action intervention needs at least two actions")
    horizons = tuple(int(horizon) for horizon in horizons)
    windows = {}
    distinct = {}
    for horizon in horizons:
        if horizon < 1 or horizon > max_horizon:
            raise ValueError(f"invalid intervention horizon {horizon}")
        if horizon == 1:
            windows[horizon] = action_windows
            distinct[horizon] = jnp.zeros((batch, roots, agents), bool)
            continue
        position = horizon - 1
        factual = action_windows[..., position].astype(jnp.int32) - int(action_low)
        in_range = (factual >= 0) & (factual < classes)
        safe_factual = jnp.clip(factual, 0, classes - 1)
        legal = action_mask[:, position : position + roots]
        offsets = jnp.arange(1, classes, dtype=jnp.int32)
        candidates = (safe_factual[..., None] + offsets) % classes
        candidate_legal = jnp.take_along_axis(legal, candidates, axis=-1)
        has_alternative = in_range & candidate_legal.any(axis=-1)
        selected = jnp.argmax(candidate_legal, axis=-1)
        alternative = jnp.take_along_axis(
            candidates, selected[..., None], axis=-1
        )[..., 0]
        replacement = jnp.where(
            has_alternative, alternative + int(action_low), action_windows[..., position]
        )
        windows[horizon] = action_windows.at[..., position].set(replacement)
        distinct[horizon] = has_alternative
    return windows, distinct


def direct_multistep_objective(
    predictions: Mapping[int, jax.Array],
    targets: Mapping[int, jax.Array],
    valid: Mapping[int, jax.Array],
    counterfactual_predictions: Mapping[int, jax.Array],
    counterfactual_valid: Mapping[int, jax.Array],
    distinct_tail: Mapping[int, jax.Array],
    *,
    horizons: Sequence[int],
    decay: float,
    action_margin: float,
):
    """Return separately normalized cosine and action-tail ranking losses.

    Geometric weights follow the ordered prediction heads rather than elapsed
    time: for horizons ``(1, 2, 4, 8)`` the normalized weights are proportional
    to ``(1, decay, decay**2, decay**3)``. EMA targets are stopped. The
    counterfactual ranking loss backpropagates through both the correct and
    factual and same-focal legal counterfactual action-tail queries.
    """

    horizons = tuple(int(horizon) for horizon in horizons)
    if not horizons or tuple(sorted(set(horizons))) != horizons:
        raise ValueError("multi-step JEPA horizons must be sorted unique positives")
    if not 0.0 < float(decay) <= 1.0:
        raise ValueError("multi-step JEPA decay must be in (0, 1]")
    if float(action_margin) < 0.0:
        raise ValueError("multi-step JEPA action margin must be nonnegative")
    expected = set(horizons)
    for name, values in (
        ("predictions", predictions),
        ("targets", targets),
        ("valid", valid),
        ("counterfactual_predictions", counterfactual_predictions),
        ("counterfactual_valid", counterfactual_valid),
        ("distinct_tail", distinct_tail),
    ):
        if set(values) != expected:
            raise ValueError(
                f"{name} horizons {sorted(values)} do not match {horizons}"
            )

    geometric = jnp.asarray(
        [float(decay) ** index for index in range(len(horizons))], f32
    )
    geometric /= geometric.sum()
    action_indices = [index for index, horizon in enumerate(horizons) if horizon > 1]
    action_geometric = jnp.zeros_like(geometric)
    if action_indices:
        selected = geometric[jnp.asarray(action_indices)]
        action_geometric = action_geometric.at[jnp.asarray(action_indices)].set(
            selected / selected.sum()
        )
    cosine_combined = None
    action_combined = None
    metrics = {}
    for index, horizon in enumerate(horizons):
        prediction = jnp.asarray(predictions[horizon], f32)
        target = sg(jnp.asarray(targets[horizon], f32))
        counterfactual = jnp.asarray(counterfactual_predictions[horizon], f32)
        weight = jnp.asarray(valid[horizon], f32)
        counterfactual_weight = jnp.asarray(counterfactual_valid[horizon], f32)
        distinct = jnp.asarray(distinct_tail[horizon], bool)
        if (
            prediction.shape != target.shape
            or counterfactual.shape != prediction.shape
        ):
            raise ValueError(
                f"H{horizon} prediction/target shapes differ: "
                f"{prediction.shape}, {target.shape}, {counterfactual.shape}"
            )
        if (
            weight.shape != prediction.shape[:-1]
            or counterfactual_weight.shape != weight.shape
            or distinct.shape != weight.shape
        ):
            raise ValueError(
                f"H{horizon} masks do not match prediction {prediction.shape}: "
                f"{weight.shape}, {counterfactual_weight.shape}, {distinct.shape}"
            )

        cosine = _cosine(prediction, target)
        distance = 1.0 - cosine
        normalized = distance * weight / jnp.maximum(weight.mean(), 1e-8)
        weighted = geometric[index] * normalized
        cosine_combined = (
            weighted if cosine_combined is None else cosine_combined + weighted
        )

        counterfactual_cosine = _cosine(counterfactual, target)
        counterfactual_weight = weight * counterfactual_weight
        if horizon == 1:
            # hidden_t already consumed a_t, so there is no future action tail.
            counterfactual_weight = jnp.zeros_like(counterfactual_weight)
        action_weight = counterfactual_weight * distinct.astype(f32)
        cosine_gap = cosine - counterfactual_cosine
        margin_loss = jax.nn.relu(float(action_margin) - cosine_gap)
        normalized_margin = margin_loss * action_weight / jnp.maximum(
            action_weight.mean(), 1e-8
        )
        weighted_margin = action_geometric[index] * normalized_margin
        action_combined = (
            weighted_margin
            if action_combined is None
            else action_combined + weighted_margin
        )

        metric_prediction = sg(prediction)
        metric_cosine = _cosine(metric_prediction, target)
        metric_counterfactual = sg(counterfactual)
        metric_counterfactual_cosine = _cosine(metric_counterfactual, target)
        prediction_change = 1.0 - _cosine(
            metric_prediction, metric_counterfactual
        )
        mean_rank = _within_team_mean_rank(metric_prediction, target, weight)
        prediction_std, prediction_rank, prediction_rank_count = _spread_metrics(
            metric_prediction, weight
        )
        target_std, target_rank, target_rank_count = _spread_metrics(target, weight)
        prefix = f"h{horizon}"
        metrics.update(
            {
                f"{prefix}_cosine": _masked_average(metric_cosine, weight),
                f"{prefix}_within_team_mean_rank": mean_rank,
                f"{prefix}_action_counterfactual_cosine_drop": _masked_average(
                    metric_cosine - metric_counterfactual_cosine, action_weight
                ),
                f"{prefix}_action_counterfactual_prediction_change": _masked_average(
                    prediction_change, action_weight
                ),
                f"{prefix}_action_correct_vs_counterfactual_gap": _masked_average(
                    sg(cosine_gap), action_weight
                ),
                f"{prefix}_action_margin_loss": _masked_average(
                    sg(margin_loss), action_weight
                ),
                f"{prefix}_action_counterfactual_legal_count": (
                    counterfactual_weight.sum()
                ),
                f"{prefix}_action_counterfactual_legal_coverage": (
                    counterfactual_weight.sum() / jnp.maximum(weight.sum(), 1.0)
                ),
                f"{prefix}_action_distinct_legal_count": action_weight.sum(),
                f"{prefix}_action_distinct_legal_coverage": (
                    action_weight.sum() / jnp.maximum(weight.sum(), 1.0)
                ),
                f"{prefix}_prediction_std": prediction_std,
                f"{prefix}_target_std": target_std,
                f"{prefix}_prediction_effective_rank": prediction_rank,
                f"{prefix}_target_effective_rank": target_rank,
                f"{prefix}_prediction_rank_sample_count": prediction_rank_count,
                f"{prefix}_target_rank_sample_count": target_rank_count,
                f"{prefix}_valid_count": weight.sum(),
                f"{prefix}_valid_fraction": weight.mean(),
            }
        )
    assert cosine_combined is not None and action_combined is not None
    metrics["weighted_cosine_loss"] = sum(
        geometric[index] * (1.0 - metrics[f"h{horizon}_cosine"])
        for index, horizon in enumerate(horizons)
    )
    metrics["weighted_action_margin_loss"] = sum(
        action_geometric[index] * metrics[f"h{horizon}_action_margin_loss"]
        for index, horizon in enumerate(horizons)
    )
    return {"cosine": cosine_combined, "action": action_combined}, metrics


def _cosine(left, right):
    left = left / jnp.maximum(jnp.linalg.norm(left, axis=-1, keepdims=True), 1e-8)
    right = right / jnp.maximum(jnp.linalg.norm(right, axis=-1, keepdims=True), 1e-8)
    return jnp.sum(left * right, axis=-1)


def _masked_average(value, weight):
    weight = weight.astype(f32)
    return (value.astype(f32) * weight).sum() / jnp.maximum(weight.sum(), 1.0)


def _within_team_mean_rank(prediction, target, valid):
    """Rank each agent's matching future target among its factual teammates."""

    prediction = prediction / jnp.maximum(
        jnp.linalg.norm(prediction, axis=-1, keepdims=True), 1e-8
    )
    target = target / jnp.maximum(
        jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-8
    )
    similarity = jnp.einsum("...id,...jd->...ij", prediction, target)
    positive = jnp.diagonal(similarity, axis1=-2, axis2=-1)
    agents = similarity.shape[-1]
    identity = jnp.eye(agents, dtype=bool)
    candidates = valid[..., None, :].astype(bool) & ~identity
    better = (similarity > positive[..., None]) & candidates
    tied = (similarity == positive[..., None]) & candidates
    rank = (
        1.0
        + better.astype(f32).sum(axis=-1)
        + 0.5 * tied.astype(f32).sum(axis=-1)
    )
    return _masked_average(rank, valid)


def _spread_metrics(value, valid, max_samples=64):
    """Return RMS feature spread and entropy effective rank on a fixed sample."""

    value = sg(jnp.asarray(value, f32)).reshape((-1, value.shape[-1]))
    weight = jnp.asarray(valid, f32).reshape(-1)
    count = jnp.maximum(weight.sum(), 1.0)
    mean = (value * weight[:, None]).sum(axis=0) / count
    variance = (jnp.square(value - mean) * weight[:, None]).sum(axis=0) / count
    spread = jnp.sqrt(jnp.maximum(variance.mean(), 0.0))

    sample_count = min(max_samples, value.shape[0])
    sample_weight, indices = jax.lax.top_k(weight, sample_count)
    sample = value[indices]
    selected = jnp.maximum(sample_weight.sum(), 1.0)
    sample_mean = (sample * sample_weight[:, None]).sum(axis=0) / selected
    sample = (sample - sample_mean) * jnp.sqrt(sample_weight[:, None])
    gram = sample @ sample.T / selected
    eigenvalue = jnp.maximum(jnp.linalg.eigvalsh(gram), 0.0)
    probability = eigenvalue / jnp.maximum(eigenvalue.sum(), 1e-8)
    entropy = -(probability * jnp.log(jnp.maximum(probability, 1e-8))).sum()
    effective_rank = jnp.where(sample_weight.sum() > 0, jnp.exp(entropy), 0.0)
    return spread, effective_rank, sample_weight.sum()


__all__ = [
    "action_window_validity",
    "aligned_action_windows",
    "same_focal_legal_action_interventions",
    "direct_multistep_objective",
]
