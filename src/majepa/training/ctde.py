"""Replay utilities for bounded self-fed CTDE prediction.

The authoritative CTDE loss remains the one-step teacher-forced objective in
``MARLCore``.  This module contains the temporal bookkeeping for an additional
two-step objective: choose valid replay anchors, gather aligned ``t:t+2``
values, detach the state produced by the first predicted transition, and place
the final-step losses back on the replay grid.

Keeping this code model agnostic makes the intended gradient boundary explicit:
the second-step loss trains the joint predictor at the last rollout step only.
It cannot update the local world model or backpropagate through the first joint
prediction.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp


f32 = jnp.float32


class TwoStepAnchors(NamedTuple):
    """Fixed-size replay anchor sample.

    ``batch`` and ``time`` address source states on a ``[B,T-2]`` anchor grid.
    ``valid`` distinguishes genuine uniformly sampled anchors from static-shape
    padding when a replay batch contains fewer valid anchors than requested.
    """

    batch: jax.Array
    time: jax.Array
    valid: jax.Array


def two_step_anchor_mask(is_first, source_valid):
    """Return sources whose complete ``t -> t+1 -> t+2`` path is in one episode.

    ``source_valid`` may be team-level ``[B,T]`` or per-agent ``[B,T,A]``.
    Per-agent validity is reduced only at the source state; an agent dying at
    either target remains a valid event to learn rather than removing the
    transition from the sample population.
    """

    first = jnp.asarray(is_first, bool)
    valid = jnp.asarray(source_valid, bool)
    if first.ndim == 3:
        first = first.any(axis=-1)
    if valid.ndim == 3:
        valid = valid.any(axis=-1)
    if first.ndim != 2 or valid.shape != first.shape:
        raise ValueError(
            "two-step anchors require is_first/source_valid [B,T] or [B,T,A], "
            f"got {jnp.shape(is_first)} and {jnp.shape(source_valid)}"
        )
    return valid[:, :-2] & ~first[:, 1:-1] & ~first[:, 2:]


def sample_two_step_anchors(key, valid, count):
    """Uniformly sample a fixed number of valid anchors without replacement.

    Invalid slots are returned only as static-shape padding when fewer than
    ``count`` valid anchors exist.  Their indices are safe to gather, and every
    downstream helper masks them with ``anchors.valid``.
    """

    valid = jnp.asarray(valid, bool)
    count = int(count)
    if valid.ndim != 2 or count < 1 or count > valid.size:
        raise ValueError(
            f"anchor sample count must be in [1,{valid.size}] for [B,L] valid, "
            f"got shape {valid.shape} and count {count}"
        )
    scores = jax.random.uniform(key, (valid.size,), dtype=f32)
    scores = jnp.where(valid.reshape(-1), scores, -jnp.ones_like(scores))
    _, indices = jax.lax.top_k(scores, count)
    length = valid.shape[1]
    return TwoStepAnchors(
        indices // length,
        indices % length,
        valid.reshape(-1)[indices],
    )


def gather_anchors(values, anchors: TwoStepAnchors, offset=0):
    """Gather a replay array or pytree at every sampled anchor plus ``offset``."""

    offset = int(offset)

    def gather(value):
        return value[anchors.batch, anchors.time + offset]

    return jax.tree.map(gather, values)


def detach_self_feed(values):
    """Apply the last-gradient-only boundary before a self-fed rollout step."""

    return jax.tree.map(jax.lax.stop_gradient, values)


def predicted_controllable_alive(current_alive, present, probability):
    """Apply the same detached, monotonic liveness rule as CTDE imagination."""

    current_alive = jnp.asarray(current_alive, bool)
    present = jnp.asarray(present, bool)
    probability = f32(probability)
    if current_alive.shape != present.shape or probability.shape != present.shape:
        raise ValueError(
            "CTDE liveness tensors must share [...,A], got "
            f"{current_alive.shape}, {present.shape}, and {probability.shape}"
        )
    return jax.lax.stop_gradient(current_alive & present & (probability >= 0.5))


def two_step_objective(
    predicted_embedding,
    target_embedding,
    auxiliary_losses: Mapping[str, jax.Array],
    anchors: TwoStepAnchors,
    supervision_valid,
    destination_valid,
    auxiliary_valid: Mapping[str, jax.Array] | None = None,
):
    """Build replay-aligned losses for the final step of a two-step rollout.

    Args:
      predicted_embedding: Joint-JEPA prediction ``[K,A,D]`` at ``t+2``.
      target_embedding: Stopped EMA encoder target with the same shape.
      auxiliary_losses: Already reduced reward, continuation, action-mask, and
        liveness head losses, each shaped ``[K,A]``.
      anchors: The sampled source coordinates on ``[B,T-2]``.
      supervision_valid: Per-sample/per-agent validity ``[K,A]``.  This is where
        the caller expresses embedding supervision semantics.
      destination_valid: Learner validity on the replay grid receiving the
        sparse losses, shaped ``[B,L,A]``.  Passing the full source-aligned
        grid lets transitions into an absorbing/dead state remain supervised
        even though that agent is no longer controllable at ``t+2``.
      auxiliary_valid: Optional per-head validity masks.  In particular,
        liveness is normally supervised for every present roster slot while
        embedding/reward/mask predictions use source-controllable slots.

    Returns:
      A loss dictionary on ``[B,L,A]`` and scalar diagnostic metrics.  Each
      dense loss is normalized so the learner's ordinary validity-masked mean
      equals the mean over sampled valid agents.
    """

    prediction = f32(predicted_embedding)
    target = jax.lax.stop_gradient(f32(target_embedding))
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            "two-step embeddings must match [K,A,D], got "
            f"{prediction.shape} and {target.shape}"
        )
    embedding_valid = jnp.asarray(supervision_valid, bool)
    destination_valid = jnp.asarray(destination_valid, bool)
    if embedding_valid.shape != prediction.shape[:2]:
        raise ValueError(
            f"supervision validity must be {prediction.shape[:2]}, got "
            f"{embedding_valid.shape}"
        )
    if destination_valid.ndim != 3:
        raise ValueError(
            f"destination validity must be [B,L,A], got {destination_valid.shape}"
        )
    outer_sample_valid = anchors.valid[:, None]
    outer_sample_valid &= destination_valid[anchors.batch, anchors.time]
    embedding_valid &= outer_sample_valid

    pred_norm = prediction / jnp.maximum(
        jnp.linalg.norm(prediction, axis=-1, keepdims=True), 1e-8
    )
    target_norm = target / jnp.maximum(
        jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-8
    )
    cosine = jnp.sum(pred_norm * target_norm, axis=-1)
    sampled_losses = {"embedding": 1.0 - cosine}
    validities = {"embedding": embedding_valid}
    auxiliary_valid = {} if auxiliary_valid is None else auxiliary_valid
    for name, value in auxiliary_losses.items():
        value = f32(value)
        if value.shape != embedding_valid.shape:
            raise ValueError(
                f"two-step {name} loss must be {embedding_valid.shape}, got "
                f"{value.shape}"
            )
        name = str(name)
        valid = jnp.asarray(auxiliary_valid.get(name, supervision_valid), bool)
        if valid.shape != embedding_valid.shape:
            raise ValueError(
                f"two-step {name} validity must be {embedding_valid.shape}, got "
                f"{valid.shape}"
            )
        sampled_losses[name] = value
        validities[name] = valid & outer_sample_valid

    losses = {
        name: _scatter_normalized(value, anchors, validities[name], destination_valid)
        for name, value in sampled_losses.items()
    }
    metrics = {
        "embedding_cosine": _masked_mean(cosine, embedding_valid),
        "valid_agents": embedding_valid.astype(f32).sum(),
        "valid_anchor_fraction": anchors.valid.astype(f32).mean(),
    }
    metrics.update(
        {
            f"{name}_loss": _masked_mean(value, validities[name])
            for name, value in sampled_losses.items()
        }
    )
    return losses, metrics


def _scatter_normalized(value, anchors, sample_valid, destination_valid):
    outer_count = destination_valid.astype(f32).sum()
    sample_count = sample_valid.astype(f32).sum()
    scale = outer_count / jnp.maximum(sample_count, 1.0)
    weighted = f32(value) * sample_valid.astype(f32) * scale
    output = jnp.zeros(destination_valid.shape, f32)
    return output.at[anchors.batch, anchors.time].add(weighted)


def _masked_mean(value, valid):
    weight = jnp.asarray(valid, bool).astype(f32)
    return (f32(value) * weight).sum() / jnp.maximum(weight.sum(), 1.0)


__all__ = [
    "TwoStepAnchors",
    "detach_self_feed",
    "gather_anchors",
    "predicted_controllable_alive",
    "sample_two_step_anchors",
    "two_step_anchor_mask",
    "two_step_objective",
]
