"""Replay anchors, team outcomes, and predicted rollout support."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


f32 = jnp.float32


def imagined_action_mask(probability, alive, seed=None):
    """Select imagined support; factual root masks never pass through here.

    None preserves the original threshold rule without consuming randomness.
    A key samples independent Bernoulli availability events. Both modes retain
    the established empty-mask fallback and absorbing dead-agent no-op support.
    """
    mask = (
        probability >= 0.5 if seed is None else jax.random.bernoulli(seed, probability)
    )
    noop = jnp.zeros_like(mask).at[..., 0].set(True)
    mask = jnp.where(mask.any(axis=-1, keepdims=True), mask, noop)
    return jax.lax.stop_gradient(jnp.where(alive[..., None], mask, noop))


class TwoStepAnchors(NamedTuple):
    """Fixed-size replay anchor sample.

    ``batch`` and ``time`` address source states on a ``[B,T-2]`` anchor grid.
    ``valid`` distinguishes genuine uniformly sampled anchors from static-shape
    padding when a replay batch contains fewer valid anchors than requested.
    """

    batch: jax.Array
    time: jax.Array
    valid: jax.Array


def broadcast_team_mean(values, present):
    """Broadcast a shared SMAC signal over the fixed, present agent roster.

    Reward and episode continuation belong to the team, including units that
    have died. Averaging present-slot predictions prevents their approximation
    errors from creating different objectives for agents in the same rollout.
    Absent padding has no contribution and receives a zero output.
    """

    values = jnp.asarray(values, f32)
    present = jnp.asarray(present, bool)
    if values.shape != present.shape:
        raise ValueError("team signals and roster must share [...,A] axes")
    total = jnp.where(present, values, 0.0).sum(axis=-1, keepdims=True)
    count = present.sum(axis=-1, keepdims=True)
    mean = total / jnp.maximum(count, 1)
    return jnp.where(present, mean, 0.0)


def shared_team_outcomes(reward, continuation, present, source_alive, next_alive):
    """Apply shared SMAC outcomes and the absorbing all-dead team boundary."""

    def any_alive(alive):
        alive = jnp.asarray(alive)
        if jnp.issubdtype(alive.dtype, jnp.bool_):
            return (alive & present).any(axis=-1, keepdims=True).astype(f32)
        probability = jnp.where(present, jnp.clip(alive, 0.0, 1.0), 0.0)
        return 1.0 - jnp.prod(1.0 - probability, axis=-1, keepdims=True)

    # Preserve the final reward on a transition out of a live team. Once the
    # source is absorbing, there can be no further reward or continuation.
    source_live = any_alive(source_alive)
    reward = broadcast_team_mean(reward, present) * source_live
    continuation = (
        broadcast_team_mean(continuation, present) * source_live * any_alive(next_alive)
    )
    return reward, continuation


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


__all__ = [
    "TwoStepAnchors",
    "broadcast_team_mean",
    "detach_self_feed",
    "gather_anchors",
    "predicted_controllable_alive",
    "sample_two_step_anchors",
    "shared_team_outcomes",
]
