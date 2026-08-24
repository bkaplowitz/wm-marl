"""Functional policy-churn regularization for discrete local actors."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def categorical_forward_kl(reference, current, action_key: str):
    """Return ``KL(stop_gradient(reference) || current)`` per actor state."""

    reference_logits = jax.lax.stop_gradient(reference[action_key].logits).astype(
        jnp.float32
    )
    current_logits = current[action_key].logits.astype(jnp.float32)
    reference_logprob = jax.nn.log_softmax(reference_logits, axis=-1)
    current_logprob = jax.nn.log_softmax(current_logits, axis=-1)
    reference_prob = jnp.exp(reference_logprob)
    divergence = jnp.sum(
        reference_prob * (reference_logprob - current_logprob), axis=-1
    )
    return jnp.maximum(divergence, 0.0)


def relative_churn_scale(
    policy_magnitude,
    churn_loss,
    *,
    beta: float,
    maximum: float,
    epsilon: float,
):
    """Scale churn to a bounded fraction of the score-function objective."""

    scale = (
        beta
        * jax.lax.stop_gradient(policy_magnitude)
        / (jax.lax.stop_gradient(churn_loss) + epsilon)
    )
    return jnp.clip(scale, 0.0, maximum)


__all__ = ["categorical_forward_kl", "relative_churn_scale"]
