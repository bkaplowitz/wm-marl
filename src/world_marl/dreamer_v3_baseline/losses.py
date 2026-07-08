from __future__ import annotations

import jax
import jax.numpy as jnp


def symlog(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * (jnp.expm1(jnp.abs(x)))


def two_hot(
    values: jax.Array,
    *,
    num_bins: int,
    lower: float,
    upper: float,
) -> jax.Array:
    if num_bins <= 1:
        raise ValueError("num_bins must be greater than one")
    if lower >= upper:
        raise ValueError("lower must be smaller than upper")

    clipped = jnp.clip(values, lower, upper)
    position = (clipped - lower) / (upper - lower) * (num_bins - 1)
    lower_idx = jnp.floor(position).astype(jnp.int32)
    upper_idx = jnp.clip(lower_idx + 1, 0, num_bins - 1)
    upper_weight = position - lower_idx.astype(jnp.float32)
    lower_weight = 1.0 - upper_weight
    lower_hot = jax.nn.one_hot(lower_idx, num_bins, dtype=jnp.float32)
    upper_hot = jax.nn.one_hot(upper_idx, num_bins, dtype=jnp.float32)
    return lower_hot * lower_weight[..., None] + upper_hot * upper_weight[..., None]


def categorical_kl_loss(
    posterior_logits: jax.Array,
    prior_logits: jax.Array,
    *,
    free_nats: float,
) -> jax.Array:
    posterior_probs = jax.nn.softmax(posterior_logits, axis=-1)
    posterior_log_probs = jax.nn.log_softmax(posterior_logits, axis=-1)
    prior_log_probs = jax.nn.log_softmax(prior_logits, axis=-1)
    kl = jnp.sum(
        posterior_probs * (posterior_log_probs - prior_log_probs),
        axis=(-2, -1),
    )
    if free_nats > 0.0:
        kl = jnp.maximum(kl, free_nats)
    return jnp.mean(kl)
