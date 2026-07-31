"""Dreamer-style symlog two-hot distributions for task-scale robustness."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from world_marl.dreamarl.config import DistributionConfig


def symlog(value: jax.Array) -> jax.Array:
    return jnp.sign(value) * jnp.log1p(jnp.abs(value))


def symexp(value: jax.Array) -> jax.Array:
    return jnp.sign(value) * jnp.expm1(jnp.abs(value))


def two_hot(target: jax.Array, config: DistributionConfig) -> jax.Array:
    """Project scalar targets onto neighboring bins in symlog space."""

    transformed = jnp.clip(symlog(target), config.low, config.high)
    position = (
        (transformed - config.low)
        / (config.high - config.low)
        * (config.bins - 1)
    )
    lower = jnp.floor(position).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, config.bins - 1)
    upper_weight = position - lower.astype(position.dtype)
    lower_weight = 1.0 - upper_weight
    return (
        jax.nn.one_hot(lower, config.bins) * lower_weight[..., None]
        + jax.nn.one_hot(upper, config.bins) * upper_weight[..., None]
    )


def two_hot_loss(
    logits: jax.Array,
    target: jax.Array,
    config: DistributionConfig,
) -> jax.Array:
    labels = two_hot(target, config)
    return -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)


def two_hot_mean(
    logits: jax.Array,
    config: DistributionConfig,
) -> jax.Array:
    support = jnp.linspace(config.low, config.high, config.bins)
    transformed = jnp.sum(jax.nn.softmax(logits, axis=-1) * support, axis=-1)
    return symexp(transformed)
