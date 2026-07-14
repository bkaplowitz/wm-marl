from __future__ import annotations

import jax
import jax.numpy as jnp

from world_marl.dreamer_v3_baseline.config import DreamerV3Config


def symlog(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * (jnp.expm1(jnp.abs(x)))


def symexp_two_hot_support(
    *,
    num_bins: int,
    lower: float = -20.0,
    upper: float = 20.0,
) -> jax.Array:
    if num_bins <= 1:
        raise ValueError("num_bins must be greater than one")
    if lower >= upper:
        raise ValueError("lower must be smaller than upper")
    if lower != -upper:
        raise ValueError("DreamerV3 symexp support must be symmetric around zero")
    if num_bins % 2:
        half = jnp.linspace(lower, 0.0, (num_bins - 1) // 2 + 1, dtype=jnp.float32)
        half = symexp(half)
        return jnp.concatenate([half, -half[:-1][::-1]], axis=0)
    half = jnp.linspace(lower, 0.0, num_bins // 2, dtype=jnp.float32)
    half = symexp(half)
    return jnp.concatenate([half, -half[::-1]], axis=0)


def reconstruction_loss(
    predictions: jax.Array,
    observations: jax.Array,
    config: DreamerV3Config,
) -> jax.Array:
    targets = observations if config.is_image_observation else symlog(observations)
    squared_error = jnp.square(predictions - targets)
    event_axes = tuple(
        range(squared_error.ndim - len(config.observation_shape), squared_error.ndim)
    )
    return jnp.mean(jnp.sum(squared_error, axis=event_axes))


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


def symexp_two_hot(
    values: jax.Array,
    *,
    num_bins: int,
    lower: float = -20.0,
    upper: float = 20.0,
) -> jax.Array:
    """DreamerV3 two-hot targets over exponentially spaced value bins."""
    support = symexp_two_hot_support(
        num_bins=num_bins,
        lower=lower,
        upper=upper,
    )
    values = jnp.clip(values.astype(jnp.float32), support[0], support[-1])
    below = jnp.sum(support <= values[..., None], axis=-1) - 1
    above = num_bins - jnp.sum(support > values[..., None], axis=-1)
    below = jnp.clip(below, 0, num_bins - 1)
    above = jnp.clip(above, 0, num_bins - 1)
    equal = below == above
    distance_below = jnp.where(equal, 1.0, jnp.abs(support[below] - values))
    distance_above = jnp.where(equal, 1.0, jnp.abs(support[above] - values))
    total_distance = distance_below + distance_above
    weight_below = distance_above / total_distance
    weight_above = distance_below / total_distance
    return (
        jax.nn.one_hot(below, num_bins, dtype=jnp.float32) * weight_below[..., None]
        + jax.nn.one_hot(above, num_bins, dtype=jnp.float32) * weight_above[..., None]
    )


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


def balanced_categorical_kl_loss(
    posterior_logits: jax.Array,
    prior_logits: jax.Array,
    *,
    free_nats: float,
    dynamics_scale: float,
    representation_scale: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    dynamics_kl = categorical_kl_loss(
        jax.lax.stop_gradient(posterior_logits),
        prior_logits,
        free_nats=free_nats,
    )
    representation_kl = categorical_kl_loss(
        posterior_logits,
        jax.lax.stop_gradient(prior_logits),
        free_nats=free_nats,
    )
    total = dynamics_scale * dynamics_kl + representation_scale * representation_kl
    return total, dynamics_kl, representation_kl
