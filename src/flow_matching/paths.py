"""Gaussian conditional probability path formulas."""

import jax
import jax.numpy as jnp


def alpha(t: jax.Array) -> jax.Array:
    """Linear interpolation schedule alpha_t = t."""
    return t


def alpha_dt(t: jax.Array) -> jax.Array:
    """Time derivative of alpha_t."""
    return jnp.ones_like(t)


def beta(t: jax.Array, eps: float = 1e-4) -> jax.Array:
    """Noise schedule beta_t = sqrt(1 - t), with a small numerical floor."""
    return jnp.sqrt(jnp.maximum(1.0 - t, eps))


def beta_dt(t: jax.Array, eps: float = 1e-4) -> jax.Array:
    """Time derivative of beta_t, matching the original PyTorch notebook convention."""
    return -0.5 / (jnp.sqrt(jnp.maximum(1.0 - t, eps)) + eps)


def sample_conditional_path(key: jax.Array, x1: jax.Array, t: jax.Array) -> jax.Array:
    """Sample x_t ~ N(alpha_t x1, beta_t^2 I)."""
    key1, key2 = jax.random.split(key)
    epsilon = jax.random.normal(key1, x1.shape)
    return alpha(t) * x1 + beta(t) * epsilon  # xt


def conditional_vector_field(xt: jax.Array, x1: jax.Array, t: jax.Array) -> jax.Array:
    """Evaluate the conditional vector field u_t(x_t | x1)."""
    dlogbt = beta_dt(t) / beta(t)
    return (alpha_dt(t) - dlogbt * alpha(t)) * x1 + dlogbt * xt


def conditional_score(xt: jax.Array, x1: jax.Array, t: jax.Array) -> jax.Array:
    """Evaluate the conditional score for the Gaussian path."""
    return (alpha(t) * x1 - xt) / (beta(t) ** 2)
