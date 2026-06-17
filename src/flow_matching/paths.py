"""Conditional probability paths: gaussian (VP) and linear (OT) bridges."""

import jax
import jax.numpy as jnp


def alpha(t: jax.Array) -> jax.Array:
    """Linear interpolation schedule alpha_t = t."""
    return t


def alpha_dt(t: jax.Array) -> jax.Array:
    """Time derivative of alpha_t."""
    return jnp.ones_like(t)


def gaussian_beta(t: jax.Array, eps: float = 1e-4) -> jax.Array:
    """Noise schedule beta_t = sqrt(1 - t), with a small numerical floor."""
    return jnp.sqrt(jnp.maximum(1.0 - t, eps))


def gaussian_beta_dt(t: jax.Array, eps: float = 1e-4) -> jax.Array:
    """Time derivative of beta_t."""
    return -0.5 / (jnp.sqrt(jnp.maximum(1.0 - t, eps)) + eps)


def linear_beta(t: jax.Array, eps: float = 1e-4) -> jax.Array:
    """OT schedule beta_t = 1 - t."""
    return jnp.maximum(1.0 - t, eps)


def linear_beta_dt(t: jax.Array, eps: float = 1e-4) -> jax.Array:
    """d beta_t / dt for the linear bridge."""
    return -jnp.ones_like(t)


def flow_schedule(flow_type: str = "gaussian"):
    """Map a flow type to its (alpha, alpha_dt, beta, beta_dt) callables."""
    if flow_type == "gaussian":
        return alpha, alpha_dt, gaussian_beta, gaussian_beta_dt
    if flow_type == "linear":
        return alpha, alpha_dt, linear_beta, linear_beta_dt
    raise ValueError(f"unknown flow_type {flow_type!r}")


def sample_conditional_path(
    key: jax.Array,
    x1: jax.Array,
    t: jax.Array,
    alpha=alpha,
    beta=gaussian_beta,
) -> jax.Array:
    """Sample x_t ~ N(alpha_t x1, beta_t^2 I)."""
    key, key_epsilon = jax.random.split(key)
    epsilon = jax.random.normal(key_epsilon, x1.shape)
    return alpha(t) * x1 + beta(t) * epsilon  # xt


def conditional_vector_field(
    xt: jax.Array,
    x1: jax.Array,
    t: jax.Array,
    alpha=alpha,
    alpha_dt=alpha_dt,
    beta=gaussian_beta,
    beta_dt=gaussian_beta_dt,
) -> jax.Array:
    """Evaluate the conditional vector field u_t(x_t | x1)."""
    dlogbt = beta_dt(t) / beta(t)
    return (alpha_dt(t) - dlogbt * alpha(t)) * x1 + dlogbt * xt


def conditional_score(xt: jax.Array, x1: jax.Array, t: jax.Array) -> jax.Array:
    """Evaluate the conditional score for the Gaussian path."""
    return (alpha(t) * x1 - xt) / (gaussian_beta(t) ** 2)
