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


def sample_discrete_conditional_path(
    key: jax.Array,
    z: jax.Array,
    t: jax.Array,
    num_categories: int,
) -> jax.Array:
    """Sample x_t from the factorized mixture path (discrete.md Alg 8, lines 5-9).

    Per factor, keep the clean token z_j with prob kappa_t (= ``alpha(t) = t``) and
    otherwise draw a noise token from the uniform source p_init. This is the
    discrete twin of :func:`sample_conditional_path`.

    ``z`` holds integer tokens of shape ``(B, d)``; ``t`` is ``(B, 1)``. Returns
    integer tokens ``x_t`` of shape ``(B, d)``.
    """
    key_mask, key_noise = jax.random.split(key)
    kappa = alpha(t)  # (B, 1), broadcast over the d factors
    mask = jax.random.bernoulli(key_mask, kappa, z.shape)
    noise = jax.random.randint(key_noise, z.shape, 0, num_categories)
    return jnp.where(mask, z, noise)


def mixture_path_rates(
    posterior: jax.Array,
    t: jax.Array,
    eps: float = 1e-4,
) -> jax.Array:
    """Off-diagonal CTMC jump rates q_j(v) from the denoising posterior.

    The conditional generator "jump to the clean token z at rate kappa_dot/(1 -
    kappa)" generates the mixture path, so the marginal rate is the posterior-
    average ``p_{1|t}(v|x) * kappa_dot/(1 - kappa)``. With the linear schedule
    ``kappa_t = t`` (``kappa_dot = 1``) this is ``posterior / (1 - t)``. The
    ``eps`` floor mirrors the schedule guards in this module and is inert on the
    left-endpoint sampling grid where ``1 - t >= 1/steps``. Discrete twin of
    :func:`conditional_vector_field`.
    """
    return posterior / jnp.maximum(1.0 - t, eps)
