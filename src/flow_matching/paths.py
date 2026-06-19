"""Conditional probability paths: gaussian (VP) and linear (OT) bridges."""

import jax
import jax.numpy as jnp


def alpha(t: jax.Array) -> jax.Array:
    """Linear interpolation schedule alpha_t = t. Used for the Gaussian path and the linear path."""
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
    """Return (alpha, alpha_dt, beta, beta_dt) schedule for a given flow type."""
    if flow_type == "gaussian":
        return alpha, alpha_dt, gaussian_beta, gaussian_beta_dt
    if flow_type == "linear":
        return alpha, alpha_dt, linear_beta, linear_beta_dt
    raise ValueError(f"unknown flow_type {flow_type!r}")


# Continuous flow matching


def sample_conditional_path(
    key: jax.Array,
    x1: jax.Array,
    t: jax.Array,
    alpha=alpha,
    beta=gaussian_beta,
) -> jax.Array:
    """Sample x_t ~ N(α_t x1, β_t^2 I)."""
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
    """Evaluate the analytic conditional vector field  u_t(x_t | x1)."""
    dlogbt = beta_dt(t) / beta(t)
    return (alpha_dt(t) - dlogbt * alpha(t)) * x1 + dlogbt * xt


def conditional_score(xt: jax.Array, x1: jax.Array, t: jax.Array) -> jax.Array:
    r"""Evaluate the conditional score for the Gaussian path
    The conditional score is the gradient of the log of the conditional probability density and given by:
    $α_t (x_1 - x_t) / (β_t^2 I)$
    """
    return (alpha(t) * x1 - xt) / (gaussian_beta(t) ** 2)


# Discrete flow matching


def sample_discrete_conditional_path(
    key: jax.Array,
    z: jax.Array,  # tokenized data
    t: jax.Array,
    num_categories: int,
) -> jax.Array:
    r"""Sample x_t from the factorized mixture path for each data dimension `j`.

    Per factor, keep the clean token z_j with prob kappa_t (= ``alpha(t) = t``) and
    otherwise draw a noise token from the uniform source p_init. This is the
    discrete twin of :func:`sample_conditional_path`.

    ``z`` holds integer tokens of shape ``(B, d)``; ``t`` is ``(B, 1)``. Returns
    integer tokens ``x_t`` of shape ``(B, d)``.

    Analytic form:
    $p(x_t | z, t) = α_t * z_t + (1 - α_t) \, ϵ, where ϵ = x_0 ∼ \text{Uniform}(0, 1)$

    """
    key_mask, key_noise = jax.random.split(key)
    # probability of being kept (sample x_t with probability kappa_t else sample from uniform source)
    kappa = alpha(t)
    # sample whether to keep the clean token or sample from uniform source
    mask = jax.random.bernoulli(key_mask, kappa, z.shape)
    # sample from uniform source
    x0 = jax.random.randint(key_noise, z.shape, 0, num_categories)
    return jnp.where(mask, z, x0)


# We use CTMC: rate matrix is given by $Q_t(y|x) \geq 0$ prob of jump x->y and $Q_t(x|x) = -\sum_{y \neq x} Q_t(y|x)$ for stay at x for all x.
# Poisson intensities.
# By definition $Q_t(y|x) = \frac{d}{dh}p_{t+h|t}(X_{t+h}=y|X_t=x)|_{h=0} = Q_t(y|x)$ for all x,y, $t \geq 0$
# Therefore similarly, the probability of staying at x is $\sum_{y \neq x} Q_t(y|x) =  \frac{d}{dh}(1 - p_{t+h|t}(X_{t+h}=y|X_t=x)|_{h=0}) = -Q(x|x)$
# Taking a first order approximation of changing to $y$:
# $p_{t+h|t}(X_{t+h}=y|X_t=x) = p_{t|t}(X_t=y|X_t=x)Q_t(y|x)h + O(h^2) = 1_{y=x} + Q_t(y|x)h + O(h^2) \approx 1_{y=x} + h *Q_t(y|x)$
# Sampling from 1_{y=x} + h * Q_t(y|x) is exactly the mixture path sampling step.


def factorized_jump_rates(
    posterior: jax.Array,
    t: jax.Array,
    eps: float = 1e-4,
    alpha=alpha,
    alpha_dt=alpha_dt,
) -> jax.Array:
    """Off-diagonal CTMC jump rates q_j(v) using the model.

    With linear schedule, ``α_t = t``, this is ``(p_{1|t}(z_j=v_i | z, t)  - δ_{z_j=v_i})/ (1 - t) the probability of drawing the data token v_i at position j. The
    ``eps`` floor mirrors the schedule guards in this module and is inert on the
    left-endpoint sampling grid where ``1 - t >= 1/steps``. Discrete twin of
    :func:`conditional_vector_field`. We are off diagonal, so $\delta_{z_j=v_i}$ is 0.


    With $a_t = t$ (and hence $\dot{a_t} = 1$) the agent stays on the clean token $z_t$. With probability $1 - a_t$ (and hence rate $-1$) the agent samples a noise token from the uniform source.

    Therefore, the jump rate between the clean token $z_t$ and the noise token $\epsilon$ is given by:
    $q_j(v) = p(x_t | z, t) / (1 - α_t)$, where p(x_t | z, t) = α_t * z_t + (1 - α_t) * ϵ, ϵ ∼ \text{Uniform}(0, 1)
    """
    return (alpha_dt(t)) / (jnp.maximum(1.0 - alpha(t), eps)) * posterior
