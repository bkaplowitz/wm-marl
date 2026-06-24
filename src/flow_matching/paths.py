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


# LLaDA2.0 absorbing (masked) discrete diffusion (arXiv 2512.15745)
#
# Convention note: the continuous / CTMC flow above uses ``alpha(t)=t`` as the
# *keep* probability with t=1 == clean. The masked-diffusion schedule below also
# treats ``alpha`` as the keep probability, but with the LLaDA time direction
# t=0 == clean, t=1 == fully masked (linear: ``alpha_t = 1 - t``). A token is
# replaced by the absorbing ``[MASK]`` token w.p. the mask rate ``1 - alpha_t``,
# and the BDLM / SFT loss weights each masked-token cross-entropy by
# ``w(t) = -alpha'_t / (1 - alpha_t)`` (eqs 1, 5). ``[MASK]`` has id
# ``num_categories`` so the vocabulary is ``num_categories + 1``.


def mask_schedule(name: str = "linear"):
    """Return ``(alpha, alpha_dt)`` keep-probability schedule for masked diffusion.

    ``alpha(0)=1`` (clean) decreasing to ``alpha(1)=0`` (fully masked). ``linear``
    is the LLaDA2.0 default; ``cosine`` is provided as a non-linear alternative so
    the time reweighting can be exercised against a different schedule.
    """
    if name == "linear":
        return (lambda t: 1.0 - t, lambda t: -jnp.ones_like(t))
    if name == "cosine":
        return (
            lambda t: jnp.cos(0.5 * jnp.pi * t),
            lambda t: -0.5 * jnp.pi * jnp.sin(0.5 * jnp.pi * t),
        )
    raise ValueError(f"unknown mask schedule {name!r}")


def absorbing_loss_weight(t, alpha, alpha_dt, eps: float = 1e-4):
    """BDLM / MDLM per-masked-token time weight ``w(t) = -alpha'_t / (1 - alpha_t)``.

    Linear schedule -> ``1/t`` (mask rate ``1 - alpha_t = t``). The denominator is
    the mask rate; the ``eps`` floor mirrors :func:`factorized_jump_rates` and the
    schedule guards above, keeping the weight finite as ``t -> 0``.
    """
    return -alpha_dt(t) / jnp.maximum(1.0 - alpha(t), eps)


def sample_t_in_bandwidth(
    key, shape, alpha_min: float = 0.15, alpha_max: float = 0.95
):
    """Sample ``t ~ U[alpha_min, alpha_max]`` (mask-ratio bandwidth, §5.1).

    Standard discrete diffusion samples ``t ~ U[0, 1]``; clipping to a band avoids
    the trivial near-zero / near-full masking regimes (and keeps ``1/t`` finite).
    For the linear schedule the mask rate equals ``t``, so ``[alpha_min, alpha_max]``
    is directly the mask-rate band.
    """
    return jax.random.uniform(key, shape, minval=alpha_min, maxval=alpha_max)


def sample_absorbing_path(key, x0, t, num_categories: int, alpha=None):
    """Absorbing forward path: token -> ``[MASK]=num_categories`` w.p. ``1-alpha(t)``.

    ``x0`` integer tokens ``(B, d)``, ``t`` shape ``(B, 1)``. Returns ``(x_t,
    masked)`` where ``masked`` is the boolean mask of corrupted positions.
    Absorbing twin of :func:`sample_discrete_conditional_path`.
    """
    if alpha is None:
        alpha, _ = mask_schedule("linear")
    masked = jax.random.bernoulli(key, 1.0 - alpha(t), x0.shape)
    x_t = jnp.where(masked, num_categories, x0)
    return x_t, masked


def complementary_absorbing_pair(key, x0, t, num_categories: int, alpha=None):
    """Complementary masking (§5.1): a random mask and its logical inverse.

    Returns ``(x_t, x_t_comp, masked, masked_comp)`` with ``masked_comp = ~masked``
    so every position is presented uncorrupted in exactly one member of the pair.
    ``x_t`` masks at rate ``1-alpha(t)``; the complement masks at the inverse rate.
    """
    if alpha is None:
        alpha, _ = mask_schedule("linear")
    masked = jax.random.bernoulli(key, 1.0 - alpha(t), x0.shape)
    masked_comp = jnp.logical_not(masked)
    x_t = jnp.where(masked, num_categories, x0)
    x_t_comp = jnp.where(masked_comp, num_categories, x0)
    return x_t, x_t_comp, masked, masked_comp


def block_diffusion_attention_mask(
    prefix_len: int,
    block_size,
    n_response: int,
    *,
    doc_ids=None,
    include_clean_copy: bool = True,
):
    """Block-diffusion attention mask (eq 3) over ``[c ; x_t ; x_0]`` (+ eq-4 doc mask).

    Layout: ``prefix_len`` clean condition tokens ``c`` (block −1, always visible),
    then ``n_response`` noisy tokens ``x_t``, then (training only) ``n_response``
    clean tokens ``x_0``. ``block_size`` may be a traced scalar: the block index is
    ``arange(n_response)//block_size`` and only the mask *contents* depend on it,
    never the (static) shape ``L``.

    Training (``include_clean_copy=True``, eq 3):
      * ``x_t`` block ``b`` -> ``x_t`` same block (M_BD), strictly earlier ``x_0``
        blocks (M_OBC), and all of ``c``.
      * ``x_0`` block ``b`` -> ``x_0`` blocks ``<= b`` (M_BC) and all of ``c``.
      * ``c`` -> ``c`` only; ``x_0`` never attends to ``x_t`` (clean->noisy = 0).
    Inference (``include_clean_copy=False``): drop the ``x_0`` copy; the response is
    block-causal (block ``b`` attends to blocks ``<= b`` + ``c``) for block-by-block
    decoding. Returns a ``(L, L)`` boolean array, ``True`` == attend.
    """
    resp_blk = jnp.arange(n_response) // block_size
    prefix_blk = jnp.full(prefix_len, -1, dtype=resp_blk.dtype)
    if include_clean_copy:
        region = jnp.concatenate(
            [
                jnp.zeros(prefix_len, dtype=jnp.int32),
                jnp.ones(n_response, dtype=jnp.int32),
                jnp.full(n_response, 2, dtype=jnp.int32),
            ]
        )
        blk = jnp.concatenate([prefix_blk, resp_blk, resp_blk])
    else:
        region = jnp.concatenate(
            [
                jnp.zeros(prefix_len, dtype=jnp.int32),
                jnp.ones(n_response, dtype=jnp.int32),
            ]
        )
        blk = jnp.concatenate([prefix_blk, resp_blk])

    rq, rk = region[:, None], region[None, :]
    bq, bk = blk[:, None], blk[None, :]
    is_prefix_key = rk == 0

    if include_clean_copy:
        q_prefix = (rq == 0) & is_prefix_key
        q_noisy = (rq == 1) & (
            is_prefix_key
            | ((rk == 1) & (bk == bq))  # M_BD: same noisy block
            | ((rk == 2) & (bk < bq))  # M_OBC: strictly earlier clean block
        )
        q_clean = (rq == 2) & (
            is_prefix_key | ((rk == 2) & (bk <= bq))  # M_BC: own + preceding clean
        )
        mask = q_prefix | q_noisy | q_clean
    else:
        q_prefix = (rq == 0) & is_prefix_key
        q_resp = (rq == 1) & (
            is_prefix_key | ((rk == 1) & (bk <= bq))  # block-causal response
        )
        mask = q_prefix | q_resp

    if doc_ids is not None:
        mask = mask & (doc_ids[:, None] == doc_ids[None, :])
    return mask
