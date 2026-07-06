"""LLaDA2.0 absorbing (masked) discrete diffusion (arXiv 2512.15745).

Convention note: the continuous / CTMC flow in :mod:`flow_matching.paths` uses
``alpha(t)=t`` as the *keep* probability with t=1 == clean. The masked-diffusion
schedule here also treats ``alpha`` as the keep probability, but with the LLaDA
time direction t=0 == clean, t=1 == fully masked (linear: ``alpha_t = 1 - t``).
A token is replaced by the absorbing ``[MASK]`` token w.p. the mask rate
``1 - alpha_t``, and the BDLM / SFT loss weights each masked-token cross-entropy
by ``w(t) = -alpha'_t / (1 - alpha_t)`` (eqs 1, 5). ``[MASK]`` has id
``num_categories`` so the vocabulary is ``num_categories + 1``.
"""

import jax
import jax.numpy as jnp


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
    the mask rate; the ``eps`` floor mirrors
    :func:`flow_matching.paths.factorized_jump_rates` and the schedule guards
    there, keeping the weight finite as ``t -> 0``.
    """
    return -alpha_dt(t) / jnp.maximum(1.0 - alpha(t), eps)


def sample_t_in_bandwidth(key, shape, alpha_min: float = 0.15, alpha_max: float = 0.95):
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
    Absorbing twin of :func:`flow_matching.paths.sample_discrete_conditional_path`.
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
