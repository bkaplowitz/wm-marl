"""Block-by-block sampling for the LLaDA2.0 arm (arXiv 2512.15745, §5.4)."""

from typing import Any

import jax
import jax.numpy as jnp


def sample_llada2_block_diffusion(
    apply_fn: Any,
    params: Any,
    key: jax.Array,
    prev_tokens: jax.Array,
    action_ids: jax.Array,
    *,
    num_factors: int,
    num_categories: int,
    block_size: int,
    steps_per_block: int,
    confidence_threshold: float,
) -> jax.Array:
    """LLaDA2.0 block-by-block hybrid-confidence sampler (§5.4).

    Absorbing block-wise twin of
    :func:`flow_matching.simulate.sample_marginal_discrete_flow_model`. Start
    from an all-``[MASK]`` response and decode one block at a time, conditioned on
    the clean prefix (prev-state + action tokens) and previously committed clean
    blocks via the block-causal inference mask (``include_clean_copy=False``). Within
    a block, ``steps_per_block`` refinement passes commit positions whose certainty
    (max softmax prob) clears ``confidence_threshold``, with a per-row top-k fallback
    so progress is guaranteed even when nothing clears the threshold (§5.4 hybrid
    acceptance); a final force-commit fills any block stragglers. Predictions are
    drawn stochastically (categorical) while certainty drives the commit decision —
    committed positions are high-certainty, where the draw ≈ argmax. Every buffer
    keeps a fixed ``(B, num_factors)`` shape: only the commit *contents* are
    data-dependent (boolean ops + ``jnp.where``), never shapes, so the whole routine
    jits/scans cleanly. ``block_size`` is a static inference setting (the WSD
    curriculum only varies it during training). Returns ``(B, num_factors)`` clean
    integer tokens.
    """
    mask_token = num_categories
    batch = prev_tokens.shape[0]
    n_blocks = (num_factors + block_size - 1) // block_size
    quota = max(1, -(-block_size // steps_per_block))  # ceil: per-step commit budget
    topk = min(quota, num_factors)
    block_of = jnp.arange(num_factors) // block_size  # (d,)

    def forward(tokens: jax.Array, draw_key: jax.Array):
        logits, _ = apply_fn(
            {"params": params}, tokens, prev_tokens, action_ids, block_size=block_size
        )  # (B, d, V)
        conf = jnp.max(jax.nn.softmax(logits, axis=-1), axis=-1)  # certainty (B, d)
        pred = jax.random.categorical(draw_key, logits, axis=-1).astype(tokens.dtype)
        return pred, conf

    def refine_step(carry, _):
        tokens, committed, in_block, rng = carry
        rng, draw_key = jax.random.split(rng)
        pred, conf = forward(tokens, draw_key)
        candidate = in_block & ~committed  # (B, d)
        masked_conf = jnp.where(candidate, conf, -jnp.inf)
        kth = jax.lax.top_k(masked_conf, topk)[0][:, -1:]  # per-row fallback threshold
        accept = candidate & ((conf > confidence_threshold) | (masked_conf >= kth))
        tokens = jnp.where(accept, pred, tokens)
        return (tokens, committed | accept, in_block, rng), None

    def block_step(carry, b):
        tokens, committed, rng = carry
        in_block = jnp.broadcast_to((block_of == b)[None, :], tokens.shape)
        rng, inner_rng, force_key = jax.random.split(rng, 3)
        (tokens, committed, _, _), _ = jax.lax.scan(
            refine_step,
            (tokens, committed, in_block, inner_rng),
            xs=None,
            length=steps_per_block,
        )
        pred, _ = forward(tokens, force_key)  # force-commit block stragglers
        accept = in_block & ~committed
        tokens = jnp.where(accept, pred, tokens)
        return (tokens, committed | accept, rng), None

    tokens = jnp.full((batch, num_factors), mask_token, dtype=jnp.int32)
    committed = jnp.zeros((batch, num_factors), dtype=bool)
    (tokens, _, _), _ = jax.lax.scan(
        block_step, (tokens, committed, key), jnp.arange(n_blocks)
    )
    return tokens
