"""LLaDA2.0 block-diffusion backbone (arXiv 2512.15745): RoPE (+YaRN), RMSNorm,
SwiGLU MoE, time-agnostic. See :mod:`flow_matching.llada2.paths` for the eq-3
block-diffusion attention mask, :mod:`flow_matching.llada2.train` for the BDLM
loss and WSD curriculum, and :mod:`flow_matching.llada2.simulate` for the
block-by-block sampler.
"""

import math

import flax.linen as nn
import jax
import jax.numpy as jnp

from flow_matching.llada2.paths import block_diffusion_attention_mask


def rotate_half(x: jax.Array) -> jax.Array:
    """Rotate the two halves of the last axis: ``[x1, x2] -> [-x2, x1]`` (RoPE)."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def _rope_inv_freq(
    head_dim: int, base: float, scaling: float, original_max_position: int
) -> jax.Array:
    """Per-frequency inverse frequencies, with optional YaRN (NTK-by-parts) scaling.

    ``scaling <= 1`` returns plain RoPE frequencies. ``scaling > 1`` activates YaRN:
    high-frequency dims extrapolate (unchanged), low-frequency dims interpolate
    (frequency divided by ``scaling``), with a smooth ramp between, enabling
    long-context extrapolation beyond ``original_max_position``.
    """
    half = head_dim // 2
    idx = jnp.arange(half, dtype=jnp.float32)
    inv_freq = base ** (-2.0 * idx / head_dim)
    if scaling is None or scaling <= 1.0:
        return inv_freq
    beta_fast, beta_slow = 32.0, 1.0

    def correction_dim(num_rotations: float) -> float:
        return (
            head_dim
            * math.log(original_max_position / (num_rotations * 2.0 * math.pi))
            / (2.0 * math.log(base))
        )

    low = min(max(math.floor(correction_dim(beta_fast)), 0), half - 1)
    high = min(max(math.ceil(correction_dim(beta_slow)), 0), half - 1)
    ramp = jnp.clip((idx - low) / max(high - low, 1), 0.0, 1.0)
    extrapolation_mask = 1.0 - ramp  # 1 for high-freq dims (no interpolation)
    return inv_freq * extrapolation_mask + (inv_freq / scaling) * (
        1.0 - extrapolation_mask
    )


def apply_rope(
    x: jax.Array,
    positions: jax.Array,
    base: float = 10000.0,
    scaling: float = 1.0,
    original_max_position: int = 32,
) -> jax.Array:
    """Apply rotary position embedding to ``x`` shaped ``(B, L, H, head_dim)``.

    ``positions`` is ``(L,)``. ``scaling`` is the YaRN dynamic-scaling factor
    (1.0 = plain RoPE). Standalone twin of the rotation done inside
    :class:`RoPEAttention`, exposed for testing.
    """
    head_dim = x.shape[-1]
    inv_freq = _rope_inv_freq(head_dim, base, scaling, original_max_position)
    angles = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]  # (L, half)
    cos = jnp.concatenate([jnp.cos(angles), jnp.cos(angles)], axis=-1)
    sin = jnp.concatenate([jnp.sin(angles), jnp.sin(angles)], axis=-1)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return x * cos + rotate_half(x) * sin


class RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE (+YaRN) and an explicit boolean mask.

    Hand-rolled RoPE attention rather than ``nn.MultiHeadDotProductAttention`` because RoPE must be
    injected between the q/k projection and the dot product. ``mask`` is ``(L, L)``
    boolean (``True`` == attend); the YaRN temperature ``m_scale`` rescales logits
    when ``rope_scaling > 1``.
    """

    num_heads: int
    rope_base: float = 10000.0
    rope_scaling: float = 1.0
    rope_original_max_position: int = 32

    @nn.compact
    def __call__(
        self, x: jax.Array, mask: jax.Array, positions: jax.Array
    ) -> jax.Array:
        batch, length, model_dim = x.shape
        head_dim = model_dim // self.num_heads

        def proj(name: str) -> jax.Array:
            return nn.Dense(model_dim, name=name)(x).reshape(
                batch, length, self.num_heads, head_dim
            )

        q, k, v = proj("query"), proj("key"), proj("value")
        q = apply_rope(
            q,
            positions,
            self.rope_base,
            self.rope_scaling,
            self.rope_original_max_position,
        )
        k = apply_rope(
            k,
            positions,
            self.rope_base,
            self.rope_scaling,
            self.rope_original_max_position,
        )
        m_scale = (
            0.1 * math.log(self.rope_scaling) + 1.0 if self.rope_scaling > 1.0 else 1.0
        )
        scale = m_scale / math.sqrt(head_dim)
        logits = jnp.einsum("blhd,bmhd->bhlm", q, k) * scale
        logits = jnp.where(mask[None, None, :, :], logits, -1e30)
        weights = jax.nn.softmax(logits, axis=-1)
        out = jnp.einsum("bhlm,bmhd->blhd", weights, v).reshape(
            batch, length, model_dim
        )
        return nn.Dense(model_dim, name="out")(out)


class SwiGLUExpert(nn.Module):
    """SwiGLU feed-forward expert: ``down(silu(gate(x)) * up(x))``."""

    ffn_hidden_dim: int
    model_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        gate = nn.Dense(self.ffn_hidden_dim, name="gate")(x)
        up = nn.Dense(self.ffn_hidden_dim, name="up")(x)
        return nn.Dense(self.model_dim, name="down")(nn.silu(gate) * up)


class MoELayer(nn.Module):
    """Top-k token-choice mixture-of-SwiGLU-experts with a load-balancing aux loss.

    Returns ``(output, aux_loss)``. ``num_experts == 1`` degenerates to a single
    dense SwiGLU FFN (aux 0). At small scale all experts are computed densely and
    gated by the renormalized top-k router weights — identical outputs to sparse
    dispatch, simpler and jit-friendly. The Switch-style aux loss is
    ``E * sum_e f_e * P_e`` (``f_e`` = fraction of tokens selecting expert ``e`` in
    their top-k, ``P_e`` = mean router probability).
    """

    num_experts: int
    expert_top_k: int
    ffn_hidden_dim: int
    model_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        if self.num_experts == 1:
            out = SwiGLUExpert(self.ffn_hidden_dim, self.model_dim, name="expert_0")(x)
            return out, jnp.asarray(0.0)

        router_probs = jax.nn.softmax(
            nn.Dense(self.num_experts, name="router")(x), axis=-1
        )  # (B, L, E)
        top_k = min(self.expert_top_k, self.num_experts)
        topk_vals, topk_idx = jax.lax.top_k(router_probs, top_k)  # (B, L, k)
        gates = topk_vals / jnp.maximum(topk_vals.sum(axis=-1, keepdims=True), 1e-9)

        expert_outs = jnp.stack(
            [
                SwiGLUExpert(self.ffn_hidden_dim, self.model_dim, name=f"expert_{e}")(x)
                for e in range(self.num_experts)
            ],
            axis=-2,
        )  # (B, L, E, D)
        selected = jax.nn.one_hot(topk_idx, self.num_experts)  # (B, L, k, E)
        combine = jnp.einsum("blk,blke->ble", gates, selected)  # (B, L, E)
        output = jnp.einsum("ble,bled->bld", combine, expert_outs)

        dispatch = selected.sum(axis=-2)  # (B, L, E) in {0, 1}
        frac = dispatch.mean(axis=(0, 1))  # (E,)
        prob = router_probs.mean(axis=(0, 1))  # (E,)
        aux = self.num_experts * jnp.sum(frac * prob)
        return output, aux


class BlockDiffusionTransformer(nn.Module):
    """LLaDA2.0 block-diffusion backbone (arXiv 2512.15745).

    Time-agnostic: the network conditions only on the (partially-masked) token
    sequence and the clean context — there is **no timestep input** (the
    ``-alpha'/(1-alpha)`` term is a scalar loss reweight, applied in the loss).

    Maps next-state prediction onto the conditional SFT objective (eq 5): the clean
    prefix ``c`` = prev-state tokens + action tokens; the response ``x_0`` =
    next-state tokens (``d`` factors), denoised block by block. The forward
    assembles ``[c ; x_t ; x_0]`` (training) or ``[c ; x_t]`` (inference) and applies
    the eq-3 block-diffusion attention mask so block ``k``'s noisy tokens attend to
    earlier clean blocks + ``c``. Reads logits from the ``x_t`` slots -> ``(B, d, V)``
    over the ``V`` real categories (``[MASK]=V`` is never an output).
    """

    num_categories: int
    num_actions: int
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ffn_hidden_dim: int = 128
    num_experts: int = 1
    expert_top_k: int = 1
    rope_base: float = 10000.0
    rope_scaling: float = 1.0
    rope_original_max_position: int = 32

    @nn.compact
    def __call__(
        self,
        target_tokens: jax.Array,
        prev_tokens: jax.Array,
        action_ids: jax.Array,
        x0_target: jax.Array | None = None,
        *,
        block_size,
        doc_ids: jax.Array | None = None,
        mask_noise_std: float = 0.0,
        noise_rng: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        num_factors = target_tokens.shape[-1]
        token_embed = nn.Embed(
            self.num_categories + 1, self.model_dim, name="token_embed"
        )
        action_embed = nn.Embed(self.num_actions, self.model_dim, name="action_embed")

        prev_emb = token_embed(prev_tokens)  # (B, d, D)
        act_emb = action_embed(action_ids)  # (B, A, D)
        xt_emb = token_embed(target_tokens)  # (B, d, D)
        if noise_rng is not None:
            # §7.1 masked-embedding Gaussian-noise stabilizer (AR-init regime).
            is_mask = (target_tokens == self.num_categories)[..., None]
            noise = mask_noise_std * jax.random.normal(noise_rng, xt_emb.shape)
            xt_emb = xt_emb + jnp.where(is_mask, noise, 0.0)

        prefix = jnp.concatenate([prev_emb, act_emb], axis=1)  # (B, d+A, D)
        prefix_len = num_factors + action_ids.shape[-1]
        training = x0_target is not None
        if training:
            h = jnp.concatenate([prefix, xt_emb, token_embed(x0_target)], axis=1)
        else:
            h = jnp.concatenate([prefix, xt_emb], axis=1)

        positions = jnp.arange(h.shape[1])
        mask = block_diffusion_attention_mask(
            prefix_len,
            block_size,
            num_factors,
            doc_ids=doc_ids,
            include_clean_copy=training,
        )

        aux_total = jnp.asarray(0.0)
        for _ in range(self.num_layers):
            attn = RoPEAttention(
                num_heads=self.num_heads,
                rope_base=self.rope_base,
                rope_scaling=self.rope_scaling,
                rope_original_max_position=self.rope_original_max_position,
            )(nn.RMSNorm()(h), mask, positions)
            h = h + attn
            moe_out, aux = MoELayer(
                self.num_experts, self.expert_top_k, self.ffn_hidden_dim, self.model_dim
            )(nn.RMSNorm()(h))
            h = h + moe_out
            aux_total = aux_total + aux

        h = nn.RMSNorm()(h)
        target_h = h[:, prefix_len : prefix_len + num_factors, :]
        logits = nn.Dense(self.num_categories, name="head")(target_h)
        return logits, aux_total
