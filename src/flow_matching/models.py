"""Minimal Flax models for vector-field learning."""

import math
from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class MLPVectorField(nn.Module):
    """A small MLP that predicts flow u_theta(x, t). Uses SiLU activation. Represents both conditional and unconditional vector fields.

    Conditional vector field: u_theta(x, t, cond_vars)
    Unconditional vector field: u_theta(x, t)
    """

    hidden_dims: Sequence[int] = (64, 64, 64, 64)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        t: jax.Array,
        cond_vars: jax.Array | None = None,
    ) -> jax.Array:
        """Evaluate the vector field at batched positions and times.

        If cond_vars=None, unconditional vector field, else conditional.

        When ``cond_vars`` is provided it is concatenated into the trunk input,
        but the output head stays sized to ``x`` so the predicted field matches
        the target dimensionality (no conditioning leakage into the output).
        """
        parts = (x, t) if cond_vars is None else (x, t, cond_vars)
        xt = jnp.concat(parts, axis=-1)

        hidden_layers = [
            layer for dim in self.hidden_dims for layer in (nn.Dense(dim), nn.silu)
        ]
        final_layer = [nn.Dense(x.shape[-1])]
        return nn.Sequential(hidden_layers + final_layer)(xt)


class TokenizedDiscreteDenoiser(nn.Module):
    """Posterior network f_theta for discrete flow matching (discrete.md Alg 8).

    Maps integer tokens ``(B, d)`` to per-factor logits ``(B, d, V)`` via a token
    embedding rather than a one-hot input, so the input embedding and the d*V
    classification head are sized independently (unlike :class:`MLPVectorField`,
    whose head is tied to its input width).
    """

    num_categories: int
    embed_dim: int = 16
    hidden_dims: Sequence[int] = (64, 64, 64, 64)

    @nn.compact
    def __call__(
        self,
        tokens: jax.Array,
        t: jax.Array,
        cond_vars: jax.Array | None = None,
    ) -> jax.Array:
        num_factors = tokens.shape[-1]
        emb = nn.Embed(self.num_categories, self.embed_dim)(tokens)
        emb_flat = emb.reshape((tokens.shape[0], num_factors * self.embed_dim))

        parts = (emb_flat, t) if cond_vars is None else (emb_flat, t, cond_vars)
        xt = jnp.concat(parts, axis=-1)

        hidden_layers = [
            layer for dim in self.hidden_dims for layer in (nn.Dense(dim), nn.silu)
        ]
        head = [nn.Dense(num_factors * self.num_categories)]
        logits = nn.Sequential(hidden_layers + head)(xt)
        return logits.reshape((tokens.shape[0], num_factors, self.num_categories))


def sinusoidal_time_embedding(t: jax.Array, dim: int) -> jax.Array:
    half = dim // 2
    freqs = jnp.exp(-math.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
    angles = t * freqs
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    if emb.shape[-1] < dim:
        emb = jnp.pad(emb, ((0, 0), (0, dim - emb.shape[-1])))
    return emb


class TokenizedDiscreteTransformer(nn.Module):
    """Transformer posterior network f_theta for discrete flow matching (Alg 8).

    Treats the d factors as a length-d sequence with bidirectional self-attention
    and a prepended conditioning token carrying (sinusoidal-t, cond_vars). Honors
    the same ``(tokens, t, cond_vars) -> (B, d, V)`` contract as
    :class:`TokenizedDiscreteDenoiser`; dropout-free so the apply signature needs
    no rng.
    """

    num_categories: int
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    mlp_ratio: int = 4

    @nn.compact
    def __call__(
        self,
        tokens: jax.Array,
        t: jax.Array,
        cond_vars: jax.Array | None = None,
    ) -> jax.Array:
        num_factors = tokens.shape[-1]
        h = nn.Embed(self.num_categories, self.model_dim)(tokens)
        pos = self.param(
            "pos_emb", nn.initializers.normal(0.02), (num_factors, self.model_dim)
        )
        h = h + pos

        c = nn.Dense(self.model_dim)(sinusoidal_time_embedding(t, self.model_dim))
        if cond_vars is not None:
            c = c + nn.Dense(self.model_dim)(cond_vars)
        h = jnp.concatenate([c[:, None, :], h], axis=1)

        for _ in range(self.num_layers):
            x = nn.LayerNorm()(h)
            h = h + nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(
                x, x, deterministic=True
            )
            y = nn.LayerNorm()(h)
            y = nn.Dense(self.mlp_ratio * self.model_dim)(y)
            y = nn.gelu(y)
            h = h + nn.Dense(self.model_dim)(y)

        h = nn.LayerNorm()(h)[:, 1:, :]
        return nn.Dense(self.num_categories)(h)
