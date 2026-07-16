"""Jafar spatiotemporal transformer and vector quantizer.

Adapted from ``FLAIROx/jafar`` at commit
``5ff9fc7d5d744c8c2797ba3ad0a095ed7f2e2665``, path ``utils/nn.py``.
Integration changes: package-local typing and public names; model equations,
initializers, normalization, attention order, and stop-gradient boundaries are
preserved.
"""

import math

from flax import linen as nn
import jax
import jax.numpy as jnp


class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000

    def setup(self) -> None:
        encoding = jnp.zeros((self.max_len, self.d_model))
        position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model)
        )
        encoding = encoding.at[:, 0::2].set(jnp.sin(position * div_term))
        encoding = encoding.at[:, 1::2].set(jnp.cos(position * div_term))
        self.encoding = encoding

    def __call__(self, values: jax.Array) -> jax.Array:
        return values + self.encoding[: values.shape[2]]


class STBlock(nn.Module):
    dim: int
    num_heads: int
    dropout: float

    @nn.remat
    @nn.compact
    def __call__(self, values: jax.Array) -> jax.Array:
        residual = PositionalEncoding(self.dim)(values)
        residual = nn.LayerNorm()(residual)
        residual = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
        )(residual)
        values = values + residual

        values = values.swapaxes(1, 2)
        residual = PositionalEncoding(self.dim)(values)
        residual = nn.LayerNorm()(residual)
        causal_mask = jnp.tri(residual.shape[-2])
        residual = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
        )(residual, mask=causal_mask)
        values = values + residual
        values = values.swapaxes(1, 2)

        residual = nn.LayerNorm()(values)
        residual = nn.Dense(self.dim)(residual)
        residual = nn.gelu(residual)
        return values + residual


class STTransformer(nn.Module):
    model_dim: int
    out_dim: int
    num_blocks: int
    num_heads: int
    dropout: float

    @nn.compact
    def __call__(self, values: jax.Array) -> jax.Array:
        values = nn.Sequential(
            [
                nn.LayerNorm(),
                nn.Dense(self.model_dim),
                nn.LayerNorm(),
            ]
        )(values)
        for _ in range(self.num_blocks):
            values = STBlock(
                dim=self.model_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(values)
        return nn.Dense(self.out_dim)(values)


def normalize(values: jax.Array) -> jax.Array:
    return values / (jnp.linalg.norm(values, ord=2, axis=-1, keepdims=True) + 1e-8)


class VectorQuantizer(nn.Module):
    latent_dim: int
    num_latents: int
    dropout: float

    def setup(self) -> None:
        self.codebook = normalize(
            self.param(
                "codebook",
                nn.initializers.lecun_uniform(),
                (self.num_latents, self.latent_dim),
            )
        )
        self.drop = nn.Dropout(self.dropout, deterministic=False)

    def __call__(
        self,
        values: jax.Array,
        training: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        embeddings = normalize(values)
        codebook = normalize(self.codebook)
        distance = -jnp.matmul(embeddings, codebook.T)
        if training:
            distance = self.drop(distance)

        indices = jnp.argmin(distance, axis=-1)
        codes = self.codebook[indices]
        quantized = embeddings + jax.lax.stop_gradient(codes - embeddings)
        return quantized, codes, embeddings, indices

    def get_codes(self, indices: jax.Array) -> jax.Array:
        return self.codebook[indices]
