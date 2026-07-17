"""Jasmine axial transformer and attention primitives.

Adapted from ``p-doom/jasmine`` at commit
``420859bc99eecf6b07a7e9edf65d5d145935f1e1``, path
``jasmine/utils/nn.py``. Integration changes: NNX modules and state are
translated to Flax Linen modules and parameter collections; equations,
initializers, rematerialization, float32 normalization/parameters, bf16 dense
and attention compute, and cuDNN dot-product attention behavior are preserved.
"""

from collections.abc import Callable
import math

from flax import linen as nn
import jax
import jax.numpy as jnp


def _spatiotemporal_position_encoding(
    d_model: int,
    max_len: int = 5000,
) -> Callable[[jax.Array], jax.Array]:
    encoding = jnp.zeros((max_len, d_model))
    position = jnp.arange(0, max_len, dtype=jnp.float32)[:, None]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    encoding = encoding.at[:, 0::2].set(jnp.sin(position * div_term))
    encoding = encoding.at[:, 1::2].set(jnp.cos(position * div_term))

    def encode(values: jax.Array) -> jax.Array:
        if values.ndim != 4:
            raise ValueError(f"input must be four-dimensional, got {values.shape}")
        time_steps = values.shape[1]
        spatial_patches = values.shape[2]
        values = values + encoding[None, :time_steps, None, :]
        return values + encoding[None, None, :spatial_patches, :]

    return encode


def _create_flash_attention_fn(
    use_flash_attention: bool,
    is_causal: bool,
) -> Callable[..., jax.Array]:
    def attention_fn(
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        bias: jax.Array | None = None,
        mask: jax.Array | None = None,
        **_: object,
    ) -> jax.Array:
        del mask
        implementation = "cudnn" if use_flash_attention else None

        def merge_batch_dims(values: jax.Array) -> jax.Array:
            return values.reshape((-1,) + values.shape[-3:])

        def pad_length(values: jax.Array, pad_size: int) -> jax.Array:
            return jnp.pad(values, ((0, 0), (0, pad_size), (0, 0), (0, 0)))

        original_shape = query.shape
        query_length = query.shape[-3]
        key_length = key.shape[-3]
        padded_query_length = ((query_length + 3) // 4) * 4
        padded_key_length = ((key_length + 3) // 4) * 4
        query_padding = padded_query_length - query_length
        key_padding = padded_key_length - key_length

        padded_query = pad_length(merge_batch_dims(query), query_padding)
        padded_key = pad_length(merge_batch_dims(key), key_padding)
        padded_value = pad_length(merge_batch_dims(value), key_padding)

        attention_mask = jnp.ones(
            (padded_query_length, padded_key_length),
            dtype=jnp.bool_,
        )
        attention_mask = attention_mask.at[query_length:, :].set(False)
        attention_mask = attention_mask.at[:, key_length:].set(False)
        attention_mask = attention_mask[None, None, :, :]

        padded_bias = (
            jnp.pad(
                merge_batch_dims(bias),
                ((0, 0), (0, 0), (0, query_padding), (0, key_padding)),
            )
            if bias is not None
            else None
        )
        output = jax.nn.dot_product_attention(
            query=padded_query,
            key=padded_key,
            value=padded_value,
            bias=padded_bias,
            mask=attention_mask,
            implementation=implementation,
            is_causal=is_causal,
        )
        return output[..., :query_length, :, :].reshape(original_shape)

    return attention_fn


class AxialBlock(nn.Module):
    dim: int
    ffn_dim: int
    num_heads: int
    dropout: float
    param_dtype: jnp.dtype
    dtype: jnp.dtype
    use_flash_attention: bool
    spatial_causal: bool
    temporal_causal: bool

    @nn.remat
    @nn.compact
    def __call__(self, values: jax.Array) -> jax.Array:
        residual = nn.LayerNorm(
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
            name="spatial_norm",
        )(values)
        residual = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            attention_fn=_create_flash_attention_fn(
                self.use_flash_attention,
                is_causal=self.spatial_causal,
            ),
            deterministic=True,
            name="spatial_attention",
        )(residual)
        values = values + residual

        values = values.swapaxes(1, 2)
        residual = nn.LayerNorm(
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
            name="temporal_norm",
        )(values)
        residual = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            attention_fn=_create_flash_attention_fn(
                self.use_flash_attention,
                is_causal=self.temporal_causal,
            ),
            deterministic=True,
            name="temporal_attention",
        )(residual)
        values = values + residual
        values = values.swapaxes(1, 2)

        residual = nn.LayerNorm(
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
            name="ffn_norm",
        )(values)
        residual = nn.Dense(
            self.ffn_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            name="ffn_dense1",
        )(residual)
        residual = jax.nn.gelu(residual)
        residual = nn.Dense(
            self.dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            name="ffn_dense2",
        )(residual)
        return values + residual


class AxialTransformer(nn.Module):
    input_dim: int
    model_dim: int
    ffn_dim: int
    out_dim: int
    num_blocks: int
    num_heads: int
    dropout: float
    param_dtype: jnp.dtype
    dtype: jnp.dtype
    use_flash_attention: bool
    spatial_causal: bool
    temporal_causal: bool
    max_len: int = 5000

    @nn.compact
    def __call__(self, values: jax.Array) -> jax.Array:
        values = nn.LayerNorm(
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
            name="input_norm1",
        )(values)
        values = nn.Dense(
            self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            name="input_dense",
        )(values)
        values = nn.LayerNorm(
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
            name="input_norm2",
        )(values)
        values = _spatiotemporal_position_encoding(
            self.model_dim,
            max_len=self.max_len,
        )(values)
        for index in range(self.num_blocks):
            values = AxialBlock(
                dim=self.model_dim,
                ffn_dim=self.ffn_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                param_dtype=self.param_dtype,
                dtype=self.dtype,
                use_flash_attention=self.use_flash_attention,
                spatial_causal=self.spatial_causal,
                temporal_causal=self.temporal_causal,
                name=f"blocks_{index}",
            )(values)
        return nn.Dense(
            self.out_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            name="output_dense",
        )(values)


def normalize(values: jax.Array) -> jax.Array:
    return values / (jnp.linalg.norm(values, ord=2, axis=-1, keepdims=True) + 1e-8)


class VectorQuantizer(nn.Module):
    latent_dim: int
    num_latents: int
    dropout: float
    dtype: jnp.dtype

    def setup(self) -> None:
        def init_codebook(rng: jax.Array, shape: tuple[int, int]) -> jax.Array:
            return normalize(nn.initializers.normal(stddev=1)(rng, shape))

        self.codebook = self.param(
            "codebook",
            init_codebook,
            (self.num_latents, self.latent_dim),
        )
        self.drop = nn.Dropout(self.dropout)

    def __call__(
        self,
        values: jax.Array,
        training: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        embeddings = normalize(values.astype(self.dtype))
        codebook = normalize(self.codebook.astype(self.dtype))
        distance = -jnp.matmul(embeddings, codebook.T)
        if training:
            distance = self.drop(distance, deterministic=False)

        indices = jnp.argmin(distance, axis=-1)
        codes = codebook[indices]
        quantized = embeddings + jax.lax.stop_gradient(codes - embeddings)
        return quantized, codes, embeddings, indices

    def get_codes(self, indices: jax.Array) -> jax.Array:
        return self.codebook[indices]
