"""Spatiotemporal axial transformer used by the Genie2 public-source baseline.

The architecture follows the Apache-2.0 Jasmine implementation's AxialTransformer:
https://github.com/p-doom/jasmine/blob/main/jasmine/utils/nn.py
This is a native Flax Linen implementation for the repository's JAX/Flax versions.
"""

from __future__ import annotations

from functools import partial
import math

from flax import linen as nn
import jax
import jax.numpy as jnp


def dtype_from_name(name: str) -> jnp.dtype:
    try:
        return getattr(jnp, name)
    except AttributeError as error:
        raise ValueError(f"unsupported JAX dtype {name!r}") from error


def spatiotemporal_position_encoding(
    time_steps: int,
    num_patches: int,
    model_dim: int,
    dtype: jnp.dtype,
) -> jax.Array:
    positions = jnp.arange(max(time_steps, num_patches), dtype=jnp.float32)[:, None]
    frequencies = jnp.exp(
        jnp.arange(0, model_dim, 2, dtype=jnp.float32)
        * (-math.log(10_000.0) / model_dim)
    )
    table = jnp.zeros((positions.shape[0], model_dim), dtype=jnp.float32)
    table = table.at[:, 0::2].set(jnp.sin(positions * frequencies))
    table = table.at[:, 1::2].set(jnp.cos(positions * frequencies))
    temporal = table[:time_steps][None, :, None, :]
    spatial = table[:num_patches][None, None, :, :]
    return (temporal + spatial).astype(dtype)


class AxialTransformerBlock(nn.Module):
    model_dim: int
    ffn_dim: int
    num_heads: int
    dropout: float = 0.0
    temporal_causal: bool = True
    spatial_causal: bool = False
    compute_dtype: str = "bfloat16"
    parameter_dtype: str = "float32"

    @partial(nn.remat, static_argnums=(2,))
    @nn.compact
    def __call__(self, inputs: jax.Array, training: bool) -> jax.Array:
        compute_dtype = dtype_from_name(self.compute_dtype)
        parameter_dtype = dtype_from_name(self.parameter_dtype)
        batch_size, time_steps, num_patches, _ = inputs.shape
        x = inputs.astype(compute_dtype)

        spatial = nn.LayerNorm(dtype=jnp.float32, param_dtype=parameter_dtype)(x)
        spatial = spatial.reshape(
            (batch_size * time_steps, num_patches, self.model_dim)
        )
        spatial_mask = None
        if self.spatial_causal:
            spatial_mask = nn.make_causal_mask(
                jnp.ones((batch_size * time_steps, num_patches), dtype=bool)
            )
        spatial = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            dtype=compute_dtype,
            param_dtype=parameter_dtype,
        )(spatial, spatial, mask=spatial_mask, deterministic=not training)
        x = x + spatial.reshape((batch_size, time_steps, num_patches, self.model_dim))

        temporal = x.swapaxes(1, 2)
        temporal = nn.LayerNorm(dtype=jnp.float32, param_dtype=parameter_dtype)(
            temporal
        )
        temporal = temporal.reshape(
            (batch_size * num_patches, time_steps, self.model_dim)
        )
        temporal_mask = None
        if self.temporal_causal:
            temporal_mask = nn.make_causal_mask(
                jnp.ones((batch_size * num_patches, time_steps), dtype=bool)
            )
        temporal = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            dtype=compute_dtype,
            param_dtype=parameter_dtype,
        )(temporal, temporal, mask=temporal_mask, deterministic=not training)
        temporal = temporal.reshape(
            (batch_size, num_patches, time_steps, self.model_dim)
        ).swapaxes(1, 2)
        x = x + temporal

        hidden = nn.LayerNorm(dtype=jnp.float32, param_dtype=parameter_dtype)(x)
        hidden = nn.Dense(
            self.ffn_dim,
            dtype=compute_dtype,
            param_dtype=parameter_dtype,
        )(hidden)
        hidden = nn.gelu(hidden)
        hidden = nn.Dense(
            self.model_dim,
            dtype=compute_dtype,
            param_dtype=parameter_dtype,
        )(hidden)
        hidden = nn.Dropout(self.dropout)(hidden, deterministic=not training)
        return x + hidden


class AxialTransformer(nn.Module):
    input_dim: int
    model_dim: int
    ffn_dim: int
    output_dim: int
    num_blocks: int
    num_heads: int
    dropout: float = 0.0
    temporal_causal: bool = True
    spatial_causal: bool = False
    compute_dtype: str = "bfloat16"
    parameter_dtype: str = "float32"

    @nn.compact
    def __call__(self, inputs: jax.Array, *, training: bool = False) -> jax.Array:
        if inputs.ndim != 4 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected (batch,time,patch,{self.input_dim}), got {inputs.shape}"
            )
        compute_dtype = dtype_from_name(self.compute_dtype)
        parameter_dtype = dtype_from_name(self.parameter_dtype)
        x = nn.LayerNorm(
            dtype=jnp.float32, param_dtype=parameter_dtype, name="input_norm"
        )(inputs)
        x = nn.Dense(
            self.model_dim,
            dtype=compute_dtype,
            param_dtype=parameter_dtype,
            name="input_projection",
        )(x)
        x = nn.LayerNorm(
            dtype=jnp.float32,
            param_dtype=parameter_dtype,
            name="projected_norm",
        )(x)
        x = x + spatiotemporal_position_encoding(
            x.shape[1], x.shape[2], self.model_dim, compute_dtype
        )
        for index in range(self.num_blocks):
            x = AxialTransformerBlock(
                model_dim=self.model_dim,
                ffn_dim=self.ffn_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                temporal_causal=self.temporal_causal,
                spatial_causal=self.spatial_causal,
                compute_dtype=self.compute_dtype,
                parameter_dtype=self.parameter_dtype,
                name=f"block_{index}",
            )(x, training)
        return nn.Dense(
            self.output_dim,
            dtype=compute_dtype,
            param_dtype=parameter_dtype,
            name="output_projection",
        )(x)
