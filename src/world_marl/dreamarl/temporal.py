"""Bounded causal Transformer with equivalent sequence and KV-cache paths."""

from __future__ import annotations

from typing import NamedTuple

from flax import linen as nn
import jax
import jax.numpy as jnp


class TemporalCache(NamedTuple):
    keys: jax.Array
    values: jax.Array
    valid: jax.Array
    position: jax.Array


def episode_positions(is_first: jax.Array) -> jax.Array:
    """Return zero-based positions that restart at each environment reset."""

    def advance(previous, reset):
        current = jnp.where(reset, 0, previous + 1)
        return current, current

    initial = -jnp.ones((is_first.shape[1],), jnp.int32)
    _, positions = jax.lax.scan(advance, initial, is_first)
    return positions


def segment_causal_mask(is_first: jax.Array) -> jax.Array:
    """Create ``[batch, 1, query, key]`` reset-aware causal attention masks."""

    segments = jnp.cumsum(is_first.astype(jnp.int32), axis=0)
    same_segment = segments[:, None, :] == segments[None, :, :]
    causal = jnp.arange(is_first.shape[0])[:, None] >= jnp.arange(
        is_first.shape[0]
    )[None, :]
    return jnp.transpose(same_segment & causal[..., None], (2, 0, 1))[:, None]


class TemporalLayer(nn.Module):
    model_dim: int
    num_heads: int
    mlp_ratio: int

    def setup(self) -> None:
        self.attention_norm = nn.RMSNorm(name="attention_norm")
        self.qkv = nn.Dense(
            3 * self.model_dim,
            precision=jax.lax.Precision.HIGHEST,
            name="qkv",
        )
        self.attention_out = nn.Dense(
            self.model_dim,
            precision=jax.lax.Precision.HIGHEST,
            name="attention_out",
        )
        self.ffn_norm = nn.RMSNorm(name="ffn_norm")
        self.ffn_in = nn.Dense(
            2 * self.mlp_ratio * self.model_dim,
            precision=jax.lax.Precision.HIGHEST,
            name="ffn_in",
        )
        self.ffn_out = nn.Dense(
            self.model_dim,
            precision=jax.lax.Precision.HIGHEST,
            name="ffn_out",
        )

    def __call__(self, inputs: jax.Array, mask: jax.Array) -> jax.Array:
        batch, length, _ = inputs.shape
        head_dim = self.model_dim // self.num_heads
        residual = inputs
        qkv = self.qkv(self.attention_norm(inputs))
        query, key, value = [
            item.reshape((batch, length, self.num_heads, head_dim))
            for item in jnp.split(qkv, 3, axis=-1)
        ]
        logits = jnp.einsum(
            "bqhd,bkhd->bhqk",
            query,
            key,
            precision=jax.lax.Precision.HIGHEST,
        )
        logits = logits.astype(jnp.float32) / jnp.sqrt(float(head_dim))
        logits = jnp.where(mask, logits, -1e30)
        weights = jax.nn.softmax(logits, axis=-1).astype(inputs.dtype)
        attended = jnp.einsum(
            "bhqk,bkhd->bqhd",
            weights,
            value,
            precision=jax.lax.Precision.HIGHEST,
        )
        attended = attended.reshape((batch, length, self.model_dim))
        x = residual + self.attention_out(attended)
        residual = x
        value, gate = jnp.split(self.ffn_in(self.ffn_norm(x)), 2, axis=-1)
        return residual + self.ffn_out(value * nn.silu(gate))

    def step(
        self,
        inputs: jax.Array,
        cached_keys: jax.Array,
        cached_values: jax.Array,
        valid: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        batch = inputs.shape[0]
        head_dim = self.model_dim // self.num_heads
        residual = inputs
        qkv = self.qkv(self.attention_norm(inputs))
        query, key, value = [
            item.reshape((batch, self.num_heads, head_dim))
            for item in jnp.split(qkv, 3, axis=-1)
        ]
        keys = jnp.concatenate([cached_keys[:, 1:], key[:, None]], axis=1)
        values = jnp.concatenate([cached_values[:, 1:], value[:, None]], axis=1)
        logits = jnp.einsum(
            "bhd,bkhd->bhk",
            query,
            keys,
            precision=jax.lax.Precision.HIGHEST,
        )
        logits = logits.astype(jnp.float32) / jnp.sqrt(float(head_dim))
        logits = jnp.where(valid[:, None], logits, -1e30)
        weights = jax.nn.softmax(logits, axis=-1).astype(inputs.dtype)
        attended = jnp.einsum(
            "bhk,bkhd->bhd",
            weights,
            values,
            precision=jax.lax.Precision.HIGHEST,
        )
        x = residual + self.attention_out(attended.reshape((batch, self.model_dim)))
        residual = x
        value_ff, gate = jnp.split(
            self.ffn_in(self.ffn_norm(x)), 2, axis=-1
        )
        return residual + self.ffn_out(value_ff * nn.silu(gate)), keys, values


class CausalKVTransformer(nn.Module):
    """Shared local temporal model for replay, collection, and imagination."""

    pair_dim: int
    model_dim: int
    num_layers: int
    num_heads: int
    mlp_ratio: int
    context_length: int

    def setup(self) -> None:
        self.pair_projection = nn.Dense(
            self.model_dim,
            precision=jax.lax.Precision.HIGHEST,
            name="pair_projection",
        )
        self.start_token = self.param(
            "start_token", nn.initializers.normal(0.02), (self.model_dim,)
        )
        self.position_embedding = self.param(
            "position_embedding",
            nn.initializers.normal(0.02),
            (self.context_length, self.model_dim),
        )
        self.layers = tuple(
            TemporalLayer(
                self.model_dim,
                self.num_heads,
                self.mlp_ratio,
                name=f"layer_{index}",
            )
            for index in range(self.num_layers)
        )
        self.output_norm = nn.RMSNorm(name="output_norm")

    def initial(self, batch_size: int) -> TemporalCache:
        head_dim = self.model_dim // self.num_heads
        return TemporalCache(
            keys=jnp.zeros(
                (
                    batch_size,
                    self.num_layers,
                    self.context_length,
                    self.num_heads,
                    head_dim,
                ),
                jnp.float32,
            ),
            values=jnp.zeros(
                (
                    batch_size,
                    self.num_layers,
                    self.context_length,
                    self.num_heads,
                    head_dim,
                ),
                jnp.float32,
            ),
            valid=jnp.zeros((batch_size, self.context_length), bool),
            position=-jnp.ones((batch_size,), jnp.int32),
        )

    def __call__(self, previous_pairs: jax.Array, is_first: jax.Array) -> jax.Array:
        """Compute strict-history states for ``[time, batch, pair]`` inputs."""

        if previous_pairs.ndim != 3 or previous_pairs.shape[-1] != self.pair_dim:
            raise ValueError("previous_pairs must have shape [time,batch,pair_dim]")
        if is_first.shape != previous_pairs.shape[:2]:
            raise ValueError("is_first must match [time,batch]")
        if previous_pairs.shape[0] > self.context_length:
            raise ValueError("sequence exceeds bounded temporal context")
        projected = self.pair_projection(previous_pairs)
        tokens = jnp.where(is_first[..., None], self.start_token, projected)
        positions = episode_positions(is_first)
        x = tokens + self.position_embedding[positions % self.context_length]
        x = jnp.swapaxes(x, 0, 1)
        mask = segment_causal_mask(is_first)
        for layer in self.layers:
            x = layer(x, mask)
        return jnp.swapaxes(self.output_norm(x), 0, 1)

    def step(
        self,
        cache: TemporalCache,
        previous_pair: jax.Array,
        is_first: jax.Array,
    ) -> tuple[TemporalCache, jax.Array]:
        """Advance one state using a bounded per-layer KV cache."""

        if previous_pair.shape != (previous_pair.shape[0], self.pair_dim):
            raise ValueError("previous_pair must have shape [batch,pair_dim]")
        if is_first.shape != previous_pair.shape[:1]:
            raise ValueError("is_first must have shape [batch]")
        keep = ~is_first
        keys = jnp.where(keep[:, None, None, None, None], cache.keys, 0)
        values = jnp.where(keep[:, None, None, None, None], cache.values, 0)
        valid = jnp.where(keep[:, None], cache.valid, False)
        position = jnp.where(is_first, 0, cache.position + 1)
        valid = jnp.concatenate(
            [valid[:, 1:], jnp.ones((valid.shape[0], 1), bool)], axis=1
        )

        projected = self.pair_projection(previous_pair)
        x = jnp.where(is_first[:, None], self.start_token, projected)
        x = x + self.position_embedding[position % self.context_length]
        next_keys = []
        next_values = []
        for index, layer in enumerate(self.layers):
            x, layer_keys, layer_values = layer.step(
                x, keys[:, index], values[:, index], valid
            )
            next_keys.append(layer_keys)
            next_values.append(layer_values)
        return (
            TemporalCache(
                keys=jnp.stack(next_keys, axis=1),
                values=jnp.stack(next_values, axis=1),
                valid=valid,
                position=position,
            ),
            self.output_norm(x),
        )
