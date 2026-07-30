"""Causal temporal state for the stochastic JEPA Transformer.

The public interface encodes the project's temporal contract directly. Given
pair ``x_t = concat(z_t, a_t)``, the state returned at index ``t`` is allowed
to depend on ``x_<t`` only. Episode starts replace the shifted input with a
learned start token and isolate attention from the preceding episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from flax import linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class TemporalConfig:
    """Shape and capacity of the causal temporal dynamics module."""

    pair_dim: int
    model_dim: int = 512
    state_dim: int = 4096
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    context_length: int = 64

    def __post_init__(self) -> None:
        positive = {
            "pair_dim": self.pair_dim,
            "model_dim": self.model_dim,
            "state_dim": self.state_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "context_length": self.context_length,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")


class TemporalCache(NamedTuple):
    """Fixed-size history used by recurrent imagination and collection."""

    tokens: jax.Array
    valid: jax.Array
    positions: jax.Array
    position: jax.Array


def episode_positions(is_first: jax.Array) -> jax.Array:
    """Return zero-based positions that restart at every episode boundary."""
    if is_first.ndim != 2:
        raise ValueError(f"expected [batch,time] is_first, got {is_first.shape}")

    def advance(previous: jax.Array, reset: jax.Array):
        current = jnp.where(reset, 0, previous + 1)
        return current, current

    initial = -jnp.ones(is_first.shape[0], jnp.int32)
    _, positions = jax.lax.scan(advance, initial, is_first.T)
    return positions.T


def segment_causal_mask(is_first: jax.Array) -> jax.Array:
    """Build a causal mask that cannot cross episode boundaries."""
    if is_first.ndim != 2:
        raise ValueError(f"expected [batch,time] is_first, got {is_first.shape}")
    segments = jnp.cumsum(is_first.astype(jnp.int32), axis=1)
    same_episode = segments[:, :, None] == segments[:, None, :]
    length = is_first.shape[1]
    causal = jnp.arange(length)[None, :] <= jnp.arange(length)[:, None]
    return same_episode & causal[None]


class TemporalBlock(nn.Module):
    """Pre-normalized causal Transformer block."""

    config: TemporalConfig

    @nn.compact
    def __call__(self, inputs: jax.Array, mask: jax.Array) -> jax.Array:
        cfg = self.config
        residual = inputs
        x = nn.RMSNorm(name="attention_norm")(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            qkv_features=cfg.model_dim,
            out_features=cfg.model_dim,
            dropout_rate=0.0,
            precision=jax.lax.Precision.HIGHEST,
            name="attention",
        )(x, x, mask=mask, deterministic=True)
        x = residual + x
        residual = x
        x = nn.RMSNorm(name="ffn_norm")(x)
        x = nn.Dense(
            2 * cfg.mlp_ratio * cfg.model_dim,
            precision=jax.lax.Precision.HIGHEST,
            name="ffn_in",
        )(x)
        value, gate = jnp.split(x, 2, axis=-1)
        x = value * nn.gelu(gate)
        x = nn.Dense(
            cfg.model_dim,
            precision=jax.lax.Precision.HIGHEST,
            name="ffn_out",
        )(x)
        return residual + x


class CausalTemporalTransformer(nn.Module):
    """Transformer state shared by posterior inference and imagination.

    ``__call__`` is the parallel training path. ``step`` is the recurrent path
    used during collection and imagination. The two paths intentionally share
    every parameter and are required to be numerically equivalent while the
    active episode fits within ``context_length``.
    """

    config: TemporalConfig

    def initial(self, batch_size: int) -> TemporalCache:
        cfg = self.config
        return TemporalCache(
            tokens=jnp.zeros(
                (batch_size, cfg.context_length, cfg.model_dim), jnp.float32
            ),
            valid=jnp.zeros((batch_size, cfg.context_length), bool),
            positions=jnp.zeros((batch_size, cfg.context_length), jnp.int32),
            position=-jnp.ones((batch_size,), jnp.int32),
        )

    @nn.compact
    def __call__(self, pairs: jax.Array, is_first: jax.Array) -> jax.Array:
        """Compute ``h_t = Transformer((z,a)_<t)`` for a replay sequence."""
        cfg = self.config
        self._validate_parallel_inputs(pairs, is_first)
        pair_projection = nn.Dense(
            cfg.model_dim,
            precision=jax.lax.Precision.HIGHEST,
            name="pair_projection",
        )
        projected = jax.vmap(pair_projection, in_axes=1, out_axes=1)(pairs)
        shifted = jnp.concatenate(
            [jnp.zeros_like(projected[:, :1]), projected[:, :-1]], axis=1
        )
        start = self.param(
            "start_token", nn.initializers.normal(0.02), (cfg.model_dim,)
        )
        tokens = jnp.where(is_first[..., None], start, shifted)
        positions = episode_positions(is_first)
        sequence_length = pairs.shape[1]
        padding = cfg.context_length - sequence_length
        tokens = jnp.pad(tokens, ((0, 0), (padding, 0), (0, 0)))
        positions = jnp.pad(positions, ((0, 0), (padding, 0)))
        sequence_mask = segment_causal_mask(is_first)
        mask = jnp.pad(
            sequence_mask,
            ((0, 0), (padding, 0), (padding, 0)),
            constant_values=False,
        )[:, None]
        hidden = self._transform(tokens, positions, mask)[:, -sequence_length:]
        state_projection = nn.Dense(
            cfg.state_dim,
            precision=jax.lax.Precision.HIGHEST,
            name="state_projection",
        )
        return jax.vmap(state_projection, in_axes=1, out_axes=1)(hidden)

    @nn.compact
    def step(
        self,
        cache: TemporalCache,
        previous_pair: jax.Array,
        is_first: jax.Array,
    ) -> tuple[TemporalCache, jax.Array]:
        """Advance one recurrent state without observing the current pair."""
        cfg = self.config
        self._validate_step_inputs(cache, previous_pair, is_first)
        projected = nn.Dense(
            cfg.model_dim,
            precision=jax.lax.Precision.HIGHEST,
            name="pair_projection",
        )(previous_pair)
        start = self.param(
            "start_token", nn.initializers.normal(0.02), (cfg.model_dim,)
        )
        token = jnp.where(is_first[:, None], start, projected)

        keep = ~is_first[:, None]
        tokens = jnp.where(keep[..., None], cache.tokens, 0)
        valid = jnp.where(keep, cache.valid, False)
        positions = jnp.where(keep, cache.positions, 0)
        position = jnp.where(is_first, 0, cache.position + 1)

        tokens = jnp.concatenate([tokens[:, 1:], token[:, None]], axis=1)
        valid = jnp.concatenate(
            [valid[:, 1:], jnp.ones((valid.shape[0], 1), bool)], axis=1
        )
        positions = jnp.concatenate([positions[:, 1:], position[:, None]], axis=1)
        length = cfg.context_length
        causal = jnp.arange(length)[None, :] <= jnp.arange(length)[:, None]
        mask = causal[None] & valid[:, None, :]
        hidden = self._transform(tokens, positions, mask[:, None])
        state = nn.Dense(
            cfg.state_dim,
            precision=jax.lax.Precision.HIGHEST,
            name="state_projection",
        )(hidden[:, -1])
        return TemporalCache(tokens, valid, positions, position), state

    def _transform(
        self, tokens: jax.Array, positions: jax.Array, mask: jax.Array
    ) -> jax.Array:
        cfg = self.config
        position_table = self.param(
            "position_embedding",
            nn.initializers.normal(0.02),
            (cfg.context_length, cfg.model_dim),
        )
        x = tokens + position_table[positions % cfg.context_length]
        for index in range(cfg.num_layers):
            x = TemporalBlock(cfg, name=f"block_{index}")(x, mask)
        return nn.RMSNorm(name="output_norm")(x)

    def _validate_parallel_inputs(self, pairs: jax.Array, is_first: jax.Array) -> None:
        cfg = self.config
        if pairs.ndim != 3 or pairs.shape[-1] != cfg.pair_dim:
            raise ValueError(
                f"expected pairs [batch,time,{cfg.pair_dim}], got {pairs.shape}"
            )
        if is_first.shape != pairs.shape[:2]:
            raise ValueError(
                f"is_first shape {is_first.shape} does not match {pairs.shape[:2]}"
            )
        if pairs.shape[1] > cfg.context_length:
            raise ValueError(
                "parallel sequence exceeds context_length; chunk it at replay "
                "boundaries before applying the temporal model"
            )

    def _validate_step_inputs(
        self,
        cache: TemporalCache,
        previous_pair: jax.Array,
        is_first: jax.Array,
    ) -> None:
        cfg = self.config
        batch = previous_pair.shape[0]
        if previous_pair.shape != (batch, cfg.pair_dim):
            raise ValueError(
                f"expected previous_pair [batch,{cfg.pair_dim}], "
                f"got {previous_pair.shape}"
            )
        if is_first.shape != (batch,):
            raise ValueError(f"expected is_first [batch], got {is_first.shape}")
        expected = (batch, cfg.context_length, cfg.model_dim)
        if cache.tokens.shape != expected:
            raise ValueError(
                f"expected cache tokens {expected}, got {cache.tokens.shape}"
            )
