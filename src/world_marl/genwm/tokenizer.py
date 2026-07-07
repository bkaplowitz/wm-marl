"""Per-dimension quantile tokenizer for continuous observation/action vectors.

The generative token arms (discrete-transformer CTMC flow, LLaDA2 block
diffusion) model per-factor categoricals, so continuous vectors must be binned.
Edges/centers are fit host-side from replay data quantiles (numpy, once per fit
phase); encode/decode are pure jnp so the tokenizer can flow through jit as a
traced pytree.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class QuantileTokenizer:
    """Maps float vectors to per-dimension quantile-bin tokens and back.

    ``edges`` are the interior bin boundaries ``[dim, num_bins - 1]``;
    ``centers`` are the decode values ``[dim, num_bins]`` (the within-bin
    median). Registered as a flax pytree so it can be passed as a traced jit
    argument.
    """

    edges: jax.Array
    centers: jax.Array

    @property
    def num_bins(self) -> int:
        return self.centers.shape[-1]

    @property
    def dim(self) -> int:
        return self.centers.shape[0]


def fit_quantile_tokenizer(samples: np.ndarray, num_bins: int) -> QuantileTokenizer:
    """Fit per-dimension bin edges and centers from data quantiles.

    ``samples`` is ``[N, dim]``. Edges sit at quantiles ``i / num_bins``
    (i = 1..num_bins-1) and centers at ``(i + 0.5) / num_bins``, so bins are
    equal-mass under the fitting distribution. Dimensions with constant (or
    heavily duplicated) data produce repeated edges: encode then collapses the
    mass into fewer effective bins, and the duplicated centers decode to the
    same value, so round trips stay consistent.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(f"samples must be [N, dim], got shape {samples.shape}")
    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2, got {num_bins}")
    edge_quantiles = np.arange(1, num_bins) / num_bins
    center_quantiles = (np.arange(num_bins) + 0.5) / num_bins
    edges = np.quantile(samples, edge_quantiles, axis=0).T
    centers = np.quantile(samples, center_quantiles, axis=0).T
    return QuantileTokenizer(
        edges=jnp.asarray(edges, dtype=jnp.float32),
        centers=jnp.asarray(centers, dtype=jnp.float32),
    )


def encode_tokens(tokenizer: QuantileTokenizer, values: jax.Array) -> jax.Array:
    """``[..., dim]`` floats -> ``[..., dim]`` int32 bin ids."""
    flat = values.reshape((-1, tokenizer.dim))
    tokens = jax.vmap(
        lambda edges, column: jnp.searchsorted(edges, column, side="right"),
        in_axes=(0, 1),
        out_axes=1,
    )(tokenizer.edges, flat)
    return tokens.reshape(values.shape).astype(jnp.int32)


def decode_tokens(tokenizer: QuantileTokenizer, tokens: jax.Array) -> jax.Array:
    """``[..., dim]`` int32 bin ids -> ``[..., dim]`` float32 bin centers."""
    flat = tokens.reshape((-1, tokenizer.dim))
    values = jax.vmap(
        lambda centers, column: centers[column],
        in_axes=(0, 1),
        out_axes=1,
    )(tokenizer.centers, flat)
    return values.reshape(tokens.shape).astype(jnp.float32)
