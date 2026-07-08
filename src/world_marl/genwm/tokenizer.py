"""Tokenizers mapping continuous vectors to per-factor token ids and back.

The generative token arms (discrete-transformer CTMC flow, LLaDA2 block
diffusion) model per-factor categoricals, so continuous vectors must be
discretized. ``QuantileTokenizer`` bins each dimension independently on fixed
data quantiles; ``CodebookTokenizer`` exposes a learned VQ codebook (fit by
:mod:`world_marl.genwm.genie`) where each token decodes to a ``code_dim``
embedding vector rather than a scalar. Both are flax pytrees so they flow
through jit as traced arguments; ``encode_tokens``/``decode_tokens`` dispatch
on the tokenizer type.
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


@struct.dataclass
class CodebookTokenizer:
    """Maps flattened codebook embeddings to token ids and back.

    ``codebook`` is ``[num_bins, code_dim]``. Values are ``[..., dim * code_dim]``
    floats; encode is nearest-neighbor over codebook rows, which is the exact
    inverse of decode on any point that lies on the codebook, so round trips
    through the token world model are lossless.
    """

    codebook: jax.Array

    @property
    def num_bins(self) -> int:
        return self.codebook.shape[0]

    @property
    def code_dim(self) -> int:
        return self.codebook.shape[-1]


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


def encode_tokens(
    tokenizer: QuantileTokenizer | CodebookTokenizer, values: jax.Array
) -> jax.Array:
    """``[..., dim]`` floats (``[..., dim * code_dim]`` for codebooks) -> int32 ids."""
    if isinstance(tokenizer, CodebookTokenizer):
        code_dim = tokenizer.code_dim
        rows = values.reshape((*values.shape[:-1], -1, code_dim))
        distances = jnp.sum((rows[..., None, :] - tokenizer.codebook) ** 2, axis=-1)
        return jnp.argmin(distances, axis=-1).astype(jnp.int32)
    flat = values.reshape((-1, tokenizer.dim))
    tokens = jax.vmap(
        lambda edges, column: jnp.searchsorted(edges, column, side="right"),
        in_axes=(0, 1),
        out_axes=1,
    )(tokenizer.edges, flat)
    return tokens.reshape(values.shape).astype(jnp.int32)


def decode_tokens(
    tokenizer: QuantileTokenizer | CodebookTokenizer, tokens: jax.Array
) -> jax.Array:
    """``[..., dim]`` int32 ids -> bin centers (``[..., dim * code_dim]`` embeddings)."""
    if isinstance(tokenizer, CodebookTokenizer):
        embeddings = tokenizer.codebook[tokens]
        return embeddings.reshape((*tokens.shape[:-1], -1)).astype(jnp.float32)
    flat = tokens.reshape((-1, tokenizer.dim))
    values = jax.vmap(
        lambda centers, column: centers[column],
        in_axes=(0, 1),
        out_axes=1,
    )(tokenizer.centers, flat)
    return values.reshape(tokens.shape).astype(jnp.float32)
