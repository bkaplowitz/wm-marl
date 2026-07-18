from __future__ import annotations

import jax

from .oracle import RSSM_SOURCE_SPEC


def ninjax_scan_sample_keys(root_seed: jax.Array, length: int) -> jax.Array:
    if length <= 0:
        raise ValueError("scan length must be positive")
    contexts = jax.random.split(root_seed, length + 1)[1:]
    return jax.vmap(lambda key: jax.random.split(key, 16)[1])(contexts)


__all__ = ["RSSM_SOURCE_SPEC", "ninjax_scan_sample_keys"]
