from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp


def sample_next_observation(
    autoencoder_apply: Callable[..., tuple[jnp.ndarray, jnp.ndarray]],
    autoencoder_params: Any,
    next_latents: jnp.ndarray,
    *,
    observation_shape: tuple[int, ...] = (6, 6, 3),
) -> jnp.ndarray:
    dummy = jnp.zeros((next_latents.shape[0], *observation_shape), dtype=jnp.float32)
    variables = autoencoder_params
    _, decoded = autoencoder_apply(variables, dummy, method=None)
    del next_latents
    return decoded
