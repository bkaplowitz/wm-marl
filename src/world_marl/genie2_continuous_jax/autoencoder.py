from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class ContinuousLatentAutoencoder(nn.Module):
    latent_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
        *,
        decode_latents: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if observations.ndim < 2:
            raise ValueError("observations must have shape (batch, ...)")

        obs_shape = observations.shape[1:]
        x = observations.astype(jnp.float32).reshape((observations.shape[0], -1))
        for dim in self.hidden_dims:
            x = nn.silu(nn.Dense(dim)(x))
        latents = nn.Dense(self.latent_dim, name="latent")(x)

        y = latents if decode_latents is None else decode_latents.astype(jnp.float32)
        for dim in reversed(tuple(self.hidden_dims)):
            y = nn.silu(nn.Dense(dim)(y))
        recon_flat = nn.Dense(int(jnp.prod(jnp.asarray(obs_shape))), name="decoder")(y)
        if len(obs_shape) == 3 and obs_shape[-1] in {1, 3, 4}:
            recon_flat = nn.sigmoid(recon_flat)
        reconstructions = recon_flat.reshape((observations.shape[0], *obs_shape))
        return latents, reconstructions


def reconstruction_loss(
    observations: jax.Array,
    reconstructions: jax.Array,
) -> jax.Array:
    return jnp.mean(jnp.square(observations.astype(jnp.float32) - reconstructions))
