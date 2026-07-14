from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class ContinuousLAM(nn.Module):
    latent_action_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(
        self,
        prev_latents: jax.Array,
        next_latents: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        if prev_latents.shape != next_latents.shape:
            raise ValueError("prev_latents and next_latents must have identical shapes")
        x = jnp.concatenate(
            [
                prev_latents.astype(jnp.float32),
                next_latents.astype(jnp.float32),
                next_latents.astype(jnp.float32) - prev_latents.astype(jnp.float32),
            ],
            axis=-1,
        )
        for dim in self.hidden_dims:
            x = nn.silu(nn.Dense(dim)(x))
        mean = nn.Dense(self.latent_action_dim, name="mean")(x)
        log_std = nn.Dense(self.latent_action_dim, name="log_std")(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class LatentActionReconstructor(nn.Module):
    latent_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(
        self,
        prev_latents: jax.Array,
        latent_actions: jax.Array,
    ) -> jax.Array:
        x = jnp.concatenate(
            [prev_latents.astype(jnp.float32), latent_actions.astype(jnp.float32)],
            axis=-1,
        )
        for dim in self.hidden_dims:
            x = nn.silu(nn.Dense(dim)(x))
        return nn.Dense(self.latent_dim, name="next_latent")(x)


def sample_latent_actions(
    key: jax.Array,
    mean: jax.Array,
    log_std: jax.Array,
) -> jax.Array:
    return mean + jax.random.normal(key, mean.shape, dtype=mean.dtype) * jnp.exp(
        log_std
    )


def lam_kl_loss(mean: jax.Array, log_std: jax.Array) -> jax.Array:
    variance = jnp.exp(2.0 * log_std)
    return 0.5 * jnp.mean(jnp.square(mean) + variance - 1.0 - jnp.log(variance))
