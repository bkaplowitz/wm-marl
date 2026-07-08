from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class CausalLatentDynamics(nn.Module):
    latent_dim: int
    latent_action_dim: int
    model_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    max_context: int = 32
    mlp_ratio: int = 4

    @nn.compact
    def __call__(
        self,
        latent_history: jax.Array,
        latent_actions: jax.Array,
        noise_level: jax.Array | None = None,
        condition_keep_mask: jax.Array | None = None,
    ) -> jax.Array:
        if latent_history.ndim != 3:
            raise ValueError("latent_history must have shape (batch, time, latent_dim)")
        if latent_actions.ndim != 3:
            raise ValueError(
                "latent_actions must have shape (batch, time, latent_action_dim)"
            )
        if latent_history.shape[:2] != latent_actions.shape[:2]:
            raise ValueError("latent_history and latent_actions must share batch/time")
        if latent_history.shape[-1] != self.latent_dim:
            raise ValueError("latent_history last dimension must match latent_dim")
        if latent_actions.shape[-1] != self.latent_action_dim:
            raise ValueError(
                "latent_actions last dimension must match latent_action_dim"
            )

        batch_size, time_steps, _ = latent_history.shape
        if time_steps > self.max_context:
            raise ValueError("time dimension exceeds max_context")

        actions = latent_actions.astype(jnp.float32)
        if condition_keep_mask is not None:
            keep = condition_keep_mask.astype(jnp.float32).reshape((batch_size, 1, 1))
            actions = actions * keep

        h = nn.Dense(self.model_dim, name="latent_in")(
            latent_history.astype(jnp.float32)
        )
        h = h + nn.Dense(self.model_dim, name="action_in")(actions)
        positions = self.param(
            "position_embedding",
            nn.initializers.normal(0.02),
            (self.max_context, self.model_dim),
        )
        h = h + positions[:time_steps][None, :, :]
        if noise_level is not None:
            noise = noise_level.astype(jnp.float32).reshape((batch_size, 1))
            h = h + nn.Dense(self.model_dim, name="noise_in")(noise)[:, None, :]

        causal_mask = nn.make_causal_mask(
            jnp.ones((batch_size, time_steps), dtype=bool)
        )
        for _ in range(self.num_layers):
            z = nn.LayerNorm()(h)
            h = h + nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(
                z, z, mask=causal_mask, deterministic=True
            )
            y = nn.LayerNorm()(h)
            y = nn.silu(nn.Dense(self.mlp_ratio * self.model_dim)(y))
            h = h + nn.Dense(self.model_dim)(y)

        return nn.Dense(self.latent_dim, name="next_latent_head")(
            nn.LayerNorm()(h[:, -1])
        )


def classifier_free_guidance(
    *,
    conditioned: jax.Array,
    unconditioned: jax.Array,
    guidance_scale: float,
) -> jax.Array:
    return unconditioned + guidance_scale * (conditioned - unconditioned)


def dynamics_mse_loss(prediction: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean(jnp.square(prediction - target))
