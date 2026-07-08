from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class RewardContinueHead(nn.Module):
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(
        self,
        latents: jax.Array,
        latent_actions: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        x = jnp.concatenate(
            [latents.astype(jnp.float32), latent_actions.astype(jnp.float32)],
            axis=-1,
        )
        for dim in self.hidden_dims:
            x = nn.silu(nn.Dense(dim)(x))
        reward = nn.Dense(1, name="reward")(x)[..., 0]
        continue_logit = nn.Dense(1, name="continue")(x)[..., 0]
        return reward, continue_logit
