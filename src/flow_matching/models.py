"""Minimal Flax models for vector-field learning."""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class MLPVectorField(nn.Module):
    """A small MLP that predicts u_theta(x, t). Uses SiLU activation."""

    hidden_dims: Sequence[int] = (64, 64, 64, 64)

    @nn.compact
    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Evaluate the vector field at batched positions and times."""
        xt = jnp.concat((x, t), axis=-1)  # concatenate `x`, `t`

        hidden_layers = [
            layer for dim in self.hidden_dims for layer in (nn.Dense(dim), nn.silu)
        ]
        final_layer = [nn.Dense(x.shape[-1])]
        return nn.Sequential(hidden_layers + final_layer)(xt)
