"""Minimal Flax models for vector-field learning."""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class MLPVectorField(nn.Module):
    """A small MLP that predicts u_theta(x, t). Uses SiLU activation."""

    hidden_dims: Sequence[int] = (64, 64, 64, 64)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        t: jax.Array,
        cond_vars: jax.Array | None = None,
    ) -> jax.Array:
        """Evaluate the vector field at batched positions and times.

        If cond_vars=None, unconditional vector field, else conditional.

        When ``cond_vars`` is provided it is concatenated into the trunk input,
        but the output head stays sized to ``x`` so the predicted field matches
        the target dimensionality (no conditioning leakage into the output).
        """
        parts = (x, t) if cond_vars is None else (x, t, cond_vars)
        xt = jnp.concat(parts, axis=-1)

        hidden_layers = [
            layer for dim in self.hidden_dims for layer in (nn.Dense(dim), nn.silu)
        ]
        final_layer = [nn.Dense(x.shape[-1])]
        return nn.Sequential(hidden_layers + final_layer)(xt)
