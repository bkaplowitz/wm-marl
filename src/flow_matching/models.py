"""Minimal Flax models for vector-field learning."""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class MLPVectorField(nn.Module):
    """A small MLP that predicts flow u_theta(x, t). Uses SiLU activation. Represents both conditional and unconditional vector fields.

    Conditional vector field: u_theta(x, t, cond_vars)
    Unconditional vector field: u_theta(x, t)
    """

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


class TokenizedDiscreteDenoiser(nn.Module):
    """Posterior network f_theta for discrete flow matching (discrete.md Alg 8).

    Maps integer tokens ``(B, d)`` to per-factor logits ``(B, d, V)`` via a token
    embedding rather than a one-hot input, so the input embedding and the d*V
    classification head are sized independently (unlike :class:`MLPVectorField`,
    whose head is tied to its input width).
    """

    num_categories: int
    embed_dim: int = 16
    hidden_dims: Sequence[int] = (64, 64, 64, 64)

    @nn.compact
    def __call__(
        self,
        tokens: jax.Array,
        t: jax.Array,
        cond_vars: jax.Array | None = None,
    ) -> jax.Array:
        num_factors = tokens.shape[-1]
        emb = nn.Embed(self.num_categories, self.embed_dim)(tokens)
        emb_flat = emb.reshape((tokens.shape[0], num_factors * self.embed_dim))

        parts = (emb_flat, t) if cond_vars is None else (emb_flat, t, cond_vars)
        xt = jnp.concat(parts, axis=-1)

        hidden_layers = [
            layer for dim in self.hidden_dims for layer in (nn.Dense(dim), nn.silu)
        ]
        head = [nn.Dense(num_factors * self.num_categories)]
        logits = nn.Sequential(hidden_layers + head)(xt)
        return logits.reshape((tokens.shape[0], num_factors, self.num_categories))
