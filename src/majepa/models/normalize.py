"""Masked normalization retained from the pre-PPO MA-JEPA implementation."""

import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp


class Normalize(embodied.jax.Normalize):
    """Exclude invalid agent rows from the optional return-scale control."""

    rate: float = 0.01
    limit: float = 1.0
    perclo: float = 5.0
    perchi: float = 95.0
    debias: bool = False

    def __call__(self, x, update, mask=None):
        if update:
            self.update(x, mask)
        return self.stats()

    def update(self, x, mask=None):
        if mask is None:
            return super().update(x)
        if self.impl != "perc":
            raise ValueError("Masked PPO return normalization requires percentiles")
        x = jax.lax.stop_gradient(jnp.asarray(x, jnp.float32))
        mask = jnp.broadcast_to(jnp.asarray(mask, bool), x.shape)
        axes = embodied.jax.internal.get_data_axes()
        if axes:
            x = jax.lax.all_gather(x, axes)
            mask = jax.lax.all_gather(mask, axes)
        available = mask.any()
        values = jnp.where(mask, x, jnp.nan)
        for variable, percentile in ((self.lo, self.perclo), (self.hi, self.perchi)):
            old = variable.read()
            target = jnp.nanpercentile(values, percentile)
            candidate = (1 - self.rate) * old + self.rate * target
            variable.write(nn.where(available, candidate, old))
        if self.debias:
            old = self.corr.read()
            self.corr.write(nn.where(available, (1 - self.rate) * old + self.rate, old))
