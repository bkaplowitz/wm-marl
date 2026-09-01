"""Normalization statistics with explicit variable-liveness masking."""

import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp


f32 = jnp.float32
sg = jax.lax.stop_gradient


class Normalize(embodied.jax.Normalize):
    """Pinned Dreamer normalizer that excludes invalid agent samples."""

    rate: float = 0.01
    limit: float = 1e-8
    perclo: float = 5.0
    perchi: float = 95.0
    debias: bool = True

    def __call__(self, x, update, mask=None):
        if update:
            self.update(x, mask)
        return self.stats()

    def update(self, x, mask=None):
        if mask is None:
            return super().update(x)
        x = sg(f32(x))
        mask = jnp.broadcast_to(jnp.asarray(mask, bool), x.shape)
        axes = embodied.jax.internal.get_data_axes()
        if axes:
            x = jax.lax.all_gather(x, axes)
            mask = jax.lax.all_gather(mask, axes)
        count = mask.sum()
        available = count > 0

        if self.impl == "none":
            pass
        elif self.impl == "meanstd":
            mean = (jnp.where(mask, x, 0).sum() / jnp.maximum(count, 1)).astype(f32)
            sqrs = (
                jnp.where(mask, jnp.square(x), 0).sum() / jnp.maximum(count, 1)
            ).astype(f32)
            self._masked_update(self.mean, mean, available)
            self._masked_update(self.sqrs, sqrs, available)
        elif self.impl == "perc":
            values = jnp.where(mask, x, jnp.nan)
            lo = jnp.nanpercentile(values, self.perclo)
            hi = jnp.nanpercentile(values, self.perchi)
            self._masked_update(self.lo, lo, available)
            self._masked_update(self.hi, hi, available)
        else:
            raise NotImplementedError(self.impl)
        if self.debias and self.impl != "none":
            self._masked_update(self.corr, jnp.asarray(1.0, f32), available)

    def _masked_update(self, variable, value, available):
        old = variable.read()
        candidate = (1 - self.rate) * old + self.rate * sg(value)
        variable.write(nn.where(available, candidate, old))


__all__ = ["Normalize"]
