"""Shared vector-observation encoder; agent histories remain separate."""

import embodied.jax.nets as nn
import jax.numpy as jnp
import ninjax as nj


class Encoder(nj.Module):
    units: int = 1024
    norm: str = "rms"
    act: str = "silu"
    layers: int = 3
    symlog: bool = True

    def __init__(self, obs_space, **kw):
        if not obs_space or any(len(space.shape) > 2 for space in obs_space.values()):
            raise ValueError("The maintained encoder accepts vector observations")
        self.obs_space = obs_space
        self.veckeys = list(obs_space)
        self.kw = kw

    @property
    def entry_space(self):
        return {}

    def initial(self, batch_size):
        return {}

    def truncate(self, entries, carry=None):
        return {}

    def __call__(self, carry, obs, reset, training, single=False):
        bdims = 1 if single else 2
        vspace = {key: self.obs_space[key] for key in self.veckeys}
        vecs = {key: obs[key] for key in self.veckeys}
        squish = nn.symlog if self.symlog else lambda value: value
        x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
        x = x.reshape((-1, *x.shape[bdims:]))
        for index in range(self.layers):
            x = self.sub(f"mlp{index}", nn.Linear, self.units, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f"mlp{index}norm", nn.Norm, self.norm)(x))
        x = jnp.concatenate([x], -1)
        return carry, {}, x.reshape((*reset.shape, *x.shape[1:]))

    def calculate_encoder_output_dim(self):
        return int(self.units)
