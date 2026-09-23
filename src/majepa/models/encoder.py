"""DreamerV3 convolutional encoder used by canonical MA-JEPA."""

import embodied.jax.nets as nn
import jax.numpy as jnp
import ninjax as nj


class Encoder(nj.Module):
    units: int = 1024
    norm: str = "rms"
    act: str = "gelu"
    layers: int = 3
    symlog: bool = True

    def __init__(self, obs_space, **kw):
        if not obs_space or any(len(space.shape) > 2 for space in obs_space.values()):
            raise ValueError("The SMAC encoder expects nonempty vector observations")
        self.obs_space = obs_space
        self.veckeys = [
            key for key, space in obs_space.items() if len(space.shape) <= 2
        ]
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
        outs = []
        bshape = reset.shape

        vspace = {key: self.obs_space[key] for key in self.veckeys}
        vecs = {key: obs[key] for key in self.veckeys}
        squish = nn.symlog if self.symlog else lambda value: value
        x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
        x = x.reshape((-1, *x.shape[bdims:]))
        for index in range(self.layers):
            x = self.sub(f"mlp{index}", nn.Linear, self.units, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f"mlp{index}norm", nn.Norm, self.norm)(x))
        outs.append(x)

        x = jnp.concatenate(outs, -1)
        return carry, {}, x.reshape((*bshape, *x.shape[1:]))

    def calculate_encoder_output_dim(self):
        total = self.units
        return int(total)
