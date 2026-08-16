"""DreamerV3 convolutional encoder used by canonical DreaMARL."""

import embodied.jax.nets as nn
import jax.numpy as jnp
import ninjax as nj


class Encoder(nj.Module):
    units: int = 1024
    norm: str = "rms"
    act: str = "gelu"
    depth: int = 64
    mults: tuple = (2, 3, 4, 4)
    layers: int = 3
    kernel: int = 5
    symlog: bool = True
    outer: bool = False
    strided: bool = False

    def __init__(self, obs_space, **kw):
        assert all(len(space.shape) <= 3 for space in obs_space.values()), obs_space
        self.obs_space = obs_space
        self.veckeys = [
            key for key, space in obs_space.items() if len(space.shape) <= 2
        ]
        self.imgkeys = [
            key for key, space in obs_space.items() if len(space.shape) == 3
        ]
        self.depths = tuple(self.depth * mult for mult in self.mults)
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

        if self.veckeys:
            vspace = {key: self.obs_space[key] for key in self.veckeys}
            vecs = {key: obs[key] for key in self.veckeys}
            squish = nn.symlog if self.symlog else lambda value: value
            x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
            x = x.reshape((-1, *x.shape[bdims:]))
            for index in range(self.layers):
                x = self.sub(f"mlp{index}", nn.Linear, self.units, **self.kw)(x)
                x = nn.act(self.act)(self.sub(f"mlp{index}norm", nn.Norm, self.norm)(x))
            outs.append(x)

        if self.imgkeys:
            images = [obs[key] for key in sorted(self.imgkeys)]
            assert all(image.dtype == jnp.uint8 for image in images)
            x = nn.cast(jnp.concatenate(images, -1), force=True) / 255 - 0.5
            x = x.reshape((-1, *x.shape[bdims:]))
            for index, depth in enumerate(self.depths):
                if self.outer and index == 0:
                    x = self.sub(
                        f"cnn{index}", nn.Conv2D, depth, self.kernel, **self.kw
                    )(x)
                elif self.strided:
                    x = self.sub(
                        f"cnn{index}", nn.Conv2D, depth, self.kernel, 2, **self.kw
                    )(x)
                else:
                    x = self.sub(
                        f"cnn{index}", nn.Conv2D, depth, self.kernel, **self.kw
                    )(x)
                    batch, height, width, channels = x.shape
                    x = x.reshape((batch, height // 2, 2, width // 2, 2, channels)).max(
                        (2, 4)
                    )
                x = nn.act(self.act)(self.sub(f"cnn{index}norm", nn.Norm, self.norm)(x))
            assert 3 <= x.shape[-3] <= 16, x.shape
            assert 3 <= x.shape[-2] <= 16, x.shape
            outs.append(x.reshape((x.shape[0], -1)))

        x = jnp.concatenate(outs, -1)
        return carry, {}, x.reshape((*bshape, *x.shape[1:]))

    def calculate_encoder_output_dim(self):
        total = self.units if self.veckeys else 0
        if self.imgkeys:
            height, width, depth = self.image_grid_shape()
            total += height * width * depth
        return int(total)

    def image_grid_shape(self):
        spatial_shapes = {self.obs_space[key].shape[:2] for key in self.imgkeys}
        if len(spatial_shapes) != 1:
            raise ValueError("image observations must share one spatial shape")
        height, width = next(iter(spatial_shapes))
        for index in range(len(self.depths)):
            if self.outer and index == 0:
                continue
            if self.strided:
                height = (height + 1) // 2
                width = (width + 1) // 2
            else:
                height //= 2
                width //= 2
        return int(height), int(width), int(self.depths[-1])

    def spatial_tokens(self, tokens):
        height, width, depth = self.image_grid_shape()
        offset = self.units if self.veckeys else 0
        values = tokens[..., offset : offset + height * width * depth]
        return values.reshape((*tokens.shape[:-1], height * width, depth))
