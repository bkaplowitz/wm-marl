"""Shared local observation encoder and report-only decoder for DreaMARL."""

import math

import einops
import embodied
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np


class Encoder(nj.Module):
    """Encode one agent's local vector or image observation."""

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
        if not all(len(space.shape) <= 3 for space in obs_space.values()):
            raise ValueError(obs_space)
        self.obs_space = obs_space
        self.veckeys = [key for key, space in obs_space.items() if len(space.shape) <= 2]
        self.imgkeys = [key for key, space in obs_space.items() if len(space.shape) == 3]
        self.depths = tuple(self.depth * multiplier for multiplier in self.mults)
        self.kw = kw

    @property
    def entry_space(self):
        return {}

    def initial(self, batch_size):
        del batch_size
        return {}

    def truncate(self, entries, carry=None):
        del entries, carry
        return {}

    def __call__(self, carry, obs, reset, training, single=False):
        del training
        batch_dimensions = 1 if single else 2
        outputs = []
        batch_shape = reset.shape

        if self.veckeys:
            spaces = {key: self.obs_space[key] for key in self.veckeys}
            vectors = {key: obs[key] for key in self.veckeys}
            transform = nn.symlog if self.symlog else lambda value: value
            value = nn.DictConcat(spaces, 1, squish=transform)(vectors)
            value = value.reshape((-1, *value.shape[batch_dimensions:]))
            for index in range(self.layers):
                value = self.sub(
                    f"mlp{index}", nn.Linear, self.units, **self.kw
                )(value)
                value = nn.act(self.act)(
                    self.sub(f"mlp{index}norm", nn.Norm, self.norm)(value)
                )
            outputs.append(value)

        if self.imgkeys:
            images = [obs[key] for key in sorted(self.imgkeys)]
            if not all(image.dtype == jnp.uint8 for image in images):
                raise TypeError([image.dtype for image in images])
            value = nn.cast(jnp.concatenate(images, -1), force=True) / 255 - 0.5
            value = value.reshape((-1, *value.shape[batch_dimensions:]))
            for index, depth in enumerate(self.depths):
                if self.outer and index == 0:
                    value = self.sub(
                        f"cnn{index}", nn.Conv2D, depth, self.kernel, **self.kw
                    )(value)
                elif self.strided:
                    value = self.sub(
                        f"cnn{index}", nn.Conv2D, depth, self.kernel, 2, **self.kw
                    )(value)
                else:
                    value = self.sub(
                        f"cnn{index}", nn.Conv2D, depth, self.kernel, **self.kw
                    )(value)
                    batch, height, width, channels = value.shape
                    value = value.reshape(
                        (batch, height // 2, 2, width // 2, 2, channels)
                    ).max((2, 4))
                value = nn.act(self.act)(
                    self.sub(f"cnn{index}norm", nn.Norm, self.norm)(value)
                )
            if not (3 <= value.shape[-3] <= 16 and 3 <= value.shape[-2] <= 16):
                raise ValueError(value.shape)
            outputs.append(value.reshape((value.shape[0], -1)))

        if not outputs:
            raise ValueError("encoder requires at least one observation input")
        tokens = jnp.concatenate(outputs, -1)
        tokens = tokens.reshape((*batch_shape, *tokens.shape[1:]))
        return carry, {}, tokens

    def calculate_output_dim(self):
        """Return the exact width emitted by the configured encoder."""

        width = self.units if self.veckeys else 0
        if self.imgkeys:
            height, image_width = self.obs_space[self.imgkeys[0]].shape[:2]
            reductions = len(self.depths) - int(bool(self.outer))
            height //= 2**reductions
            image_width //= 2**reductions
            width += height * image_width * self.depths[-1]
        return int(width)


class Decoder(nj.Module):
    """Decode local joint-world features without shaping the world latent."""

    units: int = 1024
    norm: str = "rms"
    act: str = "gelu"
    outscale: float = 1.0
    depth: int = 64
    mults: tuple = (2, 3, 4, 4)
    layers: int = 3
    kernel: int = 5
    symlog: bool = True
    bspace: int = 8
    outer: bool = False
    strided: bool = False

    def __init__(self, obs_space, **kw):
        if not all(len(space.shape) <= 3 for space in obs_space.values()):
            raise ValueError(obs_space)
        self.obs_space = obs_space
        self.veckeys = [key for key, space in obs_space.items() if len(space.shape) <= 2]
        self.imgkeys = [key for key, space in obs_space.items() if len(space.shape) == 3]
        self.depths = tuple(self.depth * multiplier for multiplier in self.mults)
        self.image_depth = sum(obs_space[key].shape[-1] for key in self.imgkeys)
        self.image_resolution = (
            self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
        )
        self.kw = kw

    @property
    def entry_space(self):
        return {}

    def initial(self, batch_size):
        del batch_size
        return {}

    def truncate(self, entries, carry=None):
        del entries, carry
        return {}

    def __call__(self, carry, feature, reset, training, single=False):
        del training, single
        if feature["deter"].shape[-1] % self.bspace:
            raise ValueError((feature["deter"].shape, self.bspace))
        reconstructions = {}
        batch_shape = reset.shape
        inputs = [nn.cast(feature[key]) for key in ("stoch", "deter")]
        inputs = [value.reshape((math.prod(batch_shape), -1)) for value in inputs]
        inputs = jnp.concatenate(inputs, -1)

        if self.veckeys:
            spaces = {key: self.obs_space[key] for key in self.veckeys}
            outputs = {
                key: (
                    "categorical"
                    if space.discrete
                    else ("symlog_mse" if self.symlog else "mse")
                )
                for key, space in spaces.items()
            }
            kwargs = dict(**self.kw, act=self.act, norm=self.norm)
            value = self.sub(
                "mlp", nn.MLP, self.layers, self.units, **kwargs
            )(inputs)
            value = value.reshape((*batch_shape, *value.shape[1:]))
            kwargs = dict(**self.kw, outscale=self.outscale)
            reconstructions.update(
                self.sub(
                    "vec", embodied.jax.DictHead, spaces, outputs, **kwargs
                )(value)
            )

        if self.imgkeys:
            factor = 2 ** (len(self.depths) - int(bool(self.outer)))
            minimum = [int(size // factor) for size in self.image_resolution]
            if not (3 <= minimum[0] <= 16 and 3 <= minimum[1] <= 16):
                raise ValueError(minimum)
            shape = (*minimum, self.depths[-1])
            if self.bspace:
                units, groups = math.prod(shape), self.bspace
                deter, stochastic = nn.cast(
                    (feature["deter"], feature["stoch"])
                )
                stochastic = stochastic.reshape((*stochastic.shape[:-2], -1))
                deter = deter.reshape((-1, deter.shape[-1]))
                stochastic = stochastic.reshape((-1, stochastic.shape[-1]))
                deter = self.sub(
                    "sp0", nn.BlockLinear, units, groups, **self.kw
                )(deter)
                deter = einops.rearrange(
                    deter,
                    "... (g h w c) -> ... h w (g c)",
                    h=minimum[0],
                    w=minimum[1],
                    g=groups,
                )
                stochastic = self.sub(
                    "sp1", nn.Linear, 2 * self.units, **self.kw
                )(stochastic)
                stochastic = nn.act(self.act)(
                    self.sub("sp1norm", nn.Norm, self.norm)(stochastic)
                )
                stochastic = self.sub("sp2", nn.Linear, shape, **self.kw)(stochastic)
                value = nn.act(self.act)(
                    self.sub("spnorm", nn.Norm, self.norm)(deter + stochastic)
                )
            else:
                kwargs = dict(**self.kw, act=self.act, norm=self.norm)
                value = self.sub("space", nn.Linear, shape, **kwargs)(inputs)
                value = nn.act(self.act)(
                    self.sub("spacenorm", nn.Norm, self.norm)(value)
                )

            for index, depth in reversed(list(enumerate(self.depths[:-1]))):
                if self.strided:
                    kwargs = dict(**self.kw, transp=True)
                    value = self.sub(
                        f"conv{index}", nn.Conv2D, depth, self.kernel, 2, **kwargs
                    )(value)
                else:
                    value = value.repeat(2, -2).repeat(2, -3)
                    value = self.sub(
                        f"conv{index}", nn.Conv2D, depth, self.kernel, **self.kw
                    )(value)
                value = nn.act(self.act)(
                    self.sub(f"conv{index}norm", nn.Norm, self.norm)(value)
                )
            kwargs = dict(**self.kw, outscale=self.outscale)
            if self.outer:
                value = self.sub(
                    "imgout", nn.Conv2D, self.image_depth, self.kernel, **kwargs
                )(value)
            elif self.strided:
                kwargs["transp"] = True
                value = self.sub(
                    "imgout", nn.Conv2D, self.image_depth, self.kernel, 2, **kwargs
                )(value)
            else:
                value = value.repeat(2, -2).repeat(2, -3)
                value = self.sub(
                    "imgout", nn.Conv2D, self.image_depth, self.kernel, **kwargs
                )(value)
            value = jax.nn.sigmoid(value)
            value = value.reshape((*batch_shape, *value.shape[1:]))
            split = np.cumsum(
                [self.obs_space[key].shape[-1] for key in self.imgkeys][:-1]
            )
            for key, output in zip(self.imgkeys, jnp.split(value, split, -1)):
                output = embodied.jax.outs.MSE(output)
                reconstructions[key] = embodied.jax.outs.Agg(
                    output, 3, jnp.sum
                )

        return carry, {}, reconstructions
