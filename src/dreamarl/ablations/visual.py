"""Alternative visual encoders and reconstruction decoder for ablations.

Canonical DreaMARL imports its CNN from ``dreamarl.models.visual`` instead.
"""

import math

import einops
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np


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
        assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
        self.obs_space = obs_space
        self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
        self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
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
            vspace = {k: self.obs_space[k] for k in self.veckeys}
            vecs = {k: obs[k] for k in self.veckeys}
            squish = nn.symlog if self.symlog else lambda x: x
            x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
            x = x.reshape((-1, *x.shape[bdims:]))
            for i in range(self.layers):
                x = self.sub(f"mlp{i}", nn.Linear, self.units, **self.kw)(x)
                x = nn.act(self.act)(self.sub(f"mlp{i}norm", nn.Norm, self.norm)(x))
            outs.append(x)

        if self.imgkeys:
            K = self.kernel
            imgs = [obs[k] for k in sorted(self.imgkeys)]
            assert all(x.dtype == jnp.uint8 for x in imgs)
            x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
            x = x.reshape((-1, *x.shape[bdims:]))
            for i, depth in enumerate(self.depths):
                if self.outer and i == 0:
                    x = self.sub(f"cnn{i}", nn.Conv2D, depth, K, **self.kw)(x)
                elif self.strided:
                    x = self.sub(f"cnn{i}", nn.Conv2D, depth, K, 2, **self.kw)(x)
                else:
                    x = self.sub(f"cnn{i}", nn.Conv2D, depth, K, **self.kw)(x)
                    B, H, W, C = x.shape
                    x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
                x = nn.act(self.act)(self.sub(f"cnn{i}norm", nn.Norm, self.norm)(x))
            assert 3 <= x.shape[-3] <= 16, x.shape
            assert 3 <= x.shape[-2] <= 16, x.shape
            x = x.reshape((x.shape[0], -1))
            outs.append(x)

        x = jnp.concatenate(outs, -1)
        tokens = x.reshape((*bshape, *x.shape[1:]))
        entries = {}
        return carry, entries, tokens

    def calculate_encoder_output_dim(self):
        """Return the exact concatenated width produced by this configuration."""

        total_dim = self.units if self.veckeys else 0
        if not self.imgkeys:
            return int(total_dim)

        height, width, depth = self.image_grid_shape()
        total_dim += height * width * depth
        return int(total_dim)

    def image_grid_shape(self):
        """Return the final CNN grid before it is flattened into one vector."""

        if not self.imgkeys:
            raise ValueError("encoder has no image observations")

        spatial_shapes = {self.obs_space[key].shape[:2] for key in self.imgkeys}
        if len(spatial_shapes) != 1:
            raise ValueError(
                "all image observations must share one spatial shape, got "
                f"{sorted(spatial_shapes)}"
            )
        height, width = next(iter(spatial_shapes))
        for index in range(len(self.depths)):
            if self.outer and index == 0:
                continue
            if self.strided:
                height = (height + 1) // 2
                width = (width + 1) // 2
            else:
                if height % 2 or width % 2:
                    raise ValueError(
                        "non-strided encoder pooling requires even spatial "
                        f"dimensions at layer {index}, got {(height, width)}"
                    )
                height //= 2
                width //= 2
        return int(height), int(width), int(self.depths[-1])

    def spatial_tokens(self, tokens):
        """Recover final CNN patch tokens from the flattened encoder output."""

        height, width, depth = self.image_grid_shape()
        offset = self.units if self.veckeys else 0
        image_width = height * width * depth
        image_values = tokens[..., offset : offset + image_width]
        return image_values.reshape((*tokens.shape[:-1], height * width, depth))


class ViTEncoder(nj.Module):
    """Compact spatial ViT with the same flattened interface as the CNN.

    The default 8-pixel patches produce 64 spatial tokens. A final per-token
    projection keeps the complete image representation at 4096 values, so the
    temporal model and control heads are unchanged in encoder comparisons.
    """

    units: int = 1024
    layers: int = 6
    model: int = 256
    heads: int = 8
    ffup: int = 4
    patch: int = 8
    token_dim: int = 64
    norm: str = "rms"
    act: str = "silu"
    winit: str = "trunc_normal_in"
    symlog: bool = True

    def __init__(self, obs_space, **kw):
        assert all(len(space.shape) <= 3 for space in obs_space.values()), obs_space
        if self.model % self.heads:
            raise ValueError("ViT model width must be divisible by attention heads")
        self.obs_space = obs_space
        self.veckeys = [
            key for key, space in obs_space.items() if len(space.shape) <= 2
        ]
        self.imgkeys = [
            key for key, space in obs_space.items() if len(space.shape) == 3
        ]
        self.kw = kw
        if self.imgkeys:
            self.image_grid_shape()

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
        bdims = 1 if single else 2
        bshape = reset.shape
        outs = []

        if self.veckeys:
            spaces = {key: self.obs_space[key] for key in self.veckeys}
            values = {key: obs[key] for key in self.veckeys}
            squish = nn.symlog if self.symlog else lambda value: value
            x = nn.DictConcat(spaces, 1, squish=squish)(values)
            x = x.reshape((-1, *x.shape[bdims:]))
            for index in range(3):
                x = self.sub(f"vec{index}", nn.Linear, self.units, winit=self.winit)(x)
                x = nn.act(self.act)(self.sub(f"vec{index}norm", nn.Norm, self.norm)(x))
            outs.append(x)

        if self.imgkeys:
            x = self._encode_image_tokens(obs, bdims=bdims)
            outs.append(x.reshape((x.shape[0], -1)))

        tokens = jnp.concatenate(outs, axis=-1).reshape((*bshape, -1))
        return carry, {}, tokens

    def _encode_image_tokens(self, obs, *, bdims):
        images = [obs[key] for key in sorted(self.imgkeys)]
        assert all(image.dtype == jnp.uint8 for image in images)
        x = nn.cast(jnp.concatenate(images, axis=-1), force=True) / 255 - 0.5
        x = x.reshape((-1, *x.shape[bdims:]))
        batch, _, _, channels = x.shape
        grid_height, grid_width, _ = self.image_grid_shape()
        token_count = grid_height * grid_width
        patch = self.patch
        x = x.reshape((batch, grid_height, patch, grid_width, patch, channels))
        x = x.transpose((0, 1, 3, 2, 4, 5)).reshape(
            (batch, token_count, patch * patch * channels)
        )
        x = self.sub("patch_projection", nn.Linear, self.model, winit=self.winit)(x)
        position = self.value(
            "position",
            nn.init("trunc_normal"),
            (token_count, self.model),
            jnp.float32,
        )
        x = x + nn.cast(position)[None]

        for index in range(self.layers):
            with nj.scope(f"layer{index}"):
                residual = x
                normed = self.sub("attention_norm", nn.Norm, self.norm)(x)
                qkv = self.sub("qkv", nn.Linear, 3 * self.model, winit=self.winit)(
                    normed
                )
                qkv = qkv.reshape(
                    (
                        batch,
                        token_count,
                        3,
                        self.heads,
                        self.model // self.heads,
                    )
                )
                query, key, value = [qkv[:, :, part] for part in range(3)]
                logits = jnp.einsum("bnhd,bmhd->bhnm", query, key)
                logits = jnp.float32(logits) / math.sqrt(key.shape[-1])
                weights = jax.nn.softmax(logits, axis=-1).astype(x.dtype)
                attended = jnp.einsum("bhnm,bmhd->bnhd", weights, value)
                attended = attended.reshape((batch, token_count, self.model))
                x = residual + self.sub(
                    "attention_out", nn.Linear, self.model, winit=self.winit
                )(attended)

                residual = x
                x = self.sub("ffn_norm", nn.Norm, self.norm)(x)
                x = self.sub(
                    "ffn_in",
                    nn.Linear,
                    self.model * self.ffup,
                    winit=self.winit,
                )(x)
                x = nn.act(self.act)(x)
                x = self.sub("ffn_out", nn.Linear, self.model, winit=self.winit)(x)
                x = residual + x

        x = self.sub("output_norm", nn.Norm, self.norm)(x)
        if x.shape[-1] != self.token_dim:
            x = self.sub(
                "token_projection", nn.Linear, self.token_dim, winit=self.winit
            )(x)
        return x

    def calculate_encoder_output_dim(self):
        total_dim = self.units if self.veckeys else 0
        if self.imgkeys:
            height, width, depth = self.image_grid_shape()
            total_dim += height * width * depth
        return int(total_dim)

    def image_grid_shape(self):
        if not self.imgkeys:
            raise ValueError("encoder has no image observations")
        spatial_shapes = {self.obs_space[key].shape[:2] for key in self.imgkeys}
        if len(spatial_shapes) != 1:
            raise ValueError(
                "all image observations must share one spatial shape, got "
                f"{sorted(spatial_shapes)}"
            )
        height, width = next(iter(spatial_shapes))
        if height % self.patch or width % self.patch:
            raise ValueError(
                "image dimensions must be divisible by ViT patch size, got "
                f"image={(height, width)}, patch={self.patch}"
            )
        return height // self.patch, width // self.patch, int(self.token_dim)

    def spatial_tokens(self, tokens):
        height, width, depth = self.image_grid_shape()
        offset = self.units if self.veckeys else 0
        image_width = height * width * depth
        image_values = tokens[..., offset : offset + image_width]
        return image_values.reshape((*tokens.shape[:-1], height * width, depth))


class Decoder(nj.Module):
    """Official DreamerV3 observation decoder used as a control objective."""

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
        assert all(len(space.shape) <= 3 for space in obs_space.values()), obs_space
        self.obs_space = obs_space
        self.veckeys = [
            key for key, space in obs_space.items() if len(space.shape) <= 2
        ]
        self.imgkeys = [
            key for key, space in obs_space.items() if len(space.shape) == 3
        ]
        self.depths = tuple(self.depth * multiplier for multiplier in self.mults)
        self.imgdep = sum(obs_space[key].shape[-1] for key in self.imgkeys)
        self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
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

    def __call__(self, carry, feat, reset, training, single=False):
        del training, single
        assert feat["deter"].shape[-1] % self.bspace == 0
        kernel = self.kernel
        recons = {}
        bshape = reset.shape
        inp = [nn.cast(feat[key]) for key in ("stoch", "deter")]
        inp = [value.reshape((math.prod(bshape), -1)) for value in inp]
        inp = jnp.concatenate(inp, -1)

        if self.veckeys:
            spaces = {key: self.obs_space[key] for key in self.veckeys}
            outputs = {
                key: "categorical"
                if space.discrete
                else ("symlog_mse" if self.symlog else "mse")
                for key, space in spaces.items()
            }
            kwargs = dict(**self.kw, act=self.act, norm=self.norm)
            value = self.sub("mlp", nn.MLP, self.layers, self.units, **kwargs)(inp)
            value = value.reshape((*bshape, *value.shape[1:]))
            kwargs = dict(**self.kw, outscale=self.outscale)
            recons.update(
                self.sub("vec", embodied.jax.DictHead, spaces, outputs, **kwargs)(value)
            )

        if self.imgkeys:
            factor = 2 ** (len(self.depths) - int(bool(self.outer)))
            minres = [int(value // factor) for value in self.imgres]
            assert 3 <= minres[0] <= 16, minres
            assert 3 <= minres[1] <= 16, minres
            shape = (*minres, self.depths[-1])
            if self.bspace:
                units, groups = math.prod(shape), self.bspace
                deter, stoch = nn.cast((feat["deter"], feat["stoch"]))
                stoch = stoch.reshape((*stoch.shape[:-2], -1))
                deter = deter.reshape((-1, deter.shape[-1]))
                stoch = stoch.reshape((-1, stoch.shape[-1]))
                deter = self.sub("sp0", nn.BlockLinear, units, groups, **self.kw)(deter)
                deter = einops.rearrange(
                    deter,
                    "... (g h w c) -> ... h w (g c)",
                    h=minres[0],
                    w=minres[1],
                    g=groups,
                )
                stoch = self.sub("sp1", nn.Linear, 2 * self.units, **self.kw)(stoch)
                stoch = nn.act(self.act)(self.sub("sp1norm", nn.Norm, self.norm)(stoch))
                stoch = self.sub("sp2", nn.Linear, shape, **self.kw)(stoch)
                value = nn.act(self.act)(
                    self.sub("spnorm", nn.Norm, self.norm)(deter + stoch)
                )
            else:
                kwargs = dict(**self.kw, act=self.act, norm=self.norm)
                value = self.sub("space", nn.Linear, shape, **kwargs)(inp)
                value = nn.act(self.act)(
                    self.sub("spacenorm", nn.Norm, self.norm)(value)
                )
            for index, depth in reversed(list(enumerate(self.depths[:-1]))):
                if self.strided:
                    kwargs = dict(**self.kw, transp=True)
                    value = self.sub(
                        f"conv{index}", nn.Conv2D, depth, kernel, 2, **kwargs
                    )(value)
                else:
                    value = value.repeat(2, -2).repeat(2, -3)
                    value = self.sub(
                        f"conv{index}", nn.Conv2D, depth, kernel, **self.kw
                    )(value)
                value = nn.act(self.act)(
                    self.sub(f"conv{index}norm", nn.Norm, self.norm)(value)
                )
            if self.outer:
                kwargs = dict(**self.kw, outscale=self.outscale)
                value = self.sub("imgout", nn.Conv2D, self.imgdep, kernel, **kwargs)(
                    value
                )
            elif self.strided:
                kwargs = dict(**self.kw, outscale=self.outscale, transp=True)
                value = self.sub("imgout", nn.Conv2D, self.imgdep, kernel, 2, **kwargs)(
                    value
                )
            else:
                value = value.repeat(2, -2).repeat(2, -3)
                kwargs = dict(**self.kw, outscale=self.outscale)
                value = self.sub("imgout", nn.Conv2D, self.imgdep, kernel, **kwargs)(
                    value
                )
            value = jax.nn.sigmoid(value)
            value = value.reshape((*bshape, *value.shape[1:]))
            split = np.cumsum(
                [self.obs_space[key].shape[-1] for key in self.imgkeys][:-1]
            )
            for key, output in zip(self.imgkeys, jnp.split(value, split, -1)):
                output = embodied.jax.outs.MSE(output)
                recons[key] = embodied.jax.outs.Agg(output, 3, jnp.sum)

        return carry, {}, recons
