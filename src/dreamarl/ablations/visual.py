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
    pool: str = "tokens"
    position: str = "learned"
    hierarchical: tuple = ()
    modality_embedding: bool = False

    def __init__(self, obs_space, **kw):
        assert all(len(space.shape) <= 3 for space in obs_space.values()), obs_space
        if self.model % self.heads:
            raise ValueError("ViT model width must be divisible by attention heads")
        if self.pool not in {"tokens", "cls"}:
            raise ValueError("ViT pooling must be tokens or cls")
        if self.position not in {"learned", "rope3d"}:
            raise ValueError("ViT position must be learned or rope3d")
        if self.pool == "cls" and self.position != "learned":
            raise ValueError("CLS pooling requires learned positional embeddings")
        if any(index < 0 or index >= self.layers for index in self.hierarchical):
            raise ValueError("hierarchical ViT layers must be valid block indices")
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
            x = self._encode_image_tokens(obs, bdims=bdims, project=True)
            outs.append(x.reshape((x.shape[0], -1)))

        tokens = jnp.concatenate(outs, axis=-1).reshape((*bshape, -1))
        return carry, {}, tokens

    @staticmethod
    def _rotate_axis(values, positions, start, width):
        """Apply the adjacent-pair RoPE convention used by V-JEPA 2.1."""

        if not width:
            return values
        section = values[..., start : start + width]
        omega = jnp.arange(width // 2, dtype=jnp.float32) / (width / 2.0)
        omega = 1.0 / jnp.power(10000.0, omega)
        angles = positions.astype(jnp.float32)[None, :, None, None] * omega
        sine = jnp.repeat(jnp.sin(angles), 2, axis=-1).astype(values.dtype)
        cosine = jnp.repeat(jnp.cos(angles), 2, axis=-1).astype(values.dtype)
        paired = section.reshape((*section.shape[:-1], width // 2, 2))
        rotated = jnp.stack((-paired[..., 1], paired[..., 0]), axis=-1)
        rotated = rotated.reshape(section.shape)
        section = section * cosine + rotated * sine
        return jnp.concatenate(
            [values[..., :start], section, values[..., start + width :]], axis=-1
        )

    def _rope3d(self, values, grid_height, grid_width):
        """V-JEPA 2.1 RoPE with zero time and height/width positions."""

        head_dim = values.shape[-1]
        axis_dim = 2 * ((head_dim // 3) // 2)
        token_ids = jnp.arange(grid_height * grid_width)
        positions = (
            jnp.zeros_like(token_ids),
            token_ids // grid_width,
            token_ids % grid_width,
        )
        for axis, position in enumerate(positions):
            values = self._rotate_axis(
                values, position, axis * axis_dim, axis_dim
            )
        return values

    def _encode_image_tokens(self, obs, *, bdims, visible=None, project=True):
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
        if self.pool == "cls":
            cls = self.value(
                "cls_token", nn.init("trunc_normal"), (1, self.model), jnp.float32
            )
            cls = jnp.broadcast_to(nn.cast(cls), (batch, 1, self.model))
            x = jnp.concatenate([cls, x], axis=1)
        sequence_count = x.shape[1]
        if self.position == "learned":
            position = self.value(
                "position",
                nn.init("trunc_normal"),
                (sequence_count, self.model),
                jnp.float32,
            )
            x = x + nn.cast(position)[None]
        if self.modality_embedding:
            modality = self.value(
                "image_modality",
                nn.Initializer("normal", "none", 1e-6),
                (1, self.model),
                jnp.float32,
            )
            x = x + nn.cast(modality)[None]

        valid = None
        if visible is not None:
            valid = visible.reshape((batch, token_count)).astype(bool)
            if self.pool == "cls":
                valid = jnp.concatenate([jnp.ones((batch, 1), bool), valid], axis=1)
            x = jnp.where(valid[..., None], x, jnp.zeros_like(x))

        hierarchical = []
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
                        sequence_count,
                        3,
                        self.heads,
                        self.model // self.heads,
                    )
                )
                query, key, value = [qkv[:, :, part] for part in range(3)]
                if self.position == "rope3d":
                    query = self._rope3d(query, grid_height, grid_width)
                    key = self._rope3d(key, grid_height, grid_width)
                logits = jnp.einsum("bnhd,bmhd->bhnm", query, key)
                logits = jnp.float32(logits) / math.sqrt(key.shape[-1])
                if valid is not None:
                    logits = jnp.where(valid[:, None, None, :], logits, -1e30)
                weights = jax.nn.softmax(logits, axis=-1).astype(x.dtype)
                attended = jnp.einsum("bhnm,bmhd->bnhd", weights, value)
                attended = attended.reshape((batch, sequence_count, self.model))
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
                if valid is not None:
                    x = jnp.where(valid[..., None], x, jnp.zeros_like(x))
                if index in self.hierarchical:
                    hierarchical.append(
                        self.sub(f"hierarchical_norm{index}", nn.Norm, self.norm)(x)
                    )

        if hierarchical:
            x = jnp.concatenate(hierarchical, axis=-1)
        else:
            x = self.sub("output_norm", nn.Norm, self.norm)(x)
        if self.pool == "cls":
            x = x[:, 0]
        if project and x.shape[-1] != self.token_dim:
            x = self.sub("token_projection", nn.Linear, self.token_dim, winit=self.winit)(x)
        if valid is not None:
            x = jnp.where(valid[..., None], x, jnp.zeros_like(x))
        return x

    def visible_spatial_tokens(self, obs, reset, visible, *, single=False):
        """Encode only visible context tokens while preserving static shapes."""

        bdims = 1 if single else 2
        expected = (*reset.shape, *self.image_grid_shape()[:2])
        if visible.shape != expected:
            raise ValueError(
                f"visible token mask must have shape {expected}, got {visible.shape}"
            )
        if self.pool != "tokens":
            raise ValueError("visible spatial tokens require token pooling")
        encoded = self._encode_image_tokens(
            obs, bdims=bdims, visible=visible, project=not bool(self.hierarchical)
        )
        return encoded.reshape((*reset.shape, encoded.shape[-2], encoded.shape[-1]))

    def full_predictor_tokens(self, obs, reset, *, single=False):
        """Return full hierarchical targets for the V-JEPA 2.1 predictor."""

        if self.pool != "tokens":
            raise ValueError("predictor tokens require token pooling")
        bdims = 1 if single else 2
        encoded = self._encode_image_tokens(obs, bdims=bdims, project=False)
        return encoded.reshape((*reset.shape, encoded.shape[-2], encoded.shape[-1]))

    @property
    def predictor_token_dim(self):
        return self.model * len(self.hierarchical) if self.hierarchical else self.token_dim

    def calculate_encoder_output_dim(self):
        total_dim = self.units if self.veckeys else 0
        if self.imgkeys:
            height, width, depth = self.image_grid_shape()
            total_dim += depth if self.pool == "cls" else height * width * depth
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
        if self.pool != "tokens":
            raise ValueError("CLS encoders do not expose flattened spatial tokens")
        height, width, depth = self.image_grid_shape()
        offset = self.units if self.veckeys else 0
        image_width = height * width * depth
        image_values = tokens[..., offset : offset + image_width]
        return image_values.reshape((*tokens.shape[:-1], height * width, depth))


class SpatialTokenPredictor(nj.Module):
    """Predict target tokens from visible context tokens and position queries."""

    grid: tuple = (14, 14)
    input_dim: int = 64
    model: int = 384
    layers: int = 12
    heads: int = 12
    ffup: int = 4
    norm: str = "layer1em6"
    act: str = "gelu"
    winit: str = "trunc_normal_in"

    def __init__(self, **kw):
        if self.model % self.heads:
            raise ValueError("predictor model width must divide attention heads")
        self.kw = kw

    def __call__(self, context, visible, target):
        leading = context.shape[:-2]
        token_count = math.prod(self.grid)
        if context.shape[-2:] != (token_count, self.input_dim):
            raise ValueError(
                "context token shape must end in "
                f"{(token_count, self.input_dim)}, got {context.shape}"
            )
        expected_mask = (*leading, *self.grid)
        if visible.shape != expected_mask or target.shape != expected_mask:
            raise ValueError(
                "predictor masks must match context leading dimensions and grid"
            )

        batch = math.prod(leading)
        context = nn.cast(context.reshape((batch, token_count, self.input_dim)))
        visible = visible.reshape((batch, token_count)).astype(bool)
        target = target.reshape((batch, token_count)).astype(bool)
        context = self.sub(
            "context_projection", nn.Linear, self.model, winit=self.winit
        )(context)
        mask_token = self.value(
            "mask_token",
            nn.init("zeros"),
            (1, self.model),
            jnp.float32,
        )
        context = jnp.where(visible[..., None], context, jnp.zeros_like(context))
        queries = jnp.where(target[..., None], nn.cast(mask_token)[None], 0.0)
        x = jnp.concatenate([context, queries], axis=1)
        valid = jnp.concatenate([visible, target], axis=1)

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
                        2 * token_count,
                        3,
                        self.heads,
                        self.model // self.heads,
                    )
                )
                query, key, value = [qkv[:, :, part] for part in range(3)]
                query = self._rope_queries(query, token_count)
                key = self._rope_queries(key, token_count)
                logits = jnp.einsum("bnhd,bmhd->bhnm", query, key)
                logits = jnp.float32(logits) / math.sqrt(key.shape[-1])
                logits = jnp.where(valid[:, None, None, :], logits, -1e30)
                weights = jax.nn.softmax(logits, axis=-1).astype(x.dtype)
                attended = jnp.einsum("bhnm,bmhd->bnhd", weights, value)
                attended = attended.reshape((batch, 2 * token_count, self.model))
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
                x = jnp.where(valid[..., None], x, jnp.zeros_like(x))

        context_output = self.sub("output_norm", nn.Norm, self.norm)(x[:, :token_count])
        query_output = self.sub("output_norm", nn.Norm, self.norm)(x[:, token_count:])
        prediction = self.sub(
            "target_projection", nn.Linear, self.input_dim, winit=self.winit
        )(query_output)
        context_prediction = self.sub(
            "target_projection", nn.Linear, self.input_dim, winit=self.winit
        )(context_output)
        prediction = jnp.where(
            target[..., None], prediction, jnp.zeros_like(prediction)
        )
        context_prediction = jnp.where(
            visible[..., None], context_prediction, jnp.zeros_like(context_prediction)
        )
        return (
            prediction.reshape((*leading, token_count, self.input_dim)),
            context_prediction.reshape((*leading, token_count, self.input_dim)),
        )

    def _rope_queries(self, values, token_count):
        """Apply official 3-axis RoPE to context and target token copies."""

        head_dim = values.shape[-1]
        axis_dim = 2 * ((head_dim // 3) // 2)
        ids = jnp.tile(jnp.arange(token_count), 2)
        positions = (jnp.zeros_like(ids), ids // self.grid[1], ids % self.grid[1])
        for axis, position in enumerate(positions):
            values = ViTEncoder._rotate_axis(
                values, position, axis * axis_dim, axis_dim
            )
        return values


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
