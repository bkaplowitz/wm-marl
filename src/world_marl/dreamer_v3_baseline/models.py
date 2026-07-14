from __future__ import annotations

import math
from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from world_marl.dreamer_v3_baseline.losses import symlog
from world_marl.dreamer_v3_baseline.rssm import BlockLinear


def _scaled_lecun_normal(scale: float):
    initializer = nn.initializers.lecun_normal()

    def init(key, shape, dtype=jnp.float32):
        return scale * initializer(key, shape, dtype)

    return init


class NormedDense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        x = nn.Dense(self.features)(inputs)
        x = nn.RMSNorm(epsilon=1e-4)(x)
        return nn.silu(x)


class DreamerEncoder(nn.Module):
    observation_shape: tuple[int, ...]
    hidden_dims: Sequence[int] = (256, 256, 256)
    cnn_depth: int = 16
    cnn_multipliers: Sequence[int] = (1, 2, 3, 4, 4)
    cnn_kernel: int = 5
    cnn_outer_stride: int = 1

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        x = observations.astype(jnp.float32)
        if len(self.observation_shape) == 3:
            x = x - 0.5
            for index, multiplier in enumerate(self.cnn_multipliers):
                stride = self.cnn_outer_stride if index == 0 else 2
                x = nn.Conv(
                    self.cnn_depth * multiplier,
                    kernel_size=(self.cnn_kernel, self.cnn_kernel),
                    strides=(stride, stride),
                    padding="SAME",
                    name=f"cnn_{index}",
                )(x)
                x = nn.silu(nn.RMSNorm(epsilon=1e-4, name=f"cnn_norm_{index}")(x))
            return x.reshape((x.shape[0], -1))

        x = symlog(x.reshape((x.shape[0], -1)))
        for index, dim in enumerate(self.hidden_dims):
            x = NormedDense(dim, name=f"mlp_{index}")(x)
        return x


class DreamerDecoder(nn.Module):
    observation_shape: tuple[int, ...]
    hidden_dims: Sequence[int] = (256, 256, 256)
    cnn_depth: int = 16
    cnn_multipliers: Sequence[int] = (1, 2, 3, 4, 4)
    cnn_kernel: int = 5
    cnn_outer_stride: int = 1
    deterministic_size: int | None = None
    stochastic_size: int | None = None
    discrete_classes: int | None = None
    blocks: int = 8
    hidden_size: int = 256

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        x = features.astype(jnp.float32)
        if len(self.observation_shape) == 3:
            height, width, channels = self.observation_shape
            scale = self.cnn_outer_stride * 2 ** (len(self.cnn_multipliers) - 1)
            if height % scale or width % scale:
                raise ValueError(
                    f"image dimensions must be divisible by {scale}, got "
                    f"{self.observation_shape}"
                )
            min_height, min_width = height // scale, width // scale
            depths = tuple(self.cnn_depth * x for x in self.cnn_multipliers)
            flat_size = min_height * min_width * depths[-1]
            if (
                self.deterministic_size is None
                or self.stochastic_size is None
                or self.discrete_classes is None
            ):
                raise ValueError("image decoder requires explicit RSSM dimensions")
            deterministic = x[..., : self.deterministic_size]
            stochastic = x[..., self.deterministic_size :]
            expected_stochastic = self.stochastic_size * self.discrete_classes
            if stochastic.shape[-1] != expected_stochastic:
                raise ValueError(
                    f"expected {expected_stochastic} stochastic features, got "
                    f"{stochastic.shape[-1]}"
                )
            deterministic_space = BlockLinear(
                flat_size,
                self.blocks,
                name="deterministic_space",
            )(deterministic)
            channels_per_block = depths[-1] // self.blocks
            deterministic_space = deterministic_space.reshape(
                (
                    x.shape[0],
                    self.blocks,
                    min_height,
                    min_width,
                    channels_per_block,
                )
            )
            deterministic_space = deterministic_space.transpose((0, 2, 3, 1, 4))
            deterministic_space = deterministic_space.reshape(
                (x.shape[0], min_height, min_width, depths[-1])
            )
            stochastic = NormedDense(
                2 * self.hidden_size,
                name="stochastic_hidden",
            )(stochastic)
            stochastic_space = nn.Dense(
                flat_size,
                name="stochastic_space",
            )(stochastic)
            stochastic_space = stochastic_space.reshape(
                (x.shape[0], min_height, min_width, depths[-1])
            )
            x = nn.silu(
                nn.RMSNorm(epsilon=1e-4, name="space_norm")(
                    deterministic_space + stochastic_space
                )
            )
            for index, depth in enumerate(reversed(depths[:-1])):
                x = nn.ConvTranspose(
                    depth,
                    kernel_size=(self.cnn_kernel, self.cnn_kernel),
                    strides=(2, 2),
                    padding="SAME",
                    name=f"cnn_transpose_{index}",
                )(x)
                x = nn.silu(nn.RMSNorm(epsilon=1e-4, name=f"cnn_norm_{index}")(x))
            x = nn.ConvTranspose(
                channels,
                kernel_size=(self.cnn_kernel, self.cnn_kernel),
                strides=(self.cnn_outer_stride, self.cnn_outer_stride),
                padding="SAME",
                name="image_output",
            )(x)
            return nn.sigmoid(x)

        for index, dim in enumerate(self.hidden_dims):
            x = NormedDense(dim, name=f"mlp_{index}")(x)
        flat_dim = math.prod(self.observation_shape)
        x = nn.Dense(flat_dim, name="vector_output")(x)
        return x.reshape((features.shape[0], *self.observation_shape))


class RewardHead(nn.Module):
    bins: int
    hidden_dims: Sequence[int] = (256,)

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        x = features.astype(jnp.float32)
        for index, dim in enumerate(self.hidden_dims):
            x = NormedDense(dim, name=f"hidden_{index}")(x)
        return nn.Dense(
            self.bins,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="logits",
        )(x)


class ContinueHead(nn.Module):
    hidden_dims: Sequence[int] = (256,)

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        x = features.astype(jnp.float32)
        for index, dim in enumerate(self.hidden_dims):
            x = NormedDense(dim, name=f"hidden_{index}")(x)
        return nn.Dense(1, name="logit")(x)[..., 0]


class DreamerActor(nn.Module):
    action_dim: int
    action_mode: str
    hidden_dims: Sequence[int] = (256, 256, 256)
    min_std: float = 0.1
    max_std: float = 1.0

    @nn.compact
    def __call__(self, features: jax.Array) -> dict[str, jax.Array]:
        x = features.astype(jnp.float32)
        for index, dim in enumerate(self.hidden_dims):
            x = NormedDense(dim, name=f"hidden_{index}")(x)
        if self.action_mode == "discrete":
            return {
                "logits": nn.Dense(
                    self.action_dim,
                    kernel_init=_scaled_lecun_normal(0.01),
                    name="logits",
                )(x)
            }
        mean = jnp.tanh(
            nn.Dense(
                self.action_dim,
                kernel_init=_scaled_lecun_normal(0.01),
                name="mean",
            )(x)
        )
        raw_std = nn.Dense(
            self.action_dim,
            kernel_init=_scaled_lecun_normal(0.01),
            name="stddev",
        )(x)
        stddev = (self.max_std - self.min_std) * jax.nn.sigmoid(raw_std + 2.0)
        stddev = stddev + self.min_std
        return {"mean": mean, "stddev": stddev}


class DreamerCritic(nn.Module):
    bins: int
    hidden_dims: Sequence[int] = (256, 256, 256)

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        x = features.astype(jnp.float32)
        for index, dim in enumerate(self.hidden_dims):
            x = NormedDense(dim, name=f"hidden_{index}")(x)
        return nn.Dense(
            self.bins,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="value_logits",
        )(x)
