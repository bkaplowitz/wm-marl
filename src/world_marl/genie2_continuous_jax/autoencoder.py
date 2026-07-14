"""Continuous frame representations for the Genie2 public-source baseline."""

from __future__ import annotations

import math
from typing import Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp

from world_marl.genie2_continuous_jax.config import AutoencoderConfig
from world_marl.genie2_continuous_jax.st_transformer import AxialTransformer


def patchify(videos: jax.Array, patch_size: int) -> jax.Array:
    if videos.ndim != 5:
        raise ValueError("videos must have shape (batch,time,height,width,channels)")
    batch, time, height, width, channels = videos.shape
    if height % patch_size or width % patch_size:
        raise ValueError("height and width must be divisible by patch_size")
    grid_height, grid_width = height // patch_size, width // patch_size
    patches = videos.reshape(
        batch,
        time,
        grid_height,
        patch_size,
        grid_width,
        patch_size,
        channels,
    )
    patches = patches.transpose((0, 1, 2, 4, 3, 5, 6))
    return patches.reshape(
        batch,
        time,
        grid_height * grid_width,
        patch_size * patch_size * channels,
    )


def unpatchify(
    patches: jax.Array,
    patch_size: int,
    height: int,
    width: int,
    channels: int,
) -> jax.Array:
    batch, time, num_patches, patch_dim = patches.shape
    grid_height, grid_width = height // patch_size, width // patch_size
    expected_dim = patch_size * patch_size * channels
    if num_patches != grid_height * grid_width or patch_dim != expected_dim:
        raise ValueError("patch tensor does not match the requested image shape")
    videos = patches.reshape(
        batch,
        time,
        grid_height,
        grid_width,
        patch_size,
        patch_size,
        channels,
    )
    videos = videos.transpose((0, 1, 2, 4, 3, 5, 6))
    return videos.reshape((batch, time, height, width, channels))


class ContinuousVideoTokenizer(nn.Module):
    """Jasmine-style ST-ViViT masked autoencoder without a codebook."""

    config: AutoencoderConfig

    def setup(self) -> None:
        patch_dim = self.config.patch_size**2 * 3
        self.encoder = AxialTransformer(
            input_dim=patch_dim,
            model_dim=self.config.model_dim,
            ffn_dim=self.config.ffn_dim,
            output_dim=self.config.latent_patch_dim,
            num_blocks=self.config.num_blocks,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            temporal_causal=True,
            spatial_causal=False,
            compute_dtype=self.config.compute_dtype,
            parameter_dtype=self.config.parameter_dtype,
            name="encoder",
        )
        self.decoder = AxialTransformer(
            input_dim=self.config.latent_patch_dim,
            model_dim=self.config.model_dim,
            ffn_dim=self.config.ffn_dim,
            output_dim=patch_dim,
            num_blocks=self.config.num_blocks,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            temporal_causal=True,
            spatial_causal=False,
            compute_dtype=self.config.compute_dtype,
            parameter_dtype=self.config.parameter_dtype,
            name="decoder",
        )
        self.mask_patch = self.param(
            "mask_patch",
            nn.initializers.lecun_uniform(),
            (1, 1, 1, patch_dim),
        )

    def encode(
        self,
        videos: jax.Array,
        *,
        key: jax.Array | None = None,
        training: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        patches = patchify(videos.astype(jnp.float32), self.config.patch_size)
        batch, time, num_patches, _ = patches.shape
        mask = jnp.zeros((batch, time, num_patches), dtype=bool)
        if training and self.config.max_mask_ratio > 0.0:
            if key is None:
                raise ValueError("training tokenizer encoding requires a PRNG key")
            probability_key, mask_key = jax.random.split(key)
            probabilities = jax.random.uniform(
                probability_key,
                (batch * time,),
                minval=0.0,
                maxval=self.config.max_mask_ratio,
            )
            keys = jax.random.split(mask_key, batch * time)
            mask = jax.vmap(
                lambda sample_key, probability: jax.random.bernoulli(
                    sample_key,
                    probability,
                    (num_patches,),
                )
            )(keys, probabilities).reshape((batch, time, num_patches))
            patches = jnp.where(mask[..., None], self.mask_patch, patches)
        latents = jnp.tanh(self.encoder(patches, training=training))
        return latents, mask

    def decode(
        self,
        latents: jax.Array,
        *,
        video_shape: tuple[int, int, int],
        training: bool = False,
    ) -> jax.Array:
        height, width, channels = video_shape
        patches = jax.nn.sigmoid(self.decoder(latents, training=training))
        return unpatchify(
            patches.astype(jnp.float32),
            self.config.patch_size,
            height,
            width,
            channels,
        )

    def __call__(
        self,
        videos: jax.Array,
        *,
        key: jax.Array | None = None,
        training: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if videos.shape[-1] != 3:
            raise ValueError("the Jasmine tokenizer substitution expects RGB frames")
        latents, mask = self.encode(videos, key=key, training=training)
        reconstructions = self.decode(
            latents,
            video_shape=tuple(int(value) for value in videos.shape[-3:]),
            training=training,
        )
        return latents, reconstructions, mask


class ContinuousLatentAutoencoder(nn.Module):
    """Vector-observation adapter extension; not the visual Genie2 tokenizer."""

    latent_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
        *,
        decode_latents: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if observations.ndim < 2:
            raise ValueError("observations must have shape (batch, ...)")
        obs_shape = observations.shape[1:]
        x = observations.astype(jnp.float32).reshape((observations.shape[0], -1))
        for dim in self.hidden_dims:
            x = nn.silu(nn.Dense(dim)(x))
        latents = jnp.tanh(nn.Dense(self.latent_dim, name="latent")(x))
        y = latents if decode_latents is None else decode_latents.astype(jnp.float32)
        for dim in reversed(tuple(self.hidden_dims)):
            y = nn.silu(nn.Dense(dim)(y))
        recon_flat = nn.Dense(math.prod(obs_shape), name="decoder")(y)
        reconstructions = recon_flat.reshape((observations.shape[0], *obs_shape))
        return latents, reconstructions


def reconstruction_loss(targets: jax.Array, reconstructions: jax.Array) -> jax.Array:
    if targets.shape != reconstructions.shape:
        raise ValueError("targets and reconstructions must share shape")
    return jnp.mean(jnp.square(targets.astype(jnp.float32) - reconstructions))
