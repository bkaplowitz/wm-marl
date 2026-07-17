"""Jasmine continuous masked-autoencoder tokenizer.

Adapted from ``p-doom/jasmine`` at commit
``420859bc99eecf6b07a7e9edf65d5d145935f1e1``, path
``jasmine/models/tokenizer.py`` (``TokenizerMAE``). Integration changes: NNX
state is translated to Linen parameters and per-frame mask sampling is exposed
as a pure helper; layouts, initializers, tanh boundary, and float32 sigmoid
decode are preserved.
"""

from flax import linen as nn
import jax
import jax.numpy as jnp

from world_marl.jasmine.nn import AxialTransformer
from world_marl.jasmine.preprocess import patchify, unpatchify


def sample_patch_mask(
    rng: jax.Array,
    batch_size: int,
    sequence_length: int,
    num_patches: int,
    max_mask_ratio: float,
) -> tuple[jax.Array, jax.Array]:
    probability_rng, mask_rng = jax.random.split(rng)
    probabilities = jax.random.uniform(
        probability_rng,
        shape=(batch_size * sequence_length,),
        minval=0.0,
        maxval=max_mask_ratio,
    )
    mask = jax.vmap(
        lambda key, probability: jax.random.bernoulli(
            key,
            probability,
            (num_patches,),
        )
    )(
        jax.random.split(mask_rng, batch_size * sequence_length),
        probabilities,
    )
    return (
        mask.reshape(batch_size, sequence_length, num_patches),
        probabilities.reshape(batch_size, sequence_length),
    )


class TokenizerMAE(nn.Module):
    in_dim: int
    model_dim: int
    ffn_dim: int
    latent_dim: int
    num_latents: int
    patch_size: int
    num_blocks: int
    num_heads: int
    dropout: float
    max_mask_ratio: float
    param_dtype: jnp.dtype
    dtype: jnp.dtype
    use_flash_attention: bool

    def setup(self) -> None:
        self.encoder = AxialTransformer(
            input_dim=self.in_dim * self.patch_size**2,
            model_dim=self.model_dim,
            ffn_dim=self.ffn_dim,
            out_dim=self.latent_dim,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            use_flash_attention=self.use_flash_attention,
            spatial_causal=False,
            temporal_causal=True,
        )
        self.out_dim = self.in_dim * self.patch_size**2
        self.decoder = AxialTransformer(
            input_dim=self.latent_dim,
            model_dim=self.model_dim,
            ffn_dim=self.ffn_dim,
            out_dim=self.out_dim,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            use_flash_attention=self.use_flash_attention,
            spatial_causal=False,
            temporal_causal=True,
        )
        self.mask_patch = self.param(
            "mask_patch",
            nn.initializers.lecun_uniform(),
            (1, 1, 1, self.out_dim),
        )

    def __call__(
        self,
        batch: dict[str, jax.Array],
        training: bool = True,
    ) -> dict[str, jax.Array]:
        height, width = batch["videos"].shape[2:4]
        outputs = self.mask_and_encode(
            batch["videos"],
            batch["rng"],
            training,
        )
        reconstruction = self.decoder(outputs["z"])
        reconstruction = nn.sigmoid(reconstruction.astype(jnp.float32))
        reconstruction = reconstruction.astype(self.dtype)
        outputs["recon"] = unpatchify(
            reconstruction,
            self.patch_size,
            height,
            width,
        )
        return outputs

    def mask_and_encode(
        self,
        videos: jax.Array,
        rng: jax.Array,
        training: bool = True,
    ) -> dict[str, jax.Array]:
        batch_size, sequence_length = videos.shape[:2]
        patches = patchify(videos, self.patch_size)
        if training:
            mask, _ = sample_patch_mask(
                rng,
                batch_size,
                sequence_length,
                patches.shape[2],
                self.max_mask_ratio,
            )
            patches = jnp.where(mask[..., None], self.mask_patch, patches)
        latents = nn.tanh(self.encoder(patches))
        return {"z": latents}

    def decode(
        self,
        latents: jax.Array,
        video_hw: tuple[int, int],
    ) -> jax.Array:
        reconstruction = self.decoder(latents)
        reconstruction = nn.sigmoid(reconstruction.astype(jnp.float32))
        reconstruction = reconstruction.astype(self.dtype)
        return unpatchify(reconstruction, self.patch_size, *video_hw)
