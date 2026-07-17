"""Jasmine discrete latent-action model.

Adapted from ``p-doom/jasmine`` at commit
``420859bc99eecf6b07a7e9edf65d5d145935f1e1``, path
``jasmine/models/lam.py``. Integration changes: NNX modules/state are
translated to Linen parameter collections; action-token layout, future-frame
selection, initializers, dtypes, VQ, and decoder equations are preserved.
"""

from flax import linen as nn
import jax
import jax.numpy as jnp

from world_marl.jasmine.nn import AxialTransformer, VectorQuantizer
from world_marl.jasmine.preprocess import patchify, unpatchify


class LatentActionModel(nn.Module):
    in_dim: int
    model_dim: int
    ffn_dim: int
    latent_dim: int
    num_latents: int
    patch_size: int
    num_blocks: int
    num_heads: int
    dropout: float
    codebook_dropout: float
    param_dtype: jnp.dtype
    dtype: jnp.dtype
    use_flash_attention: bool

    def setup(self) -> None:
        self.patch_token_dim = self.in_dim * self.patch_size**2
        self.encoder = AxialTransformer(
            input_dim=self.patch_token_dim,
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
        self.action_in = self.param(
            "action_in",
            nn.initializers.lecun_uniform(),
            (1, 1, 1, self.patch_token_dim),
        )
        self.vq = VectorQuantizer(
            latent_dim=self.latent_dim,
            num_latents=self.num_latents,
            dropout=self.codebook_dropout,
            dtype=self.dtype,
        )
        self.patch_up = nn.Dense(
            self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.action_up = nn.Dense(
            self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.decoder = AxialTransformer(
            input_dim=self.model_dim,
            model_dim=self.model_dim,
            ffn_dim=self.ffn_dim,
            out_dim=self.patch_token_dim,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            use_flash_attention=self.use_flash_attention,
            spatial_causal=False,
            temporal_causal=True,
        )

    def __call__(
        self,
        batch: dict[str, jax.Array],
        training: bool = True,
    ) -> dict[str, jax.Array]:
        height, width = batch["videos"].shape[2:4]
        outputs = self.vq_encode(batch["videos"], training)
        patches = outputs["patches"]
        actions = self.action_up(outputs["z_q"])
        projected_patches = self.patch_up(patches[:, :-1])
        actions = jnp.broadcast_to(actions, projected_patches.shape)
        del outputs["patches"]

        reconstruction = self.decoder(actions + projected_patches)
        reconstruction = nn.sigmoid(reconstruction.astype(jnp.float32))
        reconstruction = reconstruction.astype(self.dtype)
        outputs["recon"] = unpatchify(
            reconstruction,
            self.patch_size,
            height,
            width,
        )
        return outputs

    def vq_encode(
        self,
        videos: jax.Array,
        training: bool = True,
    ) -> dict[str, jax.Array]:
        batch_size, sequence_length = videos.shape[:2]
        patches = patchify(videos, self.patch_size)
        action_pad = jnp.broadcast_to(
            self.action_in,
            (batch_size, sequence_length, 1, self.patch_token_dim),
        )
        padded_patches = jnp.concatenate((action_pad, patches), axis=2)
        encoded = self.encoder(padded_patches)
        encoded = encoded[:, 1:, 0]
        encoded = encoded.reshape(
            batch_size * (sequence_length - 1),
            self.latent_dim,
        )
        z_q, codes, embeddings, indices = self.vq(encoded, training)
        z_q = z_q.reshape(
            batch_size,
            sequence_length - 1,
            1,
            self.latent_dim,
        )
        return {
            "patches": patches,
            "z_q": z_q,
            "z": codes,
            "emb": embeddings,
            "indices": indices,
        }
