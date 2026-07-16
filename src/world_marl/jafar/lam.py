"""Jafar discrete latent-action model.

Adapted from ``FLAIROx/jafar`` at commit
``5ff9fc7d5d744c8c2797ba3ad0a095ed7f2e2665``, path ``models/lam.py``.
Integration changes: package-qualified imports and typed dictionaries; action
token placement, future-frame state selection, VQ, and decoder equations are
preserved.
"""

from flax import linen as nn
import jax
import jax.numpy as jnp

from world_marl.jafar.nn import STTransformer, VectorQuantizer
from world_marl.jafar.preprocess import patchify, unpatchify


class LatentActionModel(nn.Module):
    in_dim: int
    model_dim: int
    latent_dim: int
    num_latents: int
    patch_size: int
    num_blocks: int
    num_heads: int
    dropout: float
    codebook_dropout: float

    def setup(self) -> None:
        self.patch_token_dim = self.in_dim * self.patch_size**2
        self.encoder = STTransformer(
            self.model_dim,
            self.latent_dim,
            self.num_blocks,
            self.num_heads,
            self.dropout,
        )
        self.action_in = self.param(
            "action_in",
            nn.initializers.lecun_uniform(),
            (1, 1, 1, self.patch_token_dim),
        )
        self.vq = VectorQuantizer(
            self.latent_dim,
            self.num_latents,
            self.codebook_dropout,
        )
        self.patch_up = nn.Dense(self.model_dim)
        self.action_up = nn.Dense(self.model_dim)
        self.decoder = STTransformer(
            self.model_dim,
            self.patch_token_dim,
            self.num_blocks,
            self.num_heads,
            self.dropout,
        )

    def __call__(
        self,
        batch: dict[str, jax.Array],
        training: bool = True,
    ) -> dict[str, jax.Array]:
        height, width = batch["videos"].shape[2:4]
        outputs = self.vq_encode(batch["videos"], training)
        video_action_patches = self.action_up(outputs["z_q"]) + self.patch_up(
            outputs["patches"][:, :-1]
        )
        del outputs["patches"]
        reconstruction = nn.sigmoid(self.decoder(video_action_patches))
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
