"""Jafar ST-ViViT VQ-VAE tokenizer.

Adapted from ``FLAIROx/jafar`` at commit
``5ff9fc7d5d744c8c2797ba3ad0a095ed7f2e2665``, path
``models/tokenizer.py``. Integration changes: package-qualified imports and
typed dictionaries; architecture, tensor layout, quantization, and sigmoid
decode behavior are preserved.
"""

from flax import linen as nn
import jax

from world_marl.jafar.nn import STTransformer, VectorQuantizer
from world_marl.jafar.preprocess import patchify, unpatchify


class TokenizerVQVAE(nn.Module):
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
        self.encoder = STTransformer(
            self.model_dim,
            self.latent_dim,
            self.num_blocks,
            self.num_heads,
            self.dropout,
        )
        self.vq = VectorQuantizer(
            self.latent_dim,
            self.num_latents,
            self.codebook_dropout,
        )
        self.out_dim = self.in_dim * self.patch_size**2
        self.decoder = STTransformer(
            self.model_dim,
            self.out_dim,
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
        reconstruction = nn.sigmoid(self.decoder(outputs["z_q"]))
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
        num_patches = patches.shape[2]
        embeddings = self.encoder(patches)
        embeddings = embeddings.reshape(
            batch_size * sequence_length * num_patches,
            self.latent_dim,
        )
        z_q, codes, embeddings, indices = self.vq(embeddings, training)
        z_q = z_q.reshape(
            batch_size,
            sequence_length,
            num_patches,
            self.latent_dim,
        )
        indices = indices.reshape(batch_size, sequence_length, num_patches)
        return {"z_q": z_q, "z": codes, "emb": embeddings, "indices": indices}

    def decode(
        self,
        indices: jax.Array,
        video_hw: tuple[int, int],
    ) -> jax.Array:
        latents = self.vq.codebook[indices]
        reconstruction = nn.sigmoid(self.decoder(latents))
        return unpatchify(reconstruction, self.patch_size, *video_hw)
