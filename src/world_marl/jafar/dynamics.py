"""Jafar MaskGIT dynamics model.

Adapted from ``FLAIROx/jafar`` at commit
``5ff9fc7d5d744c8c2797ba3ad0a095ed7f2e2665``, path
``models/dynamics.py``. Integration changes: package-qualified imports,
testable mask-probability helper, and the approved interpretation of
``mask_limit`` as the upper bound (the pinned source passes it as
``jax.random.uniform(..., minval=mask_limit)``). The first-frame exclusion,
action alignment, embedding, and transformer equations are preserved.
"""

from flax import linen as nn
import jax
import jax.numpy as jnp

from world_marl.jafar.nn import STTransformer


def sample_mask_probability(rng: jax.Array, mask_limit: float) -> jax.Array:
    return jax.random.uniform(rng, minval=0.0, maxval=mask_limit)


class DynamicsMaskGIT(nn.Module):
    model_dim: int
    num_latents: int
    num_blocks: int
    num_heads: int
    dropout: float
    mask_limit: float

    def setup(self) -> None:
        self.dynamics = STTransformer(
            self.model_dim,
            self.num_latents,
            self.num_blocks,
            self.num_heads,
            self.dropout,
        )
        self.patch_embed = nn.Embed(self.num_latents, self.model_dim)
        self.mask_token = self.param(
            "mask_token",
            nn.initializers.lecun_uniform(),
            (1, 1, 1, self.model_dim),
        )
        self.action_up = nn.Dense(self.model_dim)

    def __call__(
        self,
        batch: dict[str, jax.Array],
        training: bool = True,
    ) -> dict[str, jax.Array | None]:
        video_embeddings = self.patch_embed(batch["video_tokens"])
        if training:
            probability_rng, mask_rng = jax.random.split(batch["mask_rng"])
            mask_probability = sample_mask_probability(
                probability_rng,
                self.mask_limit,
            )
            mask = jax.random.bernoulli(
                mask_rng,
                mask_probability,
                video_embeddings.shape[:-1],
            )
            mask = mask.at[:, 0].set(False)
            video_embeddings = jnp.where(
                mask[..., None],
                self.mask_token,
                video_embeddings,
            )
        else:
            mask = None

        action_embeddings = self.action_up(batch["latent_actions"])
        video_embeddings = video_embeddings + jnp.pad(
            action_embeddings,
            ((0, 0), (1, 0), (0, 0), (0, 0)),
        )
        logits = self.dynamics(video_embeddings)
        return {"token_logits": logits, "mask": mask}
