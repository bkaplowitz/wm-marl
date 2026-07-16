"""Composed Jafar tokenizer, LAM, dynamics, and sampler.

Adapted from ``FLAIROx/jafar`` at commit
``5ff9fc7d5d744c8c2797ba3ad0a095ed7f2e2665``, path ``genie.py``.
Integration changes: the public model is named for Jafar, immutable config
objects construct the components, and both MaskGIT refinement and
autoregressive frame generation use ``jax.lax.scan``. Each outer step reruns
the source tokenizer and complete refinement sampler before decoding pixels.
"""

from flax import linen as nn
import jax
import jax.numpy as jnp

from world_marl.jafar.config import DynamicsConfig, LAMConfig, TokenizerConfig
from world_marl.jafar.dynamics import DynamicsMaskGIT
from world_marl.jafar.lam import LatentActionModel
from world_marl.jafar.sampling import unmasked_ratio
from world_marl.jafar.tokenizer import TokenizerVQVAE


class JafarWorldModel(nn.Module):
    tokenizer_config: TokenizerConfig
    lam_config: LAMConfig
    dynamics_config: DynamicsConfig

    def setup(self) -> None:
        tokenizer = self.tokenizer_config
        self.tokenizer = TokenizerVQVAE(
            in_dim=tokenizer.in_dim,
            model_dim=tokenizer.model_dim,
            latent_dim=tokenizer.latent_dim,
            num_latents=tokenizer.num_latents,
            patch_size=tokenizer.patch_size,
            num_blocks=tokenizer.num_blocks,
            num_heads=tokenizer.num_heads,
            dropout=tokenizer.dropout,
            codebook_dropout=tokenizer.codebook_dropout,
        )
        lam = self.lam_config
        self.lam = LatentActionModel(
            in_dim=lam.in_dim,
            model_dim=lam.model_dim,
            latent_dim=lam.latent_dim,
            num_latents=lam.num_latents,
            patch_size=lam.patch_size,
            num_blocks=lam.num_blocks,
            num_heads=lam.num_heads,
            dropout=lam.dropout,
            codebook_dropout=lam.codebook_dropout,
        )
        dynamics = self.dynamics_config
        self.dynamics = DynamicsMaskGIT(
            model_dim=dynamics.model_dim,
            num_latents=dynamics.num_latents,
            num_blocks=dynamics.num_blocks,
            num_heads=dynamics.num_heads,
            dropout=dynamics.dropout,
            mask_limit=dynamics.mask_limit,
        )

    def __call__(
        self,
        batch: dict[str, jax.Array],
        training: bool = True,
    ) -> dict[str, jax.Array | None]:
        tokenizer_outputs = self.tokenizer.vq_encode(
            batch["videos"],
            training=False,
        )
        lam_outputs = self.lam.vq_encode(batch["videos"], training=False)
        dynamics_inputs = {
            "video_tokens": jax.lax.stop_gradient(tokenizer_outputs["indices"]),
            "latent_actions": jax.lax.stop_gradient(lam_outputs["z_q"]),
            "mask_rng": batch["mask_rng"],
        }
        outputs = dict(dynamics_inputs)
        outputs.update(self.dynamics(dynamics_inputs, training=training))
        mle_indices = jnp.argmax(outputs["token_logits"], axis=-1)
        outputs["recon"] = self.tokenizer.decode(
            mle_indices,
            batch["videos"].shape[2:4],
        )
        return outputs

    def sample(
        self,
        batch: dict[str, jax.Array],
        seq_len: int,
        steps: int = 25,
        temperature: float = 1.0,
        sample_argmax: bool = False,
    ) -> jax.Array:
        context = batch["videos"]
        batch_size, context_frames, height, width, channels = context.shape
        if seq_len < context_frames:
            raise ValueError("seq_len must be at least the number of context frames")
        if batch["latent_actions"].shape != (batch_size, seq_len - 1):
            raise ValueError("latent_actions must have shape (batch, seq_len - 1)")

        pixel_history = jnp.concatenate(
            [
                context,
                jnp.zeros(
                    (
                        batch_size,
                        seq_len - context_frames,
                        height,
                        width,
                        channels,
                    ),
                    dtype=context.dtype,
                ),
            ],
            axis=1,
        )
        action_tokens = self.lam.vq.get_codes(batch["latent_actions"])
        action_tokens = action_tokens[:, :, None, :]
        action_embeddings = self.dynamics.action_up(action_tokens)
        padded_action_embeddings = jnp.pad(
            action_embeddings,
            ((0, 0), (1, 0), (0, 0), (0, 0)),
        )

        def autoregressive_step(
            carry: tuple[jax.Array, jax.Array],
            frame_index: jax.Array,
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            rng, pixels = carry
            tokenizer_outputs = self.tokenizer.vq_encode(pixels, training=False)
            token_indices = tokenizer_outputs["indices"]
            past_frame_mask = jnp.arange(seq_len)[None, :, None] < frame_index
            token_indices = jnp.where(past_frame_mask, token_indices, 0)
            current_indices = jnp.zeros_like(token_indices[:, 0])
            current_mask = jnp.ones_like(current_indices, dtype=jnp.bool_)

            def refinement_step(
                refinement_carry: tuple[jax.Array, jax.Array, jax.Array],
                step: jax.Array,
            ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
                refinement_rng, final_indices, mask = refinement_carry
                video_indices = token_indices.at[:, frame_index].set(final_indices)
                video_embeddings = self.dynamics.patch_embed(video_indices)
                masked_frame = jnp.where(
                    mask[..., None],
                    self.dynamics.mask_token[0],
                    video_embeddings[:, frame_index],
                )
                video_embeddings = video_embeddings.at[:, frame_index].set(masked_frame)
                video_embeddings = video_embeddings + padded_action_embeddings

                ratio = unmasked_ratio(step, steps)
                step_temperature = temperature * (1.0 - ratio)
                final_logits = (
                    self.dynamics.dynamics(video_embeddings)[:, frame_index]
                    / step_temperature
                )

                if sample_argmax:
                    sampled_indices = jnp.argmax(final_logits, axis=-1)
                else:
                    refinement_rng, sample_rng = jax.random.split(refinement_rng)
                    sampled_indices = jnp.where(
                        step == steps - 1,
                        jnp.argmax(final_logits, axis=-1),
                        jax.random.categorical(sample_rng, final_logits),
                    )

                probabilities = jax.nn.softmax(final_logits)
                selected_probabilities = jnp.take_along_axis(
                    probabilities,
                    sampled_indices[..., None],
                    axis=-1,
                )[..., 0]
                selected_probabilities = selected_probabilities + (~mask)
                new_indices = jnp.where(mask, sampled_indices, final_indices)

                num_patches = selected_probabilities.shape[-1]
                num_unmasked = jnp.round(num_patches * (1.0 - ratio)).astype(int)
                rank_mask = jnp.arange(num_patches) > num_unmasked
                sorted_indices = jnp.argsort(
                    selected_probabilities,
                    axis=-1,
                    descending=True,
                )
                new_mask = jax.vmap(
                    lambda old_mask, order: old_mask.at[order].set(rank_mask)
                )(mask, sorted_indices)
                return (refinement_rng, new_indices, new_mask), None

            (rng, current_indices, _), _ = jax.lax.scan(
                refinement_step,
                (rng, current_indices, current_mask),
                jnp.arange(steps),
            )
            new_frame = self.tokenizer.decode(
                current_indices[:, None],
                (height, width),
            )[:, 0]
            pixels = pixels.at[:, frame_index].set(new_frame)
            return (rng, pixels), None

        (_, generated), _ = jax.lax.scan(
            autoregressive_step,
            (batch["rng"], pixel_history),
            jnp.arange(context_frames, seq_len),
        )
        return generated
