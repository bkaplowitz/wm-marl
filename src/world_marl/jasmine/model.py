"""Composed Jasmine MAE, LAM, diffusion dynamics, and sampler.

Adapted from ``p-doom/jasmine`` at commit
``420859bc99eecf6b07a7e9edf65d5d145935f1e1``, path
``jasmine/models/genie.py`` (``GenieDiffusion``). Integration changes: NNX
state/transforms become Linen parameters and nested ``jax.lax.scan`` calls;
the public model is named for Jasmine and excludes ground-truth-action and CFG
paths. Tokenizer stop-gradient, optional LAM co-training, no-decoder LAM use,
noise equations, context corruption, and latent-space sampling are preserved.
"""

from flax import linen as nn
import jax
import jax.numpy as jnp

from world_marl.jasmine.config import DynamicsConfig, LAMConfig, TokenizerConfig
from world_marl.jasmine.dynamics import DynamicsDiffusion
from world_marl.jasmine.lam import LatentActionModel
from world_marl.jasmine.sampling import snapped_context_signal_level
from world_marl.jasmine.tokenizer import TokenizerMAE


class JasmineWorldModel(nn.Module):
    tokenizer_config: TokenizerConfig
    lam_config: LAMConfig
    dynamics_config: DynamicsConfig
    lam_co_train: bool = True

    def setup(self) -> None:
        if self.dynamics_config.use_gt_actions:
            raise ValueError("Jasmine does not support direct real-action conditioning")
        if self.dynamics_config.use_cfg:
            raise ValueError("Jasmine does not support classifier-free guidance")

        tokenizer = self.tokenizer_config
        self.tokenizer = TokenizerMAE(
            in_dim=tokenizer.in_dim,
            model_dim=tokenizer.model_dim,
            ffn_dim=tokenizer.ffn_dim,
            latent_dim=tokenizer.latent_dim,
            num_latents=tokenizer.num_latents,
            patch_size=tokenizer.patch_size,
            num_blocks=tokenizer.num_blocks,
            num_heads=tokenizer.num_heads,
            dropout=0.0,
            max_mask_ratio=0.0,
            param_dtype=tokenizer.param_dtype,
            dtype=tokenizer.dtype,
            use_flash_attention=tokenizer.use_flash_attention,
        )
        lam = self.lam_config
        self.lam = LatentActionModel(
            in_dim=lam.in_dim,
            model_dim=lam.model_dim,
            ffn_dim=lam.ffn_dim,
            latent_dim=lam.latent_dim,
            num_latents=lam.num_latents,
            patch_size=lam.patch_size,
            num_blocks=lam.num_blocks,
            num_heads=lam.num_heads,
            dropout=0.0,
            codebook_dropout=0.0,
            param_dtype=lam.param_dtype,
            dtype=lam.dtype,
            use_flash_attention=lam.use_flash_attention,
        )
        dynamics = self.dynamics_config
        self.dynamics = DynamicsDiffusion(
            model_dim=dynamics.model_dim,
            ffn_dim=dynamics.ffn_dim,
            latent_patch_dim=dynamics.latent_patch_dim,
            latent_action_dim=dynamics.latent_action_dim,
            num_blocks=dynamics.num_blocks,
            num_heads=dynamics.num_heads,
            denoise_steps=dynamics.denoise_steps,
            dropout=dynamics.dropout,
            param_dtype=dynamics.param_dtype,
            dtype=dynamics.dtype,
            use_flash_attention=dynamics.use_flash_attention,
        )

    def __call__(self, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        videos = batch["videos"]
        height, width = videos.shape[2:4]
        lam_outputs = self.lam.vq_encode(videos, training=False)
        latent_actions = jax.lax.cond(
            self.lam_co_train,
            lambda: lam_outputs["z_q"],
            lambda: jax.lax.stop_gradient(lam_outputs["z_q"]),
        )

        dynamics_rng, tokenizer_rng = jax.random.split(batch["rng"])
        tokenizer_outputs = self.tokenizer.mask_and_encode(
            videos,
            tokenizer_rng,
            training=False,
        )
        token_latents = jax.lax.stop_gradient(tokenizer_outputs["z"])
        predicted, signal_level = self.dynamics(
            {
                "token_latents": token_latents,
                "latent_actions": latent_actions,
                "rng": dynamics_rng,
            }
        )
        return {
            "token_latents": token_latents,
            "latent_actions": latent_actions,
            "x_pred": predicted,
            "x_gt": token_latents,
            "signal_level": signal_level,
            "recon": self.tokenizer.decode(predicted, (height, width)),
            "lam_indices": lam_outputs["indices"],
        }

    def sample(
        self,
        batch: dict[str, jax.Array],
        seq_len: int,
        diffusion_steps: int = 64,
        context_corruption: float = 0.1,
    ) -> jax.Array:
        if diffusion_steps != self.dynamics_config.denoise_steps:
            raise ValueError("diffusion_steps must match the dynamics denoise_steps")
        videos = batch["videos"]
        batch_size, context_frames, height, width, _ = videos.shape
        if seq_len < context_frames:
            raise ValueError("seq_len must be at least the number of context frames")
        if batch["latent_actions"].shape != (batch_size, seq_len - 1):
            raise ValueError("latent_actions must have shape (batch, seq_len - 1)")

        rng, tokenizer_rng, future_noise_rng = jax.random.split(batch["rng"], 3)
        token_latents = self.tokenizer.mask_and_encode(
            videos,
            tokenizer_rng,
            training=False,
        )["z"]
        _, _, num_patches, latent_dim = token_latents.shape
        future_noise = jax.random.normal(
            future_noise_rng,
            (
                batch_size,
                seq_len - context_frames,
                num_patches,
                latent_dim,
            ),
        )
        latent_history = jnp.concatenate([token_latents, future_noise], axis=1)
        action_tokens = self.lam.vq.get_codes(batch["latent_actions"])
        action_tokens = action_tokens.reshape(
            batch_size,
            seq_len - 1,
            1,
            self.dynamics_config.latent_action_dim,
        )
        action_embeddings = self.dynamics.action_up(action_tokens)
        action_embeddings = jnp.pad(
            action_embeddings,
            ((0, 0), (1, 0), (0, 0), (0, 0)),
        )
        context_signal = snapped_context_signal_level(
            diffusion_steps,
            context_corruption,
        )

        def denoise_step(
            carry: tuple[jax.Array, jax.Array],
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            latents, denoise_rng = carry
            frame_index, step = inputs
            denoise_rng, context_noise_rng = jax.random.split(denoise_rng)

            denoise_indices = jnp.full(
                (batch_size, seq_len),
                diffusion_steps - 1,
                dtype=jnp.int32,
            )
            denoise_indices = denoise_indices.at[:, frame_index].set(step)
            context_noise = jax.random.normal(
                context_noise_rng,
                (batch_size, seq_len, num_patches, latent_dim),
            )
            corrupted_context = (
                latents * context_signal + (1.0 - context_signal) * context_noise
            )
            context_mask = (jnp.arange(seq_len) < frame_index)[None, :, None, None]
            transformer_latents = jnp.where(
                context_mask,
                corrupted_context,
                latents,
            )
            denoise_embeddings = self.dynamics.timestep_embed(denoise_indices).reshape(
                batch_size,
                seq_len,
                1,
                self.dynamics_config.latent_patch_dim,
            )
            transformer_inputs = jnp.concatenate(
                [action_embeddings, denoise_embeddings, transformer_latents],
                axis=2,
            )
            predicted = self.dynamics.diffusion_transformer(transformer_inputs)[
                :, :, 2:
            ]
            latents = latents.at[:, frame_index].set(predicted[:, frame_index])
            return (latents, denoise_rng), None

        def autoregressive_step(
            carry: tuple[jax.Array, jax.Array],
            frame_index: jax.Array,
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            latents, outer_rng = carry
            outer_rng, denoise_rng = jax.random.split(outer_rng)
            frame_indices = jnp.full(
                (diffusion_steps,),
                frame_index,
                dtype=jnp.int32,
            )
            (latents, _), _ = jax.lax.scan(
                denoise_step,
                (latents, denoise_rng),
                (frame_indices, jnp.arange(diffusion_steps)),
            )
            return (latents, outer_rng), None

        rng, sample_rng = jax.random.split(rng)
        (final_latents, _), _ = jax.lax.scan(
            autoregressive_step,
            (latent_history, sample_rng),
            jnp.arange(context_frames, seq_len),
        )
        return self.tokenizer.decode(final_latents, (height, width))
