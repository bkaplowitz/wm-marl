"""Jasmine diffusion-forcing dynamics.

Adapted from ``p-doom/jasmine`` at commit
``420859bc99eecf6b07a7e9edf65d5d145935f1e1``, path
``jasmine/models/dynamics.py`` (``DynamicsDiffusion``). Integration changes:
NNX state is translated to Linen parameters and the noise/ramp equations are
public pure helpers; per-frame levels, token layout, x-prediction, initializers,
and dtypes are preserved.
"""

from flax import linen as nn
import jax
import jax.numpy as jnp

from world_marl.jasmine.nn import AxialTransformer


def linear_noise_mix(
    clean_latents: jax.Array,
    noise: jax.Array,
    signal_level: jax.Array,
) -> jax.Array:
    signal = signal_level[..., None, None]
    return (1.0 - (1.0 - 1e-5) * signal) * noise + signal * clean_latents


def ramp_weight(signal_level: jax.Array) -> jax.Array:
    return 0.9 * signal_level + 0.1


class DynamicsDiffusion(nn.Module):
    model_dim: int
    ffn_dim: int
    latent_patch_dim: int
    latent_action_dim: int
    num_blocks: int
    num_heads: int
    denoise_steps: int
    dropout: float
    param_dtype: jnp.dtype
    dtype: jnp.dtype
    use_flash_attention: bool

    def setup(self) -> None:
        self.diffusion_transformer = AxialTransformer(
            input_dim=self.latent_patch_dim,
            model_dim=self.model_dim,
            ffn_dim=self.ffn_dim,
            out_dim=self.latent_patch_dim,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            use_flash_attention=self.use_flash_attention,
            spatial_causal=False,
            temporal_causal=True,
        )
        self.action_up = nn.Dense(
            self.latent_patch_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )
        self.timestep_embed = nn.Embed(
            num_embeddings=self.denoise_steps,
            features=self.latent_patch_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

    def __call__(
        self,
        batch: dict[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        time_rng, noise_rng = jax.random.split(batch["rng"])
        latents = batch["token_latents"]
        latent_actions = batch["latent_actions"]
        batch_size, sequence_length, num_patches, latent_dim = latents.shape

        denoise_step = jax.random.randint(
            time_rng,
            (batch_size, sequence_length),
            minval=0,
            maxval=self.denoise_steps,
        )
        denoise_step_embedding = self.timestep_embed(denoise_step).reshape(
            batch_size,
            sequence_length,
            1,
            self.latent_patch_dim,
        )
        signal_level = denoise_step / self.denoise_steps
        noise = jax.random.normal(
            noise_rng,
            (batch_size, sequence_length, num_patches, latent_dim),
        )
        noised_latents = linear_noise_mix(latents, noise, signal_level)

        action_embeddings = self.action_up(latent_actions)
        padded_action_embeddings = jnp.pad(
            action_embeddings,
            ((0, 0), (1, 0), (0, 0), (0, 0)),
        )
        inputs = jnp.concatenate(
            [padded_action_embeddings, denoise_step_embedding, noised_latents],
            axis=2,
        )
        outputs = self.diffusion_transformer(inputs)
        return outputs[:, :, 2:], signal_level
