"""Autoregressive latent diffusion dynamics for the Genie2 public-source arm."""

from __future__ import annotations

from flax import linen as nn
import jax
import jax.numpy as jnp

from world_marl.genie2_continuous_jax.config import DynamicsConfig
from world_marl.genie2_continuous_jax.st_transformer import (
    AxialTransformer,
    dtype_from_name,
)


def classifier_free_guidance(
    conditioned: jax.Array,
    unconditioned: jax.Array,
    guidance_scale: float,
) -> jax.Array:
    return unconditioned + guidance_scale * (conditioned - unconditioned)


def quantized_context_signal_level(
    *,
    denoising_steps: int,
    context_corruption: float,
) -> jax.Array:
    levels = jnp.arange(denoising_steps, dtype=jnp.float32) / float(denoising_steps)
    target = jnp.asarray(1.0 - context_corruption, dtype=jnp.float32)
    return levels[jnp.argmin(jnp.abs(levels - target))]


class ActionConditionedLatentDiffusion(nn.Module):
    """Jasmine-style action-prepended axial diffusion transformer."""

    latent_patch_dim: int
    action_dim: int
    config: DynamicsConfig

    def setup(self) -> None:
        compute_dtype = dtype_from_name(self.config.compute_dtype)
        parameter_dtype = dtype_from_name(self.config.parameter_dtype)
        self.action_projection = nn.Dense(
            self.latent_patch_dim,
            dtype=compute_dtype,
            param_dtype=parameter_dtype,
            name="action_projection",
        )
        self.timestep_embedding = nn.Embed(
            num_embeddings=self.config.denoising_steps,
            features=self.latent_patch_dim,
            dtype=compute_dtype,
            param_dtype=parameter_dtype,
            name="timestep_embedding",
        )
        self.transformer = AxialTransformer(
            input_dim=self.latent_patch_dim,
            model_dim=self.config.model_dim,
            ffn_dim=self.config.ffn_dim,
            output_dim=self.latent_patch_dim,
            num_blocks=self.config.num_blocks,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            temporal_causal=True,
            spatial_causal=False,
            compute_dtype=self.config.compute_dtype,
            parameter_dtype=self.config.parameter_dtype,
            name="diffusion_transformer",
        )

    def predict_x(
        self,
        noised_latents: jax.Array,
        actions: jax.Array,
        denoising_steps: jax.Array,
        *,
        condition_keep_mask: jax.Array,
        training: bool,
    ) -> jax.Array:
        if noised_latents.ndim != 4:
            raise ValueError("latents must have shape (batch,time,patch,latent)")
        batch, time, _, latent_dim = noised_latents.shape
        if latent_dim != self.latent_patch_dim:
            raise ValueError("latent patch width does not match the dynamics model")
        if actions.shape != (batch, time - 1, self.action_dim):
            raise ValueError(
                f"actions must have shape {(batch, time - 1, self.action_dim)}, "
                f"got {actions.shape}"
            )
        if denoising_steps.shape != (batch, time):
            raise ValueError("denoising_steps must have shape (batch,time)")
        if condition_keep_mask.shape == (batch,):
            condition_keep_mask = condition_keep_mask[:, None]
        if condition_keep_mask.shape != (batch, time - 1):
            raise ValueError("condition_keep_mask must have shape (batch,time-1)")

        actions = actions.astype(jnp.float32) * condition_keep_mask[..., None]
        action_tokens = self.action_projection(actions)[:, :, None, :]
        action_tokens = jnp.pad(action_tokens, ((0, 0), (1, 0), (0, 0), (0, 0)))
        timestep_tokens = self.timestep_embedding(denoising_steps)[:, :, None, :]
        inputs = jnp.concatenate(
            [action_tokens, timestep_tokens, noised_latents],
            axis=2,
        )
        outputs = self.transformer(inputs, training=training)
        return outputs[:, :, 2:].astype(jnp.float32)

    def __call__(
        self,
        latents: jax.Array,
        actions: jax.Array,
        *,
        key: jax.Array,
        training: bool,
    ) -> dict[str, jax.Array]:
        batch, time, num_patches, latent_dim = latents.shape
        time_key, noise_key, dropout_key = jax.random.split(key, 3)
        denoising_steps = jax.random.randint(
            time_key,
            (batch, time),
            minval=0,
            maxval=self.config.denoising_steps,
        )
        signal_level = denoising_steps.astype(jnp.float32) / float(
            self.config.denoising_steps
        )
        signal = signal_level[:, :, None, None]
        noise = jax.random.normal(
            noise_key,
            (batch, time, num_patches, latent_dim),
            dtype=jnp.float32,
        )
        noised_latents = (
            1.0 - (1.0 - 1e-5) * signal
        ) * noise + signal * latents.astype(jnp.float32)
        if training:
            condition_keep = jax.random.bernoulli(
                dropout_key,
                1.0 - self.config.classifier_free_dropout,
                (batch, 1),
            ).astype(jnp.float32)
            condition_keep = jnp.broadcast_to(condition_keep, (batch, time - 1))
        else:
            condition_keep = jnp.ones((batch, time - 1), dtype=jnp.float32)
        prediction = self.predict_x(
            noised_latents,
            actions,
            denoising_steps,
            condition_keep_mask=condition_keep,
            training=training,
        )
        return {
            "x_prediction": prediction,
            "x_target": latents.astype(jnp.float32),
            "signal_level": signal_level,
            "noise": noise,
            "noised_latents": noised_latents,
            "condition_keep_mask": condition_keep,
        }


def diffusion_forcing_loss(
    outputs: dict[str, jax.Array],
    *,
    ramp_weight: bool,
    valid_mask: jax.Array | None = None,
) -> jax.Array:
    per_frame = jnp.mean(
        jnp.square(outputs["x_prediction"] - outputs["x_target"]),
        axis=(2, 3),
    )
    if ramp_weight:
        per_frame = per_frame * (0.9 * outputs["signal_level"] + 0.1)
    if valid_mask is None:
        return jnp.mean(per_frame)
    weights = valid_mask.astype(per_frame.dtype)
    return jnp.sum(per_frame * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def autoregressive_sample(
    apply_fn,
    variables,
    context_latents: jax.Array,
    actions: jax.Array,
    *,
    key: jax.Array,
    num_future_frames: int,
    config: DynamicsConfig,
) -> jax.Array:
    if num_future_frames <= 0:
        raise ValueError("num_future_frames must be positive")
    batch, context_time, num_patches, latent_dim = context_latents.shape
    total_time = context_time + num_future_frames
    if actions.shape[:2] != (batch, total_time - 1):
        raise ValueError("actions must cover every generated transition")
    key, pad_key = jax.random.split(key)
    pad = jax.random.normal(
        pad_key,
        (batch, num_future_frames, num_patches, latent_dim),
        dtype=jnp.float32,
    )
    initial_latents = jnp.concatenate([context_latents, pad], axis=1)
    context_signal = quantized_context_signal_level(
        denoising_steps=config.denoising_steps,
        context_corruption=config.context_corruption,
    )

    def generate_frame(
        carry: tuple[jax.Array, jax.Array],
        frame_index: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        latents, frame_key = carry

        def denoise(
            denoise_carry: tuple[jax.Array, jax.Array],
            denoising_step: jax.Array,
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            current_latents, denoise_key = denoise_carry
            denoise_key, context_noise_key = jax.random.split(denoise_key)
            context_noise = jax.random.normal(
                context_noise_key,
                current_latents.shape,
                dtype=jnp.float32,
            )
            corrupted_context = (
                context_signal * current_latents
                + (1.0 - context_signal) * context_noise
            )
            is_context = jnp.arange(total_time) < frame_index
            model_latents = jnp.where(
                is_context[None, :, None, None],
                corrupted_context,
                current_latents,
            )
            steps = (
                jnp.full(
                    (batch, total_time),
                    config.denoising_steps - 1,
                    dtype=jnp.int32,
                )
                .at[:, frame_index]
                .set(denoising_step)
            )
            conditioned = apply_fn(
                variables,
                model_latents,
                actions,
                steps,
                condition_keep_mask=jnp.ones(
                    (batch, total_time - 1), dtype=jnp.float32
                ),
                training=False,
                method=ActionConditionedLatentDiffusion.predict_x,
            )
            unconditioned = apply_fn(
                variables,
                model_latents,
                actions,
                steps,
                condition_keep_mask=jnp.zeros(
                    (batch, total_time - 1), dtype=jnp.float32
                ),
                training=False,
                method=ActionConditionedLatentDiffusion.predict_x,
            )
            prediction = classifier_free_guidance(
                conditioned,
                unconditioned,
                config.guidance_scale,
            )
            current_latents = current_latents.at[:, frame_index].set(
                prediction[:, frame_index]
            )
            return (current_latents, denoise_key), None

        frame_key, inner_key = jax.random.split(frame_key)
        (latents, _), _ = jax.lax.scan(
            denoise,
            (latents, inner_key),
            jnp.arange(config.denoising_steps, dtype=jnp.int32),
        )
        return (latents, frame_key), None

    (latents, _), _ = jax.lax.scan(
        generate_frame,
        (initial_latents, key),
        jnp.arange(context_time, total_time, dtype=jnp.int32),
    )
    return latents


def dynamics_mse_loss(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must share shape")
    return jnp.mean(jnp.square(predictions - targets))


class CausalLatentDynamics(nn.Module):
    """Legacy flat-latent extension retained only for old experiment loading."""

    latent_dim: int
    latent_action_dim: int
    model_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    max_context: int = 32

    @nn.compact
    def __call__(
        self,
        latent_history: jax.Array,
        latent_actions: jax.Array,
        noise_level: jax.Array | None = None,
        condition_keep_mask: jax.Array | None = None,
        query_latent: jax.Array | None = None,
        context_mask: jax.Array | None = None,
    ) -> jax.Array:
        del context_mask
        actions = latent_actions.astype(jnp.float32)
        if condition_keep_mask is not None:
            actions = actions * condition_keep_mask[:, None, None]
        x = nn.Dense(self.model_dim)(latent_history) + nn.Dense(self.model_dim)(actions)
        if query_latent is not None:
            x = x.at[:, -1].add(nn.Dense(self.model_dim)(query_latent))
        if noise_level is not None:
            x = x + nn.Dense(self.model_dim)(noise_level[:, None])[:, None]
        for _ in range(self.num_layers):
            y = nn.LayerNorm()(x)
            mask = nn.make_causal_mask(jnp.ones(x.shape[:2], dtype=bool))
            x = x + nn.MultiHeadDotProductAttention(self.num_heads)(y, y, mask=mask)
            y = nn.gelu(nn.Dense(4 * self.model_dim)(nn.LayerNorm()(x)))
            x = x + nn.Dense(self.model_dim)(y)
        return nn.Dense(self.latent_dim)(nn.LayerNorm()(x[:, -1]))
