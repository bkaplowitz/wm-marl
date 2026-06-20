"""Decoder-free isotropy-JEPA model for vector observations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class JepaConfig:
    observation_dim: int
    action_dim: int
    latent_dim: int = 128
    model_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    max_horizon: int = 1
    context_window: int = 1
    learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    isotropy_weight: float = 0.05
    reward_weight: float = 1.0
    continue_weight: float = 1.0
    gamma: float = 0.99
    lambda_return: float = 0.95
    entropy_coef: float = 0.01

    def __post_init__(self) -> None:
        if self.max_horizon != 1:
            raise ValueError(
                "Milestone 1 supports max_horizon=1 only; "
                "multi-step action-conditioned overshooting is not implemented."
            )
        if self.context_window != 1:
            raise ValueError(
                "Milestone 1 supports context_window=1 only; "
                "real-history imagination initialization is not implemented."
            )


class MLPEncoder(nn.Module):
    latent_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        x = observations.astype(jnp.float32)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.latent_dim)(x)
        return nn.LayerNorm()(x)


class MLPHead(nn.Module):
    output_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        return nn.Dense(self.output_dim)(x)


class TransformerBlock(nn.Module):
    model_dim: int
    num_heads: int
    mlp_ratio: int

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        h = nn.LayerNorm()(x)
        x = x + nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(
            h,
            h,
            mask=mask,
            deterministic=True,
        )
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.mlp_ratio * self.model_dim)(h)
        h = nn.gelu(h)
        return x + nn.Dense(self.model_dim)(h)


class JepaWorldModel(nn.Module):
    config: JepaConfig

    def setup(self) -> None:
        self.encoder = MLPEncoder(
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.model_dim,
            name="encoder",
        )
        self.latent_proj = nn.Dense(self.config.model_dim, name="latent_proj")
        self.action_embed = nn.Embed(
            self.config.action_dim,
            self.config.model_dim,
            name="action_embed",
        )
        self.horizon_embed = nn.Embed(
            self.config.max_horizon + 1,
            self.config.model_dim,
            name="horizon_embed",
        )
        self.blocks = [
            TransformerBlock(
                model_dim=self.config.model_dim,
                num_heads=self.config.num_heads,
                mlp_ratio=self.config.mlp_ratio,
                name=f"block_{index}",
            )
            for index in range(self.config.num_layers)
        ]
        self.dynamics_norm = nn.LayerNorm(name="dynamics_norm")
        self.predictor = MLPHead(
            self.config.latent_dim,
            self.config.model_dim,
            name="predictor",
        )
        self.predictor_norm = nn.LayerNorm(name="predictor_norm")
        self.reward_head = MLPHead(1, self.config.model_dim, name="reward_head")
        self.continue_head = MLPHead(1, self.config.model_dim, name="continue_head")
        self.actor_head = MLPHead(
            self.config.action_dim,
            self.config.model_dim,
            name="actor_head",
        )
        self.value_head = MLPHead(1, self.config.model_dim, name="value_head")

    def __call__(self, observations: jax.Array) -> jax.Array:
        return self.encode(observations)

    def initialize(
        self,
        observations: jax.Array,
        actions: jax.Array,
        *,
        chunk_length: int,
    ) -> dict[str, jax.Array]:
        outputs = self.sequence_outputs(
            observations,
            actions,
            chunk_length=chunk_length,
        )
        logits, values = self.actor_value_from_obs(observations[:, 0])
        outputs["actor_logits"] = logits
        outputs["values"] = values
        return outputs

    def encode(self, observations: jax.Array) -> jax.Array:
        flat = observations.reshape((-1, self.config.observation_dim))
        latents = self.encoder(flat)
        return latents.reshape((*observations.shape[:-1], self.config.latent_dim))

    def actor_value_from_obs(
        self,
        observations: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        z = self.encode(observations)
        return self.actor_value_from_latent(z)

    def actor_value_from_latent(
        self,
        latents: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        flat = latents.reshape((-1, self.config.latent_dim))
        logits = self.actor_head(flat)
        values = jnp.squeeze(self.value_head(flat), axis=-1)
        return (
            logits.reshape((*latents.shape[:-1], self.config.action_dim)),
            values.reshape(latents.shape[:-1]),
        )

    def sequence_outputs(
        self,
        observations: jax.Array,
        actions: jax.Array,
        *,
        chunk_length: int,
        dones: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        latents = self.encode(observations)
        context_latents = latents[:, :chunk_length]
        context_actions = actions[:, :chunk_length]
        context_dones = None if dones is None else dones[:, :chunk_length]
        h = self.dynamics_hidden(
            context_latents,
            context_actions,
            dones=context_dones,
        )

        predictions = []
        targets = []
        for horizon in range(1, self.config.max_horizon + 1):
            horizon_ids = jnp.full(
                (h.shape[0], h.shape[1]),
                horizon,
                dtype=jnp.int32,
            )
            h_k = h + self.horizon_embed(horizon_ids)
            predictions.append(self.predict_latent(h_k))
            targets.append(
                jax.lax.stop_gradient(latents[:, horizon : horizon + chunk_length])
            )
        predicted_latents = jnp.stack(predictions, axis=2)
        target_latents = jnp.stack(targets, axis=2)
        rewards = jnp.squeeze(self.reward_head(h), axis=-1)
        continues = jnp.squeeze(self.continue_head(h), axis=-1)
        return {
            "context_latents": context_latents,
            "predicted_latents": predicted_latents,
            "target_latents": target_latents,
            "reward_logits": rewards,
            "continue_logits": continues,
            "hidden": h,
        }

    def dynamics_hidden(
        self,
        latents: jax.Array,
        actions: jax.Array,
        *,
        dones: jax.Array | None = None,
    ) -> jax.Array:
        if latents.ndim != 3:
            raise ValueError("latents must be shaped [batch, time, latent_dim]")
        tokens = self.latent_proj(latents) + self.action_embed(actions)
        time = tokens.shape[1]
        positions = sinusoidal_position_embedding(time, self.config.model_dim)
        h = tokens + positions[None, :, :]
        mask = causal_attention_mask(
            time,
            context_window=self.config.context_window,
            dones=dones,
        )
        for block in self.blocks:
            h = block(h, mask)
        return self.dynamics_norm(h)

    def predict_latent(self, hidden: jax.Array) -> jax.Array:
        return self.predictor_norm(self.predictor(hidden))

    def predict_next_from_history(
        self,
        latent_history: jax.Array,
        action_history: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        h = self.dynamics_hidden(latent_history, action_history)
        last_h = h[:, -1]
        horizon_ids = jnp.ones((last_h.shape[0],), dtype=jnp.int32)
        z_next = self.predict_latent(last_h + self.horizon_embed(horizon_ids))
        reward = jnp.squeeze(self.reward_head(last_h), axis=-1)
        continue_logit = jnp.squeeze(self.continue_head(last_h), axis=-1)
        return z_next, reward, continue_logit


def causal_attention_mask(
    time: int,
    *,
    context_window: int,
    dones: jax.Array | None = None,
) -> jax.Array:
    positions = jnp.arange(time)
    causal = positions[None, :] <= positions[:, None]
    local = (positions[:, None] - positions[None, :]) < context_window
    mask = causal & local
    mask = mask[None, None, :, :]
    if dones is None:
        return mask

    previous_dones = jnp.concatenate(
        [
            jnp.zeros((dones.shape[0], 1), dtype=dones.dtype),
            dones[:, : time - 1],
        ],
        axis=1,
    )
    segment_ids = jnp.cumsum(previous_dones.astype(jnp.int32), axis=1)
    same_segment = segment_ids[:, None, :, None] == segment_ids[:, None, None, :]
    return mask & same_segment


def sinusoidal_position_embedding(time: int, dim: int) -> jax.Array:
    half = dim // 2
    positions = jnp.arange(time, dtype=jnp.float32)[:, None]
    freqs = jnp.exp(-math.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
    angles = positions * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    if emb.shape[-1] < dim:
        emb = jnp.pad(emb, ((0, 0), (0, dim - emb.shape[-1])))
    return emb
