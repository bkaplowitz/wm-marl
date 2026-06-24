"""Representation-space SIGReg/JEPA model for vector observations."""

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
    action_mode: str = "discrete"
    latent_dim: int = 128
    model_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: int = 4
    max_horizon: int = 1
    context_window: int = 1
    learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    regularizer: str = "sigreg"
    regularizer_weight: float = 0.05
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    reward_weight: float = 1.0
    continue_weight: float = 1.0
    dynamics_ensemble_size: int = 1
    gamma: float = 0.99
    lambda_return: float = 0.95
    residual_dynamics: bool = True
    target_gradient: str = "stopgrad"

    def __post_init__(self) -> None:
        if self.action_mode not in ("discrete", "continuous"):
            raise ValueError("action_mode must be one of: discrete, continuous")
        if self.regularizer not in ("sigreg", "none"):
            raise ValueError("regularizer must be one of: sigreg, none")
        if self.target_gradient not in ("stopgrad", "symmetric"):
            raise ValueError("target_gradient must be one of: stopgrad, symmetric")
        if self.sigreg_knots < 2:
            raise ValueError("sigreg_knots must be >= 2")
        if self.sigreg_num_proj < 1:
            raise ValueError("sigreg_num_proj must be >= 1")
        if self.max_horizon < 1:
            raise ValueError("max_horizon must be >= 1")
        if self.context_window < 1:
            raise ValueError("context_window must be >= 1")
        if self.dynamics_ensemble_size < 1:
            raise ValueError("dynamics_ensemble_size must be >= 1")


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
        if self.config.action_mode == "discrete":
            self.action_embed = nn.Embed(
                self.config.action_dim,
                self.config.model_dim,
                name="action_embed",
            )
        else:
            self.action_encoder_hidden = nn.Dense(
                self.config.model_dim,
                name="action_encoder_hidden",
            )
            self.action_encoder_out = nn.Dense(
                self.config.model_dim,
                name="action_encoder_out",
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
        if self.config.dynamics_ensemble_size == 1:
            self.predictor = MLPHead(
                self.config.latent_dim,
                self.config.model_dim,
                name="predictor",
            )
            self.predictor_norm = nn.LayerNorm(name="predictor_norm")
            self.reward_head = MLPHead(1, self.config.model_dim, name="reward_head")
            self.continue_head = MLPHead(1, self.config.model_dim, name="continue_head")
        else:
            self.predictors = [
                MLPHead(
                    self.config.latent_dim,
                    self.config.model_dim,
                    name=f"predictor_{index}",
                )
                for index in range(self.config.dynamics_ensemble_size)
            ]
            self.predictor_norms = [
                nn.LayerNorm(name=f"predictor_norm_{index}")
                for index in range(self.config.dynamics_ensemble_size)
            ]
            self.reward_heads = [
                MLPHead(1, self.config.model_dim, name=f"reward_head_{index}")
                for index in range(self.config.dynamics_ensemble_size)
            ]
            self.continue_heads = [
                MLPHead(1, self.config.model_dim, name=f"continue_head_{index}")
                for index in range(self.config.dynamics_ensemble_size)
            ]
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
        z0 = self.encode(observations[:, 0])
        logits, values = self.actor_value_from_latent(z0)
        outputs["actor_logits"] = logits
        outputs["values"] = values
        return outputs

    def encode(self, observations: jax.Array) -> jax.Array:
        flat = observations.reshape((-1, self.config.observation_dim))
        latents = self.encoder(flat)
        return latents.reshape((*observations.shape[:-1], self.config.latent_dim))

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
        max_horizon = self.config.max_horizon
        if latents.shape[1] < chunk_length + max_horizon:
            raise ValueError("observations must cover chunk_length + max_horizon")
        if actions.shape[1] < chunk_length + max_horizon - 1:
            raise ValueError("actions must cover chunk_length + max_horizon - 1")

        context_latents = latents[:, :chunk_length]
        latent_history = history_windows(
            context_latents,
            self.config.context_window,
            pad="edge",
        )
        action_history = history_windows(
            actions[:, :chunk_length],
            self.config.context_window,
            pad="zero",
        )
        done_history = (
            None
            if dones is None
            else history_windows(
                dones[:, :chunk_length],
                self.config.context_window,
                pad="done",
            )
        )
        predictions = []
        targets = []
        rewards = []
        continues = []
        batch_size = latents.shape[0]
        flat_batch = batch_size * chunk_length
        for step_index in range(max_horizon):
            flat_latents = latent_history.reshape(
                (flat_batch, self.config.context_window, self.config.latent_dim)
            )
            flat_actions = action_history.reshape(
                (flat_batch, self.config.context_window, *actions.shape[2:])
            )
            flat_dones = (
                None
                if done_history is None
                else done_history.reshape((flat_batch, self.config.context_window))
            )
            h = self.dynamics_hidden(flat_latents, flat_actions, dones=flat_dones)
            last_h = h[:, -1]
            horizon_ids = jnp.full((flat_batch,), step_index + 1, dtype=jnp.int32)
            current_latents = flat_latents[:, -1]
            head_predictions = []
            head_rewards = []
            head_continues = []
            for head_index in range(self.config.dynamics_ensemble_size):
                next_latents = self.predict_latent(
                    last_h + self.horizon_embed(horizon_ids),
                    current_latents=current_latents,
                    head_index=head_index,
                )
                head_predictions.append(
                    next_latents.reshape(
                        (batch_size, chunk_length, self.config.latent_dim)
                    )
                )
                head_rewards.append(
                    jnp.squeeze(
                        self.reward_from_hidden(last_h, head_index=head_index),
                        axis=-1,
                    ).reshape((batch_size, chunk_length))
                )
                head_continues.append(
                    jnp.squeeze(
                        self.continue_from_hidden(last_h, head_index=head_index),
                        axis=-1,
                    ).reshape((batch_size, chunk_length))
                )
            head_predictions = jnp.stack(head_predictions, axis=2)
            predictions.append(head_predictions)
            target = latents[:, step_index + 1 : step_index + 1 + chunk_length]
            if self.config.target_gradient == "stopgrad":
                target = jax.lax.stop_gradient(target)
            targets.append(target)
            rewards.append(jnp.stack(head_rewards, axis=2))
            continues.append(jnp.stack(head_continues, axis=2))
            if step_index + 1 < max_horizon:
                latent_history = append_history(
                    latent_history,
                    jnp.mean(predictions[-1], axis=2),
                )
                action_history = append_history(
                    action_history,
                    actions[:, step_index + 1 : step_index + 1 + chunk_length],
                )
                if done_history is not None:
                    done_history = append_history(
                        done_history,
                        dones[:, step_index + 1 : step_index + 1 + chunk_length],
                    )
        predicted_latents = jnp.stack(predictions, axis=2)
        reward_logits = jnp.stack(rewards, axis=2)
        continue_logits = jnp.stack(continues, axis=2)
        if self.config.dynamics_ensemble_size == 1:
            predicted_latents = predicted_latents[..., 0, :]
            reward_logits = reward_logits[..., 0]
            continue_logits = continue_logits[..., 0]
        return {
            "context_latents": context_latents,
            "predicted_latents": predicted_latents,
            "target_latents": jnp.stack(targets, axis=2),
            "reward_logits": reward_logits,
            "continue_logits": continue_logits,
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
        tokens = self.latent_proj(latents) + self.action_tokens(actions)
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

    def predict_latent(
        self,
        hidden: jax.Array,
        *,
        current_latents: jax.Array | None = None,
        head_index: int = 0,
    ) -> jax.Array:
        if self.config.dynamics_ensemble_size == 1:
            update = self.predictor(hidden)
        else:
            update = self.predictors[head_index](hidden)
        if self.config.residual_dynamics and current_latents is not None:
            update = current_latents + update
        if self.config.dynamics_ensemble_size == 1:
            return self.predictor_norm(update)
        return self.predictor_norms[head_index](update)

    def reward_from_hidden(self, hidden: jax.Array, *, head_index: int = 0) -> jax.Array:
        if self.config.dynamics_ensemble_size == 1:
            return self.reward_head(hidden)
        return self.reward_heads[head_index](hidden)

    def continue_from_hidden(
        self,
        hidden: jax.Array,
        *,
        head_index: int = 0,
    ) -> jax.Array:
        if self.config.dynamics_ensemble_size == 1:
            return self.continue_head(hidden)
        return self.continue_heads[head_index](hidden)

    def predict_next_from_history(
        self,
        latent_history: jax.Array,
        action_history: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if self.config.dynamics_ensemble_size > 1:
            z_ensemble, reward_ensemble, continue_logit_ensemble = (
                self.predict_next_ensemble_from_history(latent_history, action_history)
            )
            continue_prob = jnp.mean(jax.nn.sigmoid(continue_logit_ensemble), axis=0)
            continue_logit = jnp.log(
                jnp.clip(continue_prob, 1e-6, 1.0 - 1e-6)
                / jnp.clip(1.0 - continue_prob, 1e-6, 1.0)
            )
            return (
                jnp.mean(z_ensemble, axis=0),
                jnp.mean(reward_ensemble, axis=0),
                continue_logit,
            )

        h = self.dynamics_hidden(latent_history, action_history)
        last_h = h[:, -1]
        horizon_ids = jnp.ones((last_h.shape[0],), dtype=jnp.int32)
        z_next = self.predict_latent(
            last_h + self.horizon_embed(horizon_ids),
            current_latents=latent_history[:, -1],
        )
        reward = jnp.squeeze(self.reward_from_hidden(last_h), axis=-1)
        continue_logit = jnp.squeeze(self.continue_from_hidden(last_h), axis=-1)
        return z_next, reward, continue_logit

    def predict_next_ensemble_from_history(
        self,
        latent_history: jax.Array,
        action_history: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        h = self.dynamics_hidden(latent_history, action_history)
        last_h = h[:, -1]
        horizon_ids = jnp.ones((last_h.shape[0],), dtype=jnp.int32)
        z_next = []
        rewards = []
        continue_logits = []
        for head_index in range(self.config.dynamics_ensemble_size):
            z_next.append(
                self.predict_latent(
                    last_h + self.horizon_embed(horizon_ids),
                    current_latents=latent_history[:, -1],
                    head_index=head_index,
                )
            )
            rewards.append(
                jnp.squeeze(
                    self.reward_from_hidden(last_h, head_index=head_index),
                    axis=-1,
                )
            )
            continue_logits.append(
                jnp.squeeze(
                    self.continue_from_hidden(last_h, head_index=head_index),
                    axis=-1,
                )
            )
        return (
            jnp.stack(z_next, axis=0),
            jnp.stack(rewards, axis=0),
            jnp.stack(continue_logits, axis=0),
        )

    def action_tokens(self, actions: jax.Array) -> jax.Array:
        if self.config.action_mode == "discrete":
            return self.action_embed(actions.astype(jnp.int32))
        if actions.ndim != 3:
            raise ValueError(
                "continuous actions must be shaped [batch, time, action_dim]"
            )
        x = actions.astype(jnp.float32)
        x = self.action_encoder_hidden(x)
        x = nn.gelu(x)
        return self.action_encoder_out(x)


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


def history_windows(values: jax.Array, window: int, *, pad: str) -> jax.Array:
    """Return per-position windows ending at each position in ``values``.

    The result is shaped [batch, time, window, ...]. Left padding is only used for
    the first few stream positions where less than ``window`` real history exists.
    """

    if window < 1:
        raise ValueError("window must be >= 1")
    if pad == "edge":
        pad_values = jnp.repeat(values[:, :1], repeats=window - 1, axis=1)
    elif pad == "zero":
        pad_values = jnp.zeros(
            (values.shape[0], window - 1, *values.shape[2:]), dtype=values.dtype
        )
    elif pad == "done":
        pad_values = jnp.ones(
            (values.shape[0], window - 1, *values.shape[2:]), dtype=values.dtype
        )
    else:
        raise ValueError("pad must be one of: edge, zero, done")
    padded = jnp.concatenate([pad_values, values], axis=1)
    return jnp.stack(
        [padded[:, index : index + values.shape[1]] for index in range(window)],
        axis=2,
    )


def append_history(history: jax.Array, values: jax.Array) -> jax.Array:
    return jnp.concatenate([history[:, :, 1:], values[:, :, None]], axis=2)


def sinusoidal_position_embedding(time: int, dim: int) -> jax.Array:
    half = dim // 2
    positions = jnp.arange(time, dtype=jnp.float32)[:, None]
    freqs = jnp.exp(-math.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
    angles = positions * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    if emb.shape[-1] < dim:
        emb = jnp.pad(emb, ((0, 0), (0, dim - emb.shape[-1])))
    return emb
