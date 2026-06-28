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
    model_grad_clip_norm: float = 100.0
    actor_grad_clip_norm: float = 10.0
    critic_grad_clip_norm: float = 100.0
    stochastic_actor: bool = False
    actor_log_std_min: float = -5.0
    actor_log_std_max: float = 2.0
    regularizer: str = "sigreg"
    regularizer_weight: float = 0.05
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    reward_weight: float = 1.0
    continue_weight: float = 1.0
    reward_prediction_mode: str = "mse"
    value_prediction_mode: str = "mse"
    twohot_bins: int = 41
    twohot_min: float = -20.0
    twohot_max: float = 20.0
    clip_imagined_rewards: bool = False
    imagined_reward_min: float = 0.0
    imagined_reward_max: float = 1.0
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
        if self.reward_prediction_mode not in ("mse", "symlog_twohot"):
            raise ValueError(
                "reward_prediction_mode must be one of: mse, symlog_twohot"
            )
        if self.value_prediction_mode not in ("mse", "symlog_twohot"):
            raise ValueError("value_prediction_mode must be one of: mse, symlog_twohot")
        if self.twohot_bins < 3:
            raise ValueError("twohot_bins must be >= 3")
        if self.twohot_min >= self.twohot_max:
            raise ValueError("twohot_min must be < twohot_max")
        if self.imagined_reward_min >= self.imagined_reward_max:
            raise ValueError("imagined_reward_min must be < imagined_reward_max")
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
        if self.model_grad_clip_norm < 0.0:
            raise ValueError("model_grad_clip_norm must be >= 0")
        if self.actor_grad_clip_norm < 0.0:
            raise ValueError("actor_grad_clip_norm must be >= 0")
        if self.critic_grad_clip_norm < 0.0:
            raise ValueError("critic_grad_clip_norm must be >= 0")
        if self.actor_log_std_min >= self.actor_log_std_max:
            raise ValueError("actor_log_std_min must be < actor_log_std_max")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if (self.model_dim // self.num_heads) % 2 != 0:
            raise ValueError("per-head dimension must be even for RoPE")


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
        head_dim = self.model_dim // self.num_heads
        query = nn.DenseGeneral(
            (self.num_heads, head_dim),
            axis=-1,
            use_bias=False,
            name="query",
        )(h)
        key = nn.DenseGeneral(
            (self.num_heads, head_dim),
            axis=-1,
            use_bias=False,
            name="key",
        )(h)
        value = nn.DenseGeneral(
            (self.num_heads, head_dim),
            axis=-1,
            use_bias=False,
            name="value",
        )(h)
        query = apply_rotary_position_embedding(query)
        key = apply_rotary_position_embedding(key)
        attention = nn.dot_product_attention(
            query,
            key,
            value,
            mask=mask,
            deterministic=True,
        )
        attention = nn.DenseGeneral(
            self.model_dim,
            axis=(-2, -1),
            name="attention_out",
        )(attention)
        x = x + attention
        h = nn.LayerNorm()(x)
        h = nn.Dense(
            2 * self.mlp_ratio * self.model_dim,
            name="geglu_in",
        )(h)
        value, gate = jnp.split(h, 2, axis=-1)
        h = value * nn.gelu(gate)
        h = nn.Dense(
            self.model_dim,
            name="geglu_out",
        )(h)
        return x + h


class JepaWorldModel(nn.Module):
    config: JepaConfig

    def setup(self) -> None:
        reward_output_dim = prediction_output_dim(
            self.config.reward_prediction_mode,
            self.config.twohot_bins,
        )
        value_output_dim = prediction_output_dim(
            self.config.value_prediction_mode,
            self.config.twohot_bins,
        )
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
            self.reward_head = MLPHead(
                reward_output_dim,
                self.config.model_dim,
                name="reward_head",
            )
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
                MLPHead(
                    reward_output_dim,
                    self.config.model_dim,
                    name=f"reward_head_{index}",
                )
                for index in range(self.config.dynamics_ensemble_size)
            ]
            self.continue_heads = [
                MLPHead(1, self.config.model_dim, name=f"continue_head_{index}")
                for index in range(self.config.dynamics_ensemble_size)
            ]
        actor_output_dim = self.config.action_dim
        if self.config.action_mode == "continuous" and self.config.stochastic_actor:
            actor_output_dim = 2 * self.config.action_dim
        self.actor_head = MLPHead(
            actor_output_dim,
            self.config.model_dim,
            name="actor_head",
        )
        self.value_head = MLPHead(
            value_output_dim,
            self.config.model_dim,
            name="value_head",
        )

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
        means, _, values = self.actor_value_stats_from_latent(latents)
        return means, values

    def actor_value_stats_from_latent(
        self,
        latents: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        logits, value_logits = self.actor_value_logits_from_latent(latents)
        if self.config.action_mode == "continuous" and self.config.stochastic_actor:
            means, log_stds = jnp.split(logits, 2, axis=-1)
            log_stds = jnp.clip(
                log_stds,
                self.config.actor_log_std_min,
                self.config.actor_log_std_max,
            )
        else:
            means = logits
            log_stds = jnp.zeros_like(means)
        values = scalar_prediction_from_logits(
            value_logits,
            mode=self.config.value_prediction_mode,
            num_bins=self.config.twohot_bins,
            low=self.config.twohot_min,
            high=self.config.twohot_max,
        )
        return means, log_stds, values

    def actor_value_logits_from_latent(
        self,
        latents: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        flat = latents.reshape((-1, self.config.latent_dim))
        logits = self.actor_head(flat)
        value_logits = self.value_head(flat)
        return (
            logits.reshape((*latents.shape[:-1], logits.shape[-1])),
            value_logits.reshape(
                (*latents.shape[:-1], value_logits.shape[-1]),
            ),
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
        reward_logits = []
        reward_values = []
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
            # Recurrent imagination calls the next-step API repeatedly with the
            # one-step horizon id. Train the autoregressive path with the same
            # convention so actor rollouts do not see a different interface.
            horizon_ids = jnp.ones((flat_batch,), dtype=jnp.int32)
            current_latents = flat_latents[:, -1]
            head_predictions = []
            head_reward_logits = []
            head_reward_values = []
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
                current_reward_logits = self.reward_from_hidden(
                    last_h,
                    head_index=head_index,
                )
                head_reward_logits.append(
                    current_reward_logits.reshape(
                        (batch_size, chunk_length, current_reward_logits.shape[-1])
                    )
                )
                head_reward_values.append(
                    self.reward_value_from_logits(current_reward_logits).reshape(
                        (batch_size, chunk_length)
                    )
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
            reward_logits.append(jnp.stack(head_reward_logits, axis=2))
            reward_values.append(jnp.stack(head_reward_values, axis=2))
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
        reward_logits = jnp.stack(reward_logits, axis=2)
        reward_values = jnp.stack(reward_values, axis=2)
        continue_logits = jnp.stack(continues, axis=2)
        if self.config.dynamics_ensemble_size == 1:
            predicted_latents = predicted_latents[..., 0, :]
            reward_logits = reward_logits[..., 0, :]
            reward_values = reward_values[..., 0]
            continue_logits = continue_logits[..., 0]
        if self.config.reward_prediction_mode == "mse":
            reward_logits = jnp.squeeze(reward_logits, axis=-1)
        return {
            "context_latents": context_latents,
            "predicted_latents": predicted_latents,
            "target_latents": jnp.stack(targets, axis=2),
            "reward_logits": reward_logits,
            "reward_values": reward_values,
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
        mask = causal_attention_mask(
            time,
            context_window=self.config.context_window,
            dones=dones,
        )
        h = tokens
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

    def reward_from_hidden(
        self, hidden: jax.Array, *, head_index: int = 0
    ) -> jax.Array:
        if self.config.dynamics_ensemble_size == 1:
            return self.reward_head(hidden)
        return self.reward_heads[head_index](hidden)

    def reward_value_from_logits(self, logits: jax.Array) -> jax.Array:
        return scalar_prediction_from_logits(
            logits,
            mode=self.config.reward_prediction_mode,
            num_bins=self.config.twohot_bins,
            low=self.config.twohot_min,
            high=self.config.twohot_max,
        )

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
        reward = self.reward_value_from_logits(self.reward_from_hidden(last_h))
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
                self.reward_value_from_logits(
                    self.reward_from_hidden(last_h, head_index=head_index)
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


def prediction_output_dim(mode: str, num_bins: int) -> int:
    if mode == "mse":
        return 1
    if mode == "symlog_twohot":
        return num_bins
    raise ValueError(f"unknown prediction mode: {mode}")


def symlog(values: jax.Array) -> jax.Array:
    return jnp.sign(values) * jnp.log1p(jnp.abs(values))


def symexp(values: jax.Array) -> jax.Array:
    return jnp.sign(values) * jnp.expm1(jnp.abs(values))


def twohot_support(num_bins: int, low: float, high: float) -> jax.Array:
    return jnp.linspace(low, high, num_bins, dtype=jnp.float32)


def symlog_twohot(
    values: jax.Array, *, num_bins: int, low: float, high: float
) -> jax.Array:
    encoded = jnp.clip(symlog(values), low, high)
    bin_width = (high - low) / float(num_bins - 1)
    position = (encoded - low) / bin_width
    lower = jnp.floor(position).astype(jnp.int32)
    lower = jnp.clip(lower, 0, num_bins - 1)
    upper = jnp.clip(lower + 1, 0, num_bins - 1)
    upper_weight = position - lower.astype(encoded.dtype)
    lower_weight = 1.0 - upper_weight
    return (
        jax.nn.one_hot(lower, num_bins, dtype=encoded.dtype) * lower_weight[..., None]
        + jax.nn.one_hot(upper, num_bins, dtype=encoded.dtype) * upper_weight[..., None]
    )


def symlog_twohot_decode(
    logits: jax.Array, *, num_bins: int, low: float, high: float
) -> jax.Array:
    probs = jax.nn.softmax(logits, axis=-1)
    support = twohot_support(num_bins, low, high).astype(logits.dtype)
    encoded = jnp.sum(probs * support, axis=-1)
    return symexp(encoded)


def scalar_prediction_from_logits(
    logits: jax.Array,
    *,
    mode: str,
    num_bins: int,
    low: float,
    high: float,
) -> jax.Array:
    if mode == "mse":
        return jnp.squeeze(logits, axis=-1)
    if mode == "symlog_twohot":
        return symlog_twohot_decode(logits, num_bins=num_bins, low=low, high=high)
    raise ValueError(f"unknown prediction mode: {mode}")


def apply_rotary_position_embedding(x: jax.Array) -> jax.Array:
    """Apply RoPE to q/k tensors shaped [batch, time, heads, head_dim]."""

    dim = x.shape[-1]
    half = dim // 2
    time = x.shape[1]
    positions = jnp.arange(time, dtype=jnp.float32)[:, None]
    freqs = jnp.exp(-math.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
    angles = positions * freqs[None, :]
    sin = jnp.sin(angles)[None, :, None, :]
    cos = jnp.cos(angles)[None, :, None, :]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = jnp.stack(
        [
            x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos,
        ],
        axis=-1,
    )
    return rotated.reshape(x.shape)
