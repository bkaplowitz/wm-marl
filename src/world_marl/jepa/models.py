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
    optimizer_warmup_steps: int = 0
    adaptive_grad_clip: float = 0.0
    optimizer_epsilon: float = 1e-5
    actor_hidden_dim: int = 0
    critic_hidden_dim: int = 0
    actor_num_layers: int = 1
    critic_num_layers: int = 1
    actor_layer_norm: bool = False
    critic_layer_norm: bool = False
    actor_log_std_min: float = -5.0
    actor_log_std_max: float = 2.0
    actor_output_scale: float = 1.0
    value_output_scale: float = 1.0
    reward_output_scale: float = 1.0
    regularizer_weight: float = 0.05
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    reward_weight: float = 1.0
    continue_weight: float = 1.0
    twohot_bins: int = 41
    twohot_min: float = -20.0
    twohot_max: float = 20.0
    gamma: float = 0.99
    lambda_return: float = 0.95

    def __post_init__(self) -> None:
        if self.action_mode not in ("discrete", "continuous"):
            raise ValueError("action_mode must be one of: discrete, continuous")
        if self.twohot_bins < 3:
            raise ValueError("twohot_bins must be >= 3")
        if self.twohot_min >= self.twohot_max:
            raise ValueError("twohot_min must be < twohot_max")
        if self.sigreg_knots < 2:
            raise ValueError("sigreg_knots must be >= 2")
        if self.sigreg_num_proj < 1:
            raise ValueError("sigreg_num_proj must be >= 1")
        if self.max_horizon < 1:
            raise ValueError("max_horizon must be >= 1")
        if self.context_window < 1:
            raise ValueError("context_window must be >= 1")
        if self.model_grad_clip_norm < 0.0:
            raise ValueError("model_grad_clip_norm must be >= 0")
        if self.actor_grad_clip_norm < 0.0:
            raise ValueError("actor_grad_clip_norm must be >= 0")
        if self.critic_grad_clip_norm < 0.0:
            raise ValueError("critic_grad_clip_norm must be >= 0")
        if self.optimizer_warmup_steps < 0:
            raise ValueError("optimizer_warmup_steps must be >= 0")
        if self.adaptive_grad_clip < 0.0:
            raise ValueError("adaptive_grad_clip must be >= 0")
        if self.optimizer_epsilon <= 0.0:
            raise ValueError("optimizer_epsilon must be > 0")
        if self.actor_hidden_dim < 0:
            raise ValueError("actor_hidden_dim must be >= 0")
        if self.critic_hidden_dim < 0:
            raise ValueError("critic_hidden_dim must be >= 0")
        if self.actor_num_layers < 1:
            raise ValueError("actor_num_layers must be >= 1")
        if self.critic_num_layers < 1:
            raise ValueError("critic_num_layers must be >= 1")
        if self.actor_log_std_min >= self.actor_log_std_max:
            raise ValueError("actor_log_std_min must be < actor_log_std_max")
        for name in (
            "actor_output_scale",
            "value_output_scale",
            "reward_output_scale",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if (self.model_dim // self.num_heads) % 2 != 0:
            raise ValueError("per-head dimension must be even for RoPE")


def apply_activation(x: jax.Array) -> jax.Array:
    return nn.silu(x)


def normalization_module(*, name: str):
    return nn.RMSNorm(name=name)


def apply_normalization(
    x: jax.Array,
    *,
    name: str | None = None,
) -> jax.Array:
    return nn.RMSNorm(name=name)(x)


def scaled_kernel_init(scale: float):
    if scale == 0.0:
        return nn.initializers.zeros_init()
    return nn.initializers.variance_scaling(
        scale**2,
        "fan_in",
        "truncated_normal",
    )


class MLPEncoder(nn.Module):
    latent_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        x = symlog(observations.astype(jnp.float32))
        x = nn.Dense(self.hidden_dim)(x)
        x = apply_activation(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = apply_activation(x)
        x = nn.Dense(self.latent_dim)(x)
        return apply_normalization(x)


class MLPHead(nn.Module):
    output_dim: int
    hidden_dim: int
    num_layers: int = 1
    use_layer_norm: bool = False
    output_scale: float = 1.0

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        default_layout = self.num_layers == 1 and not self.use_layer_norm
        if self.use_layer_norm:
            x = apply_normalization(x, name="input_norm")
        for index in range(self.num_layers):
            if default_layout:
                x = nn.Dense(self.hidden_dim)(x)
            else:
                x = nn.Dense(self.hidden_dim, name=f"hidden_{index}")(x)
            x = apply_activation(x)
        output_kwargs = {}
        if self.output_scale != 1.0:
            output_kwargs["kernel_init"] = scaled_kernel_init(self.output_scale)
        if default_layout:
            return nn.Dense(self.output_dim, **output_kwargs)(x)
        return nn.Dense(self.output_dim, name="out", **output_kwargs)(x)


class TransformerBlock(nn.Module):
    model_dim: int
    num_heads: int
    mlp_ratio: int

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        h = apply_normalization(x)
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
        h = apply_normalization(x)
        h = nn.Dense(
            2 * self.mlp_ratio * self.model_dim,
            name="geglu_in",
        )(h)
        value, gate = jnp.split(h, 2, axis=-1)
        h = value * apply_activation(gate)
        h = nn.Dense(
            self.model_dim,
            name="geglu_out",
        )(h)
        return x + h


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
        self.dynamics_norm = normalization_module(name="dynamics_norm")
        self.predictor = MLPHead(
            self.config.latent_dim,
            self.config.model_dim,
            name="predictor",
        )
        self.predictor_norm = normalization_module(name="predictor_norm")
        self.reward_head = MLPHead(
            self.config.twohot_bins,
            self.config.model_dim,
            output_scale=self.config.reward_output_scale,
            name="reward_head",
        )
        self.continue_head = MLPHead(
            1,
            self.config.model_dim,
            name="continue_head",
        )
        actor_output_dim = self.config.action_dim
        if self.config.action_mode == "continuous":
            actor_output_dim = 2 * self.config.action_dim
        actor_hidden_dim = self.config.actor_hidden_dim or self.config.model_dim
        critic_hidden_dim = self.config.critic_hidden_dim or self.config.model_dim
        self.actor_head = MLPHead(
            actor_output_dim,
            actor_hidden_dim,
            num_layers=self.config.actor_num_layers,
            use_layer_norm=self.config.actor_layer_norm,
            output_scale=self.config.actor_output_scale,
            name="actor_head",
        )
        self.value_head = MLPHead(
            self.config.twohot_bins,
            critic_hidden_dim,
            num_layers=self.config.critic_num_layers,
            use_layer_norm=self.config.critic_layer_norm,
            output_scale=self.config.value_output_scale,
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
        means, log_stds = self.actor_stats_from_latent(latents)
        values = self.value_from_latent(latents)
        return means, log_stds, values

    def actor_stats_from_latent(
        self,
        latents: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        logits = self.actor_logits_from_latent(latents)
        if self.config.action_mode == "continuous":
            means, log_stds = jnp.split(logits, 2, axis=-1)
            log_stds = jnp.clip(
                log_stds,
                self.config.actor_log_std_min,
                self.config.actor_log_std_max,
            )
        else:
            means = logits
            log_stds = jnp.zeros_like(means)
        return means, log_stds

    def value_from_latent(self, latents: jax.Array) -> jax.Array:
        value_logits = self.value_logits_from_latent(latents)
        values = scalar_prediction_from_logits(
            value_logits,
            num_bins=self.config.twohot_bins,
            low=self.config.twohot_min,
            high=self.config.twohot_max,
        )
        return values

    def actor_value_logits_from_latent(
        self,
        latents: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return (
            self.actor_logits_from_latent(latents),
            self.value_logits_from_latent(latents),
        )

    def actor_logits_from_latent(self, latents: jax.Array) -> jax.Array:
        flat = latents.reshape((-1, self.config.latent_dim))
        logits = self.actor_head(flat)
        return logits.reshape((*latents.shape[:-1], logits.shape[-1]))

    def value_logits_from_latent(self, latents: jax.Array) -> jax.Array:
        flat = latents.reshape((-1, self.config.latent_dim))
        value_logits = self.value_head(flat)
        return value_logits.reshape(
            (*latents.shape[:-1], value_logits.shape[-1]),
        )

    def sequence_outputs(
        self,
        observations: jax.Array,
        actions: jax.Array,
        *,
        chunk_length: int,
        is_last: jax.Array | None = None,
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
        last_history = (
            None
            if is_last is None
            else history_windows(
                is_last[:, :chunk_length],
                self.config.context_window,
                pad="last",
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
            flat_is_last = (
                None
                if last_history is None
                else last_history.reshape((flat_batch, self.config.context_window))
            )
            h = self.dynamics_hidden(
                flat_latents,
                flat_actions,
                is_last=flat_is_last,
            )
            last_h = h[:, -1]
            # Recurrent imagination calls the next-step API repeatedly with the
            # one-step horizon id. Train the autoregressive path with the same
            # convention so actor rollouts do not see a different interface.
            horizon_ids = jnp.ones((flat_batch,), dtype=jnp.int32)
            current_latents = flat_latents[:, -1]
            next_latents = self.predict_latent(
                last_h + self.horizon_embed(horizon_ids),
                current_latents=current_latents,
            )
            predictions.append(
                next_latents.reshape((batch_size, chunk_length, self.config.latent_dim))
            )
            targets.append(
                jax.lax.stop_gradient(
                    latents[:, step_index + 1 : step_index + 1 + chunk_length]
                )
            )
            current_reward_logits = self.reward_from_hidden(last_h)
            reward_logits.append(
                current_reward_logits.reshape(
                    (batch_size, chunk_length, current_reward_logits.shape[-1])
                )
            )
            reward_values.append(
                self.reward_value_from_logits(current_reward_logits).reshape(
                    (batch_size, chunk_length)
                )
            )
            continues.append(
                jnp.squeeze(
                    self.continue_from_hidden(last_h),
                    axis=-1,
                ).reshape((batch_size, chunk_length))
            )
            if step_index + 1 < max_horizon:
                latent_history = append_history(
                    latent_history,
                    predictions[-1],
                )
                action_history = append_history(
                    action_history,
                    actions[:, step_index + 1 : step_index + 1 + chunk_length],
                )
                if last_history is not None:
                    last_history = append_history(
                        last_history,
                        is_last[
                            :,
                            step_index + 1 : step_index + 1 + chunk_length,
                        ],
                    )
        predicted_latents = jnp.stack(predictions, axis=2)
        reward_logits = jnp.stack(reward_logits, axis=2)
        reward_values = jnp.stack(reward_values, axis=2)
        continue_logits = jnp.stack(continues, axis=2)
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
        is_last: jax.Array | None = None,
    ) -> jax.Array:
        if latents.ndim != 3:
            raise ValueError("latents must be shaped [batch, time, latent_dim]")
        tokens = self.latent_proj(latents) + self.action_tokens(actions)
        time = tokens.shape[1]
        mask = causal_attention_mask(
            time,
            context_window=self.config.context_window,
            is_last=is_last,
        )
        h = tokens
        for block in self.blocks:
            h = block(h, mask)
        return self.dynamics_norm(h)

    def predict_latent(
        self,
        hidden: jax.Array,
        *,
        current_latents: jax.Array,
    ) -> jax.Array:
        return self.predictor_norm(current_latents + self.predictor(hidden))

    def reward_from_hidden(self, hidden: jax.Array) -> jax.Array:
        return self.reward_head(hidden)

    def reward_value_from_logits(self, logits: jax.Array) -> jax.Array:
        return scalar_prediction_from_logits(
            logits,
            num_bins=self.config.twohot_bins,
            low=self.config.twohot_min,
            high=self.config.twohot_max,
        )

    def continue_from_hidden(self, hidden: jax.Array) -> jax.Array:
        return self.continue_head(hidden)

    def predict_next_from_history(
        self,
        latent_history: jax.Array,
        action_history: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
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

    def action_tokens(self, actions: jax.Array) -> jax.Array:
        if self.config.action_mode == "discrete":
            return self.action_embed(actions.astype(jnp.int32))
        if actions.ndim != 3:
            raise ValueError(
                "continuous actions must be shaped [batch, time, action_dim]"
            )
        x = actions.astype(jnp.float32)
        x = self.action_encoder_hidden(x)
        x = apply_activation(x)
        return self.action_encoder_out(x)


def causal_attention_mask(
    time: int,
    *,
    context_window: int,
    is_last: jax.Array | None = None,
) -> jax.Array:
    positions = jnp.arange(time)
    causal = positions[None, :] <= positions[:, None]
    local = (positions[:, None] - positions[None, :]) < context_window
    mask = causal & local
    mask = mask[None, None, :, :]
    if is_last is None:
        return mask

    previous_is_last = jnp.concatenate(
        [
            jnp.zeros((is_last.shape[0], 1), dtype=is_last.dtype),
            is_last[:, : time - 1],
        ],
        axis=1,
    )
    segment_ids = jnp.cumsum(previous_is_last.astype(jnp.int32), axis=1)
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
    elif pad == "last":
        pad_values = jnp.ones(
            (values.shape[0], window - 1, *values.shape[2:]), dtype=values.dtype
        )
    else:
        raise ValueError("pad must be one of: edge, zero, last")
    padded = jnp.concatenate([pad_values, values], axis=1)
    return jnp.stack(
        [padded[:, index : index + values.shape[1]] for index in range(window)],
        axis=2,
    )


def append_history(history: jax.Array, values: jax.Array) -> jax.Array:
    return jnp.concatenate([history[:, :, 1:], values[:, :, None]], axis=2)


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
    num_bins: int,
    low: float,
    high: float,
) -> jax.Array:
    return symlog_twohot_decode(logits, num_bins=num_bins, low=low, high=high)


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
