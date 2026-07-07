"""Flax modules for the single-agent generative world-model arms."""

from __future__ import annotations

from collections.abc import Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from flow_matching.models import sinusoidal_time_embedding


class ContinuousTokenTransformer(nn.Module):
    """Continuous-target twin of ``flow_matching.models.TokenizedDiscreteTransformer``.

    Treats the d observation dimensions as a length-d sequence: each scalar is
    lifted with a shared Dense instead of an ``nn.Embed`` lookup, and the
    per-token head is Dense(1) (a velocity) instead of Dense(V) (logits).
    Everything else — learned absolute positions, the prepended [COND] token
    carrying (sinusoidal-t, cond_vars), the pre-LN attention/FFN stack — mirrors
    the discrete twin so the two arms are capacity-matched by construction.
    Honors the ``MLPVectorField`` contract ``(x, t, cond_vars) -> (B, d)``, so
    the existing conditional flow-matching loss/train/sample functions apply
    unchanged.
    """

    model_dim: int = 64
    num_heads: int = 4
    ffn_hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        t: jax.Array,
        cond_vars: jax.Array | None = None,
    ) -> jax.Array:
        num_factors = x.shape[-1]
        h = nn.Dense(self.model_dim)(x[..., None])
        pos = self.param(
            "pos_emb", nn.initializers.normal(0.02), (num_factors, self.model_dim)
        )
        h = h + pos

        c = nn.Dense(self.model_dim)(sinusoidal_time_embedding(t, self.model_dim))
        if cond_vars is not None:
            c = c + nn.Dense(self.model_dim)(cond_vars)
        h = jnp.concatenate([c[:, None, :], h], axis=1)

        for ffn_dim in self.ffn_hidden_dims:
            z = nn.LayerNorm()(h)
            h = h + nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(
                z, z, deterministic=True
            )
            y = nn.LayerNorm()(h)
            y = nn.silu(nn.Dense(ffn_dim)(y))
            h = h + nn.Dense(self.model_dim)(y)

        h = nn.LayerNorm()(h)[:, 1:, :]
        return nn.Dense(1)(h)[..., 0]


class RewardContinueHead(nn.Module):
    """Predicts (reward, continue-logit) from (observation, action features).

    Trained on real replay transitions and consumed inside the imagined
    rollout, where its parameters are traced — unlike the coins pipeline's
    ``reward_done_fn``, which is a static jit argument and therefore cannot
    learn. Conditioning on (s, a) rather than (s, a, s') keeps terminal-step
    replay targets valid: with auto-resetting envs the stored next observation
    at a terminal step is the post-reset one.
    """

    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
        action_features: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        x = jnp.concatenate([observations, action_features], axis=-1)
        for dim in self.hidden_dims:
            x = nn.silu(nn.Dense(dim)(x))
        reward = nn.Dense(1)(x)[..., 0]
        continue_logit = nn.Dense(1)(x)[..., 0]
        return reward, continue_logit


class GaussianMLPActorCritic(nn.Module):
    """Diag-Gaussian twin of ``algs.networks.MLPActorCritic`` for continuous actions.

    Same trunk widths (128 shared, 64 per head) and orthogonal inits; the actor
    head emits a mean and a state-independent learned log-std. Samples are
    unsquashed — callers clip to the action bounds before stepping, and PPO
    ratios use the pre-clip log-probs (standard clipped-Gaussian PPO practice).
    """

    action_dim: int
    activation: str = "relu"
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray
    ) -> tuple[distrax.MultivariateNormalDiag, jnp.ndarray]:
        if self.activation == "tanh":
            activation = nn.tanh
        elif self.activation == "relu":
            activation = nn.relu
        else:
            raise ValueError(f"unsupported activation {self.activation!r}")

        x = observations.astype(jnp.float32)
        x = x.reshape((x.shape[0], -1))
        embedding = nn.Dense(
            features=128,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
            name="shared_dense",
        )(x)
        embedding = activation(embedding)

        actor_hidden = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(embedding)
        actor_hidden = activation(actor_hidden)
        mean = nn.Dense(
            features=self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_hidden)
        log_std = self.param("log_std", constant(0.0), (self.action_dim,))
        scale = jnp.exp(jnp.clip(log_std, self.log_std_min, self.log_std_max))
        policy = distrax.MultivariateNormalDiag(
            loc=mean, scale_diag=jnp.broadcast_to(scale, mean.shape)
        )

        critic_hidden = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(embedding)
        critic_hidden = activation(critic_hidden)
        value = nn.Dense(
            features=1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic_hidden)

        return policy, jnp.squeeze(value, axis=-1)
