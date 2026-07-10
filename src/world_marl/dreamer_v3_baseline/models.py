from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class DreamerEncoder(nn.Module):
    embedding_dim: int
    hidden_dims: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        x = observations.astype(jnp.float32).reshape((observations.shape[0], -1))
        for dim in self.hidden_dims:
            x = nn.silu(nn.Dense(dim)(x))
        return nn.Dense(self.embedding_dim)(x)


class DreamerDecoder(nn.Module):
    observation_shape: tuple[int, ...]
    hidden_dims: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        x = features.astype(jnp.float32)
        for dim in self.hidden_dims:
            x = nn.silu(nn.Dense(dim)(x))
        flat_dim = 1
        for dim in self.observation_shape:
            flat_dim *= dim
        recon = nn.Dense(flat_dim)(x)
        if len(self.observation_shape) == 3 and self.observation_shape[-1] in {1, 3, 4}:
            recon = nn.sigmoid(recon)
        return recon.reshape((features.shape[0], *self.observation_shape))


class RewardHead(nn.Module):
    bins: int
    hidden_dims: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        x = features.astype(jnp.float32)
        for dim in self.hidden_dims:
            x = nn.silu(nn.Dense(dim)(x))
        return nn.Dense(self.bins)(x)


class ContinueHead(nn.Module):
    hidden_dims: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        x = features.astype(jnp.float32)
        for dim in self.hidden_dims:
            x = nn.silu(nn.Dense(dim)(x))
        return nn.Dense(1)(x)[..., 0]


class DreamerActor(nn.Module):
    action_dim: int
    action_mode: str
    hidden_dims: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, features: jax.Array) -> dict[str, jax.Array]:
        x = features.astype(jnp.float32)
        for dim in self.hidden_dims:
            x = nn.silu(nn.LayerNorm()(nn.Dense(dim)(x)))
        if self.action_mode == "discrete":
            return {"logits": nn.Dense(self.action_dim, name="logits")(x)}
        mean = nn.Dense(self.action_dim, name="mean")(x)
        log_std = jnp.clip(nn.Dense(self.action_dim, name="log_std")(x), -5.0, 2.0)
        return {"mean": mean, "log_std": log_std}


class DreamerCritic(nn.Module):
    bins: int
    hidden_dims: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        x = features.astype(jnp.float32)
        for dim in self.hidden_dims:
            x = nn.silu(nn.LayerNorm()(nn.Dense(dim)(x)))
        return nn.Dense(self.bins, name="value_logits")(x)
