"""Genie-style transformer VQ-VAE observation tokenizer.

Analog of the original Genie video tokenizer for state vectors: each
observation dimension is one token, a pre-LN transformer encoder maps the
tokens to ``code_dim`` latents, each latent snaps to its nearest row of a
learned vector codebook (straight-through estimator), and a transformer
decoder reconstructs the observation. Training is stage-wise as in Genie —
reconstruction + VQ codebook/commitment losses only; the token world model
trains on the detached code ids via the unchanged ``genwm_train_step``. The
fitted codebook is exported as a :class:`CodebookTokenizer`, so the policy and
reward/continue head consume flattened codebook embeddings
(``obs_dim * code_dim`` floats) and imagination stays token-native.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

COMMITMENT_BETA = 0.25


class _TokenTransformer(nn.Module):
    """Pre-LN attention/FFN trunk over per-dimension tokens with a linear head."""

    model_dim: int
    num_heads: int
    num_layers: int
    mlp_ratio: int
    out_dim: int

    @nn.compact
    def __call__(self, tokens: jax.Array) -> jax.Array:
        h = nn.Dense(self.model_dim)(tokens)
        pos_emb = self.param(
            "pos_emb",
            nn.initializers.normal(stddev=0.02),
            (tokens.shape[-2], self.model_dim),
        )
        h = h + pos_emb
        for _ in range(self.num_layers):
            z = nn.LayerNorm()(h)
            h = h + nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(
                z, z, deterministic=True
            )
            y = nn.LayerNorm()(h)
            y = nn.silu(nn.Dense(self.mlp_ratio * self.model_dim)(y))
            h = h + nn.Dense(self.model_dim)(y)
        return nn.Dense(self.out_dim)(nn.LayerNorm()(h))


class GenieTokenizer(nn.Module):
    obs_dim: int
    codebook_size: int
    code_dim: int
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    mlp_ratio: int = 4

    def setup(self) -> None:
        self.encoder = _TokenTransformer(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_ratio=self.mlp_ratio,
            out_dim=self.code_dim,
        )
        self.decoder = _TokenTransformer(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_ratio=self.mlp_ratio,
            out_dim=1,
        )
        self.codebook = self.param(
            "codebook",
            nn.initializers.normal(stddev=0.02),
            (self.codebook_size, self.code_dim),
        )

    def quantize(self, latents: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Nearest codebook row per token, with a straight-through gradient."""
        distances = jnp.sum((latents[..., None, :] - self.codebook) ** 2, axis=-1)
        ids = jnp.argmin(distances, axis=-1).astype(jnp.int32)
        codes = self.codebook[ids]
        quantized = latents + jax.lax.stop_gradient(codes - latents)
        return quantized, codes, ids

    def __call__(
        self, observations: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        latents = self.encoder(observations[..., None])
        quantized, codes, ids = self.quantize(latents)
        recon = self.decoder(quantized)[..., 0]
        return recon, latents, codes, ids

    def encode(self, observations: jax.Array) -> jax.Array:
        """``[..., obs_dim]`` floats -> ``[..., obs_dim * code_dim]`` embeddings."""
        latents = self.encoder(observations[..., None])
        quantized, _, _ = self.quantize(latents)
        return quantized.reshape((*observations.shape[:-1], -1))


def create_genie_state(
    key: jax.Array, module: GenieTokenizer, *, learning_rate: float
) -> TrainState:
    params = module.init(key, jnp.zeros((1, module.obs_dim), dtype=jnp.float32))[
        "params"
    ]
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=optax.adam(learning_rate)
    )


@jax.jit
def genie_train_step(
    state: TrainState, observations: jax.Array
) -> tuple[TrainState, dict[str, jax.Array]]:
    """One VQ-VAE update: recon MSE + ||sg(z) - e||^2 + beta * ||z - sg(e)||^2."""

    def loss_fn(params):
        recon, latents, codes, _ = state.apply_fn({"params": params}, observations)
        recon_loss = jnp.mean((recon - observations) ** 2)
        codebook_loss = jnp.mean((jax.lax.stop_gradient(latents) - codes) ** 2)
        commit_loss = jnp.mean((latents - jax.lax.stop_gradient(codes)) ** 2)
        total = recon_loss + codebook_loss + COMMITMENT_BETA * commit_loss
        return total, {
            "genie_total_loss": total,
            "genie_recon_loss": recon_loss,
            "genie_codebook_loss": codebook_loss,
            "genie_commit_loss": commit_loss,
        }

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), metrics


def make_genie_encode(module: GenieTokenizer):
    """One jitted ``(params, observations) -> embeddings`` closure per run.

    ``scan_rollout`` caches compiled programs by ``id(fn)``, so callers must
    create this once and reuse it; the encoder params ride through as a traced
    argument so co-training never recompiles.
    """

    @jax.jit
    def encode(params, observations: jax.Array):
        return module.apply(
            {"params": params}, observations, method=GenieTokenizer.encode
        )

    return encode
