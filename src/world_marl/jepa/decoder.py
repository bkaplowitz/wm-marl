"""Post-hoc observation decoder used only as a visual JEPA diagnostic.

The world model itself stays reconstruction-free (LeJEPA-style): this decoder
is fit after world-model training, on frozen latents, and its gradients never
touch the encoder or dynamics. It exists so imagined open-loop rollouts can be
decoded back to observation space and compared frame-by-frame against the real
environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.replay import ReplayBatch
from world_marl.jepa.training import (
    JepaTrainState,
    _normalize,
    open_loop_predicted_latents,
)


@dataclass(frozen=True)
class DecoderConfig:
    latent_dim: int
    observation_dim: int
    hidden_dim: int = 256
    learning_rate: float = 1e-3
    grad_clip_norm: float = 100.0

    def __post_init__(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.observation_dim < 1:
            raise ValueError("observation_dim must be >= 1")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if self.grad_clip_norm < 0.0:
            raise ValueError("grad_clip_norm must be >= 0")


class ObservationDecoder(nn.Module):
    """MLP probe mirroring MLPEncoder, mapping latents back to observations."""

    observation_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, latents: jax.Array) -> jax.Array:
        x = latents.astype(jnp.float32)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        return nn.Dense(self.observation_dim)(x)


def create_decoder_train_state(key: jax.Array, config: DecoderConfig) -> TrainState:
    module = ObservationDecoder(
        observation_dim=config.observation_dim,
        hidden_dim=config.hidden_dim,
    )
    params = module.init(
        key,
        jnp.zeros((1, config.latent_dim), dtype=jnp.float32),
    )["params"]
    transforms = []
    if config.grad_clip_norm > 0.0:
        transforms.append(optax.clip_by_global_norm(config.grad_clip_norm))
    transforms.append(optax.adam(config.learning_rate))
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=optax.chain(*transforms),
    )


@jax.jit
def encode_observations(state: JepaTrainState, observations: jax.Array) -> jax.Array:
    return state.apply_fn(
        {"params": state.params},
        observations,
        method=JepaWorldModel.encode,
    )


@jax.jit
def train_decoder_step(
    decoder_state: TrainState,
    latents: jax.Array,
    observations: jax.Array,
) -> tuple[TrainState, jax.Array]:
    def loss_fn(params):
        recon = decoder_state.apply_fn({"params": params}, latents)
        return jnp.mean(jnp.square(recon - observations.astype(recon.dtype)))

    loss, grads = jax.value_and_grad(loss_fn)(decoder_state.params)
    return decoder_state.apply_gradients(grads=grads), loss


@jax.jit
def decoder_reconstruction_mse(
    decoder_state: TrainState,
    latents: jax.Array,
    observations: jax.Array,
) -> jax.Array:
    recon = decoder_state.apply_fn({"params": decoder_state.params}, latents)
    return jnp.mean(jnp.square(recon - observations.astype(recon.dtype)))


@partial(jax.jit, static_argnames=("config", "horizon"))
def decode_open_loop_rollout(
    state: JepaTrainState,
    decoder_state: TrainState,
    batch: ReplayBatch,
    config: JepaConfig,
    *,
    horizon: int,
) -> dict[str, jax.Array]:
    """Decode an imagined open-loop rollout next to the real trajectory.

    ``imagined_observations`` decodes latents rolled forward by the dynamics
    from encoded context (replay actions, no re-encoding), while
    ``reconstructed_observations`` decodes the encoder's latents of the real
    future observations, isolating decoder error from dynamics error.
    """
    predicted, encoded, validity = open_loop_predicted_latents(
        state,
        batch,
        config,
        horizon=horizon,
    )
    context_window = config.context_window

    def decode(latents: jax.Array) -> jax.Array:
        return decoder_state.apply_fn({"params": decoder_state.params}, latents)

    target = encoded[:, context_window : context_window + horizon]
    cosine = jnp.sum(_normalize(predicted) * _normalize(target), axis=-1)
    return {
        "context_observations": batch.observations[:, :context_window],
        "real_observations": batch.observations[
            :, context_window : context_window + horizon
        ],
        "decoded_context": decode(encoded[:, :context_window]),
        "reconstructed_observations": decode(target),
        "imagined_observations": decode(predicted),
        "open_loop_cosine": cosine,
        "validity": validity,
    }


def select_display_trajectories(
    batch: ReplayBatch,
    *,
    context_window: int,
    horizon: int,
    count: int,
) -> ReplayBatch:
    """Pick sampled windows for display, preferring ones without resets."""
    if count < 1:
        raise ValueError("count must be >= 1")
    dones = np.asarray(batch.dones)[:, : context_window + horizon - 1]
    order = np.argsort(dones.sum(axis=1), kind="stable")
    chosen = np.sort(order[:count])
    return ReplayBatch(
        observations=batch.observations[chosen],
        actions=batch.actions[chosen],
        rewards=batch.rewards[chosen],
        is_last=batch.is_last[chosen],
        is_terminal=batch.is_terminal[chosen],
    )
