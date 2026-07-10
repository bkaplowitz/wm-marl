from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from world_marl.genie2_continuous_jax.autoencoder import (
    ContinuousLatentAutoencoder,
    reconstruction_loss,
)
from world_marl.genie2_continuous_jax.config import Genie2ContinuousConfig
from world_marl.genie2_continuous_jax.dynamics import (
    CausalLatentDynamics,
    classifier_free_guidance,
    dynamics_mse_loss,
)
from world_marl.genie2_continuous_jax.lam import (
    ContinuousLAM,
    LatentActionReconstructor,
    lam_kl_loss,
    sample_latent_actions,
)
from world_marl.genie2_continuous_jax.rl_heads import RewardContinueHead
from world_marl.world_model_foundation.replay import WorldModelSequenceBatch


class Genie2WorldModel(nn.Module):
    observation_shape: tuple[int, ...]
    config: Genie2ContinuousConfig

    def setup(self) -> None:
        self.autoencoder = ContinuousLatentAutoencoder(
            latent_dim=self.config.autoencoder.latent_dim,
            hidden_dims=self.config.autoencoder.hidden_dims,
            name="autoencoder",
        )
        self.lam = ContinuousLAM(
            latent_action_dim=self.config.lam.latent_action_dim,
            hidden_dims=self.config.lam.hidden_dims,
            log_std_min=self.config.lam.log_std_min,
            log_std_max=self.config.lam.log_std_max,
            name="lam",
        )
        self.lam_reconstructor = LatentActionReconstructor(
            latent_dim=self.config.autoencoder.latent_dim,
            hidden_dims=self.config.lam.hidden_dims,
            name="lam_reconstructor",
        )
        self.dynamics = CausalLatentDynamics(
            latent_dim=self.config.autoencoder.latent_dim,
            latent_action_dim=self.config.lam.latent_action_dim,
            model_dim=self.config.dynamics.model_dim,
            num_heads=self.config.dynamics.num_heads,
            num_layers=self.config.dynamics.num_layers,
            max_context=self.config.dynamics.max_context,
            name="dynamics",
        )
        self.heads = RewardContinueHead(
            hidden_dims=self.config.reward_continue_hidden_dims,
            name="reward_continue_head",
        )

    def encode(self, observations: jax.Array) -> jax.Array:
        latents, _ = self.autoencoder(observations)
        return latents

    def decode(self, latents: jax.Array) -> jax.Array:
        dummy = jnp.zeros(
            (latents.shape[0], *self.observation_shape),
            dtype=jnp.float32,
        )
        _, observations = self.autoencoder(dummy, decode_latents=latents)
        return observations

    def infer_latent_actions(
        self,
        prev_latents: jax.Array,
        next_latents: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return self.lam(prev_latents, next_latents)

    def predict_next(
        self,
        latent_history: jax.Array,
        latent_actions: jax.Array,
    ) -> jax.Array:
        batch_size = latent_history.shape[0]
        sample = jnp.zeros(
            (batch_size, self.config.autoencoder.latent_dim),
            dtype=jnp.float32,
        )
        step_size = 1.0 / self.config.dynamics.sampling_steps
        for step in range(self.config.dynamics.sampling_steps):
            time = jnp.full(
                (batch_size,),
                (step + 0.5) * step_size,
                dtype=jnp.float32,
            )
            conditioned = self.dynamics(
                latent_history,
                latent_actions,
                time,
                jnp.ones((batch_size,), dtype=jnp.float32),
                sample,
            )
            unconditioned = self.dynamics(
                latent_history,
                latent_actions,
                time,
                jnp.zeros((batch_size,), dtype=jnp.float32),
                sample,
            )
            velocity = classifier_free_guidance(
                conditioned=conditioned,
                unconditioned=unconditioned,
                guidance_scale=self.config.dynamics.guidance_scale,
            )
            sample = sample + step_size * velocity
        return sample

    def predict_reward_continue(
        self,
        latents: jax.Array,
        latent_actions: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return self.heads(latents, latent_actions)

    def __call__(
        self,
        observations: jax.Array,
        rewards: jax.Array,
        continues: jax.Array,
    ) -> dict[str, jax.Array]:
        del rewards, continues
        time_steps, batch_size = observations.shape[:2]
        flat_obs = observations.reshape(
            (time_steps * batch_size, *self.observation_shape)
        )
        latents_flat, recon_flat = self.autoencoder(flat_obs)
        latents = latents_flat.reshape((time_steps, batch_size, -1))
        reconstructions = recon_flat.reshape(
            (time_steps, batch_size, *self.observation_shape)
        )
        prev_latents = latents[:-1].reshape((-1, latents.shape[-1]))
        next_latents = latents[1:].reshape((-1, latents.shape[-1]))
        mean, log_std = self.infer_latent_actions(prev_latents, next_latents)
        latent_actions = sample_latent_actions(
            jax.random.PRNGKey(0),
            mean,
            log_std,
        )
        lam_predicted_next = self.lam_reconstructor(prev_latents, latent_actions)
        conditioned_actions = jax.lax.stop_gradient(latent_actions)
        history = latents[:-1].transpose((1, 0, 2))
        action_history = conditioned_actions.reshape(
            (time_steps - 1, batch_size, self.config.lam.latent_action_dim)
        ).transpose((1, 0, 2))
        target_next = latents[-1]
        noise_key, time_key, dropout_key = jax.random.split(jax.random.PRNGKey(1), 3)
        base_noise = jax.random.normal(noise_key, target_next.shape)
        flow_time = jax.random.uniform(
            time_key,
            (batch_size,),
            minval=0.01,
            maxval=0.99,
        )
        noisy_next = (1.0 - flow_time[:, None]) * base_noise + flow_time[
            :, None
        ] * target_next
        target_velocity = target_next - base_noise
        condition_keep_mask = jax.random.bernoulli(
            dropout_key,
            p=1.0 - self.config.dynamics.classifier_free_dropout,
            shape=(batch_size,),
        )
        predicted_velocity = self.dynamics(
            history,
            action_history,
            flow_time,
            condition_keep_mask,
            noisy_next,
        )
        predicted_next = noisy_next + (1.0 - flow_time[:, None]) * predicted_velocity
        reward_pred, continue_logit = self.predict_reward_continue(
            next_latents,
            conditioned_actions,
        )
        return {
            "latents": latents,
            "reconstructions": reconstructions,
            "lam_mean": mean,
            "lam_log_std": log_std,
            "latent_actions": latent_actions,
            "lam_predicted_next": lam_predicted_next,
            "predicted_next": predicted_next,
            "target_next": target_next,
            "predicted_velocity": predicted_velocity,
            "target_velocity": target_velocity,
            "flow_time": flow_time,
            "reward_pred": reward_pred,
            "continue_logit": continue_logit,
        }


def create_genie2_train_state(
    key: jax.Array,
    *,
    observation_shape: tuple[int, ...],
    config: Genie2ContinuousConfig,
    learning_rate: float,
) -> TrainState:
    model = Genie2WorldModel(observation_shape=observation_shape, config=config)
    dummy_obs = jnp.zeros((2, 1, *observation_shape), dtype=jnp.float32)
    dummy_rewards = jnp.zeros((2, 1), dtype=jnp.float32)
    dummy_continues = jnp.ones((2, 1), dtype=jnp.float32)
    params = model.init(key, dummy_obs, dummy_rewards, dummy_continues)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(learning_rate),
    )


def genie2_loss(
    params: Any,
    state: TrainState,
    batch: WorldModelSequenceBatch,
    config: Genie2ContinuousConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    observations = jnp.asarray(batch.observations, dtype=jnp.float32)
    rewards = jnp.asarray(batch.rewards, dtype=jnp.float32)
    continues = jnp.asarray(batch.continues, dtype=jnp.float32)
    outputs = state.apply_fn(params, observations, rewards, continues)
    reconstruction = reconstruction_loss(observations, outputs["reconstructions"])
    lam_kl = lam_kl_loss(outputs["lam_mean"], outputs["lam_log_std"])
    lam_reconstruction = dynamics_mse_loss(
        outputs["lam_predicted_next"],
        outputs["latents"][1:].reshape((-1, config.autoencoder.latent_dim)),
    )
    lam_loss = lam_reconstruction + config.lam.kl_scale * lam_kl
    dynamics_loss = dynamics_mse_loss(
        outputs["predicted_velocity"],
        outputs["target_velocity"],
    )
    reward_targets = rewards[1:].reshape((-1,))
    continue_targets = continues[1:].reshape((-1,))
    reward_loss = jnp.mean(jnp.square(outputs["reward_pred"] - reward_targets))
    continue_loss = jnp.mean(
        optax.sigmoid_binary_cross_entropy(
            outputs["continue_logit"],
            continue_targets,
        )
    )
    loss = reconstruction + lam_loss + dynamics_loss + reward_loss + continue_loss
    metrics = {
        "loss": loss,
        "reconstruction_loss": reconstruction,
        "lam_kl_loss": lam_kl,
        "lam_reconstruction_loss": lam_reconstruction,
        "dynamics_loss": dynamics_loss,
        "flow_velocity_loss": dynamics_loss,
        "reward_loss": reward_loss,
        "continue_loss": continue_loss,
    }
    return loss, metrics


def genie2_train_step(
    state: TrainState,
    batch: WorldModelSequenceBatch,
    config: Genie2ContinuousConfig,
) -> tuple[TrainState, dict[str, jax.Array]]:
    (_, metrics), grads = jax.value_and_grad(genie2_loss, has_aux=True)(
        state.params,
        state,
        batch,
        config,
    )
    return state.apply_gradients(grads=grads), metrics


def train_genie2_world_model(
    *,
    batch: WorldModelSequenceBatch,
    observation_shape: tuple[int, ...],
    config: Genie2ContinuousConfig,
    train_steps: int,
    learning_rate: float,
    seed: int,
) -> tuple[TrainState, list[dict[str, float]]]:
    state = create_genie2_train_state(
        jax.random.PRNGKey(seed),
        observation_shape=observation_shape,
        config=config,
        learning_rate=learning_rate,
    )
    metrics: list[dict[str, float]] = []
    for step in range(train_steps):
        state, step_metrics = genie2_train_step(state, batch, config)
        metrics.append(
            {
                "step": step,
                **{name: float(value) for name, value in step_metrics.items()},
            }
        )
    return state, metrics
