from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from world_marl.dreamer_v3_baseline.config import DreamerV3Config
from world_marl.dreamer_v3_baseline.losses import categorical_kl_loss, symlog, two_hot
from world_marl.dreamer_v3_baseline.models import (
    ContinueHead,
    DreamerDecoder,
    DreamerEncoder,
    RewardHead,
)
from world_marl.dreamer_v3_baseline.rssm import (
    DreamerRSSM,
    flatten_rssm_state,
    initial_rssm_state,
)
from world_marl.world_model_foundation.replay import WorldModelSequenceBatch


class DreamerWorldModel(nn.Module):
    config: DreamerV3Config

    @nn.compact
    def __call__(
        self, observations: jax.Array, actions: jax.Array
    ) -> dict[str, jax.Array]:
        time_steps, batch_size = observations.shape[:2]
        encoder = DreamerEncoder(
            self.config.encoder.embedding_dim,
            hidden_dims=self.config.encoder.hidden_dims,
            name="encoder",
        )
        rssm = DreamerRSSM(
            self.config.rssm,
            action_dim=self.config.action_dim,
            name="rssm",
        )
        decoder = DreamerDecoder(
            self.config.observation_shape,
            name="decoder",
        )
        reward_head = RewardHead(
            self.config.reward_head.bins,
            hidden_dims=self.config.reward_head.hidden_dims,
            name="reward_head",
        )
        continue_head = ContinueHead(
            hidden_dims=self.config.continue_head.hidden_dims,
            name="continue_head",
        )
        prev_state = initial_rssm_state(batch_size=batch_size, config=self.config.rssm)
        reconstructions = []
        reward_logits = []
        continue_logits = []
        prior_logits = []
        posterior_logits = []
        features = []
        for t in range(time_steps):
            embed = encoder(observations[t])
            action_features = jax.nn.one_hot(
                actions[t].astype(jnp.int32), self.config.action_dim
            )
            prior, posterior = rssm(prev_state, action_features, embed)
            feature = flatten_rssm_state(posterior)
            reconstructions.append(decoder(feature))
            reward_logits.append(reward_head(feature))
            continue_logits.append(continue_head(feature))
            prior_logits.append(prior.logits)
            posterior_logits.append(posterior.logits)
            features.append(feature)
            prev_state = posterior
        return {
            "reconstructions": jnp.stack(reconstructions),
            "reward_logits": jnp.stack(reward_logits),
            "continue_logits": jnp.stack(continue_logits),
            "prior_logits": jnp.stack(prior_logits),
            "posterior_logits": jnp.stack(posterior_logits),
            "features": jnp.stack(features),
        }


def create_dreamer_train_state(
    key: jax.Array,
    config: DreamerV3Config,
    *,
    learning_rate: float,
) -> TrainState:
    model = DreamerWorldModel(config)
    dummy_obs = jnp.zeros((1, 1, *config.observation_shape), dtype=jnp.float32)
    dummy_actions = jnp.zeros((1, 1), dtype=jnp.int32)
    params = model.init(key, dummy_obs, dummy_actions)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(learning_rate),
    )


def dreamer_world_model_loss(
    params: Any,
    state: TrainState,
    batch: WorldModelSequenceBatch,
    config: DreamerV3Config,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    observations = jnp.asarray(batch.observations, dtype=jnp.float32)
    actions = jnp.asarray(batch.actions, dtype=jnp.int32)
    rewards = jnp.asarray(batch.rewards, dtype=jnp.float32)
    continues = jnp.asarray(batch.continues, dtype=jnp.float32)
    outputs = state.apply_fn(params, observations, actions)
    reconstruction_loss = jnp.mean(
        jnp.square(outputs["reconstructions"] - observations)
    )
    reward_targets = two_hot(
        symlog(rewards),
        num_bins=config.reward_head.bins,
        lower=-20.0,
        upper=20.0,
    )
    reward_loss = -jnp.mean(
        jnp.sum(
            reward_targets * jax.nn.log_softmax(outputs["reward_logits"], axis=-1),
            axis=-1,
        )
    )
    continue_loss = jnp.mean(
        optax.sigmoid_binary_cross_entropy(outputs["continue_logits"], continues)
    )
    kl_loss = categorical_kl_loss(
        outputs["posterior_logits"],
        outputs["prior_logits"],
        free_nats=config.kl_free_nats,
    )
    loss = reconstruction_loss + reward_loss + continue_loss + kl_loss
    metrics = {
        "loss": loss,
        "reconstruction_loss": reconstruction_loss,
        "reward_loss": reward_loss,
        "continue_loss": continue_loss,
        "kl_loss": kl_loss,
    }
    return loss, metrics


def dreamer_train_step(
    state: TrainState,
    batch: WorldModelSequenceBatch,
    config: DreamerV3Config,
) -> tuple[TrainState, dict[str, jax.Array]]:
    (_, metrics), grads = jax.value_and_grad(dreamer_world_model_loss, has_aux=True)(
        state.params,
        state,
        batch,
        config,
    )
    return state.apply_gradients(grads=grads), metrics


def train_dreamer_world_model(
    *,
    batch: WorldModelSequenceBatch,
    config: DreamerV3Config,
    train_steps: int,
    learning_rate: float,
    seed: int,
) -> tuple[TrainState, list[dict[str, float]]]:
    state = create_dreamer_train_state(
        jax.random.PRNGKey(seed),
        config,
        learning_rate=learning_rate,
    )
    metrics: list[dict[str, float]] = []
    for step in range(train_steps):
        state, step_metrics = dreamer_train_step(state, batch, config)
        metrics.append({"step": step, **{k: float(v) for k, v in step_metrics.items()}})
    return state, metrics
