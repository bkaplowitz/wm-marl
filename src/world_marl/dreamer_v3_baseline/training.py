from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from world_marl.dreamer_v3_baseline.config import DreamerV3Config
from world_marl.dreamer_v3_baseline.losses import (
    balanced_categorical_kl_loss,
    symlog,
    two_hot,
)
from world_marl.dreamer_v3_baseline.models import (
    ContinueHead,
    DreamerDecoder,
    DreamerEncoder,
    RewardHead,
)
from world_marl.dreamer_v3_baseline.rssm import (
    DreamerRSSM,
    RSSMState,
    flatten_rssm_state,
    initial_rssm_state,
    reset_rssm_state,
)
from world_marl.world_model_foundation.replay import WorldModelSequenceBatch


class DreamerWorldModel(nn.Module):
    config: DreamerV3Config

    def setup(self) -> None:
        self.encoder = DreamerEncoder(
            self.config.encoder.embedding_dim,
            hidden_dims=self.config.encoder.hidden_dims,
            name="encoder",
        )
        self.rssm = DreamerRSSM(
            self.config.rssm,
            action_dim=self.config.action_dim,
            name="rssm",
        )
        self.decoder = DreamerDecoder(
            self.config.observation_shape,
            name="decoder",
        )
        self.reward_head = RewardHead(
            self.config.reward_head.bins,
            hidden_dims=self.config.reward_head.hidden_dims,
            name="reward_head",
        )
        self.continue_head = ContinueHead(
            hidden_dims=self.config.continue_head.hidden_dims,
            name="continue_head",
        )

    def observe_step(
        self,
        prev_state: RSSMState,
        action_features: jax.Array,
        observations: jax.Array,
    ) -> tuple[RSSMState, RSSMState, dict[str, jax.Array]]:
        embed = self.encoder(observations)
        prior, posterior = self.rssm(prev_state, action_features, embed)
        feature = flatten_rssm_state(posterior)
        return prior, posterior, self.predict(feature)

    def imagine_step(
        self,
        prev_state: RSSMState,
        action_features: jax.Array,
    ) -> tuple[RSSMState, dict[str, jax.Array]]:
        prior = self.rssm.prior(prev_state, action_features)
        return prior, self.predict(flatten_rssm_state(prior))

    def predict(self, features: jax.Array) -> dict[str, jax.Array]:
        return {
            "features": features,
            "reconstructions": self.decoder(features),
            "reward_logits": self.reward_head(features),
            "continue_logits": self.continue_head(features),
        }

    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        is_first: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        time_steps, batch_size = observations.shape[:2]
        if is_first is None:
            is_first = jnp.zeros((time_steps, batch_size), dtype=bool)
        prev_state = initial_rssm_state(batch_size=batch_size, config=self.config.rssm)
        reconstructions = []
        reward_logits = []
        continue_logits = []
        prior_logits = []
        posterior_logits = []
        features = []
        deterministic = []
        stochastic = []
        for t in range(time_steps):
            prev_state = reset_rssm_state(
                prev_state,
                is_first[t],
                config=self.config.rssm,
            )
            action_features = dreamer_action_features(actions[t], self.config)
            prior, posterior, prediction = self.observe_step(
                prev_state,
                action_features,
                observations[t],
            )
            reconstructions.append(prediction["reconstructions"])
            reward_logits.append(prediction["reward_logits"])
            continue_logits.append(prediction["continue_logits"])
            prior_logits.append(prior.logits)
            posterior_logits.append(posterior.logits)
            features.append(prediction["features"])
            deterministic.append(posterior.deterministic)
            stochastic.append(posterior.stochastic)
            prev_state = posterior
        return {
            "reconstructions": jnp.stack(reconstructions),
            "reward_logits": jnp.stack(reward_logits),
            "continue_logits": jnp.stack(continue_logits),
            "prior_logits": jnp.stack(prior_logits),
            "posterior_logits": jnp.stack(posterior_logits),
            "features": jnp.stack(features),
            "deterministic": jnp.stack(deterministic),
            "stochastic": jnp.stack(stochastic),
        }


def dreamer_action_features(actions: jax.Array, config: DreamerV3Config) -> jax.Array:
    if config.action_mode == "discrete":
        return jax.nn.one_hot(actions.astype(jnp.int32), config.action_dim)
    return actions.astype(jnp.float32).reshape((actions.shape[0], config.action_dim))


def create_dreamer_train_state(
    key: jax.Array,
    config: DreamerV3Config,
    *,
    learning_rate: float,
) -> TrainState:
    model = DreamerWorldModel(config)
    dummy_obs = jnp.zeros((1, 1, *config.observation_shape), dtype=jnp.float32)
    if config.action_mode == "discrete":
        dummy_actions = jnp.zeros((1, 1), dtype=jnp.int32)
    else:
        dummy_actions = jnp.zeros((1, 1, config.action_dim), dtype=jnp.float32)
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
    action_dtype = jnp.int32 if config.action_mode == "discrete" else jnp.float32
    actions = jnp.asarray(batch.actions, dtype=action_dtype)
    rewards = jnp.asarray(batch.rewards, dtype=jnp.float32)
    continues = jnp.asarray(batch.continues, dtype=jnp.float32)
    is_first = jnp.asarray(batch.is_first, dtype=bool)
    outputs = state.apply_fn(params, observations, actions, is_first)
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
    continue_losses = optax.sigmoid_binary_cross_entropy(
        outputs["continue_logits"], continues
    )
    continue_mask = 1.0 - is_first.astype(jnp.float32)
    continue_loss = jnp.sum(continue_losses * continue_mask) / jnp.maximum(
        jnp.sum(continue_mask), 1.0
    )
    kl_loss, dynamics_kl_loss, representation_kl_loss = balanced_categorical_kl_loss(
        outputs["posterior_logits"],
        outputs["prior_logits"],
        free_nats=config.kl_free_nats,
        dynamics_scale=config.dynamics_kl_scale,
        representation_scale=config.representation_kl_scale,
    )
    loss = reconstruction_loss + reward_loss + continue_loss + kl_loss
    metrics = {
        "loss": loss,
        "reconstruction_loss": reconstruction_loss,
        "reward_loss": reward_loss,
        "continue_loss": continue_loss,
        "kl_loss": kl_loss,
        "dynamics_kl_loss": dynamics_kl_loss,
        "representation_kl_loss": representation_kl_loss,
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
