"""World-model losses and target-network updates for DreaMARL."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from world_marl.dreamarl.config import DreaMARLConfig
from world_marl.dreamarl.contracts import JaxMultiAgentSequenceBatch
from world_marl.dreamarl.distributions import two_hot_loss
from world_marl.dreamarl.world_model import (
    DreaMARLWorldModel,
    SequenceInference,
    SharedObservationEncoder,
    categorical_probabilities,
)


class WorldModelLossOutput(NamedTuple):
    loss: jax.Array
    metrics: dict[str, jax.Array]
    inference: SequenceInference


def world_model_loss(
    model: DreaMARLWorldModel,
    params: Any,
    target_encoder_params: Any,
    batch: JaxMultiAgentSequenceBatch,
    key: jax.Array,
    config: DreaMARLConfig,
) -> WorldModelLossOutput:
    """Compute independently normalized DreaMARL prediction losses."""

    inference = model.apply({"params": params}, batch, key)
    prediction = inference.predictions
    target_encoder = SharedObservationEncoder(config.encoder)
    target_embedding = target_encoder.apply(
        {"params": target_encoder_params}, batch.next_observations
    )
    target_embedding = jax.lax.stop_gradient(target_embedding)

    transition_mask = batch.valid.astype(jnp.float32)
    agent_mask = transition_mask[..., None] * batch.agent_alive.astype(jnp.float32)
    next_agent_mask = (
        transition_mask[..., None] * batch.next_agent_alive.astype(jnp.float32)
    )

    predicted = _unit(prediction.next_embedding)
    target = _unit(target_embedding)
    jepa_per_agent = 1.0 - jnp.sum(predicted * target, axis=-1)
    jepa = _masked_mean(jepa_per_agent, next_agent_mask)

    dynamics_kl = _categorical_kl(
        jax.lax.stop_gradient(prediction.posterior_logits),
        prediction.prior_logits,
        config.dynamics.unimix,
    )
    representation_kl = _categorical_kl(
        prediction.posterior_logits,
        jax.lax.stop_gradient(prediction.prior_logits),
        config.dynamics.unimix,
    )
    free_nats = config.world_model_loss.free_nats
    dynamics_kl = _masked_mean(jnp.maximum(dynamics_kl, free_nats), agent_mask)
    representation_kl = _masked_mean(
        jnp.maximum(representation_kl, free_nats), agent_mask
    )

    team_reward = _masked_mean(
        two_hot_loss(
            prediction.team_reward_logits,
            batch.team_rewards,
            config.distribution,
        ),
        transition_mask,
    )
    agent_reward = _masked_mean(
        two_hot_loss(
            prediction.agent_reward_logits,
            batch.rewards,
            config.distribution,
        ),
        agent_mask,
    )
    continuation_target = (~batch.is_terminal).astype(jnp.float32)
    continuation = _masked_mean(
        optax.sigmoid_binary_cross_entropy(
            prediction.continuation_logit, continuation_target
        ),
        transition_mask,
    )
    alive = _masked_mean(
        optax.sigmoid_binary_cross_entropy(
            prediction.next_alive_logit,
            batch.next_agent_alive.astype(jnp.float32),
        ),
        transition_mask[..., None],
    )

    if batch.next_action_mask.shape[-1] == config.action_dim:
        action_mask = _masked_mean(
            optax.sigmoid_binary_cross_entropy(
                prediction.next_action_mask_logit,
                batch.next_action_mask.astype(jnp.float32),
            ),
            next_agent_mask[..., None],
        )
    else:
        action_mask = jnp.asarray(0.0, jnp.float32)

    weights = config.world_model_loss
    terms = {
        "jepa": jepa,
        "dynamics_kl": dynamics_kl,
        "representation_kl": representation_kl,
        "team_reward": team_reward,
        "agent_reward": agent_reward,
        "continuation": continuation,
        "agent_alive": alive,
        "action_mask": action_mask,
    }
    total = (
        weights.jepa * jepa
        + weights.dynamics_kl * dynamics_kl
        + weights.representation_kl * representation_kl
        + weights.team_reward * team_reward
        + weights.agent_reward * agent_reward
        + weights.continuation * continuation
        + weights.agent_alive * alive
        + weights.action_mask * action_mask
    )
    metrics = {
        "loss": total,
        **{f"loss/{name}": value for name, value in terms.items()},
        "posterior/entropy": _masked_mean(
            _categorical_entropy(
                prediction.posterior_logits, config.dynamics.unimix
            ),
            agent_mask,
        ),
        "prior/entropy": _masked_mean(
            _categorical_entropy(
                prediction.prior_logits, config.dynamics.unimix
            ),
            agent_mask,
        ),
        "prediction/team_reward_mean": _masked_mean(
            prediction.team_reward, transition_mask
        ),
        "target/team_reward_mean": _masked_mean(
            batch.team_rewards, transition_mask
        ),
    }
    return WorldModelLossOutput(total, metrics, inference)


def update_ema(target: Any, online: Any, decay: float) -> Any:
    """Update a target pytree without creating optimizer state for it."""

    return jax.tree.map(
        lambda target_value, online_value: (
            decay * target_value + (1.0 - decay) * online_value
        ),
        target,
        online,
    )


def encoder_params(world_model_params: Any) -> Any:
    """Extract the online encoder subtree for target initialization and EMA."""

    return world_model_params["encoder"]


def _unit(value: jax.Array) -> jax.Array:
    return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-8)


def _masked_mean(value: jax.Array, mask: jax.Array) -> jax.Array:
    mask = jnp.broadcast_to(mask, value.shape).astype(value.dtype)
    return jnp.sum(value * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _categorical_kl(
    first_logits: jax.Array,
    second_logits: jax.Array,
    unimix: float,
) -> jax.Array:
    first = categorical_probabilities(first_logits, unimix)
    second = categorical_probabilities(second_logits, unimix)
    per_variable = jnp.sum(
        first * (jnp.log(first + 1e-8) - jnp.log(second + 1e-8)), axis=-1
    )
    return jnp.sum(per_variable, axis=-1)


def _categorical_entropy(logits: jax.Array, unimix: float) -> jax.Array:
    probabilities = categorical_probabilities(logits, unimix)
    per_variable = -jnp.sum(
        probabilities * jnp.log(probabilities + 1e-8), axis=-1
    )
    return jnp.sum(per_variable, axis=-1)
