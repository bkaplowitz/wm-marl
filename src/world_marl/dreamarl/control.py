"""Decentralized actor, centralized critic, and coherent joint imagination."""

from __future__ import annotations

from typing import Any, NamedTuple

from flax import linen as nn
import jax
import jax.numpy as jnp

from world_marl.dreamarl.config import DreaMARLConfig
from world_marl.dreamarl.distributions import two_hot_loss, two_hot_mean
from world_marl.dreamarl.temporal import TemporalLayer
from world_marl.dreamarl.world_model import DreaMARLWorldModel, WorldState


class SharedActor(nn.Module):
    """Parameter-shared decentralized policy over local latent beliefs."""

    config: DreaMARLConfig

    @nn.compact
    def __call__(
        self,
        beliefs: jax.Array,
        agent_alive: jax.Array,
        action_mask: jax.Array,
    ) -> jax.Array:
        cfg = self.config
        x = beliefs
        if cfg.dynamics.use_agent_identity:
            identity = self.param(
                "agent_identity",
                nn.initializers.normal(0.02),
                (cfg.max_agents, cfg.dynamics.model_dim),
            )
            identity = jnp.broadcast_to(
                identity, (*beliefs.shape[:2], identity.shape[-1])
            )
            x = jnp.concatenate([x, identity], axis=-1)
        x = nn.Dense(cfg.dynamics.model_dim, name="hidden_0")(x)
        x = nn.silu(nn.RMSNorm(name="norm_0")(x))
        x = nn.Dense(cfg.dynamics.model_dim, name="hidden_1")(x)
        x = nn.silu(nn.RMSNorm(name="norm_1")(x))
        logits = nn.Dense(
            cfg.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            name="logits",
        )(x)
        if action_mask.shape[-1] == 0:
            action_mask = jnp.ones((*agent_alive.shape, cfg.action_dim), bool)
        inactive_mask = jax.nn.one_hot(
            jnp.zeros(agent_alive.shape, jnp.int32), cfg.action_dim, dtype=bool
        )
        effective_mask = jnp.where(
            agent_alive[..., None], action_mask, inactive_mask
        )
        any_legal = jnp.any(effective_mask, axis=-1, keepdims=True)
        effective_mask = jnp.where(any_legal, effective_mask, True)
        return jnp.where(effective_mask, logits, -1e30)


class CentralizedCritic(nn.Module):
    """Permutation-aware team-value model over all active latent beliefs."""

    config: DreaMARLConfig

    def setup(self) -> None:
        cfg = self.config
        self.input_projection = nn.Dense(
            cfg.dynamics.model_dim, name="input_projection"
        )
        self.layer = TemporalLayer(
            cfg.dynamics.model_dim,
            cfg.dynamics.cross_agent_heads,
            cfg.dynamics.mlp_ratio,
            name="agent_attention",
        )
        self.output_norm = nn.RMSNorm(name="output_norm")
        self.value_hidden = nn.Dense(cfg.dynamics.model_dim, name="value_hidden")
        self.value_out = nn.Dense(
            cfg.distribution.bins,
            kernel_init=nn.initializers.zeros_init(),
            name="value",
        )
        if cfg.dynamics.use_agent_identity:
            self.agent_identity = self.param(
                "agent_identity",
                nn.initializers.normal(0.02),
                (cfg.max_agents, cfg.dynamics.model_dim),
            )
        else:
            self.agent_identity = None

    def __call__(self, beliefs: jax.Array, agent_alive: jax.Array) -> jax.Array:
        x = self.input_projection(beliefs)
        if self.agent_identity is not None:
            x = x + self.agent_identity[None]
        x = jnp.where(agent_alive[..., None], x, 0.0)
        mask = agent_alive[:, None, :, None] & agent_alive[:, None, None, :]
        x = self.layer(x, mask)
        x = jnp.where(agent_alive[..., None], self.output_norm(x), 0.0)
        weights = agent_alive.astype(x.dtype)
        pooled = jnp.sum(x * weights[..., None], axis=1) / jnp.maximum(
            jnp.sum(weights, axis=1, keepdims=True), 1.0
        )
        return self.value_out(nn.silu(self.value_hidden(pooled)))

    def value(
        self, beliefs: jax.Array, agent_alive: jax.Array
    ) -> jax.Array:
        return two_hot_mean(
            self(beliefs, agent_alive), self.config.distribution
        )


class ImaginationBatch(NamedTuple):
    beliefs: jax.Array
    agent_alive: jax.Array
    actions: jax.Array
    log_probability: jax.Array
    entropy: jax.Array
    rewards: jax.Array
    discounts: jax.Array
    target_values: jax.Array
    returns: jax.Array
    weights: jax.Array


class ActorLossOutput(NamedTuple):
    loss: jax.Array
    metrics: dict[str, jax.Array]
    imagination: ImaginationBatch
    normalization_low: jax.Array
    normalization_high: jax.Array


def actor_loss(
    actor: SharedActor,
    actor_params: Any,
    critic: CentralizedCritic,
    target_critic_params: Any,
    world_model: DreaMARLWorldModel,
    world_model_params: Any,
    start_state: WorldState,
    key: jax.Array,
    config: DreaMARLConfig,
    return_low: jax.Array | None = None,
    return_high: jax.Array | None = None,
    normalization_initialized: jax.Array | None = None,
) -> ActorLossOutput:
    """Train decentralized actors in one coherent imagined joint world."""

    horizon = config.imagination.horizon
    keys = jax.random.split(key, horizon)

    def imagine_step(state, sample_key):
        action_key, dynamics_key = jax.random.split(sample_key)
        belief = world_model.apply(
            {"params": world_model_params}, state, method=world_model.belief
        )
        logits = actor.apply(
            {"params": actor_params},
            belief,
            state.agent_alive,
            state.action_mask,
        )
        actions = jax.random.categorical(action_key, logits, axis=-1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        selected = jnp.take_along_axis(
            log_probs, actions[..., None], axis=-1
        )[..., 0]
        probabilities = jax.nn.softmax(logits, axis=-1)
        entropy = -jnp.sum(probabilities * log_probs, axis=-1)
        alive = state.agent_alive.astype(jnp.float32)
        count = jnp.maximum(alive.sum(axis=-1), 1.0)
        selected = (selected * alive).sum(axis=-1) / count
        entropy = (entropy * alive).sum(axis=-1) / count
        target_value = critic.apply(
            {"params": target_critic_params},
            belief,
            state.agent_alive,
            method=critic.value,
        )
        next_state, prediction = world_model.apply(
            {"params": world_model_params},
            state,
            actions,
            dynamics_key,
            method=world_model.imagine_step,
        )
        outputs = (
            belief,
            state.agent_alive,
            actions,
            selected,
            entropy,
            prediction.team_reward,
            config.imagination.discount
            * jax.nn.sigmoid(prediction.continuation_logit),
            target_value,
        )
        return next_state, outputs

    final_state, outputs = jax.lax.scan(imagine_step, start_state, keys)
    (
        beliefs,
        alive,
        actions,
        log_probability,
        entropy,
        rewards,
        discounts,
        target_values,
    ) = outputs
    final_belief = world_model.apply(
        {"params": world_model_params}, final_state, method=world_model.belief
    )
    bootstrap = critic.apply(
        {"params": target_critic_params},
        final_belief,
        final_state.agent_alive,
        method=critic.value,
    )
    returns = lambda_returns(
        rewards,
        discounts,
        target_values,
        bootstrap,
        config.imagination.lambda_,
    )
    advantage = jax.lax.stop_gradient(returns - target_values)
    batch_low = jnp.percentile(
        returns, config.imagination.return_percentile_low
    )
    batch_high = jnp.percentile(
        returns, config.imagination.return_percentile_high
    )
    if return_low is None:
        return_low = batch_low
    if return_high is None:
        return_high = batch_high
    if normalization_initialized is None:
        normalization_initialized = jnp.asarray(False)
    decay = config.imagination.return_scale_decay
    low = jnp.where(
        normalization_initialized,
        decay * return_low + (1.0 - decay) * batch_low,
        batch_low,
    )
    high = jnp.where(
        normalization_initialized,
        decay * return_high + (1.0 - decay) * batch_high,
        batch_high,
    )
    advantage = advantage / jax.lax.stop_gradient(
        jnp.maximum(high - low, 1.0)
    )
    weights = survival_weights(discounts)
    denominator = jnp.maximum(jnp.sum(weights), 1.0)
    policy_gain = jnp.sum(weights * log_probability * advantage) / denominator
    entropy_mean = jnp.sum(weights * entropy) / denominator
    loss = -policy_gain - config.imagination.entropy_coefficient * entropy_mean
    imagination = ImaginationBatch(
        beliefs=jax.lax.stop_gradient(beliefs),
        agent_alive=alive,
        actions=actions,
        log_probability=log_probability,
        entropy=entropy,
        rewards=rewards,
        discounts=discounts,
        target_values=target_values,
        returns=jax.lax.stop_gradient(returns),
        weights=jax.lax.stop_gradient(weights),
    )
    metrics = {
        "loss": loss,
        "policy_gain": policy_gain,
        "entropy": entropy_mean,
        "return_mean": jnp.mean(returns),
        "return_std": jnp.std(returns),
        "reward_mean": jnp.mean(rewards),
        "continuation_mean": jnp.mean(discounts) / config.imagination.discount,
        "return_normalization_low": low,
        "return_normalization_high": high,
        "return_normalization_range": jnp.maximum(high - low, 1.0),
    }
    return ActorLossOutput(loss, metrics, imagination, low, high)


def critic_loss(
    critic: CentralizedCritic,
    critic_params: Any,
    imagination: ImaginationBatch,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Fit the centralized critic to stopped joint lambda returns."""

    horizon, batch = imagination.beliefs.shape[:2]
    logits = critic.apply(
        {"params": critic_params},
        imagination.beliefs.reshape((horizon * batch, *imagination.beliefs.shape[2:])),
        imagination.agent_alive.reshape(
            (horizon * batch, *imagination.agent_alive.shape[2:])
        ),
    ).reshape((horizon, batch, -1))
    values = two_hot_mean(logits, critic.config.distribution)
    error = two_hot_loss(
        logits, imagination.returns, critic.config.distribution
    )
    loss = jnp.sum(imagination.weights * error) / jnp.maximum(
        jnp.sum(imagination.weights), 1.0
    )
    return loss, {
        "loss": loss,
        "value_mean": jnp.mean(values),
        "target_mean": jnp.mean(imagination.returns),
        "absolute_error": jnp.mean(jnp.abs(values - imagination.returns)),
    }


def lambda_returns(
    rewards: jax.Array,
    discounts: jax.Array,
    values: jax.Array,
    bootstrap: jax.Array,
    lambda_: float,
) -> jax.Array:
    """Compute time-major Dreamer-style lambda returns."""

    next_values = jnp.concatenate([values[1:], bootstrap[None]], axis=0)

    def step(carry, inputs):
        reward, discount, next_value = inputs
        current = reward + discount * (
            (1.0 - lambda_) * next_value + lambda_ * carry
        )
        return current, current

    _, reversed_returns = jax.lax.scan(
        step,
        bootstrap,
        (rewards[::-1], discounts[::-1], next_values[::-1]),
    )
    return reversed_returns[::-1]


def survival_weights(discounts: jax.Array) -> jax.Array:
    """Weight each imagined state by survival to that state."""

    if discounts.shape[0] == 1:
        return jnp.ones_like(discounts)
    preceding = jnp.concatenate(
        [jnp.ones_like(discounts[:1]), discounts[:-1]], axis=0
    )
    return jnp.cumprod(preceding, axis=0)
