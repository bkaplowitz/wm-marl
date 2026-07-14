"""Repository RL extension for controlling the Genie2-style simulator."""

from __future__ import annotations

from typing import Any, NamedTuple, Sequence

from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from world_marl.genie2_continuous_jax.config import Genie2ContinuousConfig
from world_marl.genie2_continuous_jax.dynamics import (
    ActionConditionedLatentDiffusion,
    classifier_free_guidance,
    quantized_context_signal_level,
)
from world_marl.genie2_continuous_jax.training import (
    Genie2TrainState,
    action_features,
    encode_genie2_observations,
    sample_genie2_latents,
)
from world_marl.world_model_foundation.replay import (
    JaxSequenceBatch,
    WorldModelSequenceBatch,
    sample_sequence_windows,
    sequence_batch_to_jax,
)


class LatentActionPolicy(nn.Module):
    """Action policy; the historical name is retained for artifact compatibility."""

    action_dim: int
    action_mode: str = "continuous"
    hidden_dims: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, pooled_latents: jax.Array) -> dict[str, jax.Array]:
        x = pooled_latents.astype(jnp.float32)
        for width in self.hidden_dims:
            x = nn.silu(nn.Dense(width)(x))
        if self.action_mode == "discrete":
            return {"logits": nn.Dense(self.action_dim, name="action_logits")(x)}
        mean = nn.Dense(self.action_dim, name="action_mean")(x)
        log_std = jnp.clip(
            nn.Dense(self.action_dim, name="action_log_std")(x), -5.0, 2.0
        )
        return {"mean": mean, "log_std": log_std}


def update_observation_history(
    history: jax.Array,
    observations: jax.Array,
    is_first: jax.Array,
) -> jax.Array:
    if history.ndim != observations.ndim + 1:
        raise ValueError("history must be time-major observations")
    if history.shape[1:] != observations.shape:
        raise ValueError("history and observations must share batch and feature shapes")
    if is_first.shape != (observations.shape[0],):
        raise ValueError("is_first must have shape (batch,)")
    reset_shape = (1, observations.shape[0], *(1,) * (observations.ndim - 1))
    reset_history = jnp.broadcast_to(observations[None], history.shape)
    history = jnp.where(is_first.reshape(reset_shape), reset_history, history)
    return jnp.concatenate([history[1:], observations[None]], axis=0)


class LatentValue(nn.Module):
    hidden_dims: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, pooled_latents: jax.Array) -> jax.Array:
        x = pooled_latents.astype(jnp.float32)
        for width in self.hidden_dims:
            x = nn.silu(nn.Dense(width)(x))
        return nn.Dense(1, name="value")(x)[..., 0]


class PolicySample(NamedTuple):
    env_action: jax.Array
    model_action: jax.Array
    log_probability: jax.Array
    entropy: jax.Array


@jax.tree_util.register_pytree_node_class
class Genie2PolicyRollout:
    def __init__(
        self,
        states: jax.Array,
        latents: jax.Array,
        environment_actions: jax.Array,
        model_actions: jax.Array,
        rewards: jax.Array,
        continues: jax.Array,
        values: jax.Array,
        returns: jax.Array,
        log_probabilities: jax.Array,
        entropies: jax.Array,
        weights: jax.Array,
    ) -> None:
        self.states = states
        self.latents = latents
        self.environment_actions = environment_actions
        self.model_actions = model_actions
        self.rewards = rewards
        self.continues = continues
        self.values = values
        self.returns = returns
        self.log_probabilities = log_probabilities
        self.entropies = entropies
        self.weights = weights

    @property
    def latent_actions(self) -> jax.Array:
        return self.environment_actions

    def tree_flatten(self):
        return (
            (
                self.states,
                self.latents,
                self.environment_actions,
                self.model_actions,
                self.rewards,
                self.continues,
                self.values,
                self.returns,
                self.log_probabilities,
                self.entropies,
                self.weights,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        del auxiliary
        return cls(*children)


def _lambda_returns(
    rewards: jax.Array,
    continues: jax.Array,
    values: jax.Array,
    bootstrap: jax.Array,
    discount: float,
    discount_lambda: float,
) -> jax.Array:
    next_values = jnp.concatenate([values[1:], bootstrap[None]], axis=0)
    discounts = discount * continues

    def step(last, inputs):
        reward, continue_, next_value = inputs
        last = reward + continue_ * (
            (1.0 - discount_lambda) * next_value + discount_lambda * last
        )
        return last, last

    _, returns = jax.lax.scan(
        step,
        bootstrap,
        (rewards[::-1], discounts[::-1], next_values[::-1]),
    )
    return returns[::-1]


def create_latent_policy_states(
    key: jax.Array,
    config: Genie2ContinuousConfig,
    *,
    learning_rate: float,
) -> tuple[TrainState, TrainState]:
    actor_key, critic_key = jax.random.split(key)
    dummy = jnp.zeros((1, config.autoencoder.latent_patch_dim), dtype=jnp.float32)
    actor = LatentActionPolicy(
        config.action_dim,
        config.action_mode,
        config.latent_policy.hidden_dims,
    )
    critic = LatentValue(config.latent_policy.hidden_dims)
    actor_params = actor.init(actor_key, dummy)
    critic_params = critic.init(critic_key, dummy)
    return (
        TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=optax.adam(learning_rate),
        ),
        TrainState.create(
            apply_fn=critic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate),
        ),
    )


def _policy_sample(
    actor_state: TrainState,
    params: Any,
    pooled_latents: jax.Array,
    config: Genie2ContinuousConfig,
    key: jax.Array,
    *,
    deterministic: bool,
) -> PolicySample:
    outputs = actor_state.apply_fn(params, jax.lax.stop_gradient(pooled_latents))
    if config.action_mode == "discrete":
        logits = outputs["logits"]
        index = (
            jnp.argmax(logits, axis=-1)
            if deterministic
            else jax.random.categorical(key, logits)
        )
        probabilities = jax.nn.softmax(logits, axis=-1)
        model_action = jax.nn.one_hot(index, config.action_dim).astype(jnp.float32)
        log_probability = jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            index[..., None],
            axis=-1,
        )[..., 0]
        entropy = -jnp.sum(probabilities * jax.nn.log_softmax(logits, axis=-1), axis=-1)
        return PolicySample(
            index.astype(jnp.int32), model_action, log_probability, entropy
        )

    raw_mean = outputs["mean"]
    low = jnp.asarray(
        config.action_low or (-1.0,) * config.action_dim,
        dtype=jnp.float32,
    )
    high = jnp.asarray(
        config.action_high or (1.0,) * config.action_dim,
        dtype=jnp.float32,
    )
    scale = 0.5 * (high - low)
    midpoint = 0.5 * (high + low)
    std = jnp.exp(outputs["log_std"])
    raw_action = (
        raw_mean
        if deterministic
        else raw_mean + std * jax.random.normal(key, raw_mean.shape)
    )
    normalized_action = jnp.tanh(raw_action)
    model_action = midpoint + scale * normalized_action

    def transformed_log_probability(values: jax.Array) -> jax.Array:
        normal_log_probability = jnp.sum(
            -0.5 * jnp.square((values - raw_mean) / std)
            - outputs["log_std"]
            - 0.5 * jnp.log(2.0 * jnp.pi),
            axis=-1,
        )
        log_tanh_jacobian = 2.0 * (
            jnp.log(2.0) - values - jax.nn.softplus(-2.0 * values)
        )
        log_scale_jacobian = jnp.log(jnp.maximum(scale, 1e-6))
        return normal_log_probability - jnp.sum(
            log_tanh_jacobian + log_scale_jacobian,
            axis=-1,
        )

    log_probability = transformed_log_probability(jax.lax.stop_gradient(raw_action))
    entropy = -transformed_log_probability(raw_action)
    return PolicySample(model_action, model_action, log_probability, entropy)


def latent_policy_action(
    actor_state: TrainState,
    latents: jax.Array,
    config: Genie2ContinuousConfig | None = None,
) -> jax.Array:
    if config is None:
        action_dim = (
            actor_state.apply_fn(actor_state.params, latents)
            .get(
                "mean",
                actor_state.apply_fn(actor_state.params, latents).get("logits"),
            )
            .shape[-1]
        )
        config = Genie2ContinuousConfig(action_dim=action_dim, observation_shape=(1,))
    sample = _policy_sample(
        actor_state,
        actor_state.params,
        latents,
        config,
        jax.random.PRNGKey(0),
        deterministic=True,
    )
    return sample.env_action


def simulate_latent_policy_rollout(
    *,
    world_model_state: Genie2TrainState,
    actor_state: TrainState,
    critic_state: TrainState,
    start_latents: jax.Array,
    start_actions: jax.Array | None = None,
    observation_shape: tuple[int, ...] | None = None,
    config: Genie2ContinuousConfig,
    horizon: int,
    key: jax.Array,
    actor_params: Any | None = None,
    critic_params: Any | None = None,
) -> Genie2PolicyRollout:
    del observation_shape
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if start_latents.ndim == 2:
        start_latents = start_latents[:, None, None, :]
    if start_latents.ndim != 4:
        raise ValueError("start_latents must be a latent patch-grid history")
    actor_params = actor_state.params if actor_params is None else actor_params
    critic_params = critic_state.params if critic_params is None else critic_params
    batch, context_time = start_latents.shape[:2]
    if start_actions is None:
        start_actions = jnp.zeros(
            (batch, context_time - 1, config.action_dim), dtype=jnp.float32
        )
    if start_actions.shape != (batch, context_time - 1, config.action_dim):
        raise ValueError("start_actions must connect the latent context frames")

    def step(carry, _):
        latent_history, action_history, rollout_key = carry
        rollout_key, action_key, model_key = jax.random.split(rollout_key, 3)
        current_latent = latent_history[:, -1]
        pooled = jnp.mean(current_latent, axis=1)
        sample = _policy_sample(
            actor_state,
            actor_params,
            pooled,
            config,
            action_key,
            deterministic=False,
        )
        generation_actions = jnp.concatenate(
            [action_history, sample.model_action[:, None]], axis=1
        )
        generated = sample_genie2_latents(
            world_model_state,
            latent_history,
            generation_actions,
            config,
            model_key,
            num_future_frames=1,
        )
        next_latent = generated[:, -1]
        reward, continue_logit = world_model_state.heads.apply_fn(
            world_model_state.heads.params,
            pooled,
            sample.model_action,
        )
        value = critic_state.apply_fn(critic_params, pooled)
        latent_history = jnp.concatenate(
            [latent_history[:, 1:], next_latent[:, None]], axis=1
        )
        if context_time > 1:
            action_history = jnp.concatenate(
                [action_history[:, 1:], sample.model_action[:, None]], axis=1
            )
        outputs = (
            current_latent,
            next_latent,
            sample.env_action,
            sample.model_action,
            reward,
            jax.nn.sigmoid(continue_logit),
            value,
            sample.log_probability,
            sample.entropy,
        )
        return (latent_history, action_history, rollout_key), outputs

    (final_history, _, _), outputs = jax.lax.scan(
        step,
        (start_latents, start_actions, key),
        None,
        length=horizon,
    )
    (
        states,
        latents,
        environment_actions,
        model_actions,
        rewards,
        continues,
        values,
        log_probabilities,
        entropies,
    ) = outputs
    bootstrap = critic_state.apply_fn(
        critic_params,
        jnp.mean(final_history[:, -1], axis=1),
    )
    returns = _lambda_returns(
        rewards,
        continues,
        values,
        bootstrap,
        config.latent_policy.discount,
        config.latent_policy.discount_lambda,
    )
    discounts = config.latent_policy.discount * continues
    weights = jnp.concatenate(
        [jnp.ones_like(discounts[:1]), jnp.cumprod(discounts[:-1], axis=0)],
        axis=0,
    )
    return Genie2PolicyRollout(
        states,
        latents,
        environment_actions,
        model_actions,
        rewards,
        continues,
        values,
        returns,
        log_probabilities,
        entropies,
        weights,
    )


def _actor_loss(
    actor_params: Any,
    *,
    world_model_state: Genie2TrainState,
    actor_state: TrainState,
    critic_state: TrainState,
    start_latents: jax.Array,
    start_actions: jax.Array,
    config: Genie2ContinuousConfig,
    horizon: int,
    key: jax.Array,
) -> tuple[jax.Array, Genie2PolicyRollout]:
    rollout = simulate_latent_policy_rollout(
        world_model_state=world_model_state,
        actor_state=actor_state,
        critic_state=critic_state,
        start_latents=start_latents,
        start_actions=start_actions,
        config=config,
        horizon=horizon,
        key=key,
        actor_params=actor_params,
    )
    advantage = jax.lax.stop_gradient(rollout.returns - rollout.values)
    loss = -jnp.mean(
        jax.lax.stop_gradient(rollout.weights)
        * (
            rollout.log_probabilities * advantage
            + config.latent_policy.entropy_scale * rollout.entropies
        )
    )
    return loss, rollout


def _score_candidate_rollouts(
    *,
    world_model_state: Genie2TrainState,
    actor_state: TrainState,
    actor_params: Any,
    start_latents: jax.Array,
    start_actions: jax.Array,
    normalized_candidates: jax.Array,
    config: Genie2ContinuousConfig,
    key: jax.Array,
    horizon: int,
) -> jax.Array:
    if config.action_mode != "continuous":
        raise ValueError("candidate rollouts require continuous actions")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if start_latents.ndim != 4:
        raise ValueError("start_latents must be a latent patch-grid history")
    batch, context_time, num_patches, latent_dim = start_latents.shape
    if start_actions.shape != (batch, context_time - 1, config.action_dim):
        raise ValueError("start_actions must connect the latent context frames")
    if normalized_candidates.ndim != 3:
        raise ValueError(
            "normalized_candidates must have shape (batch,candidate,action)"
        )
    if normalized_candidates.shape[0] != batch:
        raise ValueError("candidate and latent batch dimensions must match")
    if normalized_candidates.shape[2] != config.action_dim:
        raise ValueError("candidate actions must match config.action_dim")

    num_candidates = normalized_candidates.shape[1]
    flat_batch = batch * num_candidates
    low = jnp.asarray(
        config.action_low or (-1.0,) * config.action_dim,
        dtype=jnp.float32,
    )
    high = jnp.asarray(
        config.action_high or (1.0,) * config.action_dim,
        dtype=jnp.float32,
    )
    scale = 0.5 * (high - low)
    midpoint = 0.5 * (high + low)
    latent_history = jnp.repeat(start_latents[:, None], num_candidates, axis=1)
    latent_history = latent_history.reshape(
        (flat_batch, context_time, num_patches, latent_dim)
    )
    action_history = jnp.repeat(start_actions[:, None], num_candidates, axis=1)
    action_history = action_history.reshape(
        (flat_batch, context_time - 1, config.action_dim)
    )
    current_normalized = normalized_candidates.reshape((flat_batch, config.action_dim))
    returns = jnp.zeros((flat_batch,), dtype=jnp.float32)
    weights = jnp.ones((flat_batch,), dtype=jnp.float32)
    context_signal = quantized_context_signal_level(
        denoising_steps=config.dynamics.denoising_steps,
        context_corruption=config.dynamics.context_corruption,
    )
    context_step = config.dynamics.denoising_steps - 1
    planning_actor_params = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        actor_params,
    )

    def planning_step(carry, _):
        (
            current_history,
            current_action_history,
            normalized_action,
            cumulative_return,
            cumulative_weight,
            planning_key,
        ) = carry
        planning_key, context_key, target_key = jax.random.split(planning_key, 3)
        model_action = midpoint + scale * normalized_action
        pooled = jnp.mean(current_history[:, -1], axis=1)
        reward, continue_logit = world_model_state.heads.apply_fn(
            world_model_state.heads.params,
            pooled,
            model_action,
        )
        penalized_reward = reward - config.latent_policy.action_penalty * jnp.mean(
            jnp.square(normalized_action),
            axis=-1,
        )
        cumulative_return = cumulative_return + cumulative_weight * penalized_reward
        cumulative_weight = (
            cumulative_weight
            * config.latent_policy.discount
            * jax.nn.sigmoid(continue_logit)
        )

        context_noise = jax.random.normal(
            context_key,
            (batch, context_time, num_patches, latent_dim),
            dtype=jnp.float32,
        )
        context_noise = jnp.repeat(
            context_noise[:, None], num_candidates, axis=1
        ).reshape(current_history.shape)
        corrupted_context = (
            context_signal * current_history + (1.0 - context_signal) * context_noise
        )
        target_noise = jax.random.normal(
            target_key,
            (batch, 1, num_patches, latent_dim),
            dtype=jnp.float32,
        )
        target_noise = jnp.repeat(
            target_noise[:, None], num_candidates, axis=1
        ).reshape((flat_batch, 1, num_patches, latent_dim))
        model_latents = jnp.concatenate([corrupted_context, target_noise], axis=1)
        generation_actions = jnp.concatenate(
            [current_action_history, model_action[:, None]],
            axis=1,
        )
        denoising_steps = jnp.concatenate(
            [
                jnp.full(
                    (flat_batch, context_time),
                    context_step,
                    dtype=jnp.int32,
                ),
                jnp.zeros((flat_batch, 1), dtype=jnp.int32),
            ],
            axis=1,
        )
        condition_shape = (flat_batch, context_time)
        conditioned = world_model_state.dynamics.apply_fn(
            world_model_state.dynamics.params,
            model_latents,
            generation_actions,
            denoising_steps,
            condition_keep_mask=jnp.ones(condition_shape, dtype=jnp.float32),
            training=False,
            method=ActionConditionedLatentDiffusion.predict_x,
        )
        unconditioned = world_model_state.dynamics.apply_fn(
            world_model_state.dynamics.params,
            model_latents,
            generation_actions,
            denoising_steps,
            condition_keep_mask=jnp.zeros(condition_shape, dtype=jnp.float32),
            training=False,
            method=ActionConditionedLatentDiffusion.predict_x,
        )
        next_latent = classifier_free_guidance(
            conditioned,
            unconditioned,
            config.dynamics.guidance_scale,
        )[:, -1]
        current_history = jnp.concatenate(
            [current_history[:, 1:], next_latent[:, None]],
            axis=1,
        )
        current_action_history = generation_actions[:, 1:]
        next_pooled = jnp.mean(next_latent, axis=1)
        actor_outputs = actor_state.apply_fn(
            planning_actor_params,
            jax.lax.stop_gradient(next_pooled),
        )
        normalized_action = jnp.tanh(actor_outputs["mean"])
        return (
            current_history,
            current_action_history,
            normalized_action,
            cumulative_return,
            cumulative_weight,
            planning_key,
        ), None

    (_, _, _, returns, _, _), _ = jax.lax.scan(
        planning_step,
        (
            latent_history,
            action_history,
            current_normalized,
            returns,
            weights,
            key,
        ),
        None,
        length=horizon,
    )
    return jax.lax.stop_gradient(returns.reshape((batch, num_candidates)))


def _candidate_distill_loss(
    actor_params: Any,
    *,
    world_model_state: Genie2TrainState,
    actor_state: TrainState,
    start_latents: jax.Array,
    start_actions: jax.Array,
    config: Genie2ContinuousConfig,
    key: jax.Array,
    num_candidates: int,
    candidate_min_gap: float,
    horizon: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    if config.action_mode != "continuous":
        raise ValueError("candidate-distill requires continuous actions")
    if num_candidates < 2:
        raise ValueError("num_candidates must be at least two")

    pooled_latents = jnp.mean(start_latents[:, -1], axis=1)
    low = jnp.asarray(
        config.action_low or (-1.0,) * config.action_dim,
        dtype=jnp.float32,
    )
    high = jnp.asarray(
        config.action_high or (1.0,) * config.action_dim,
        dtype=jnp.float32,
    )
    scale = 0.5 * (high - low)
    midpoint = 0.5 * (high + low)
    candidate_key, planning_key = jax.random.split(key)
    normalized_candidates = jax.random.uniform(
        candidate_key,
        (pooled_latents.shape[0], num_candidates, config.action_dim),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    scores = _score_candidate_rollouts(
        world_model_state=world_model_state,
        actor_state=actor_state,
        actor_params=actor_params,
        start_latents=start_latents,
        start_actions=start_actions,
        normalized_candidates=normalized_candidates,
        config=config,
        key=planning_key,
        horizon=horizon,
    )
    best_index = jnp.argmax(scores, axis=1)
    best_actions = jnp.take_along_axis(
        normalized_candidates,
        best_index[:, None, None],
        axis=1,
    )[:, 0]
    sorted_scores = jnp.sort(scores, axis=1)
    top_gap = sorted_scores[:, -1] - sorted_scores[:, -2]
    score_range = sorted_scores[:, -1] - sorted_scores[:, 0]
    weights = (top_gap >= candidate_min_gap).astype(jnp.float32)

    actor_outputs = actor_state.apply_fn(
        actor_params,
        jax.lax.stop_gradient(pooled_latents),
    )
    normalized_actions = jnp.tanh(actor_outputs["mean"])
    imitation_error = jnp.mean(
        jnp.square(normalized_actions - jax.lax.stop_gradient(best_actions)),
        axis=-1,
    )
    active = jnp.maximum(jnp.sum(weights), 1.0)
    actor_loss = jnp.sum(weights * imitation_error) / active
    action_l2 = jnp.mean(jnp.square(normalized_actions))
    loss = actor_loss + config.latent_policy.action_penalty * action_l2
    environment_actions = midpoint + scale * normalized_actions
    entropy = jnp.sum(
        actor_outputs["log_std"] + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e),
        axis=-1,
    )
    return loss, {
        "actor_loss": actor_loss,
        "critic_loss": jnp.asarray(0.0, dtype=loss.dtype),
        "imagined_reward": jnp.mean(jnp.max(scores, axis=1)),
        "imagined_value": jnp.asarray(0.0, dtype=loss.dtype),
        "imagined_continue": jnp.asarray(1.0, dtype=loss.dtype),
        "actor_entropy": jnp.mean(entropy),
        "candidate_best_score": jnp.mean(jnp.max(scores, axis=1)),
        "candidate_mean_score": jnp.mean(scores),
        "candidate_top_gap": jnp.mean(top_gap),
        "candidate_score_range": jnp.mean(score_range),
        "candidate_active_fraction": jnp.mean(weights),
        "action_l2": action_l2,
        "action_mean": jnp.mean(environment_actions),
        "action_std": jnp.std(environment_actions),
        "action_abs_mean": jnp.mean(jnp.abs(environment_actions)),
        "action_saturation_fraction": jnp.mean(
            (jnp.abs(normalized_actions) >= 0.95).astype(jnp.float32)
        ),
    }


def _critic_loss(
    critic_params: Any,
    critic_state: TrainState,
    rollout: Genie2PolicyRollout,
) -> jax.Array:
    pooled = jnp.mean(jax.lax.stop_gradient(rollout.states), axis=2)
    values = critic_state.apply_fn(critic_params, pooled)
    return jnp.mean(
        jax.lax.stop_gradient(rollout.weights)
        * jnp.square(values - jax.lax.stop_gradient(rollout.returns))
    )


def train_genie2_latent_policy(
    *,
    world_model_state: Genie2TrainState,
    batch: WorldModelSequenceBatch,
    observation_shape: tuple[int, ...],
    config: Genie2ContinuousConfig,
    train_steps: int,
    learning_rate: float,
    imagination_horizon: int | None = None,
    objective: str = "reinforce",
    num_candidates: int = 64,
    candidate_min_gap: float = 0.0,
    seed: int,
) -> tuple[TrainState, TrainState, list[dict[str, float]], Genie2PolicyRollout]:
    if observation_shape != config.observation_shape:
        raise ValueError("observation_shape must match config")
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if objective not in {"reinforce", "candidate-distill"}:
        raise ValueError(f"unsupported policy objective: {objective}")
    if objective == "candidate-distill" and config.action_mode != "continuous":
        raise ValueError("candidate-distill requires continuous actions")
    horizon = imagination_horizon or config.latent_policy.imagination_horizon
    key = jax.random.PRNGKey(seed)
    key, init_key, encode_key = jax.random.split(key, 3)
    actor_state, critic_state = create_latent_policy_states(
        init_key,
        config,
        learning_rate=learning_rate,
    )
    replay = sequence_batch_to_jax(batch)
    latents = jax.jit(
        lambda state, observations, latent_key: encode_genie2_observations(
            state,
            observations,
            config,
            latent_key,
        )
    )(world_model_state, replay.observations, encode_key)
    context_time = min(latents.shape[1], config.dynamics.max_context)
    latent_replay = replay._replace(
        observations=jnp.swapaxes(jax.lax.stop_gradient(latents), 0, 1),
        actions=action_features(replay.actions, config),
    )

    def sample_contexts(
        replay_batch: JaxSequenceBatch,
        sample_key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        contexts = sample_sequence_windows(
            replay_batch,
            sample_key,
            sequence_length=context_time,
            batch_size=config.latent_policy.batch_size,
            require_same_episode=True,
            force_first=False,
        )
        start_latents = jnp.swapaxes(contexts.observations, 0, 1)
        start_actions = jnp.swapaxes(contexts.actions[:-1], 0, 1)
        return start_latents, start_actions

    def run_updates(actor, critic, update_key, replay_batch):
        def update(carry, _):
            current_actor, current_critic, step_key = carry
            step_key, sample_key, actor_key = jax.random.split(step_key, 3)
            start_latents, start_actions = sample_contexts(
                replay_batch,
                sample_key,
            )
            if objective == "candidate-distill":
                (actor_loss, metrics), actor_gradients = jax.value_and_grad(
                    _candidate_distill_loss,
                    has_aux=True,
                )(
                    current_actor.params,
                    world_model_state=world_model_state,
                    actor_state=current_actor,
                    start_latents=start_latents,
                    start_actions=start_actions,
                    config=config,
                    key=actor_key,
                    num_candidates=num_candidates,
                    candidate_min_gap=candidate_min_gap,
                    horizon=horizon,
                )
                del actor_loss
                current_actor = current_actor.apply_gradients(grads=actor_gradients)
            else:
                (actor_loss, rollout), actor_gradients = jax.value_and_grad(
                    _actor_loss,
                    has_aux=True,
                )(
                    current_actor.params,
                    world_model_state=world_model_state,
                    actor_state=current_actor,
                    critic_state=current_critic,
                    start_latents=start_latents,
                    start_actions=start_actions,
                    config=config,
                    horizon=horizon,
                    key=actor_key,
                )
                current_actor = current_actor.apply_gradients(grads=actor_gradients)
                critic_loss, critic_gradients = jax.value_and_grad(_critic_loss)(
                    current_critic.params,
                    current_critic,
                    rollout,
                )
                current_critic = current_critic.apply_gradients(grads=critic_gradients)
                metrics = {
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "imagined_reward": jnp.mean(rollout.rewards),
                    "imagined_value": jnp.mean(rollout.values),
                    "imagined_continue": jnp.mean(rollout.continues),
                    "actor_entropy": jnp.mean(rollout.entropies),
                }
            return (current_actor, current_critic, step_key), metrics

        (actor, critic, update_key), metrics = jax.lax.scan(
            update,
            (actor, critic, update_key),
            None,
            length=train_steps,
        )
        update_key, sample_key, rollout_key = jax.random.split(update_key, 3)
        start_latents, start_actions = sample_contexts(replay_batch, sample_key)
        rollout = simulate_latent_policy_rollout(
            world_model_state=world_model_state,
            actor_state=actor,
            critic_state=critic,
            start_latents=start_latents,
            start_actions=start_actions,
            config=config,
            horizon=horizon,
            key=rollout_key,
        )
        return actor, critic, metrics, rollout

    actor_state, critic_state, metric_arrays, rollout = jax.jit(run_updates)(
        actor_state,
        critic_state,
        key,
        latent_replay,
    )
    host_metrics = jax.device_get(metric_arrays)
    metrics = [
        {
            "step": step + 1,
            **{name: float(values[step]) for name, values in host_metrics.items()},
        }
        for step in range(train_steps)
    ]
    return actor_state, critic_state, metrics, rollout
