"""Training utilities for representation-space SIGReg/JEPA models."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict, freeze, unfreeze

from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.replay import ReplayBatch

ControlMode = Literal[
    "none",
    "no-action-world-model",
    "shuffled-action-replay",
    "frozen-random-world-model",
]
PolicyReturnMode = Literal["reward-only", "lambda"]

MODEL_GROUPS = frozenset(
    {
        "encoder",
        "latent_proj",
        "action_embed",
        "action_encoder_hidden",
        "action_encoder_out",
        "transformer_blocks",
        "horizon_embed",
        "dynamics_norm",
        "predictor",
        "predictor_norm",
        "reward_head",
        "continue_head",
    }
)
ONLINE_FROZEN_ENCODER_MODEL_GROUPS = MODEL_GROUPS - {"encoder"}
ACTOR_GROUPS = frozenset({"actor_head"})
CRITIC_GROUPS = frozenset({"value_head"})
ENSEMBLE_GROUP_PREFIXES = {
    "predictor": "predictor_",
    "predictor_norm": "predictor_norm_",
    "reward_head": "reward_head_",
    "continue_head": "continue_head_",
}


@struct.dataclass
class JepaTrainState:
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: FrozenDict
    control_alignment: jax.Array
    control_scale: jax.Array
    control_bias: jax.Array
    model_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    model_opt_state: optax.OptState
    frozen_encoder_model_tx: optax.GradientTransformation = struct.field(
        pytree_node=False
    )
    frozen_encoder_model_opt_state: optax.OptState
    actor_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    actor_opt_state: optax.OptState
    critic_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    critic_opt_state: optax.OptState

    def apply_model_gradients(self, grads) -> "JepaTrainState":
        updates, opt_state = self.model_tx.update(
            grads,
            self.model_opt_state,
            self.params,
        )
        return self.replace(
            step=self.step + 1,
            params=optax.apply_updates(self.params, updates),
            model_opt_state=opt_state,
        )

    def apply_frozen_encoder_model_gradients(self, grads) -> "JepaTrainState":
        updates, opt_state = self.frozen_encoder_model_tx.update(
            grads,
            self.frozen_encoder_model_opt_state,
            self.params,
        )
        return self.replace(
            step=self.step + 1,
            params=optax.apply_updates(self.params, updates),
            frozen_encoder_model_opt_state=opt_state,
        )

    def apply_actor_gradients(self, grads) -> "JepaTrainState":
        updates, opt_state = self.actor_tx.update(
            grads,
            self.actor_opt_state,
            self.params,
        )
        return self.replace(
            step=self.step + 1,
            params=optax.apply_updates(self.params, updates),
            actor_opt_state=opt_state,
        )

    def apply_critic_gradients(self, grads) -> "JepaTrainState":
        updates, opt_state = self.critic_tx.update(
            grads,
            self.critic_opt_state,
            self.params,
        )
        return self.replace(
            step=self.step + 1,
            params=optax.apply_updates(self.params, updates),
            critic_opt_state=opt_state,
        )


def create_jepa_train_state(
    key: jax.Array,
    config: JepaConfig,
) -> JepaTrainState:
    model = JepaWorldModel(config)
    # JepaConfig deliberately does not store chunk_length as a model invariant.
    # Use max_horizon + 1 positions for init so every head/submodule is touched.
    init_length = max(config.max_horizon + 1, 2)
    if config.action_mode == "discrete":
        init_actions = jnp.zeros((1, init_length - 1), dtype=jnp.int32)
    else:
        init_actions = jnp.zeros(
            (1, init_length - 1, config.action_dim),
            dtype=jnp.float32,
        )
    params = model.init(
        key,
        jnp.zeros((1, init_length, config.observation_dim), dtype=jnp.float32),
        init_actions,
        chunk_length=1,
        method=JepaWorldModel.initialize,
    )["params"]
    params = freeze(params)
    model_tx = _masked_adam(params, MODEL_GROUPS, config.learning_rate)
    frozen_encoder_model_tx = _masked_adam(
        params,
        ONLINE_FROZEN_ENCODER_MODEL_GROUPS,
        config.learning_rate,
    )
    actor_tx = _masked_adam(params, ACTOR_GROUPS, config.actor_learning_rate)
    critic_tx = _masked_adam(params, CRITIC_GROUPS, config.actor_learning_rate)
    return JepaTrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        control_alignment=jnp.eye(config.latent_dim, dtype=jnp.float32),
        control_scale=jnp.asarray(1.0, dtype=jnp.float32),
        control_bias=jnp.zeros((config.latent_dim,), dtype=jnp.float32),
        model_tx=model_tx,
        model_opt_state=model_tx.init(params),
        frozen_encoder_model_tx=frozen_encoder_model_tx,
        frozen_encoder_model_opt_state=frozen_encoder_model_tx.init(params),
        actor_tx=actor_tx,
        actor_opt_state=actor_tx.init(params),
        critic_tx=critic_tx,
        critic_opt_state=critic_tx.init(params),
    )


@partial(
    jax.jit,
    static_argnames=(
        "config",
        "chunk_length",
        "control",
        "freeze_encoder",
        "latent_anchor_weight",
    ),
)
def train_model_step(
    state: JepaTrainState,
    key: jax.Array,
    batch: ReplayBatch,
    config: JepaConfig,
    *,
    chunk_length: int,
    control: ControlMode = "none",
    freeze_encoder: bool = False,
    latent_anchor_params: FrozenDict | None = None,
    latent_anchor_alignment: jax.Array | None = None,
    latent_anchor_scale: jax.Array | None = None,
    latent_anchor_bias: jax.Array | None = None,
    latent_anchor_weight: float = 0.0,
) -> tuple[JepaTrainState, dict[str, jax.Array]]:
    def loss_fn(params):
        loss, metrics = world_model_loss(
            params,
            state.apply_fn,
            key,
            batch,
            config,
            chunk_length=chunk_length,
            control=control,
        )
        if latent_anchor_params is not None and latent_anchor_weight > 0.0:
            alignment = (
                state.control_alignment
                if latent_anchor_alignment is None
                else latent_anchor_alignment
            )
            scale = (
                state.control_scale
                if latent_anchor_scale is None
                else latent_anchor_scale
            )
            bias = state.control_bias if latent_anchor_bias is None else latent_anchor_bias
            anchor_loss = latent_anchor_loss(
                params,
                state.apply_fn,
                latent_anchor_params,
                alignment,
                scale,
                bias,
                batch.observations,
                config,
            )
            loss = loss + latent_anchor_weight * anchor_loss
            metrics = {
                **metrics,
                "model/latent_anchor_loss": anchor_loss,
                "model/latent_anchor_weight": jnp.asarray(
                    latent_anchor_weight,
                    dtype=loss.dtype,
                ),
            }
        else:
            metrics = {
                **metrics,
                "model/latent_anchor_loss": jnp.asarray(0.0, dtype=loss.dtype),
                "model/latent_anchor_weight": jnp.asarray(0.0, dtype=loss.dtype),
            }
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    del loss
    if freeze_encoder:
        return state.apply_frozen_encoder_model_gradients(grads), metrics
    return state.apply_model_gradients(grads), metrics


def apply_control_alignment(
    latents: jax.Array,
    control_alignment: jax.Array,
    control_scale: jax.Array | float = 1.0,
    control_bias: jax.Array | None = None,
) -> jax.Array:
    """Map raw world latents into the stable actor/critic control interface."""

    control_latents = jnp.einsum("...d,df->...f", latents, control_alignment)
    control_latents = jnp.asarray(control_scale, dtype=control_latents.dtype) * (
        control_latents
    )
    if control_bias is not None:
        control_latents = control_latents + jnp.asarray(
            control_bias,
            dtype=control_latents.dtype,
        )
    return control_latents


def actor_value_from_control_latent(
    apply_fn,
    params: FrozenDict,
    raw_latents: jax.Array,
    control_alignment: jax.Array,
    control_scale: jax.Array | float = 1.0,
    control_bias: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    return apply_fn(
        {"params": params},
        apply_control_alignment(
            raw_latents,
            control_alignment,
            control_scale,
            control_bias,
        ),
        method=JepaWorldModel.actor_value_from_latent,
    )


def procrustes_control_alignment(
    source_latents: jax.Array,
    target_control_latents: jax.Array,
) -> jax.Array:
    """Return Q such that source_latents @ Q best matches target_control_latents."""

    source = source_latents.reshape((-1, source_latents.shape[-1])).astype(jnp.float32)
    target = target_control_latents.reshape(
        (-1, target_control_latents.shape[-1])
    ).astype(jnp.float32)
    source = source - jnp.mean(source, axis=0, keepdims=True)
    target = target - jnp.mean(target, axis=0, keepdims=True)
    cross_cov = source.T @ target
    u, _, vh = jnp.linalg.svd(cross_cov, full_matrices=False)
    return (u @ vh).astype(source_latents.dtype)


def umeyama_control_interface(
    source_latents: jax.Array,
    target_control_latents: jax.Array,
    *,
    eps: float = 1e-8,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fit scale, rotation/reflection, and shift from source to target latents.

    Returns ``(alignment, scale, bias)`` for row-vector latents:
    ``target ~= scale * (source @ alignment) + bias``.
    """

    source = source_latents.reshape((-1, source_latents.shape[-1])).astype(jnp.float32)
    target = target_control_latents.reshape(
        (-1, target_control_latents.shape[-1])
    ).astype(jnp.float32)
    source_mean = jnp.mean(source, axis=0)
    target_mean = jnp.mean(target, axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    cross_cov = centered_source.T @ centered_target / jnp.maximum(
        source.shape[0],
        1,
    )
    u, singular_values, vh = jnp.linalg.svd(cross_cov, full_matrices=False)
    alignment = u @ vh
    source_variance = jnp.mean(jnp.sum(jnp.square(centered_source), axis=-1))
    scale = jnp.sum(singular_values) / (source_variance + eps)
    bias = target_mean - scale * (source_mean @ alignment)
    return (
        alignment.astype(source_latents.dtype),
        scale.astype(source_latents.dtype),
        bias.astype(source_latents.dtype),
    )


def latent_anchor_loss(
    params: FrozenDict,
    apply_fn,
    anchor_params: FrozenDict,
    anchor_alignment: jax.Array,
    anchor_scale: jax.Array,
    anchor_bias: jax.Array,
    observations: jax.Array,
    config: JepaConfig,
) -> jax.Array:
    flat_observations = observations.reshape((-1, config.observation_dim))
    new_latents = apply_fn(
        {"params": params},
        flat_observations,
        method=JepaWorldModel.encode,
    )
    new_latents = apply_control_alignment(
        new_latents,
        anchor_alignment,
        anchor_scale,
        anchor_bias,
    )
    teacher_params = jax.tree_util.tree_map(jax.lax.stop_gradient, anchor_params)
    target_latents = apply_fn(
        {"params": teacher_params},
        flat_observations,
        method=JepaWorldModel.encode,
    )
    target_latents = jax.lax.stop_gradient(
        apply_control_alignment(
            target_latents,
            anchor_alignment,
            anchor_scale,
            anchor_bias,
        )
    )
    cosine = jnp.sum(_normalize(new_latents) * _normalize(target_latents), axis=-1)
    return jnp.mean(1.0 - cosine)


def world_model_loss(
    params: FrozenDict,
    apply_fn,
    key: jax.Array,
    batch: ReplayBatch,
    config: JepaConfig,
    *,
    chunk_length: int,
    control: ControlMode,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    action_key, regularizer_key = jax.random.split(key)
    actions = _controlled_actions(action_key, batch.actions, config, control)
    outputs = apply_fn(
        {"params": params},
        batch.observations,
        actions,
        chunk_length=chunk_length,
        dones=batch.dones,
        method=JepaWorldModel.sequence_outputs,
    )
    pred = _normalize(outputs["predicted_latents"])
    target = _normalize(outputs["target_latents"])
    ensemble_axis = pred.ndim == target.ndim + 1
    if ensemble_axis:
        target = target[..., None, :]

    max_horizon = config.max_horizon
    reward_targets = jnp.stack(
        [
            batch.rewards[:, offset : offset + chunk_length]
            for offset in range(max_horizon)
        ],
        axis=2,
    )
    done_targets = jnp.stack(
        [
            batch.dones[:, offset : offset + chunk_length]
            for offset in range(max_horizon)
        ],
        axis=2,
    )
    continue_targets = 1.0 - done_targets
    reward_pred = outputs["reward_logits"]
    continue_logits = outputs["continue_logits"]
    if ensemble_axis:
        reward_targets = reward_targets[..., None]
        continue_targets = continue_targets[..., None]
    validity = prediction_validity(batch.dones, chunk_length, max_horizon)
    if ensemble_axis:
        validity = validity[..., None]
    reward_loss = masked_mean(jnp.square(reward_pred - reward_targets), validity)
    continue_loss = masked_mean(
        optax.sigmoid_binary_cross_entropy(continue_logits, continue_targets),
        validity,
    )
    cosine = jnp.sum(pred * target, axis=-1)
    jepa_loss = masked_mean(1.0 - cosine, validity)
    jepa_cosine = masked_mean(cosine, validity)

    regularizer, regularizer_name, collapse = representation_regularizer(
        outputs["context_latents"],
        regularizer_key,
        config,
    )
    total_loss = (
        jepa_loss
        + config.regularizer_weight * regularizer
        + config.reward_weight * reward_loss
        + config.continue_weight * continue_loss
    )
    constant_continue = jnp.full_like(continue_targets, jnp.mean(continue_targets))
    constant_reward = jnp.full_like(reward_targets, jnp.mean(reward_targets))
    terminal_logits = jnp.mean(continue_logits, axis=-1) if ensemble_axis else continue_logits
    metrics = {
        "model/total_loss": total_loss,
        "model/jepa_loss": jepa_loss,
        "model/jepa_pred_cosine": jepa_cosine,
        "model/jepa_valid_fraction": jnp.mean(validity),
        "model/regularizer_loss": regularizer,
        f"model/{regularizer_name}_loss": regularizer,
        "model/reward_loss": reward_loss,
        "model/reward_constant_mse": masked_mean(
            jnp.square(constant_reward - reward_targets),
            validity,
        ),
        "model/continue_loss": continue_loss,
        **ensemble_prediction_metrics(
            outputs["predicted_latents"],
            outputs["reward_logits"],
            outputs["continue_logits"],
        ),
        "model/continue_constant_bce": masked_mean(
            optax.sigmoid_binary_cross_entropy(
                jnp.log(constant_continue / (1.0 - constant_continue + 1e-6) + 1e-6),
                continue_targets,
            ),
            validity,
        ),
        **terminal_prediction_metrics(terminal_logits, done_targets),
        **{f"collapse/{key}": value for key, value in collapse.items()},
    }
    return total_loss, metrics


def reset_policy_heads(
    state: JepaTrainState,
    key: jax.Array,
    config: JepaConfig,
) -> JepaTrainState:
    fresh = create_jepa_train_state(key, config)
    raw = unfreeze(state.params)
    raw["actor_head"] = unfreeze(fresh.params["actor_head"])
    raw["value_head"] = unfreeze(fresh.params["value_head"])
    params = freeze(raw)
    return state.replace(
        params=params,
        actor_opt_state=state.actor_tx.init(params),
        critic_opt_state=state.critic_tx.init(params),
    )


@partial(
    jax.jit,
    static_argnames=(
        "config",
        "imag_horizon",
        "control",
        "policy_return_mode",
        "behavior_distill_weight",
    ),
)
def continuous_policy_train_step(
    state: JepaTrainState,
    key: jax.Array,
    start_observations: jax.Array,
    config: JepaConfig,
    action_low: jax.Array,
    action_high: jax.Array,
    *,
    imag_horizon: int,
    control: ControlMode = "none",
    policy_return_mode: PolicyReturnMode = "reward-only",
    value_clip: float = 100.0,
    action_saturation_threshold: float = 0.95,
    start_actions: jax.Array | None = None,
    behavior_teacher_params: FrozenDict | None = None,
    behavior_teacher_control_alignment: jax.Array | None = None,
    behavior_teacher_control_scale: jax.Array | None = None,
    behavior_teacher_control_bias: jax.Array | None = None,
    behavior_distill_weight: float = 0.0,
    uncertainty_penalty: float = 0.0,
    uncertainty_latent_weight: float = 1.0,
    uncertainty_reward_weight: float = 1.0,
    uncertainty_continue_weight: float = 1.0,
    uncertainty_threshold: float = float("inf"),
    uncertainty_budget: float = float("inf"),
) -> tuple[JepaTrainState, dict[str, jax.Array]]:
    if config.action_mode != "continuous":
        raise ValueError("continuous_policy_train_step requires continuous actions")
    del key

    def actor_loss_fn(params):
        rollout = continuous_imagine_rollout(
            params,
            state.apply_fn,
            start_observations,
            config,
            action_low,
            action_high,
            imag_horizon=imag_horizon,
            control=control,
            start_actions=start_actions,
            control_alignment=state.control_alignment,
            control_scale=state.control_scale,
            control_bias=state.control_bias,
            uncertainty_penalty=uncertainty_penalty,
            uncertainty_latent_weight=uncertainty_latent_weight,
            uncertainty_reward_weight=uncertainty_reward_weight,
            uncertainty_continue_weight=uncertainty_continue_weight,
            uncertainty_threshold=uncertainty_threshold,
            uncertainty_budget=uncertainty_budget,
        )
        if policy_return_mode == "reward-only":
            actor_returns = reward_only_returns(
                rollout["rewards"],
                rollout["continues"],
                gamma=config.gamma,
            )
        else:
            actor_returns = lambda_returns(
                rollout["rewards"],
                rollout["continues"],
                rollout["fixed_values"],
                rollout["fixed_last_value"],
                gamma=config.gamma,
                lambda_return=config.lambda_return,
            )
        clipped_returns = jnp.clip(actor_returns, -value_clip, value_clip)
        weights = survival_weights(rollout["continues"], gamma=config.gamma)
        actor_loss = -weighted_mean(clipped_returns, weights)
        behavior_distill_loss = jnp.asarray(0.0, dtype=actor_loss.dtype)
        if behavior_teacher_params is not None and behavior_distill_weight > 0.0:
            teacher_params = jax.tree_util.tree_map(
                jax.lax.stop_gradient,
                behavior_teacher_params,
            )
            teacher_context, _ = initial_imagination_context(
                state.apply_fn,
                teacher_params,
                start_observations,
                start_actions,
                config,
            )
            teacher_alignment = (
                state.control_alignment
                if behavior_teacher_control_alignment is None
                else behavior_teacher_control_alignment
            )
            teacher_scale = (
                state.control_scale
                if behavior_teacher_control_scale is None
                else behavior_teacher_control_scale
            )
            teacher_bias = (
                state.control_bias
                if behavior_teacher_control_bias is None
                else behavior_teacher_control_bias
            )
            teacher_logits, _ = actor_value_from_control_latent(
                state.apply_fn,
                teacher_params,
                teacher_context[:, -1],
                teacher_alignment,
                teacher_scale,
                teacher_bias,
            )
            teacher_actions = jax.lax.stop_gradient(jnp.tanh(teacher_logits))
            behavior_distill_loss = jnp.mean(
                jnp.square(rollout["normalized_actions"][0] - teacher_actions)
            )
            actor_loss = actor_loss + behavior_distill_weight * behavior_distill_loss
        action_saturation = jnp.mean(
            (
                jnp.abs(rollout["normalized_actions"]) >= action_saturation_threshold
            ).astype(jnp.float32)
        )
        finite_fraction = _all_finite_fraction(
            rollout["latents"],
            rollout["actions"],
            rollout["normalized_actions"],
            rollout["rewards"],
            rollout["continues"],
            rollout["raw_rewards"],
            rollout["values"],
            rollout["fixed_values"],
            rollout["uncertainty"],
            rollout["trusted"],
            actor_returns,
            clipped_returns,
            actor_loss,
        )
        metrics = {
            "policy/actor_loss": actor_loss,
            "policy/imagined_return": weighted_mean(actor_returns, weights),
            "policy/clipped_imagined_return": weighted_mean(clipped_returns, weights),
            "policy/imagined_reward": weighted_mean(rollout["rewards"], weights),
            "policy/raw_imagined_reward": weighted_mean(
                rollout["raw_rewards"],
                weights,
            ),
            "policy/imagined_continue": weighted_mean(rollout["continues"], weights),
            "policy/survival_weight_mean": jnp.mean(weights),
            "policy/uncertainty": weighted_mean(rollout["uncertainty"], weights),
            "policy/uncertainty_abs_max": jnp.max(jnp.abs(rollout["uncertainty"])),
            "policy/trusted_fraction": jnp.mean(rollout["trusted"]),
            "policy/uncertainty_penalty": jnp.asarray(
                uncertainty_penalty,
                dtype=actor_loss.dtype,
            ),
            "policy/return_abs_mean": jnp.mean(jnp.abs(actor_returns)),
            "policy/return_abs_max": jnp.max(jnp.abs(actor_returns)),
            "policy/value_target_abs_mean": jnp.mean(jnp.abs(clipped_returns)),
            "policy/value_target_abs_max": jnp.max(jnp.abs(clipped_returns)),
            "policy/behavior_distill_loss": behavior_distill_loss,
            "policy/behavior_distill_weight": jnp.asarray(
                behavior_distill_weight,
                dtype=actor_loss.dtype,
            ),
            "policy/action_mean": jnp.mean(rollout["actions"]),
            "policy/action_std": jnp.std(rollout["actions"]),
            "policy/action_abs_mean": jnp.mean(jnp.abs(rollout["actions"])),
            "policy/normalized_action_abs_mean": jnp.mean(
                jnp.abs(rollout["normalized_actions"])
            ),
            "policy/action_saturation_fraction": action_saturation,
            "policy/finite_fraction": finite_fraction,
        }
        critic_latents = jax.lax.stop_gradient(rollout["latents"])
        critic_targets = jax.lax.stop_gradient(clipped_returns)
        critic_weights = jax.lax.stop_gradient(weights)
        return actor_loss, (metrics, critic_latents, critic_targets, critic_weights)

    (actor_loss, actor_aux), actor_grads = jax.value_and_grad(
        actor_loss_fn,
        has_aux=True,
    )(state.params)
    del actor_loss
    metrics, critic_latents, critic_targets, critic_weights = actor_aux
    state = state.apply_actor_gradients(actor_grads)

    def critic_loss_fn(params):
        _, values = actor_value_from_control_latent(
            state.apply_fn,
            params,
            critic_latents,
            state.control_alignment,
            state.control_scale,
            state.control_bias,
        )
        value_loss = 0.5 * weighted_mean(
            jnp.square(values - critic_targets),
            critic_weights,
        )
        finite_fraction = _all_finite_fraction(values, critic_targets, value_loss)
        critic_metrics = {
            "policy/value_loss": value_loss,
            "policy/value_finite_fraction": finite_fraction,
        }
        return value_loss, critic_metrics

    (value_loss, critic_metrics), critic_grads = jax.value_and_grad(
        critic_loss_fn,
        has_aux=True,
    )(state.params)
    state = state.apply_critic_gradients(critic_grads)
    total_loss = metrics["policy/actor_loss"] + value_loss
    metrics = {
        **metrics,
        **critic_metrics,
        "policy/total_loss": total_loss,
        "policy/finite_fraction": jnp.minimum(
            metrics["policy/finite_fraction"],
            critic_metrics["policy/value_finite_fraction"],
        ),
    }
    return state, metrics


@partial(
    jax.jit,
    static_argnames=("config", "imag_horizon", "control", "num_candidates"),
)
def continuous_candidate_distill_step(
    state: JepaTrainState,
    key: jax.Array,
    start_observations: jax.Array,
    config: JepaConfig,
    action_low: jax.Array,
    action_high: jax.Array,
    *,
    imag_horizon: int,
    control: ControlMode = "none",
    num_candidates: int = 64,
    candidate_min_gap: float = 1e-3,
    action_l2_coef: float = 1e-3,
    action_saturation_threshold: float = 0.95,
    start_actions: jax.Array | None = None,
) -> tuple[JepaTrainState, dict[str, jax.Array]]:
    if config.action_mode != "continuous":
        raise ValueError(
            "continuous_candidate_distill_step requires continuous actions"
        )
    if num_candidates < 2:
        raise ValueError("num_candidates must be >= 2")
    del start_actions

    def loss_fn(params):
        model_params = jax.tree_util.tree_map(jax.lax.stop_gradient, params)
        current_observations = (
            start_observations[:, -1]
            if start_observations.ndim == 3
            else start_observations
        )
        flat_obs = current_observations.reshape((-1, config.observation_dim))
        z0 = state.apply_fn(
            {"params": model_params},
            flat_obs,
            method=JepaWorldModel.encode,
        )
        candidates = jax.random.uniform(
            key,
            (z0.shape[0], num_candidates, config.action_dim),
            minval=-1.0,
            maxval=1.0,
            dtype=jnp.float32,
        )
        scores = score_continuous_action_candidates(
            params,
            model_params,
            state.apply_fn,
            z0,
            candidates,
            config,
            action_low,
            action_high,
            imag_horizon=imag_horizon,
            control=control,
            control_alignment=state.control_alignment,
            control_scale=state.control_scale,
            control_bias=state.control_bias,
        )
        best_index = jnp.argmax(scores, axis=1)
        best_action = jnp.take_along_axis(
            candidates,
            best_index[:, None, None],
            axis=1,
        )[:, 0]
        sorted_scores = jnp.sort(scores, axis=1)
        top_gap = sorted_scores[:, -1] - sorted_scores[:, -2]
        score_range = sorted_scores[:, -1] - sorted_scores[:, 0]
        weights = (top_gap > candidate_min_gap).astype(jnp.float32)

        raw_actions, _ = actor_value_from_control_latent(
            state.apply_fn,
            params,
            z0,
            state.control_alignment,
            state.control_scale,
            state.control_bias,
        )
        normalized_actions = jnp.tanh(raw_actions)
        imitation_error = jnp.mean(
            jnp.square(normalized_actions - jax.lax.stop_gradient(best_action)),
            axis=-1,
        )
        imitation_loss = jnp.sum(weights * imitation_error) / (jnp.sum(weights) + 1e-6)
        active_fraction = jnp.mean(weights)
        action_l2 = jnp.mean(jnp.square(normalized_actions))
        total = imitation_loss + action_l2_coef * active_fraction * action_l2
        action_saturation = jnp.mean(
            (jnp.abs(normalized_actions) >= action_saturation_threshold).astype(
                jnp.float32
            )
        )
        finite_fraction = _all_finite_fraction(
            z0,
            candidates,
            scores,
            best_action,
            normalized_actions,
            imitation_loss,
            total,
        )
        metrics = {
            "policy/total_loss": total,
            "policy/actor_loss": imitation_loss,
            "policy/value_loss": jnp.asarray(0.0, dtype=total.dtype),
            "policy/candidate_best_score": jnp.mean(jnp.max(scores, axis=1)),
            "policy/candidate_mean_score": jnp.mean(scores),
            "policy/candidate_top_gap": jnp.mean(top_gap),
            "policy/candidate_score_range": jnp.mean(score_range),
            "policy/candidate_active_fraction": active_fraction,
            "policy/action_l2": action_l2,
            "policy/action_mean": jnp.mean(
                scale_normalized_actions(normalized_actions, action_low, action_high)
            ),
            "policy/action_std": jnp.std(
                scale_normalized_actions(normalized_actions, action_low, action_high)
            ),
            "policy/action_abs_mean": jnp.mean(
                jnp.abs(
                    scale_normalized_actions(
                        normalized_actions,
                        action_low,
                        action_high,
                    )
                )
            ),
            "policy/normalized_action_abs_mean": jnp.mean(jnp.abs(normalized_actions)),
            "policy/action_saturation_fraction": action_saturation,
            "policy/finite_fraction": finite_fraction,
        }
        return total, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    del loss
    return state.apply_actor_gradients(grads), metrics


def score_continuous_action_candidates(
    params: FrozenDict,
    model_params: FrozenDict,
    apply_fn,
    z0: jax.Array,
    normalized_candidates: jax.Array,
    config: JepaConfig,
    action_low: jax.Array,
    action_high: jax.Array,
    *,
    imag_horizon: int,
    control: ControlMode,
    control_alignment: jax.Array,
    control_scale: jax.Array | float = 1.0,
    control_bias: jax.Array | None = None,
) -> jax.Array:
    batch_size, num_candidates, _ = normalized_candidates.shape
    flat_z = jnp.repeat(z0[:, None, :], num_candidates, axis=1).reshape(
        (-1, config.latent_dim)
    )
    flat_actions = normalized_candidates.reshape((-1, config.action_dim))
    returns = jnp.zeros((flat_z.shape[0],), dtype=jnp.float32)
    weights = jnp.ones_like(returns)
    discount = jnp.asarray(1.0, dtype=jnp.float32)
    context = flat_z[:, None, :]
    actions = scale_normalized_actions(flat_actions, action_low, action_high)

    for _ in range(imag_horizon):
        model_actions = (
            jnp.zeros_like(actions) if control == "no-action-world-model" else actions
        )
        next_z, rewards, continue_logits = apply_fn(
            {"params": model_params},
            context,
            model_actions[:, None, :],
            method=JepaWorldModel.predict_next_from_history,
        )
        continues = jax.nn.sigmoid(continue_logits)
        returns = returns + discount * weights * rewards
        weights = weights * continues
        discount = discount * config.gamma
        raw_actions, _ = actor_value_from_control_latent(
            apply_fn,
            model_params,
            next_z,
            control_alignment,
            control_scale,
            control_bias,
        )
        actions = scale_normalized_actions(
            jnp.tanh(raw_actions),
            action_low,
            action_high,
        )
        context = next_z[:, None, :]

    return jax.lax.stop_gradient(returns.reshape((batch_size, num_candidates)))


@partial(jax.jit, static_argnames=("config", "horizon"))
def continuous_critic_warmup_step(
    state: JepaTrainState,
    batch: ReplayBatch,
    config: JepaConfig,
    *,
    horizon: int,
    value_clip: float = 100.0,
) -> tuple[JepaTrainState, dict[str, jax.Array]]:
    if config.action_mode != "continuous":
        raise ValueError("continuous_critic_warmup_step requires continuous actions")

    def loss_fn(params):
        rewards = jnp.swapaxes(batch.rewards[:, :horizon], 0, 1)
        dones = jnp.swapaxes(batch.dones[:, :horizon], 0, 1)
        continues = 1.0 - dones
        returns = reward_only_returns(rewards, continues, gamma=config.gamma)[0]
        targets = jax.lax.stop_gradient(jnp.clip(returns, -value_clip, value_clip))
        model_params = jax.tree_util.tree_map(jax.lax.stop_gradient, params)
        z = state.apply_fn(
            {"params": model_params},
            batch.observations[:, 0],
            method=JepaWorldModel.encode,
        )
        _, values = actor_value_from_control_latent(
            state.apply_fn,
            params,
            z,
            state.control_alignment,
            state.control_scale,
            state.control_bias,
        )
        value_loss = 0.5 * jnp.mean(jnp.square(values - targets))
        finite_fraction = _all_finite_fraction(values, targets, value_loss)
        metrics = {
            "critic/total_loss": value_loss,
            "critic/value_loss": value_loss,
            "critic/target_mean": jnp.mean(targets),
            "critic/target_abs_mean": jnp.mean(jnp.abs(targets)),
            "critic/target_abs_max": jnp.max(jnp.abs(targets)),
            "critic/value_mean": jnp.mean(values),
            "critic/value_abs_mean": jnp.mean(jnp.abs(values)),
            "critic/finite_fraction": finite_fraction,
        }
        return value_loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    del loss
    return state.apply_critic_gradients(grads), metrics


def continuous_imagine_rollout(
    params: FrozenDict,
    apply_fn,
    start_observations: jax.Array,
    config: JepaConfig,
    action_low: jax.Array,
    action_high: jax.Array,
    *,
    imag_horizon: int,
    control: ControlMode,
    start_actions: jax.Array | None = None,
    control_alignment: jax.Array,
    control_scale: jax.Array | float = 1.0,
    control_bias: jax.Array | None = None,
    uncertainty_penalty: float = 0.0,
    uncertainty_latent_weight: float = 1.0,
    uncertainty_reward_weight: float = 1.0,
    uncertainty_continue_weight: float = 1.0,
    uncertainty_threshold: float = float("inf"),
    uncertainty_budget: float = float("inf"),
) -> dict[str, jax.Array]:
    model_params = jax.tree_util.tree_map(jax.lax.stop_gradient, params)
    context, action_context = initial_imagination_context(
        apply_fn,
        model_params,
        start_observations,
        start_actions,
        config,
    )
    batch_size = context.shape[0]
    initial_uncertainty = jnp.zeros((batch_size,), dtype=context.dtype)
    initial_trusted = jnp.ones((batch_size,), dtype=context.dtype)

    def step(carry, _):
        context, action_context, cumulative_uncertainty, active = carry
        current_z = context[:, -1]
        raw_actions, values = actor_value_from_control_latent(
            apply_fn,
            params,
            current_z,
            control_alignment,
            control_scale,
            control_bias,
        )
        normalized_actions = jnp.tanh(raw_actions)
        actions = scale_normalized_actions(
            normalized_actions,
            action_low,
            action_high,
        )
        model_actions = (
            jnp.zeros_like(actions) if control == "no-action-world-model" else actions
        )
        model_action_context = replace_last_action_context(
            action_context,
            model_actions,
            config,
        )
        z_ensemble, reward_ensemble, continue_logit_ensemble = apply_fn(
            {"params": model_params},
            context,
            model_action_context,
            method=JepaWorldModel.predict_next_ensemble_from_history,
        )
        next_z = jnp.mean(z_ensemble, axis=0)
        raw_rewards = jnp.mean(reward_ensemble, axis=0)
        continues = jnp.mean(jax.nn.sigmoid(continue_logit_ensemble), axis=0)
        uncertainty = ensemble_transition_uncertainty(
            z_ensemble,
            reward_ensemble,
            continue_logit_ensemble,
            latent_weight=uncertainty_latent_weight,
            reward_weight=uncertainty_reward_weight,
            continue_weight=uncertainty_continue_weight,
        )
        next_cumulative_uncertainty = cumulative_uncertainty + uncertainty
        trusted = active * (
            (uncertainty <= uncertainty_threshold)
            & (next_cumulative_uncertainty <= uncertainty_budget)
        ).astype(context.dtype)
        rewards = (raw_rewards - uncertainty_penalty * uncertainty) * trusted
        continues = continues * trusted
        _, fixed_values = actor_value_from_control_latent(
            apply_fn,
            model_params,
            current_z,
            control_alignment,
            control_scale,
            control_bias,
        )
        next_context = jnp.concatenate([context[:, 1:], next_z[:, None, :]], axis=1)
        next_action_context = append_action_context(
            model_action_context,
            jnp.zeros_like(model_actions),
            config,
        )
        return (
            next_context,
            next_action_context,
            next_cumulative_uncertainty,
            trusted,
        ), {
            "latents": current_z,
            "actions": actions,
            "normalized_actions": normalized_actions,
            "values": values,
            "fixed_values": fixed_values,
            "raw_rewards": raw_rewards,
            "rewards": rewards,
            "continues": continues,
            "uncertainty": uncertainty,
            "trusted": trusted,
        }

    (final_context, _, _, _), rollout = jax.lax.scan(
        step,
        (context, action_context, initial_uncertainty, initial_trusted),
        xs=None,
        length=imag_horizon,
    )
    _, fixed_last_value = actor_value_from_control_latent(
        apply_fn,
        model_params,
        final_context[:, -1],
        control_alignment,
        control_scale,
        control_bias,
    )
    rollout["fixed_last_value"] = fixed_last_value
    return rollout


def ensemble_transition_uncertainty(
    z_ensemble: jax.Array,
    reward_ensemble: jax.Array,
    continue_logit_ensemble: jax.Array,
    *,
    latent_weight: float,
    reward_weight: float,
    continue_weight: float,
) -> jax.Array:
    normalized = _normalize(z_ensemble)
    mean_direction = jnp.mean(normalized, axis=0)
    latent_disagreement = 1.0 - jnp.sum(jnp.square(mean_direction), axis=-1)
    reward_variance = jnp.var(reward_ensemble, axis=0)
    continue_variance = jnp.var(jax.nn.sigmoid(continue_logit_ensemble), axis=0)
    return (
        latent_weight * latent_disagreement
        + reward_weight * reward_variance
        + continue_weight * continue_variance
    )


def initial_imagination_context(
    apply_fn,
    model_params: FrozenDict,
    start_observations: jax.Array,
    start_actions: jax.Array | None,
    config: JepaConfig,
) -> tuple[jax.Array, jax.Array]:
    if start_observations.ndim == 2:
        flat_obs = start_observations.reshape((-1, config.observation_dim))
        z0 = apply_fn({"params": model_params}, flat_obs, method=JepaWorldModel.encode)
        context = jnp.repeat(z0[:, None, :], repeats=config.context_window, axis=1)
        action_context = initial_action_context(z0.shape[0], config)
        if start_actions is not None:
            action_context = start_actions[:, -config.context_window :]
        return context, action_context

    if start_observations.ndim != 3:
        raise ValueError(
            "start_observations must be [batch, obs] or [batch, time, obs]"
        )
    if start_observations.shape[1] < config.context_window:
        raise ValueError("start_observations does not cover context_window")
    latents = apply_fn(
        {"params": model_params},
        start_observations[:, -config.context_window :],
        method=JepaWorldModel.encode,
    )
    if start_actions is None:
        action_context = initial_action_context(latents.shape[0], config)
    else:
        action_context = start_actions[:, -config.context_window :]
    return latents, action_context


def select_continuous_actions(
    state: JepaTrainState,
    observations: jax.Array,
    config: JepaConfig,
    action_low: jax.Array,
    action_high: jax.Array,
) -> jax.Array:
    if config.action_mode != "continuous":
        raise ValueError("select_continuous_actions requires continuous actions")
    flat_obs = observations.reshape((-1, config.observation_dim))
    z = state.apply_fn(
        {"params": state.params},
        flat_obs,
        method=JepaWorldModel.encode,
    )
    raw_actions, _ = actor_value_from_control_latent(
        state.apply_fn,
        state.params,
        z,
        state.control_alignment,
        state.control_scale,
        state.control_bias,
    )
    normalized_actions = jnp.tanh(raw_actions)
    return scale_normalized_actions(
        normalized_actions,
        action_low,
        action_high,
    ).reshape((*observations.shape[:-1], config.action_dim))


def scale_normalized_actions(
    normalized_actions: jax.Array,
    action_low: jax.Array,
    action_high: jax.Array,
) -> jax.Array:
    low = jnp.asarray(action_low, dtype=normalized_actions.dtype)
    high = jnp.asarray(action_high, dtype=normalized_actions.dtype)
    return low + 0.5 * (normalized_actions + 1.0) * (high - low)


def lambda_returns(
    rewards: jax.Array,
    continues: jax.Array,
    values: jax.Array,
    last_value: jax.Array,
    *,
    gamma: float,
    lambda_return: float,
) -> jax.Array:
    next_values = jnp.concatenate(
        [values[1:], last_value[None, ...]],
        axis=0,
    )

    def scan_fn(next_return, inputs):
        reward, cont, next_value = inputs
        bootstrap = (1.0 - lambda_return) * next_value + lambda_return * next_return
        ret = reward + gamma * cont * bootstrap
        return ret, ret

    _, returns = jax.lax.scan(
        scan_fn,
        last_value,
        (rewards[::-1], continues[::-1], next_values[::-1]),
    )
    return returns[::-1]


def reward_only_returns(
    rewards: jax.Array,
    continues: jax.Array,
    *,
    gamma: float,
) -> jax.Array:
    def scan_fn(next_return, inputs):
        reward, cont = inputs
        ret = reward + gamma * cont * next_return
        return ret, ret

    _, returns = jax.lax.scan(
        scan_fn,
        jnp.zeros_like(rewards[-1]),
        (rewards[::-1], continues[::-1]),
    )
    return returns[::-1]


@partial(jax.jit, static_argnames=("config", "horizon", "control"))
def evaluate_open_loop(
    state: JepaTrainState,
    batch: ReplayBatch,
    config: JepaConfig,
    *,
    horizon: int,
    control: ControlMode = "none",
) -> dict[str, jax.Array]:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if batch.observations.shape[1] < config.context_window + horizon:
        raise ValueError("batch is too short for context_window + horizon")

    target_z = state.apply_fn(
        {"params": state.params},
        batch.observations[:, : config.context_window + horizon],
        method=JepaWorldModel.encode,
    )
    context = target_z[:, : config.context_window]
    action_context = batch.actions[:, : config.context_window]
    if control == "no-action-world-model":
        action_context = jnp.zeros_like(action_context)
    preds = []
    for t in range(horizon):
        if t > 0:
            actions = batch.actions[:, config.context_window - 1 + t]
            if control == "no-action-world-model":
                actions = jnp.zeros_like(actions)
            action_context = append_action_context(action_context, actions, config)
        next_z, _, _ = state.apply_fn(
            {"params": state.params},
            context,
            action_context,
            method=JepaWorldModel.predict_next_from_history,
        )
        preds.append(next_z)
        context = jnp.concatenate([context[:, 1:], next_z[:, None, :]], axis=1)
    pred = _normalize(jnp.stack(preds, axis=1))
    target = _normalize(
        jax.lax.stop_gradient(
            target_z[:, config.context_window : config.context_window + horizon]
        )
    )
    validity = jnp.cumprod(
        1.0 - batch.dones[:, : config.context_window + horizon],
        axis=1,
    )[:, config.context_window - 1 : config.context_window - 1 + horizon]
    cosine = jnp.sum(pred * target, axis=-1)
    error = 1.0 - cosine
    return {
        "model/open_loop_loss": masked_mean(error, validity),
        "model/open_loop_cosine": masked_mean(cosine, validity),
        "model/open_loop_valid_fraction": jnp.mean(validity),
        "model/open_loop_finite_fraction": _all_finite_fraction(pred, target),
    }


def initial_action_context(batch_size: int, config: JepaConfig) -> jax.Array:
    if config.action_mode == "continuous":
        return jnp.zeros(
            (batch_size, config.context_window, config.action_dim),
            dtype=jnp.float32,
        )
    return jnp.zeros((batch_size, config.context_window), dtype=jnp.int32)


def append_action_context(
    action_context: jax.Array,
    actions: jax.Array,
    config: JepaConfig,
) -> jax.Array:
    if config.action_mode == "continuous":
        actions = actions.reshape((actions.shape[0], config.action_dim))
    return jnp.concatenate(
        [action_context[:, 1:], actions[:, None]],
        axis=1,
    )


def replace_last_action_context(
    action_context: jax.Array,
    actions: jax.Array,
    config: JepaConfig,
) -> jax.Array:
    if config.action_mode == "continuous":
        actions = actions.reshape((actions.shape[0], config.action_dim))
    return action_context.at[:, -1].set(actions)


def representation_regularizer(
    latents: jax.Array,
    key: jax.Array,
    config: JepaConfig,
) -> tuple[jax.Array, str, dict[str, jax.Array]]:
    collapse = latent_collapse_metrics(latents)
    if config.regularizer == "none":
        return jnp.asarray(0.0, dtype=latents.dtype), "none", collapse
    return (
        sigreg_loss(
            latents,
            key,
            knots=config.sigreg_knots,
            num_proj=config.sigreg_num_proj,
        ),
        "sigreg",
        collapse,
    )


def ensemble_prediction_metrics(
    predicted_latents: jax.Array,
    reward_logits: jax.Array,
    continue_logits: jax.Array,
) -> dict[str, jax.Array]:
    if predicted_latents.ndim != 5:
        zero = jnp.asarray(0.0, dtype=predicted_latents.dtype)
        return {
            "model/ensemble_latent_disagreement": zero,
            "model/ensemble_reward_std": zero,
            "model/ensemble_continue_std": zero,
        }

    normalized = _normalize(predicted_latents)
    mean_direction = jnp.mean(normalized, axis=-2)
    latent_disagreement = 1.0 - jnp.sum(jnp.square(mean_direction), axis=-1)
    continue_probs = jax.nn.sigmoid(continue_logits)
    return {
        "model/ensemble_latent_disagreement": jnp.mean(latent_disagreement),
        "model/ensemble_reward_std": jnp.mean(jnp.std(reward_logits, axis=-1)),
        "model/ensemble_continue_std": jnp.mean(jnp.std(continue_probs, axis=-1)),
    }


def sigreg_loss(
    latents: jax.Array,
    key: jax.Array,
    *,
    knots: int = 17,
    num_proj: int = 1024,
) -> jax.Array:
    """Sketch Isotropic Gaussian Regularizer from LeWM, in JAX.

    The official PyTorch implementation expects embeddings shaped [time, batch, dim]
    and compares random one-dimensional projections to a standard Gaussian
    characteristic function. Our model stores [batch, time, dim], so we transpose
    before applying the same statistic.
    """

    proj = jnp.swapaxes(latents, 0, 1)
    dim = proj.shape[-1]
    random_proj = jax.random.normal(key, (dim, num_proj), dtype=proj.dtype)
    random_proj = random_proj / (
        jnp.linalg.norm(random_proj, axis=0, keepdims=True) + 1e-6
    )
    t = jnp.linspace(0.0, 3.0, knots, dtype=proj.dtype)
    dt = jnp.asarray(3.0 / (knots - 1), dtype=proj.dtype)
    weights = jnp.full((knots,), 2.0 * dt, dtype=proj.dtype)
    weights = weights.at[0].set(dt)
    weights = weights.at[-1].set(dt)
    phi = jnp.exp(-jnp.square(t) / 2.0)
    weighted_window = weights * phi

    x_t = (proj @ random_proj)[..., None] * t
    err = jnp.square(jnp.mean(jnp.cos(x_t), axis=-3) - phi) + jnp.square(
        jnp.mean(jnp.sin(x_t), axis=-3)
    )
    statistic = (err @ weighted_window) * proj.shape[-2]
    return jnp.mean(statistic)


def latent_collapse_metrics(latents: jax.Array) -> dict[str, jax.Array]:
    z = latents.reshape((-1, latents.shape[-1]))
    z = z - jnp.mean(z, axis=0, keepdims=True)
    std = jnp.sqrt(jnp.var(z, axis=0) + 1e-6)
    cov = (z.T @ z) / jnp.maximum(z.shape[0] - 1, 1)
    cov_diag = jnp.diag(jnp.diag(cov))
    offdiag = cov - cov_diag
    eigvals = jnp.clip(jnp.linalg.eigvalsh(cov), min=0.0)
    total_variance = jnp.sum(eigvals)
    probs = eigvals / (total_variance + 1e-12)
    entropy = -jnp.sum(
        jnp.where(
            probs > 0.0,
            probs * jnp.log(probs + 1e-12),
            0.0,
        )
    )
    effective_rank = jnp.where(total_variance > 1e-8, jnp.exp(entropy), 0.0)
    norms = jnp.linalg.norm(latents, axis=-1)
    metrics = {
        "latent_std_mean": jnp.mean(std),
        "latent_std_min": jnp.min(std),
        "latent_cov_offdiag_norm": jnp.sqrt(jnp.sum(jnp.square(offdiag))),
        "latent_effective_rank": effective_rank,
        "latent_norm_mean": jnp.mean(norms),
        "latent_norm_std": jnp.std(norms),
    }
    return metrics


def _normalize(x: jax.Array) -> jax.Array:
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


def _controlled_actions(
    key: jax.Array,
    actions: jax.Array,
    config: JepaConfig,
    control: ControlMode,
) -> jax.Array:
    if control == "no-action-world-model":
        return jnp.zeros_like(actions)
    if control == "shuffled-action-replay":
        if config.action_mode == "continuous":
            flat = actions.reshape((-1, actions.shape[-1]))
            return jax.random.permutation(key, flat, axis=0).reshape(actions.shape)
        flat = actions.reshape((-1,))
        return jax.random.permutation(key, flat).reshape(actions.shape)
    del config
    return actions


def prediction_validity(
    dones: jax.Array,
    chunk_length: int,
    max_horizon: int,
) -> jax.Array:
    validity = []
    not_done = 1.0 - dones
    for horizon in range(1, max_horizon + 1):
        windows = [
            not_done[:, offset : offset + chunk_length] for offset in range(horizon)
        ]
        validity.append(jnp.prod(jnp.stack(windows, axis=0), axis=0))
    return jnp.stack(validity, axis=2)


def terminal_prediction_metrics(
    continue_logits: jax.Array,
    dones: jax.Array,
) -> dict[str, jax.Array]:
    terminal_targets = dones.astype(jnp.float32)
    terminal_probs = 1.0 - jax.nn.sigmoid(continue_logits)
    terminal_pred = terminal_probs >= 0.5
    terminal_true = terminal_targets >= 0.5
    nonterminal_true = ~terminal_true
    terminal_recall = jnp.sum((terminal_pred & terminal_true).astype(jnp.float32)) / (
        jnp.sum(terminal_true.astype(jnp.float32)) + 1e-6
    )
    nonterminal_recall = jnp.sum(
        ((~terminal_pred) & nonterminal_true).astype(jnp.float32)
    ) / (jnp.sum(nonterminal_true.astype(jnp.float32)) + 1e-6)
    return {
        "model/terminal_positive_fraction": jnp.mean(terminal_targets),
        "model/terminal_recall": terminal_recall,
        "model/nonterminal_recall": nonterminal_recall,
        "model/terminal_balanced_accuracy": 0.5
        * (terminal_recall + nonterminal_recall),
    }


def masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    return jnp.sum(values * mask) / (jnp.sum(mask) + 1e-6)


def survival_weights(continues: jax.Array, *, gamma: float) -> jax.Array:
    starts = jnp.ones_like(continues[:1])
    discounted_continues = gamma * continues[:-1]
    return jax.lax.stop_gradient(
        jnp.cumprod(jnp.concatenate([starts, discounted_continues], axis=0), axis=0)
    )


def weighted_mean(values: jax.Array, weights: jax.Array) -> jax.Array:
    return jnp.sum(values * weights) / (jnp.sum(weights) + 1e-6)


def _masked_adam(
    params: FrozenDict,
    trainable_groups: frozenset[str],
    learning_rate: float,
) -> optax.GradientTransformation:
    labels = _label_params(params, trainable_groups)
    return optax.multi_transform(
        {
            "train": optax.adam(learning_rate, eps=1e-5),
            "freeze": optax.set_to_zero(),
        },
        labels,
    )


def _label_params(params: FrozenDict, trainable_groups: frozenset[str]) -> FrozenDict:
    raw = unfreeze(params)
    labels = {
        key: jax.tree_util.tree_map(
            lambda _: "train"
            if _trainable_param_group(key, trainable_groups)
            else "freeze",
            value,
        )
        for key, value in raw.items()
    }
    return freeze(labels)


def _trainable_param_group(key: str, trainable_groups: frozenset[str]) -> bool:
    if key in trainable_groups:
        return True
    if key.startswith("block_") and "transformer_blocks" in trainable_groups:
        return True
    return any(
        key.startswith(prefix) and group in trainable_groups
        for group, prefix in ENSEMBLE_GROUP_PREFIXES.items()
    )


def _finite_fraction(x: jax.Array) -> jax.Array:
    return jnp.mean(jnp.isfinite(x).astype(jnp.float32))


def _all_finite_fraction(*values: jax.Array) -> jax.Array:
    fractions = [_finite_fraction(value) for value in values]
    return jnp.min(jnp.stack(fractions))
