"""Training utilities for the decoder-free isotropy-JEPA agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from typing import Any, Literal

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
    "no-sigreg",
    "weak-sigreg",
    "no-isotropy",
    "weak-isotropy",
]

MODEL_GROUPS = frozenset(
    {
        "encoder",
        "latent_proj",
        "action_embed",
        "horizon_embed",
        "dynamics_norm",
        "predictor",
        "predictor_norm",
        "reward_head",
        "continue_head",
    }
)
POLICY_GROUPS = frozenset({"actor_head", "value_head"})


@struct.dataclass
class JepaTrainState:
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: FrozenDict
    model_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    model_opt_state: optax.OptState
    policy_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    policy_opt_state: optax.OptState

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

    def apply_policy_gradients(self, grads) -> "JepaTrainState":
        updates, opt_state = self.policy_tx.update(
            grads,
            self.policy_opt_state,
            self.params,
        )
        return self.replace(
            step=self.step + 1,
            params=optax.apply_updates(self.params, updates),
            policy_opt_state=opt_state,
        )


def create_jepa_train_state(
    key: jax.Array,
    config: JepaConfig,
) -> JepaTrainState:
    model = JepaWorldModel(config)
    # JepaConfig deliberately does not store chunk_length as a model invariant.
    # Use max_horizon + 1 positions for init so every head/submodule is touched.
    init_length = max(config.max_horizon + 1, 2)
    params = model.init(
        key,
        jnp.zeros((1, init_length, config.observation_dim), dtype=jnp.float32),
        jnp.zeros((1, init_length - 1), dtype=jnp.int32),
        chunk_length=1,
        method=JepaWorldModel.initialize,
    )["params"]
    params = freeze(params)
    model_tx = _masked_adam(params, MODEL_GROUPS, config.learning_rate)
    policy_tx = _masked_adam(params, POLICY_GROUPS, config.actor_learning_rate)
    return JepaTrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        model_tx=model_tx,
        model_opt_state=model_tx.init(params),
        policy_tx=policy_tx,
        policy_opt_state=policy_tx.init(params),
    )


@partial(jax.jit, static_argnames=("config", "chunk_length", "control"))
def train_model_step(
    state: JepaTrainState,
    key: jax.Array,
    batch: ReplayBatch,
    config: JepaConfig,
    *,
    chunk_length: int,
    control: ControlMode = "none",
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
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    del loss
    return state.apply_model_gradients(grads), metrics


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
    actions = _controlled_actions(key, batch.actions, config, control)
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

    reward_targets = batch.rewards[:, :chunk_length]
    continue_targets = 1.0 - batch.dones[:, :chunk_length]
    reward_pred = outputs["reward_logits"]
    continue_logits = outputs["continue_logits"]
    reward_loss = jnp.mean(jnp.square(reward_pred - reward_targets))
    continue_loss = jnp.mean(
        optax.sigmoid_binary_cross_entropy(continue_logits, continue_targets)
    )
    validity = prediction_validity(batch.dones, chunk_length, config.max_horizon)
    cosine = jnp.sum(pred * target, axis=-1)
    jepa_loss = masked_mean(1.0 - cosine, validity)
    jepa_cosine = masked_mean(cosine, validity)

    isotropy_weight = _isotropy_weight(config, control)
    isotropy, collapse = isotropy_loss(outputs["context_latents"])
    total_loss = (
        jepa_loss
        + isotropy_weight * isotropy
        + config.reward_weight * reward_loss
        + config.continue_weight * continue_loss
    )
    constant_continue = jnp.full_like(continue_targets, jnp.mean(continue_targets))
    constant_reward = jnp.full_like(reward_targets, jnp.mean(reward_targets))
    metrics = {
        "model/total_loss": total_loss,
        "model/jepa_loss": jepa_loss,
        "model/jepa_pred_cosine": jepa_cosine,
        "model/jepa_valid_fraction": jnp.mean(validity),
        "model/isotropy_loss": isotropy,
        "model/reward_loss": reward_loss,
        "model/reward_constant_mse": jnp.mean(
            jnp.square(constant_reward - reward_targets)
        ),
        "model/continue_loss": continue_loss,
        "model/continue_constant_bce": jnp.mean(
            optax.sigmoid_binary_cross_entropy(
                jnp.log(constant_continue / (1.0 - constant_continue + 1e-6) + 1e-6),
                continue_targets,
            )
        ),
        **{f"collapse/{key}": value for key, value in collapse.items()},
    }
    return total_loss, metrics


@partial(jax.jit, static_argnames=("config", "imag_horizon", "control"))
def policy_train_step(
    state: JepaTrainState,
    key: jax.Array,
    start_observations: jax.Array,
    config: JepaConfig,
    *,
    imag_horizon: int,
    control: ControlMode = "none",
) -> tuple[JepaTrainState, dict[str, jax.Array]]:
    def loss_fn(params):
        rollout = imagine_rollout(
            params,
            state.apply_fn,
            key,
            start_observations,
            config,
            imag_horizon=imag_horizon,
            control=control,
        )
        returns = lambda_returns(
            rollout["rewards"],
            rollout["continues"],
            rollout["values"],
            rollout["last_value"],
            gamma=config.gamma,
            lambda_return=config.lambda_return,
        )
        weights = survival_weights(rollout["continues"], gamma=config.gamma)
        advantages = jax.lax.stop_gradient(returns - rollout["values"])
        actor_loss = -weighted_mean(rollout["log_probs"] * advantages, weights)
        value_loss = 0.5 * weighted_mean(
            jnp.square(rollout["values"] - jax.lax.stop_gradient(returns)),
            weights,
        )
        entropy = weighted_mean(rollout["entropies"], weights)
        total = actor_loss + value_loss - config.entropy_coef * entropy
        finite_fraction = _all_finite_fraction(
            rollout["latents"],
            rollout["rewards"],
            rollout["continues"],
            rollout["values"],
            rollout["log_probs"],
            returns,
            actor_loss,
            value_loss,
            total,
        )
        metrics = {
            "policy/total_loss": total,
            "policy/actor_loss": actor_loss,
            "policy/value_loss": value_loss,
            "policy/entropy": entropy,
            "policy/imagined_reward": weighted_mean(rollout["rewards"], weights),
            "policy/imagined_continue": weighted_mean(rollout["continues"], weights),
            "policy/survival_weight_mean": jnp.mean(weights),
            "policy/finite_fraction": finite_fraction,
        }
        return total, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    del loss
    return state.apply_policy_gradients(grads), metrics


def imagine_rollout(
    params: FrozenDict,
    apply_fn,
    key: jax.Array,
    start_observations: jax.Array,
    config: JepaConfig,
    *,
    imag_horizon: int,
    control: ControlMode,
) -> dict[str, jax.Array]:
    flat_obs = start_observations.reshape((-1, config.observation_dim))
    z0 = apply_fn({"params": params}, flat_obs, method=JepaWorldModel.encode)
    context = jnp.repeat(z0[:, None, :], repeats=config.context_window, axis=1)
    action_context = jnp.zeros(
        (z0.shape[0], config.context_window),
        dtype=jnp.int32,
    )

    def step(carry, step_key):
        context, action_context = carry
        current_z = context[:, -1]
        logits, values = apply_fn(
            {"params": params},
            current_z,
            method=JepaWorldModel.actor_value_from_latent,
        )
        actions, log_probs, entropies = sample_categorical(logits, step_key)
        model_actions = (
            jnp.zeros_like(actions) if control == "no-action-world-model" else actions
        )
        action_context = jnp.concatenate(
            [action_context[:, 1:], model_actions[:, None]],
            axis=1,
        )
        next_z, rewards, continue_logits = apply_fn(
            {"params": params},
            context,
            action_context,
            method=JepaWorldModel.predict_next_from_history,
        )
        continues = jax.nn.sigmoid(continue_logits)
        next_context = jnp.concatenate([context[:, 1:], next_z[:, None, :]], axis=1)
        return (next_context, action_context), {
            "latents": current_z,
            "actions": actions,
            "log_probs": log_probs,
            "entropies": entropies,
            "values": values,
            "rewards": rewards,
            "continues": continues,
        }

    step_keys = jax.random.split(key, imag_horizon)
    (final_context, _), rollout = jax.lax.scan(step, (context, action_context), step_keys)
    _, last_value = apply_fn(
        {"params": params},
        final_context[:, -1],
        method=JepaWorldModel.actor_value_from_latent,
    )
    rollout["last_value"] = last_value
    return rollout


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


@partial(jax.jit, static_argnames=("config", "chunk_length", "control"))
def evaluate_world_model(
    state: JepaTrainState,
    key: jax.Array,
    batch: ReplayBatch,
    config: JepaConfig,
    *,
    chunk_length: int,
    control: ControlMode = "none",
) -> dict[str, jax.Array]:
    _, metrics = world_model_loss(
        state.params,
        state.apply_fn,
        key,
        batch,
        config,
        chunk_length=chunk_length,
        control=control,
    )
    metrics["model/action_sensitivity"] = action_sensitivity(
        state.params,
        state.apply_fn,
        batch.observations[:, 0],
        config,
        control=control,
    )
    return metrics


@partial(jax.jit, static_argnames=("config", "horizon", "control"))
def evaluate_open_loop(
    state: JepaTrainState,
    batch: ReplayBatch,
    config: JepaConfig,
    *,
    horizon: int,
    control: ControlMode = "none",
) -> dict[str, jax.Array]:
    flat_obs = batch.observations[:, 0].reshape((-1, config.observation_dim))
    z0 = state.apply_fn({"params": state.params}, flat_obs, method=JepaWorldModel.encode)
    target_z = state.apply_fn(
        {"params": state.params},
        batch.observations[:, : horizon + 1],
        method=JepaWorldModel.encode,
    )
    context = jnp.repeat(z0[:, None, :], repeats=config.context_window, axis=1)
    action_context = jnp.zeros(
        (z0.shape[0], config.context_window),
        dtype=jnp.int32,
    )
    preds = []
    for t in range(horizon):
        actions = batch.actions[:, t]
        if control == "no-action-world-model":
            actions = jnp.zeros_like(actions)
        action_context = jnp.concatenate(
            [action_context[:, 1:], actions[:, None]],
            axis=1,
        )
        next_z, _, _ = state.apply_fn(
            {"params": state.params},
            context,
            action_context,
            method=JepaWorldModel.predict_next_from_history,
        )
        preds.append(next_z)
        context = jnp.concatenate([context[:, 1:], next_z[:, None, :]], axis=1)
    pred = _normalize(jnp.stack(preds, axis=1))
    target = _normalize(jax.lax.stop_gradient(target_z[:, 1 : horizon + 1]))
    validity = jnp.cumprod(1.0 - batch.dones[:, :horizon], axis=1)
    cosine = jnp.sum(pred * target, axis=-1)
    error = 1.0 - cosine
    return {
        "model/open_loop_loss": masked_mean(error, validity),
        "model/open_loop_cosine": masked_mean(cosine, validity),
        "model/open_loop_valid_fraction": jnp.mean(validity),
        "model/open_loop_finite_fraction": _all_finite_fraction(pred, target),
    }


def select_actions(
    state: JepaTrainState,
    observations: jax.Array,
    key: jax.Array,
    config: JepaConfig,
    *,
    deterministic: bool,
) -> jax.Array:
    flat_obs = observations.reshape((-1, config.observation_dim))
    logits, _ = state.apply_fn(
        {"params": state.params},
        flat_obs,
        method=JepaWorldModel.actor_value_from_obs,
    )
    if deterministic:
        actions = jnp.argmax(logits, axis=-1)
    else:
        actions = jax.random.categorical(key, logits, axis=-1)
    return actions.astype(jnp.int32).reshape(observations.shape[0], 1)


def sample_categorical(logits: jax.Array, key: jax.Array):
    actions = jax.random.categorical(key, logits, axis=-1)
    log_probs_all = jax.nn.log_softmax(logits, axis=-1)
    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jnp.take_along_axis(
        log_probs_all,
        actions[:, None],
        axis=-1,
    )[:, 0]
    entropies = -jnp.sum(probs * log_probs_all, axis=-1)
    return actions.astype(jnp.int32), log_probs, entropies


def isotropy_loss(latents: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    z = latents.reshape((-1, latents.shape[-1]))
    z = z - jnp.mean(z, axis=0, keepdims=True)
    std = jnp.sqrt(jnp.var(z, axis=0) + 1e-6)
    cov = (z.T @ z) / jnp.maximum(z.shape[0] - 1, 1)
    cov_diag = jnp.diag(jnp.diag(cov))
    offdiag = cov - cov_diag
    mean_loss = jnp.mean(jnp.square(jnp.mean(latents, axis=(0, 1))))
    std_loss = jnp.mean(jnp.square(std - 1.0))
    offdiag_loss = jnp.mean(jnp.square(offdiag))
    loss = mean_loss + std_loss + offdiag_loss
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
    return loss, metrics


def action_sensitivity(
    params: FrozenDict,
    apply_fn,
    observations: jax.Array,
    config: JepaConfig,
    *,
    control: ControlMode = "none",
) -> jax.Array:
    if control == "no-action-world-model":
        return jnp.asarray(0.0, dtype=jnp.float32)
    z = apply_fn({"params": params}, observations, method=JepaWorldModel.encode)
    context = jnp.repeat(z[:, None, :], repeats=config.context_window, axis=1)
    preds = []
    for action in range(config.action_dim):
        actions = jnp.zeros((z.shape[0], config.context_window), dtype=jnp.int32)
        actions = actions.at[:, -1].set(action)
        pred, _, _ = apply_fn(
            {"params": params},
            context,
            actions,
            method=JepaWorldModel.predict_next_from_history,
        )
        preds.append(pred)
    stacked = jnp.stack(preds, axis=1)
    center = jnp.mean(stacked, axis=1, keepdims=True)
    return jnp.mean(jnp.linalg.norm(stacked - center, axis=-1))


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
            not_done[:, offset : offset + chunk_length]
            for offset in range(horizon)
        ]
        validity.append(jnp.prod(jnp.stack(windows, axis=0), axis=0))
    return jnp.stack(validity, axis=2)


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


def _isotropy_weight(config: JepaConfig, control: ControlMode) -> float:
    if control in ("no-sigreg", "no-isotropy"):
        return 0.0
    if control in ("weak-sigreg", "weak-isotropy"):
        return config.isotropy_weight * 0.1
    return config.isotropy_weight


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
            lambda _: (
                "train"
                if key in trainable_groups
                or (key.startswith("block_") and "encoder" in trainable_groups)
                else "freeze"
            ),
            value,
        )
        for key, value in raw.items()
    }
    return freeze(labels)


def _finite_fraction(x: jax.Array) -> jax.Array:
    return jnp.mean(jnp.isfinite(x).astype(jnp.float32))


def _all_finite_fraction(*values: jax.Array) -> jax.Array:
    fractions = [_finite_fraction(value) for value in values]
    return jnp.min(jnp.stack(fractions))


def config_dict(config: JepaConfig) -> dict[str, Any]:
    return asdict(config)
