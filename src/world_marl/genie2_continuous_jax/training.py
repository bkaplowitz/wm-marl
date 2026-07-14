"""Staged training for the public-source Genie2/Jasmine diffusion baseline."""

from __future__ import annotations

from typing import Any, NamedTuple

from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from world_marl.genie2_continuous_jax.autoencoder import (
    ContinuousLatentAutoencoder,
    ContinuousVideoTokenizer,
    reconstruction_loss,
)
from world_marl.genie2_continuous_jax.config import (
    Genie2ContinuousConfig,
    StageOptimizerConfig,
)
from world_marl.genie2_continuous_jax.dynamics import (
    ActionConditionedLatentDiffusion,
    autoregressive_sample,
    diffusion_forcing_loss,
)
from world_marl.genie2_continuous_jax.rl_heads import RewardContinueHead
from world_marl.world_model_foundation.replay import (
    JaxSequenceBatch,
    WorldModelSequenceBatch,
    sample_sequence_windows,
    sequence_batch_to_jax,
)


class Genie2TrainState(NamedTuple):
    tokenizer: TrainState
    dynamics: TrainState
    heads: TrainState


def _wsd_schedule(config: StageOptimizerConfig) -> optax.Schedule:
    warmup = optax.linear_schedule(
        0.0,
        config.max_learning_rate,
        max(config.warmup_steps, 1),
    )
    stable = optax.constant_schedule(config.max_learning_rate)
    decay = optax.linear_schedule(
        config.max_learning_rate,
        0.0,
        max(config.wsd_decay_steps, 1),
    )
    return optax.join_schedules(
        [warmup, stable, decay],
        [config.warmup_steps, config.steps - config.wsd_decay_steps],
    )


def _stage_optimizer(config: StageOptimizerConfig) -> optax.GradientTransformation:
    return optax.adamw(
        _wsd_schedule(config),
        b1=config.beta1,
        b2=config.beta2,
        weight_decay=config.weight_decay,
        mu_dtype=jnp.float32,
    )


def action_features(actions: jax.Array, config: Genie2ContinuousConfig) -> jax.Array:
    if config.action_mode == "discrete":
        return jax.nn.one_hot(actions.astype(jnp.int32), config.action_dim).astype(
            jnp.float32
        )
    return actions.astype(jnp.float32).reshape((*actions.shape[:2], config.action_dim))


def genie2_transition_targets(batch: JaxSequenceBatch) -> dict[str, jax.Array]:
    return {
        "actions": batch.actions[:-1],
        "rewards": batch.rewards[1:],
        "continues": batch.continues[1:],
        "valid": ~batch.is_first[1:],
    }


def _tokenizer_module(config: Genie2ContinuousConfig) -> nn.Module:
    if config.is_image_observation:
        return ContinuousVideoTokenizer(config.autoencoder)
    return ContinuousLatentAutoencoder(
        latent_dim=config.autoencoder.latent_patch_dim,
        hidden_dims=config.autoencoder.vector_hidden_dims,
    )


def _encode_observations(
    tokenizer_state: TrainState,
    observations: jax.Array,
    config: Genie2ContinuousConfig,
    key: jax.Array,
    *,
    training: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    time, batch = observations.shape[:2]
    batch_major = jnp.swapaxes(observations.astype(jnp.float32), 0, 1)
    if config.is_image_observation:
        latents, reconstructions, mask = tokenizer_state.apply_fn(
            tokenizer_state.params,
            batch_major,
            key=key,
            training=training,
        )
    else:
        flat = batch_major.reshape((batch * time, *config.observation_shape))
        latent_flat, reconstruction_flat = tokenizer_state.apply_fn(
            tokenizer_state.params,
            flat,
        )
        latents = latent_flat.reshape(
            (batch, time, 1, config.autoencoder.latent_patch_dim)
        )
        reconstructions = reconstruction_flat.reshape(
            (batch, time, *config.observation_shape)
        )
        mask = jnp.zeros((batch, time, 1), dtype=bool)
    return latents, reconstructions, mask


def encode_genie2_observations(
    state: Genie2TrainState,
    observations: jax.Array,
    config: Genie2ContinuousConfig,
    key: jax.Array,
) -> jax.Array:
    latents, _, _ = _encode_observations(
        state.tokenizer,
        observations,
        config,
        key,
        training=False,
    )
    return latents


def decode_genie2_latents(
    state: Genie2TrainState,
    latents: jax.Array,
    config: Genie2ContinuousConfig,
) -> jax.Array:
    batch, time = latents.shape[:2]
    module = _tokenizer_module(config)
    if config.is_image_observation:
        return module.apply(
            state.tokenizer.params,
            latents,
            video_shape=config.observation_shape,
            training=False,
            method=ContinuousVideoTokenizer.decode,
        )
    flat_latents = latents.reshape((batch * time, config.autoencoder.latent_patch_dim))
    dummy = jnp.zeros((batch * time, *config.observation_shape), dtype=jnp.float32)
    _, decoded = state.tokenizer.apply_fn(
        state.tokenizer.params,
        dummy,
        decode_latents=flat_latents,
    )
    return decoded.reshape((batch, time, *config.observation_shape))


def create_genie2_train_state(
    key: jax.Array,
    *,
    observation_shape: tuple[int, ...] | None = None,
    config: Genie2ContinuousConfig,
    learning_rate: float | None = None,
) -> Genie2TrainState:
    if observation_shape is not None and observation_shape != config.observation_shape:
        raise ValueError("observation_shape must match config.observation_shape")
    tokenizer_key, dynamics_key, heads_key, data_key = jax.random.split(key, 4)
    tokenizer = _tokenizer_module(config)
    if config.is_image_observation:
        dummy_observations = jnp.zeros(
            (1, 2, *config.observation_shape), dtype=jnp.float32
        )
        tokenizer_params = tokenizer.init(
            tokenizer_key,
            dummy_observations,
            key=data_key,
            training=True,
        )
        num_patches = (
            config.observation_shape[0]
            // config.autoencoder.patch_size
            * (config.observation_shape[1] // config.autoencoder.patch_size)
        )
    else:
        dummy_observations = jnp.zeros(
            (2, *config.observation_shape), dtype=jnp.float32
        )
        tokenizer_params = tokenizer.init(tokenizer_key, dummy_observations)
        num_patches = 1
    tokenizer_optimizer = config.tokenizer_optimizer
    dynamics_optimizer = config.dynamics_optimizer
    if learning_rate is not None:
        tokenizer_optimizer = StageOptimizerConfig(
            steps=tokenizer_optimizer.steps,
            batch_size=tokenizer_optimizer.batch_size,
            max_learning_rate=learning_rate,
            warmup_steps=0,
            wsd_decay_steps=0,
        )
        dynamics_optimizer = StageOptimizerConfig(
            steps=dynamics_optimizer.steps,
            batch_size=dynamics_optimizer.batch_size,
            max_learning_rate=learning_rate,
            warmup_steps=0,
            wsd_decay_steps=0,
        )
    tokenizer_state = TrainState.create(
        apply_fn=tokenizer.apply,
        params=tokenizer_params,
        tx=_stage_optimizer(tokenizer_optimizer),
    )

    dynamics = ActionConditionedLatentDiffusion(
        latent_patch_dim=config.autoencoder.latent_patch_dim,
        action_dim=config.action_dim,
        config=config.dynamics,
    )
    dummy_latents = jnp.zeros(
        (1, 2, num_patches, config.autoencoder.latent_patch_dim),
        dtype=jnp.float32,
    )
    dummy_actions = jnp.zeros((1, 1, config.action_dim), dtype=jnp.float32)
    dynamics_params = dynamics.init(
        dynamics_key,
        dummy_latents,
        dummy_actions,
        key=data_key,
        training=True,
    )
    dynamics_state = TrainState.create(
        apply_fn=dynamics.apply,
        params=dynamics_params,
        tx=_stage_optimizer(dynamics_optimizer),
    )

    heads = RewardContinueHead(config.reward_continue_hidden_dims)
    dummy_pooled = jnp.zeros(
        (1, config.autoencoder.latent_patch_dim), dtype=jnp.float32
    )
    dummy_action = jnp.zeros((1, config.action_dim), dtype=jnp.float32)
    head_params = heads.init(heads_key, dummy_pooled, dummy_action)
    head_state = TrainState.create(
        apply_fn=heads.apply,
        params=head_params,
        tx=optax.adam(learning_rate or 1e-4),
    )
    return Genie2TrainState(tokenizer_state, dynamics_state, head_state)


def tokenizer_loss_arrays(
    params: Any,
    state: TrainState,
    batch: JaxSequenceBatch,
    config: Genie2ContinuousConfig,
    key: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    temporary = state.replace(params=params)
    _, reconstructions, mask = _encode_observations(
        temporary,
        batch.observations,
        config,
        key,
        training=True,
    )
    targets = jnp.swapaxes(batch.observations.astype(jnp.float32), 0, 1)
    loss = reconstruction_loss(targets, reconstructions)
    return loss, {
        "tokenizer_loss": loss,
        "reconstruction_loss": loss,
        "mask_fraction": jnp.mean(mask.astype(jnp.float32)),
    }


def dynamics_loss_arrays(
    params: Any,
    state: Genie2TrainState,
    batch: JaxSequenceBatch,
    config: Genie2ContinuousConfig,
    key: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    encode_key, diffusion_key = jax.random.split(key)
    latents, _, _ = _encode_observations(
        state.tokenizer,
        batch.observations,
        config,
        encode_key,
        training=False,
    )
    latents = jax.lax.stop_gradient(latents)
    targets = genie2_transition_targets(batch)
    actions = action_features(targets["actions"], config).swapaxes(0, 1)
    outputs = state.dynamics.apply_fn(
        params,
        latents,
        actions,
        key=diffusion_key,
        training=True,
    )
    valid_transitions = targets["valid"].swapaxes(0, 1)
    valid = jnp.concatenate(
        [jnp.ones_like(valid_transitions[:, :1]), valid_transitions],
        axis=1,
    )
    loss = diffusion_forcing_loss(
        outputs,
        ramp_weight=config.dynamics.ramp_weight,
        valid_mask=valid,
    )
    return loss, {
        "dynamics_loss": loss,
        "diffusion_x_loss": loss,
        "mean_signal_level": jnp.mean(outputs["signal_level"]),
        "conditioning_keep_rate": jnp.mean(outputs["condition_keep_mask"]),
        "valid_transition_rate": jnp.mean(valid_transitions),
    }


def reward_continue_loss_arrays(
    params: Any,
    state: Genie2TrainState,
    batch: JaxSequenceBatch,
    config: Genie2ContinuousConfig,
    key: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    latents, _, _ = _encode_observations(
        state.tokenizer,
        batch.observations,
        config,
        key,
        training=False,
    )
    pooled = jax.lax.stop_gradient(jnp.mean(latents[:, :-1], axis=2))
    targets = genie2_transition_targets(batch)
    actions = action_features(targets["actions"], config).swapaxes(0, 1)
    reward_prediction, continue_logits = state.heads.apply_fn(
        params,
        pooled,
        actions,
    )
    reward_targets = targets["rewards"].swapaxes(0, 1).astype(jnp.float32)
    continue_targets = targets["continues"].swapaxes(0, 1).astype(jnp.float32)
    valid = targets["valid"].swapaxes(0, 1).astype(jnp.float32)
    normalizer = jnp.maximum(jnp.sum(valid), 1.0)
    reward_loss = (
        jnp.sum(jnp.square(reward_prediction - reward_targets) * valid) / normalizer
    )
    continue_loss = (
        jnp.sum(
            optax.sigmoid_binary_cross_entropy(continue_logits, continue_targets)
            * valid
        )
        / normalizer
    )
    return reward_loss + continue_loss, {
        "reward_continue_loss": reward_loss + continue_loss,
        "reward_loss": reward_loss,
        "continue_loss": continue_loss,
        "valid_transition_rate": jnp.mean(valid),
    }


def _tokenizer_step(
    state: Genie2TrainState,
    batch: JaxSequenceBatch,
    config: Genie2ContinuousConfig,
    key: jax.Array,
) -> tuple[Genie2TrainState, dict[str, jax.Array]]:
    (loss, metrics), gradients = jax.value_and_grad(
        tokenizer_loss_arrays,
        has_aux=True,
    )(state.tokenizer.params, state.tokenizer, batch, config, key)
    del loss
    return state._replace(
        tokenizer=state.tokenizer.apply_gradients(grads=gradients)
    ), metrics


def _dynamics_step(
    state: Genie2TrainState,
    batch: JaxSequenceBatch,
    config: Genie2ContinuousConfig,
    key: jax.Array,
) -> tuple[Genie2TrainState, dict[str, jax.Array]]:
    (loss, metrics), gradients = jax.value_and_grad(
        dynamics_loss_arrays,
        has_aux=True,
    )(state.dynamics.params, state, batch, config, key)
    del loss
    return state._replace(
        dynamics=state.dynamics.apply_gradients(grads=gradients)
    ), metrics


def _heads_step(
    state: Genie2TrainState,
    batch: JaxSequenceBatch,
    config: Genie2ContinuousConfig,
    key: jax.Array,
) -> tuple[Genie2TrainState, dict[str, jax.Array]]:
    (loss, metrics), gradients = jax.value_and_grad(
        reward_continue_loss_arrays,
        has_aux=True,
    )(state.heads.params, state, batch, config, key)
    del loss
    return state._replace(heads=state.heads.apply_gradients(grads=gradients)), metrics


def _scan_phase(
    step_fn,
    state: Genie2TrainState,
    replay: JaxSequenceBatch,
    key: jax.Array,
    *,
    config: Genie2ContinuousConfig,
    train_steps: int,
    sequence_length: int,
    batch_size: int,
) -> tuple[Genie2TrainState, dict[str, jax.Array], jax.Array]:
    def update(carry: tuple[Genie2TrainState, jax.Array], _: None):
        train_state, update_key = carry
        update_key, sample_key, loss_key = jax.random.split(update_key, 3)
        train_batch = sample_sequence_windows(
            replay,
            sample_key,
            sequence_length=sequence_length,
            batch_size=batch_size,
            require_same_episode=True,
            force_first=False,
        )
        train_state, metrics = step_fn(train_state, train_batch, config, loss_key)
        return (train_state, update_key), metrics

    (state, key), metrics = jax.lax.scan(
        update,
        (state, key),
        None,
        length=train_steps,
    )
    return state, metrics, key


def scan_genie2_training_phases(
    state: Genie2TrainState,
    replay: JaxSequenceBatch,
    key: jax.Array,
    *,
    config: Genie2ContinuousConfig,
    tokenizer_steps: int,
    dynamics_steps: int,
    reward_continue_steps: int,
    sequence_length: int,
    batch_size: int,
) -> tuple[
    Genie2TrainState,
    dict[str, dict[str, jax.Array]],
    dict[str, dict[str, jax.Array]],
]:
    (
        key,
        validation_sample_key,
        tokenizer_validation_key,
        dynamics_validation_key,
        heads_validation_key,
    ) = jax.random.split(key, 5)
    validation_batch = sample_sequence_windows(
        replay,
        validation_sample_key,
        sequence_length=sequence_length,
        batch_size=batch_size,
        require_same_episode=True,
        force_first=False,
    )
    _, initial_tokenizer_metrics = tokenizer_loss_arrays(
        state.tokenizer.params,
        state.tokenizer,
        validation_batch,
        config,
        tokenizer_validation_key,
    )
    state, tokenizer_metrics, key = _scan_phase(
        _tokenizer_step,
        state,
        replay,
        key,
        config=config,
        train_steps=tokenizer_steps,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )
    _, final_tokenizer_metrics = tokenizer_loss_arrays(
        state.tokenizer.params,
        state.tokenizer,
        validation_batch,
        config,
        tokenizer_validation_key,
    )
    _, initial_dynamics_metrics = dynamics_loss_arrays(
        state.dynamics.params,
        state,
        validation_batch,
        config,
        dynamics_validation_key,
    )
    state, dynamics_metrics, key = _scan_phase(
        _dynamics_step,
        state,
        replay,
        key,
        config=config,
        train_steps=dynamics_steps,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )
    _, final_dynamics_metrics = dynamics_loss_arrays(
        state.dynamics.params,
        state,
        validation_batch,
        config,
        dynamics_validation_key,
    )
    _, initial_head_metrics = reward_continue_loss_arrays(
        state.heads.params,
        state,
        validation_batch,
        config,
        heads_validation_key,
    )
    state, head_metrics, _ = _scan_phase(
        _heads_step,
        state,
        replay,
        key,
        config=config,
        train_steps=reward_continue_steps,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )
    _, final_head_metrics = reward_continue_loss_arrays(
        state.heads.params,
        state,
        validation_batch,
        config,
        heads_validation_key,
    )
    return (
        state,
        {
            "tokenizer": tokenizer_metrics,
            "dynamics": dynamics_metrics,
            "reward_continue": head_metrics,
        },
        {
            "tokenizer": {
                "initial_loss": initial_tokenizer_metrics["tokenizer_loss"],
                "final_loss": final_tokenizer_metrics["tokenizer_loss"],
            },
            "dynamics": {
                "initial_loss": initial_dynamics_metrics["dynamics_loss"],
                "final_loss": final_dynamics_metrics["dynamics_loss"],
            },
            "reward_continue": {
                "initial_loss": initial_head_metrics["reward_continue_loss"],
                "final_loss": final_head_metrics["reward_continue_loss"],
            },
        },
    )


def genie2_train_step(
    state: Genie2TrainState,
    batch: JaxSequenceBatch,
    config: Genie2ContinuousConfig,
    key: jax.Array,
) -> tuple[Genie2TrainState, dict[str, jax.Array]]:
    tokenizer_key, dynamics_key, heads_key = jax.random.split(key, 3)
    state, tokenizer_metrics = _tokenizer_step(state, batch, config, tokenizer_key)
    state, dynamics_metrics = _dynamics_step(state, batch, config, dynamics_key)
    state, head_metrics = _heads_step(state, batch, config, heads_key)
    metrics = {**tokenizer_metrics, **dynamics_metrics, **head_metrics}
    metrics["loss"] = (
        metrics["tokenizer_loss"]
        + metrics["dynamics_loss"]
        + metrics["reward_continue_loss"]
    )
    return state, metrics


def scan_genie2_world_model_updates(
    state: Genie2TrainState,
    replay: JaxSequenceBatch,
    key: jax.Array,
    *,
    config: Genie2ContinuousConfig,
    train_steps: int,
    sequence_length: int,
    batch_size: int,
) -> tuple[Genie2TrainState, dict[str, jax.Array]]:
    def update(carry: tuple[Genie2TrainState, jax.Array], _: None):
        train_state, update_key = carry
        update_key, sample_key, loss_key = jax.random.split(update_key, 3)
        train_batch = sample_sequence_windows(
            replay,
            sample_key,
            sequence_length=sequence_length,
            batch_size=batch_size,
        )
        train_state, metrics = genie2_train_step(
            train_state, train_batch, config, loss_key
        )
        return (train_state, update_key), metrics

    (state, _), metrics = jax.lax.scan(
        update,
        (state, key),
        None,
        length=train_steps,
    )
    return state, metrics


def train_genie2_world_model(
    *,
    batch: WorldModelSequenceBatch,
    config: Genie2ContinuousConfig,
    train_steps: int,
    seed: int,
    learning_rate: float | None = None,
    sequence_length: int | None = None,
    batch_size: int | None = None,
) -> tuple[Genie2TrainState, list[dict[str, float]]]:
    replay = sequence_batch_to_jax(batch)
    sequence_length = sequence_length or min(
        config.dynamics.max_context, batch.time_steps
    )
    batch_size = batch_size or min(
        config.dynamics_optimizer.batch_size, batch.batch_size
    )
    state = create_genie2_train_state(
        jax.random.PRNGKey(seed),
        observation_shape=batch.observation_shape,
        config=config,
        learning_rate=learning_rate,
    )
    state, metric_arrays = jax.jit(
        scan_genie2_world_model_updates,
        static_argnames=("config", "train_steps", "sequence_length", "batch_size"),
    )(
        state,
        replay,
        jax.random.PRNGKey(seed + 1),
        config=config,
        train_steps=train_steps,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )
    host_metrics = jax.device_get(metric_arrays)
    metrics = [
        {
            "step": step + 1,
            **{name: float(values[step]) for name, values in host_metrics.items()},
        }
        for step in range(train_steps)
    ]
    return state, metrics


def sample_genie2_latents(
    state: Genie2TrainState,
    context_latents: jax.Array,
    actions: jax.Array,
    config: Genie2ContinuousConfig,
    key: jax.Array,
    *,
    num_future_frames: int,
) -> jax.Array:
    module = ActionConditionedLatentDiffusion(
        latent_patch_dim=config.autoencoder.latent_patch_dim,
        action_dim=config.action_dim,
        config=config.dynamics,
    )
    return autoregressive_sample(
        module.apply,
        state.dynamics.params,
        context_latents,
        actions,
        key=key,
        num_future_frames=num_future_frames,
        config=config.dynamics,
    )


def metrics_to_host(
    metrics: dict[str, dict[str, jax.Array]],
) -> dict[str, list[dict[str, float]]]:
    host = jax.device_get(metrics)
    result: dict[str, list[dict[str, float]]] = {}
    for phase, phase_metrics in host.items():
        if not phase_metrics:
            result[phase] = []
            continue
        steps = len(next(iter(phase_metrics.values())))
        result[phase] = [
            {
                "step": step + 1,
                **{name: float(values[step]) for name, values in phase_metrics.items()},
            }
            for step in range(steps)
        ]
    return result


Genie2WorldModel = ActionConditionedLatentDiffusion
genie2_loss = dynamics_loss_arrays
genie2_loss_arrays = dynamics_loss_arrays
