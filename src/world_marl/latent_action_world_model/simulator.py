"""Scanned source-sampler simulators for Jafar and Jasmine.

This repository-specific layer keeps backend state explicit: Jafar carries VQ
token history and Jasmine carries continuous MAE latent history. Every step
runs the corresponding complete source sampler, decodes HWC pixels, applies
frozen reward/continuation heads, and selects a replay context on termination.
World-model parameters are stop-gradient values throughout simulation.
"""

from typing import NamedTuple

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp

from world_marl.jafar.model import JafarWorldModel
from world_marl.jasmine.model import JasmineWorldModel
from world_marl.latent_action_world_model.heads import decode_reward


class JafarSimulatorState(NamedTuple):
    pixels: jax.Array
    token_history: jax.Array
    action_codes: jax.Array


class JafarReplayPool(NamedTuple):
    pixels: jax.Array
    token_histories: jax.Array
    action_codes: jax.Array


class JasmineSimulatorState(NamedTuple):
    pixels: jax.Array
    latent_history: jax.Array
    action_codes: jax.Array


class JasmineReplayPool(NamedTuple):
    pixels: jax.Array
    latent_histories: jax.Array
    action_codes: jax.Array


def _frozen_params(params):
    return jax.tree.map(jax.lax.stop_gradient, params)


def _jafar_tokens(
    model: JafarWorldModel,
    params,
    pixels: jax.Array,
) -> jax.Array:
    outputs = model.apply(
        {"params": _frozen_params(params)},
        pixels,
        method=lambda module, values: module.tokenizer.vq_encode(
            values, training=False
        ),
    )
    return jax.lax.stop_gradient(outputs["indices"])


def _jafar_features(
    model: JafarWorldModel,
    params,
    tokens: jax.Array,
    action_codes: jax.Array,
) -> jax.Array:
    embeddings = model.apply(
        {"params": _frozen_params(params)},
        tokens[:, -1],
        method=lambda module, indices: module.tokenizer.vq.get_codes(indices),
    )
    latent_features = embeddings.astype(jnp.float32).mean(axis=1)
    return jnp.concatenate([latent_features, jax.nn.one_hot(action_codes, 6)], axis=-1)


def _jasmine_latents(
    model: JasmineWorldModel,
    params,
    pixels: jax.Array,
) -> jax.Array:
    outputs = model.apply(
        {"params": _frozen_params(params)},
        pixels,
        method=lambda module, values: module.tokenizer.mask_and_encode(
            values,
            jax.random.PRNGKey(0),
            training=False,
        ),
    )
    return jax.lax.stop_gradient(outputs["z"])


def _jasmine_features(
    latents: jax.Array,
    action_codes: jax.Array,
) -> jax.Array:
    latent_features = latents[:, -1].astype(jnp.float32).mean(axis=1)
    return jnp.concatenate([latent_features, jax.nn.one_hot(action_codes, 6)], axis=-1)


def infer_jafar_codes(
    model: JafarWorldModel,
    params,
    videos: jax.Array,
) -> jax.Array:
    outputs = model.apply(
        {"params": _frozen_params(params)},
        videos,
        method=lambda module, values: module.lam.vq_encode(values, training=False),
    )
    return jax.lax.stop_gradient(outputs["indices"]).reshape(
        videos.shape[0], videos.shape[1] - 1
    )


def infer_jasmine_codes(
    model: JasmineWorldModel,
    params,
    videos: jax.Array,
) -> jax.Array:
    outputs = model.apply(
        {"params": _frozen_params(params)},
        videos,
        method=lambda module, values: module.lam.vq_encode(values, training=False),
    )
    return jax.lax.stop_gradient(outputs["indices"]).reshape(
        videos.shape[0], videos.shape[1] - 1
    )


def jafar_transition_features(
    model: JafarWorldModel,
    params,
    videos: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    codes = infer_jafar_codes(model, params, videos)[:, 0]
    tokens = _jafar_tokens(model, params, videos[:, 1:])
    return _jafar_features(model, params, tokens, codes), codes


def jasmine_transition_features(
    model: JasmineWorldModel,
    params,
    videos: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    codes = infer_jasmine_codes(model, params, videos)[:, 0]
    latents = _jasmine_latents(model, params, videos[:, 1:])
    return _jasmine_features(latents, codes), codes


def initialize_jafar_state(
    model: JafarWorldModel,
    params,
    pixels: jax.Array,
) -> JafarSimulatorState:
    if pixels.ndim != 5 or pixels.shape[-1] != 3:
        raise ValueError("Jafar simulator contexts must be batch-major HWC RGB")
    return JafarSimulatorState(
        pixels=pixels,
        token_history=_jafar_tokens(model, params, pixels),
        action_codes=jnp.zeros((pixels.shape[0], pixels.shape[1] - 1), dtype=jnp.int32),
    )


def create_jafar_replay_pool(
    model: JafarWorldModel,
    params,
    pixels: jax.Array,
) -> JafarReplayPool:
    state = initialize_jafar_state(model, params, pixels)
    return JafarReplayPool(
        pixels=state.pixels,
        token_histories=state.token_history,
        action_codes=state.action_codes,
    )


def initialize_jasmine_state(
    model: JasmineWorldModel,
    params,
    pixels: jax.Array,
) -> JasmineSimulatorState:
    if pixels.ndim != 5 or pixels.shape[-1] != 3:
        raise ValueError("Jasmine simulator contexts must be batch-major HWC RGB")
    return JasmineSimulatorState(
        pixels=pixels,
        latent_history=_jasmine_latents(model, params, pixels),
        action_codes=jnp.zeros((pixels.shape[0], pixels.shape[1] - 1), dtype=jnp.int32),
    )


def create_jasmine_replay_pool(
    model: JasmineWorldModel,
    params,
    pixels: jax.Array,
) -> JasmineReplayPool:
    state = initialize_jasmine_state(model, params, pixels)
    return JasmineReplayPool(
        pixels=state.pixels,
        latent_histories=state.latent_history,
        action_codes=state.action_codes,
    )


def _reset_value(
    done: jax.Array,
    generated: jax.Array,
    replay: jax.Array,
) -> jax.Array:
    mask = done.reshape((done.shape[0],) + (1,) * (generated.ndim - 1))
    return jnp.where(mask, replay, generated)


def jafar_simulator_step(
    model: JafarWorldModel,
    params,
    head_state: TrainState,
    replay_pool: JafarReplayPool,
    state: JafarSimulatorState,
    latent_codes: jax.Array,
    rng: jax.Array,
    *,
    sampler_steps: int = 25,
    sample_argmax: bool = False,
) -> tuple[JafarSimulatorState, jax.Array, jax.Array, jax.Array]:
    sampler_rng, termination_rng, reset_rng = jax.random.split(rng, 3)
    all_action_codes = jnp.concatenate(
        [state.action_codes, latent_codes[:, None]], axis=1
    )
    context_length = state.pixels.shape[1]
    generated = model.apply(
        {"params": _frozen_params(params)},
        {
            "videos": state.pixels,
            "latent_actions": all_action_codes,
            "rng": sampler_rng,
        },
        seq_len=context_length + 1,
        steps=sampler_steps,
        sample_argmax=sample_argmax,
        method=model.sample,
    )
    next_pixels = generated[:, -context_length:]
    next_tokens = _jafar_tokens(model, params, next_pixels)
    next_actions = all_action_codes[:, 1:]
    features = _jafar_features(model, params, next_tokens, latent_codes)
    head_outputs = head_state.apply_fn({"params": head_state.params}, features)
    reward = decode_reward(head_outputs.reward_logits)
    continue_probability = head_outputs.continue_probability
    done = ~jax.random.bernoulli(termination_rng, continue_probability)

    reset_indices = jax.random.randint(
        reset_rng,
        (state.pixels.shape[0],),
        minval=0,
        maxval=replay_pool.pixels.shape[0],
    )
    next_state = JafarSimulatorState(
        pixels=_reset_value(done, next_pixels, replay_pool.pixels[reset_indices]),
        token_history=_reset_value(
            done,
            next_tokens,
            replay_pool.token_histories[reset_indices],
        ),
        action_codes=_reset_value(
            done,
            next_actions,
            replay_pool.action_codes[reset_indices],
        ),
    )
    return next_state, reward, done, continue_probability


def jasmine_simulator_step(
    model: JasmineWorldModel,
    params,
    head_state: TrainState,
    replay_pool: JasmineReplayPool,
    state: JasmineSimulatorState,
    latent_codes: jax.Array,
    rng: jax.Array,
    *,
    diffusion_steps: int = 64,
    context_corruption: float = 0.1,
) -> tuple[JasmineSimulatorState, jax.Array, jax.Array, jax.Array]:
    sampler_rng, termination_rng, reset_rng = jax.random.split(rng, 3)
    all_action_codes = jnp.concatenate(
        [state.action_codes, latent_codes[:, None]], axis=1
    )
    context_length = state.pixels.shape[1]
    generated = model.apply(
        {"params": _frozen_params(params)},
        {
            "videos": state.pixels,
            "latent_actions": all_action_codes,
            "rng": sampler_rng,
        },
        seq_len=context_length + 1,
        diffusion_steps=diffusion_steps,
        context_corruption=context_corruption,
        method=model.sample,
    )
    next_pixels = generated[:, -context_length:]
    next_latents = _jasmine_latents(model, params, next_pixels)
    next_actions = all_action_codes[:, 1:]
    features = _jasmine_features(next_latents, latent_codes)
    head_outputs = head_state.apply_fn({"params": head_state.params}, features)
    reward = decode_reward(head_outputs.reward_logits)
    continue_probability = head_outputs.continue_probability
    done = ~jax.random.bernoulli(termination_rng, continue_probability)

    reset_indices = jax.random.randint(
        reset_rng,
        (state.pixels.shape[0],),
        minval=0,
        maxval=replay_pool.pixels.shape[0],
    )
    next_state = JasmineSimulatorState(
        pixels=_reset_value(done, next_pixels, replay_pool.pixels[reset_indices]),
        latent_history=_reset_value(
            done,
            next_latents,
            replay_pool.latent_histories[reset_indices],
        ),
        action_codes=_reset_value(
            done,
            next_actions,
            replay_pool.action_codes[reset_indices],
        ),
    )
    return next_state, reward, done, continue_probability
