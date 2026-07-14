from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from world_marl.dreamer_v3_baseline.config import OptimizerConfig


class ScaleByRmsState(NamedTuple):
    count: jax.Array
    nu: optax.Updates


class ScaleByMomentumState(NamedTuple):
    count: jax.Array
    mu: optax.Updates


def scale_by_per_tensor_agc(
    clip: float,
    parameter_norm_floor: float = 1e-3,
) -> optax.GradientTransformation:
    """DreamerV3 per-tensor adaptive gradient clipping.

    Ported from the official DreamerV3 JAX implementation under its MIT
    license: https://github.com/danijar/dreamerv3/blob/main/embodied/jax/opt.py
    """

    def init_fn(params: optax.Params) -> optax.EmptyState:
        del params
        return optax.EmptyState()

    def update_fn(
        updates: optax.Updates,
        state: optax.EmptyState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, optax.EmptyState]:
        if params is None:
            raise ValueError("DreamerV3 AGC requires current parameters")

        def clip_tensor(param: jax.Array, update: jax.Array) -> jax.Array:
            update_norm = jnp.linalg.norm(update.reshape(-1), ord=2)
            parameter_norm = jnp.linalg.norm(param.reshape(-1), ord=2)
            upper = clip * jnp.maximum(parameter_norm_floor, parameter_norm)
            return update / jnp.maximum(1.0, update_norm / upper)

        return jax.tree.map(clip_tensor, params, updates), state

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_rms_before_momentum(
    beta: float,
    epsilon: float,
) -> optax.GradientTransformation:
    """Bias-corrected RMS normalization used by DreamerV3 LaProp."""

    def init_fn(params: optax.Params) -> ScaleByRmsState:
        nu = jax.tree.map(lambda value: jnp.zeros_like(value, jnp.float32), params)
        return ScaleByRmsState(jnp.zeros((), jnp.int32), nu)

    def update_fn(
        updates: optax.Updates,
        state: ScaleByRmsState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, ScaleByRmsState]:
        del params
        count = optax.safe_int32_increment(state.count)
        nu = optax.update_moment(updates, state.nu, beta, 2)
        nu_hat = optax.bias_correction(nu, beta, count)
        normalized = jax.tree.map(
            lambda update, second_moment: update / (jnp.sqrt(second_moment) + epsilon),
            updates,
            nu_hat,
        )
        return normalized, ScaleByRmsState(count, nu)

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_bias_corrected_momentum(
    beta: float,
) -> optax.GradientTransformation:
    """Momentum stage applied after RMS normalization in LaProp."""

    def init_fn(params: optax.Params) -> ScaleByMomentumState:
        mu = jax.tree.map(lambda value: jnp.zeros_like(value, jnp.float32), params)
        return ScaleByMomentumState(jnp.zeros((), jnp.int32), mu)

    def update_fn(
        updates: optax.Updates,
        state: ScaleByMomentumState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, ScaleByMomentumState]:
        del params
        count = optax.safe_int32_increment(state.count)
        mu = optax.update_moment(updates, state.mu, beta, 1)
        mu_hat = optax.bias_correction(mu, beta, count)
        return mu_hat, ScaleByMomentumState(count, mu)

    return optax.GradientTransformation(init_fn, update_fn)


def dreamer_laprop(config: OptimizerConfig) -> optax.GradientTransformation:
    return optax.chain(
        scale_by_per_tensor_agc(config.agc),
        scale_by_rms_before_momentum(config.beta2, config.epsilon),
        scale_by_bias_corrected_momentum(config.beta1),
        optax.scale(-config.learning_rate),
    )
