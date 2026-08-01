from __future__ import annotations

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from world_marl.dreamarl.interaction import (
    AgentInteraction,
    InteractionResidual,
    safe_masked_softmax,
)


def _run_module(module, *args, seed=0, **kwargs):
    def function(*values):
        return module(*values, **kwargs)

    params = nj.init(function)({}, *args, seed=seed)
    _, output = nj.pure(function)(params, *args, seed=seed + 1)
    return params, output


def test_safe_masked_softmax_returns_zero_for_empty_rows() -> None:
    logits = jnp.array([[2.0, -1.0], [3.0, 4.0]])
    valid = jnp.array([[False, False], [True, False]])
    weights = safe_masked_softmax(logits, valid)
    np.testing.assert_array_equal(weights[0], 0)
    np.testing.assert_allclose(weights[1], np.array([1.0, 0.0]), atol=1e-7)


def test_single_agent_interaction_is_exact_zero() -> None:
    mixer = AgentInteraction(5, 7, units=8, heads=2, name="interaction")
    belief = jnp.ones((3, 4, 1, 5), jnp.float32)
    token = jnp.ones((3, 4, 1, 7), jnp.float32)
    valid = jnp.ones((3, 4, 1), bool)
    _, (message, has_other) = _run_module(mixer, belief, token, valid)
    np.testing.assert_array_equal(message, 0)
    np.testing.assert_array_equal(has_other, False)


def test_interaction_is_permutation_equivariant() -> None:
    mixer = AgentInteraction(5, 7, units=8, heads=2, name="interaction")
    belief = jax.random.normal(jax.random.key(1), (2, 3, 5))
    token = jax.random.normal(jax.random.key(2), (2, 3, 7))
    valid = jnp.ones((2, 3), bool)
    def function(belief_value, token_value, valid_value):
        return mixer(belief_value, token_value, valid_value)

    params = nj.init(function)({}, belief, token, valid, seed=3)
    _, (expected, expected_mask) = nj.pure(function)(
        params, belief, token, valid, seed=4
    )
    permutation = jnp.array([2, 0, 1])
    _, (actual, actual_mask) = nj.pure(function)(
        params,
        belief[:, permutation],
        token[:, permutation],
        valid[:, permutation],
        seed=4,
    )
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected[:, permutation], dtype=np.float32),
        atol=2e-5,
    )
    np.testing.assert_array_equal(actual_mask, expected_mask[:, permutation])


def test_shuffled_control_rolls_complete_environment_trajectories() -> None:
    mixer = AgentInteraction(5, 7, units=8, heads=2, name="interaction")
    belief = jax.random.normal(jax.random.key(10), (3, 4, 2, 5))
    token = jax.random.normal(jax.random.key(11), (3, 4, 2, 7))
    valid = jnp.ones((3, 4, 2), bool)
    def aligned_function(belief_value, token_value, valid_value):
        return mixer(
            belief_value, token_value, valid_value, shuffled=False
        )

    def shuffled_function(belief_value, token_value, valid_value):
        return mixer(
            belief_value, token_value, valid_value, shuffled=True
        )

    params = nj.init(aligned_function)({}, belief, token, valid, seed=12)
    _, shuffled = nj.pure(shuffled_function)(
        params, belief, token, valid, seed=13
    )
    _, explicit_roll = nj.pure(aligned_function)(
        params,
        belief,
        jnp.roll(token, 1, axis=0),
        jnp.roll(valid, 1, axis=0),
        seed=13,
    )
    np.testing.assert_allclose(
        np.asarray(shuffled[0], dtype=np.float32),
        np.asarray(explicit_roll[0], dtype=np.float32),
        atol=2e-5,
    )
    np.testing.assert_array_equal(shuffled[1], explicit_roll[1])


def test_residual_is_zero_initialized_and_singleton_gated_after_projection() -> None:
    residual = InteractionResidual(6, hidden=8, name="residual")
    local = jnp.ones((2, 5), jnp.float32)
    message = jnp.ones((2, 4), jnp.float32)
    active = jnp.ones((2, 1), bool)
    def function(local_value, message_value, active_value):
        return residual(local_value, message_value, active_value)

    params = nj.init(function)({}, local, message, active, seed=5)
    _, initialized = nj.pure(function)(params, local, message, active, seed=6)
    np.testing.assert_array_equal(initialized, 0)

    nonzero_params = jax.tree.map(jnp.ones_like, params)
    inactive = jnp.zeros((2, 1), bool)
    _, gated = nj.pure(function)(
        nonzero_params, local, message, inactive, seed=6
    )
    np.testing.assert_array_equal(gated, 0)


def test_inactive_interaction_cannot_change_local_input_gradient() -> None:
    residual = InteractionResidual(5, hidden=8, name="residual")
    local = jnp.ones((2, 5), jnp.float32)
    message = jnp.ones((2, 4), jnp.float32)
    inactive = jnp.zeros((2, 1), bool)
    def function(local_value, message_value, active_value):
        return residual(local_value, message_value, active_value)

    params = nj.init(function)({}, local, message, inactive, seed=7)

    def objective(value):
        _, delta = nj.pure(function)(params, value, message, inactive, seed=8)
        return jnp.square(value + delta).sum()

    expected = jax.grad(lambda value: jnp.square(value).sum())(local)
    actual = jax.grad(objective)(local)
    np.testing.assert_array_equal(actual, expected)


def test_mixer_receives_gradients_only_when_another_agent_exists() -> None:
    mixer = AgentInteraction(5, 7, units=8, heads=2, name="interaction")
    residual = InteractionResidual(5, hidden=8, name="residual")

    def function(belief, token, valid):
        message, active = mixer(belief, token, valid)
        return residual(belief, message, active)

    belief = jax.random.normal(jax.random.key(20), (2, 3, 5))
    token = jax.random.normal(jax.random.key(21), (2, 3, 7))
    valid = jnp.ones((2, 3), bool)
    params = nj.init(function)({}, belief, token, valid, seed=22)
    output_kernel = next(
        key for key in params if key.endswith("residual/output/kernel")
    )
    params = {
        **params,
        output_kernel: jnp.full_like(params[output_kernel], 1e-3),
    }

    def loss(parameters, belief_value, token_value, valid_value):
        _, output = nj.pure(function)(
            parameters,
            belief_value,
            token_value,
            valid_value,
            seed=23,
        )
        return jnp.square(output - 1).mean()

    active_grads = jax.grad(loss)(params, belief, token, valid)
    mixer_grad_norm = sum(
        float(jnp.square(value).sum())
        for key, value in active_grads.items()
        if key.startswith("interaction/")
    )
    assert mixer_grad_norm > 0

    singleton_belief = belief[:, :1]
    singleton_token = token[:, :1]
    singleton_valid = valid[:, :1]
    inactive_grads = jax.grad(loss)(
        params, singleton_belief, singleton_token, singleton_valid
    )
    inactive_grad_norm = sum(
        float(jnp.square(value).sum())
        for key, value in inactive_grads.items()
        if key.startswith("interaction/") or key.startswith("residual/")
    )
    assert inactive_grad_norm == 0
