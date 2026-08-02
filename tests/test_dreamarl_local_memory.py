from __future__ import annotations

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from world_marl.dreamarl.local_memory import LocalMemorySidecar


def _module():
    return LocalMemorySidecar(
        32,
        12,
        3,
        tokens=4,
        units=8,
        heads=2,
        ffup=2,
        name="memory",
    )


def _initialize(function, *args):
    params = nj.init(function)({}, *args, seed=0)
    return params, nj.pure(function)(params, *args, seed=1)[1]


def test_memory_is_strictly_local_and_shared_across_agents():
    module = _module()
    previous = jnp.zeros((2, 3, 4, 8), jnp.float32)
    observation = jax.random.normal(jax.random.key(1), (2, 3, 32))
    action = jnp.zeros((2, 3, 3), jnp.float32)
    reset = jnp.zeros((2, 3), bool)

    def function(memory, obs, act, first):
        return module.observe(memory, obs, act, first)[0]

    params, expected = _initialize(function, previous, observation, action, reset)
    changed = observation.at[:, 1].add(100)
    actual = nj.pure(function)(params, previous, changed, action, reset, seed=2)[1]
    np.testing.assert_allclose(
        np.asarray(actual[:, 0], np.float32),
        np.asarray(expected[:, 0], np.float32),
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual[:, 2], np.float32),
        np.asarray(expected[:, 2], np.float32),
        atol=2e-5,
    )
    assert not np.allclose(
        np.asarray(actual[:, 1], np.float32),
        np.asarray(expected[:, 1], np.float32),
    )


def test_reset_erases_previous_memory_from_posterior_and_prior():
    module = _module()
    first = jax.random.normal(jax.random.key(2), (2, 4, 8))
    second = first.at[1].add(100)
    observation = jnp.ones((2, 32), jnp.float32)
    belief = jnp.ones((2, 12), jnp.float32)
    action = jnp.ones((2, 3), jnp.float32)
    reset = jnp.ones((2,), bool)

    def function(memory):
        posterior, _ = module.observe(memory, observation, action, reset)
        prior = module.imagine(memory, belief, action, reset)
        return posterior, prior

    params, expected = _initialize(function, first)
    actual = nj.pure(function)(params, second, seed=3)[1]
    np.testing.assert_allclose(
        np.asarray(actual[0], np.float32),
        np.asarray(expected[0], np.float32),
        atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual[1], np.float32),
        np.asarray(expected[1], np.float32),
        atol=2e-5,
    )


def test_control_residual_is_exactly_zero_at_initialization():
    module = _module()
    memory = jax.random.normal(jax.random.key(4), (3, 4, 8))

    def function(value):
        return module.control_residual(value, 12)

    params, residual = _initialize(function, memory)
    np.testing.assert_array_equal(residual, 0)
    gate_key = next(key for key in params if key.endswith("control_gate"))
    nonzero = {**params, gate_key: jnp.asarray(0.5, jnp.float32)}
    residual = nj.pure(function)(nonzero, memory, seed=5)[1]
    assert not np.allclose(residual, 0)


def test_unified_control_state_is_active_with_matched_parameters():
    module = _module()
    memory = jax.random.normal(jax.random.key(14), (3, 4, 8))

    def residual(value):
        return module.control_residual(value, 12)

    def unified(value):
        return module.control_state(value, 12)

    residual_params = nj.init(residual)({}, memory, seed=15)
    unified_params = nj.init(unified)({}, memory, seed=15)
    assert residual_params.keys() == unified_params.keys()
    assert sum(value.size for value in residual_params.values()) == sum(
        value.size for value in unified_params.values()
    )
    state = nj.pure(unified)(unified_params, memory, seed=16)[1]
    assert state.shape == (3, 12)
    assert not np.allclose(state, 0)


def test_unified_memory_transition_does_not_use_legacy_belief():
    module = _module()
    memory = jax.random.normal(jax.random.key(17), (2, 4, 8))
    first_belief = jnp.ones((2, 12), jnp.float32)
    second_belief = first_belief * 100
    action = jnp.ones((2, 3), jnp.float32)
    reset = jnp.zeros((2,), bool)

    def function(previous, belief):
        return module.imagine(
            previous, belief, action, reset, use_belief=False
        )

    params, expected = _initialize(function, memory, first_belief)
    actual = nj.pure(function)(params, memory, second_belief, seed=18)[1]
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        atol=2e-5,
    )


def test_memory_prior_receives_gradients_before_control_gate_opens():
    module = _module()
    memory = jnp.zeros((2, 4, 8), jnp.float32)
    belief = jnp.ones((2, 12), jnp.float32)
    action = jnp.ones((2, 3), jnp.float32)
    reset = jnp.zeros((2,), bool)

    def function(previous, state, act, first):
        return module.imagine(previous, state, act, first)

    params = nj.init(function)({}, memory, belief, action, reset, seed=6)

    def objective(parameters):
        predicted = nj.pure(function)(
            parameters, memory, belief, action, reset, seed=7
        )[1]
        return jnp.square(predicted - 1).mean()

    gradients = jax.grad(objective)(params)
    prior_norm = sum(
        float(jnp.square(value).sum())
        for key, value in gradients.items()
        if "prior_" in key
    )
    assert prior_norm > 0


def test_sidecar_initialization_does_not_consume_learner_rng_stream():
    module = _module()
    memory = jnp.zeros((2, 4, 8), jnp.float32)
    belief = jnp.ones((2, 12), jnp.float32)
    action = jnp.ones((2, 3), jnp.float32)
    reset = jnp.zeros((2,), bool)

    def function(previous, state, act, first):
        return module.imagine(previous, state, act, first)

    first = nj.init(function)({}, memory, belief, action, reset, seed=0)
    second = nj.init(function)({}, memory, belief, action, reset, seed=999)
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
