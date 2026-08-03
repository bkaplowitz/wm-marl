from __future__ import annotations

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from world_marl.dreamarl.joint_context import JointTransitionContext
from world_marl.dreamarl.transformer_rssm import TransformerRSSM


def _module(num_agents=3):
    return JointTransitionContext(
        6,
        3,
        5,
        (2, 4),
        num_agents,
        units=8,
        heads=2,
        ffup=2,
        seed=11,
        name="context",
    )


def _inputs(num_agents=3):
    state = jax.random.normal(jax.random.key(1), (2 * num_agents, 6))
    memory = jax.random.normal(jax.random.key(2), (2 * num_agents, 2, 4))
    action = jax.random.normal(jax.random.key(3), (2 * num_agents, 3))
    reset = jnp.zeros((2 * num_agents,), bool)
    return state, memory, action, reset


def _initialize(module, inputs):
    def function(*values):
        return module(*values)

    params = nj.init(function)({}, *inputs, seed=0)
    outputs = nj.pure(function)(params, *inputs, seed=1)[1]
    return function, params, outputs


def _activate_gate(params):
    params = dict(params)
    key = next(key for key in params if key.endswith("/gate"))
    params[key] = jnp.asarray(0.5, jnp.float32)
    return params


def test_zero_initialized_context_exactly_contains_baseline():
    _, _, (pair, belief) = _initialize(_module(), _inputs())
    np.testing.assert_array_equal(pair, 0)
    np.testing.assert_array_equal(belief, 0)


def test_single_agent_context_remains_zero_after_gate_activation():
    function, params, _ = _initialize(_module(1), _inputs(1))
    pair, belief = nj.pure(function)(_activate_gate(params), *_inputs(1), seed=2)[1]
    assert np.isfinite(pair).all()
    assert np.isfinite(belief).all()
    np.testing.assert_array_equal(pair, 0)
    np.testing.assert_array_equal(belief, 0)


def test_context_is_permutation_equivariant():
    module = _module()
    inputs = _inputs()
    function, params, _ = _initialize(module, inputs)
    params = _activate_gate(params)
    expected = nj.pure(function)(params, *inputs, seed=3)[1]
    permutation = np.array([2, 0, 1])

    def permute(value):
        grouped = value.reshape((2, 3, *value.shape[1:]))
        return grouped[:, permutation].reshape(value.shape)

    actual = nj.pure(function)(params, *(permute(value) for value in inputs), seed=4)[1]
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_value, np.float32),
            np.asarray(permute(expected_value), np.float32),
            rtol=2e-5,
            atol=2e-5,
        )


def test_peer_action_changes_only_its_environment_context():
    module = _module()
    inputs = _inputs()
    function, params, _ = _initialize(module, inputs)
    params = _activate_gate(params)
    expected = nj.pure(function)(params, *inputs, seed=5)[1]
    changed_action = inputs[2].at[1].add(10.0)
    actual = nj.pure(function)(
        params, *(*inputs[:2], changed_action, inputs[3]), seed=6
    )[1]
    assert not np.allclose(
        np.asarray(actual[0][0], np.float32),
        np.asarray(expected[0][0], np.float32),
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_value[3:], np.float32),
            np.asarray(expected_value[3:], np.float32),
            rtol=2e-5,
            atol=2e-5,
        )


def test_zero_gate_receives_the_first_context_gradient():
    module = _module()
    inputs = _inputs()
    function, params, _ = _initialize(module, inputs)

    def objective(parameters):
        pair, belief = nj.pure(function)(parameters, *inputs, seed=7)[1]
        return jnp.square(pair - 1).mean() + jnp.square(belief - 1).mean()

    gradients = jax.grad(objective)(params)
    gate_norm = sum(
        float(jnp.square(value).sum())
        for key, value in gradients.items()
        if key.endswith("/gate")
    )
    body_norm = sum(
        float(jnp.square(value).sum())
        for key, value in gradients.items()
        if not key.endswith("/gate")
    )
    assert gate_norm > 0
    assert body_norm == 0


def _dynamics(context_enabled, num_agents=1):
    return TransformerRSSM(
        {"action": elements.Space(np.int32, (), 0, 4)},
        16,
        deter=8,
        hidden=8,
        stoch=2,
        classes=4,
        model=8,
        layers=1,
        heads=2,
        context=4,
        ffup=2,
        memory_tokens=2,
        memory_units=4,
        memory_heads=2,
        memory_ffup=2,
        num_agents=num_agents,
        joint_context_enabled=context_enabled,
        joint_context_units=8,
        joint_context_heads=2,
        name="dyn",
    )


def test_full_dynamics_has_exact_zero_gate_output_parity():
    tokens = jnp.ones((2, 3, 16), jnp.float32)
    action = {"action": jnp.zeros((2, 3), jnp.int32)}
    reset = jnp.zeros((2, 3), bool).at[:, 0].set(True)

    def run(enabled):
        dynamics = _dynamics(enabled)
        carry = dynamics.initial(2)

        def observe(current, observations, actions, first):
            return dynamics.observe(
                current, observations, actions, first, training=True
            )

        params = nj.init(observe)({}, carry, tokens, action, reset, seed=20)
        output = nj.pure(observe)(params, carry, tokens, action, reset, seed=21)[1]
        return params, output

    local_params, local = run(False)
    context_params, context = run(True)
    for key in set(local_params) & set(context_params):
        np.testing.assert_array_equal(local_params[key], context_params[key])
    for key in ("deter", "stoch", "memory"):
        np.testing.assert_array_equal(local[2][key], context[2][key])
    np.testing.assert_array_equal(context[2]["joint_context_pair"], 0)
    np.testing.assert_array_equal(context[2]["joint_context_belief"], 0)


def test_context_dynamics_imagination_keeps_agents_synchronous():
    dynamics = _dynamics(True, num_agents=2)
    carry = dynamics.initial(4)

    def policy(feature):
        return {"action": jnp.zeros(feature["deter"].shape[:-1], jnp.int32)}

    def imagine(current):
        return dynamics.imagine(current, policy, 3, training=True)

    params = nj.init(imagine)({}, carry, seed=30)
    _, feature, action = nj.pure(imagine)(params, carry, seed=31)[1]
    assert feature["deter"].shape == (4, 3, 8)
    assert feature["joint_context_pair"].shape == (4, 3, dynamics.pair_dim)
    assert feature["joint_context_belief"].shape == (
        4,
        3,
        dynamics.local_feature_dim,
    )
    assert action["action"].shape == (4, 3)
