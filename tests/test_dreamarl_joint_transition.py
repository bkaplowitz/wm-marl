from __future__ import annotations

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import elements

from world_marl.dreamarl.joint_transition import JointInteractionResidual
from world_marl.dreamarl.transformer_rssm import TransformerRSSM


def _module(num_agents=3):
    return JointInteractionResidual(
        6,
        3,
        5,
        (2, 4),
        num_agents,
        units=8,
        heads=2,
        ffup=2,
        seed=11,
        name="interaction",
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


def _activate_output(params):
    params = dict(params)
    kernel = next(key for key in params if key.endswith("output_projection/kernel"))
    values = jnp.arange(params[kernel].size, dtype=jnp.float32)
    params[kernel] = 0.01 * jnp.sin(values).reshape(params[kernel].shape)
    return params


def test_zero_initialized_residual_exactly_contains_baseline():
    function, _, (deter, memory) = _initialize(_module(), _inputs())
    del function
    np.testing.assert_array_equal(deter, 0)
    np.testing.assert_array_equal(memory, 0)


def test_single_agent_reduction_is_finite_and_exactly_zero():
    _, params, outputs = _initialize(_module(1), _inputs(1))
    params = _activate_output(params)

    def function(*values):
        return _module(1)(*values)

    deter, memory = nj.pure(function)(params, *_inputs(1), seed=2)[1]
    assert np.isfinite(deter).all()
    assert np.isfinite(memory).all()
    np.testing.assert_array_equal(deter, 0)
    np.testing.assert_array_equal(memory, 0)
    np.testing.assert_array_equal(outputs[0], 0)


def test_single_agent_reduction_stays_zero_after_arbitrary_output_update():
    module = _module(1)
    inputs = _inputs(1)
    function, params, _ = _initialize(module, inputs)
    params = _activate_output(params)
    kernel = next(key for key in params if key.endswith("output_projection/kernel"))
    params[kernel] = jnp.ones_like(params[kernel])
    deter, memory = nj.pure(function)(params, *inputs, seed=8)[1]
    np.testing.assert_array_equal(deter, 0)
    np.testing.assert_array_equal(memory, 0)


def test_interaction_is_permutation_equivariant():
    module = _module()
    inputs = _inputs()
    function, params, _ = _initialize(module, inputs)
    params = _activate_output(params)
    expected = nj.pure(function)(params, *inputs, seed=3)[1]
    permutation = np.array([2, 0, 1])

    def permute(value):
        grouped = value.reshape((2, 3, *value.shape[1:]))
        return grouped[:, permutation].reshape(value.shape)

    changed = tuple(permute(value) for value in inputs)
    actual = nj.pure(function)(params, *changed, seed=4)[1]
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
                np.asarray(actual_value, np.float32),
            np.asarray(permute(expected_value), np.float32),
            rtol=2e-5,
            atol=2e-5,
        )


def test_peer_action_changes_focal_residual():
    module = _module()
    inputs = _inputs()
    function, params, _ = _initialize(module, inputs)
    params = _activate_output(params)
    expected = nj.pure(function)(params, *inputs, seed=5)[1][0]
    changed_action = inputs[2].at[1].add(10.0)
    changed = (*inputs[:2], changed_action, inputs[3])
    actual = nj.pure(function)(params, *changed, seed=6)[1][0]
    assert not np.allclose(
        np.asarray(actual[0], np.float32),
        np.asarray(expected[0], np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(actual[3:], np.float32),
        np.asarray(expected[3:], np.float32),
        rtol=2e-5,
        atol=2e-5,
    )


def test_zero_output_projection_receives_first_update_gradient():
    module = _module()
    inputs = _inputs()
    function, params, _ = _initialize(module, inputs)

    def objective(parameters):
        deter, memory = nj.pure(function)(parameters, *inputs, seed=7)[1]
        return jnp.square(deter - 1).mean() + jnp.square(memory - 1).mean()

    gradients = jax.grad(objective)(params)
    output_norm = sum(
        float(jnp.square(value).sum())
        for key, value in gradients.items()
        if "output_projection" in key
    )
    attention_norm = sum(
        float(jnp.square(value).sum())
        for key, value in gradients.items()
        if "attention_" in key
    )
    assert output_norm > 0
    assert attention_norm == 0


def _dynamics(joint_enabled):
    action_space = {"action": elements.Space(np.int32, (), 0, 4)}
    return TransformerRSSM(
        action_space,
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
        num_agents=1,
        joint_enabled=joint_enabled,
        joint_units=8,
        joint_heads=2,
        name="dyn",
    )


def test_full_dynamics_has_exact_single_agent_output_parity():
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
        output = nj.pure(observe)(
            params, carry, tokens, action, reset, seed=21
        )[1]
        return params, output

    local_params, local = run(False)
    joint_params, joint = run(True)
    for key in local_params:
        np.testing.assert_array_equal(local_params[key], joint_params[key])
    for key in ("deter", "stoch", "memory"):
        np.testing.assert_array_equal(local[2][key], joint[2][key])
    np.testing.assert_array_equal(joint[2]["interaction_deter"], 0)
    np.testing.assert_array_equal(joint[2]["interaction_memory"], 0)


def test_full_dynamics_imagination_keeps_agents_synchronous():
    dynamics = TransformerRSSM(
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
        num_agents=2,
        joint_enabled=True,
        joint_units=8,
        joint_heads=2,
        name="dyn",
    )
    carry = dynamics.initial(4)

    def policy(feature):
        return {"action": jnp.zeros(feature["deter"].shape[:-1], jnp.int32)}

    def imagine(current):
        return dynamics.imagine(current, policy, 3, training=True)

    params = nj.init(imagine)({}, carry, seed=30)
    _, feature, action = nj.pure(imagine)(params, carry, seed=31)[1]
    assert feature["deter"].shape == (4, 3, 8)
    assert feature["interaction_deter"].shape == (4, 3, 8)
    assert action["action"].shape == (4, 3)
