from __future__ import annotations

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from world_marl.dreamarl.joint_model import JointWorldModel
from world_marl.dreamarl.local_belief import LocalBelief


ACTION_SPACE = {"action": elements.Space(np.int32, (), 0, 4)}


def _joint(num_agents=3):
    return JointWorldModel(
        ACTION_SPACE,
        embedding_dim=12,
        belief_dim=8,
        num_agents=num_agents,
        units=16,
        layers=1,
        heads=2,
        ffup=2,
        stoch=2,
        classes=4,
        hidden=16,
        name="world",
    )


def _run(function, *args, seed=0):
    params = nj.init(function)({}, *args, seed=seed)
    return params, nj.pure(function)(params, *args, seed=seed + 1)[1]


def test_local_belief_has_no_cross_agent_information_path() -> None:
    module = LocalBelief(
        ACTION_SPACE,
        embedding_dim=12,
        units=8,
        layers=1,
        heads=2,
        context=4,
        ffup=2,
        name="belief",
    )
    carry = module.initial(6)
    embedding = jax.random.normal(jax.random.key(1), (6, 12))
    action = {"action": jnp.zeros((6,), jnp.int32)}
    reset = jnp.zeros((6,), bool)

    def step(state, observation, act, first):
        return module.observe(
            state, observation, act, first, training=False, single=True
        )[1]

    params, expected = _run(step, carry, embedding, action, reset)
    changed = embedding.at[1].add(100)
    actual = nj.pure(step)(params, carry, changed, action, reset, seed=2)[1]
    expected_belief = np.asarray(expected, np.float32)
    actual_belief = np.asarray(actual, np.float32)
    np.testing.assert_allclose(
        actual_belief[[0, 2, 3, 4, 5]],
        expected_belief[[0, 2, 3, 4, 5]],
        atol=2e-5,
    )
    assert not np.allclose(actual_belief[1], expected_belief[1])


def test_joint_prior_is_permutation_equivariant() -> None:
    module = _joint(3)
    carry = module.initial(2)
    action = {"action": jnp.asarray([[0, 1, 2], [3, 2, 1]], jnp.int32)}
    reset = jnp.zeros((2,), bool)

    def step(state, act, first):
        return module.imagine_step(state, act, first, training=False)

    params, expected = _run(step, carry, action, reset, seed=10)
    permutation = np.asarray([2, 0, 1])
    permuted_action = {"action": action["action"][:, permutation]}
    actual = nj.pure(step)(params, carry, permuted_action, reset, seed=11)[1]
    np.testing.assert_allclose(
        np.asarray(actual["global"], np.float32),
        np.asarray(expected["global"], np.float32),
        atol=3e-5,
        rtol=3e-5,
    )
    for key in ("deter", "logit"):
        np.testing.assert_allclose(
            np.asarray(actual[key], np.float32),
            np.asarray(expected[key][:, permutation], np.float32),
            atol=3e-5,
            rtol=3e-5,
        )


def test_one_agent_action_can_change_multiple_predicted_agents() -> None:
    module = _joint(3)
    carry = module.initial(2)
    action = {"action": jnp.zeros((2, 3), jnp.int32)}
    reset = jnp.zeros((2,), bool)

    def step(state, act, first):
        return module.imagine_step(state, act, first, training=False)

    params, expected = _run(step, carry, action, reset, seed=20)
    changed_action = {"action": action["action"].at[0, 0].set(3)}
    actual = nj.pure(step)(params, carry, changed_action, reset, seed=21)[1]
    difference = np.linalg.norm(
        np.asarray(actual["deter"][0] - expected["deter"][0], np.float32), axis=-1
    )
    assert (difference > 1e-6).sum() >= 2
    np.testing.assert_allclose(
        np.asarray(actual["deter"][1], np.float32),
        np.asarray(expected["deter"][1], np.float32),
        atol=3e-5,
        rtol=3e-5,
    )


def test_joint_state_has_one_global_and_complete_agent_axis() -> None:
    module = _joint(5)
    state = module.initial(4)
    assert state["global"].shape == (4, 16)
    assert state["deter"].shape == (4, 5, 16)
    assert state["stoch"].shape == (4, 5, 2, 4)
    assert state["logit"].shape == (4, 5, 2, 4)


def test_single_agent_joint_world_is_well_defined() -> None:
    module = _joint(1)
    carry = module.initial(3)
    action = {"action": jnp.zeros((3, 1), jnp.int32)}
    reset = jnp.zeros((3,), bool)

    def step(state, act, first):
        return module.imagine_step(state, act, first, training=False)

    _, output = _run(step, carry, action, reset, seed=30)
    assert output["global"].shape == (3, 16)
    assert output["deter"].shape == (3, 1, 16)
    assert np.isfinite(np.asarray(output["deter"], np.float32)).all()
