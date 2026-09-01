"""Focused semantic gates for causally coupled TBv2 and multi-step JEPA."""

from __future__ import annotations

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from majepa.marl.axes import TeamAxis
from majepa.marl.core import MARLCore
from majepa.models.multistep_jepa import (
    ActionConditionedMultiStepJEPA,
    TeammateActionPlanGRU,
)


def _plan_model(max_horizon=4, peers=2):
    return TeammateActionPlanGRU(
        4,
        0,
        peers,
        max_horizon,
        units=6,
        name="plan",
    )


def _set_nonzero_plan_delta(params):
    params = dict(params)
    params["plan/delta_logits/kernel"] = jnp.ones_like(
        params["plan/delta_logits/kernel"]
    )
    return params


def _set_nonzero_belief_path(params):
    changed = {}
    for key, value in params.items():
        if "belief_" not in key:
            changed[key] = value
            continue
        sequence = jnp.arange(value.size, dtype=jnp.float32).reshape(value.shape)
        changed[key] = ((sequence + 1.0) / max(value.size, 1)).astype(value.dtype)
    return changed


def _set_random_belief_path(params, seed=0):
    changed = dict(params)
    for index, key in enumerate(sorted(params)):
        if "belief_" in key:
            changed[key] = (
                0.2
                * jax.random.normal(
                    jax.random.fold_in(jax.random.key(seed), index),
                    params[key].shape,
                    dtype=jnp.float32,
                )
            ).astype(params[key].dtype)
    return changed


def test_qplan_zero_delta_prior_and_causal_own_prefix_alignment() -> None:
    model = _plan_model()
    root = jax.random.normal(jax.random.key(1), (1, 2, 3, 5))
    actions = jnp.zeros((1, 2, 3, 4), jnp.int32)
    q0 = jax.random.normal(jax.random.key(2), (1, 2, 3, 2, 4))
    q0_context = q0 - q0.mean(axis=-1, keepdims=True)

    def predict(window):
        return model(root, window, q0, q0_context)

    params = nj.init(lambda: predict(actions))({}, seed=3)
    initial = nj.pure(lambda: predict(actions))(params, seed=4)[1]
    np.testing.assert_array_equal(
        initial,
        jnp.broadcast_to(q0[..., None, :, :], initial.shape),
    )

    params = _set_nonzero_plan_delta(params)
    base = nj.pure(lambda: predict(actions))(params, seed=5)[1]
    changed_a_t = actions.at[..., 0].set(1)
    changed_a_t1 = actions.at[..., 1].set(2)
    changed_unused = actions.at[..., 3].set(3)
    output_a_t = nj.pure(lambda: predict(changed_a_t))(params, seed=5)[1]
    output_a_t1 = nj.pure(lambda: predict(changed_a_t1))(params, seed=5)[1]
    output_unused = nj.pure(lambda: predict(changed_unused))(params, seed=5)[1]

    assert not np.allclose(np.asarray(base[..., 0, :, :]), output_a_t[..., 0, :, :])
    np.testing.assert_array_equal(base[..., 0, :, :], output_a_t1[..., 0, :, :])
    assert not np.allclose(np.asarray(base[..., 1:, :, :]), output_a_t1[..., 1:, :, :])
    np.testing.assert_array_equal(base, output_unused)


def test_qplan_stops_local_root_and_q0_but_ce_trains_plan_parameters() -> None:
    model = _plan_model()
    root = jax.random.normal(jax.random.key(10), (1, 1, 3, 5))
    actions = jnp.zeros((1, 1, 3, 4), jnp.int32)
    q0 = jax.random.normal(jax.random.key(11), (1, 1, 3, 2, 4))

    def objective(local_root, baseline):
        context = baseline - baseline.mean(axis=-1, keepdims=True)
        logits = model(local_root, actions, baseline, context)
        return -jax.nn.log_softmax(logits, axis=-1)[..., 1].mean()

    params = nj.init(lambda: objective(root, q0))({}, seed=12)

    def pure_objective(local_root, baseline):
        return nj.pure(lambda: objective(local_root, baseline))(params, seed=13)[1]

    root_grad, q0_grad = jax.grad(pure_objective, (0, 1))(root, q0)
    np.testing.assert_array_equal(root_grad, jnp.zeros_like(root_grad))
    np.testing.assert_array_equal(q0_grad, jnp.zeros_like(q0_grad))

    def parameter_loss(state):
        return nj.pure(lambda: objective(root, q0))(state, seed=14)[1]

    parameter_grad = jax.grad(parameter_loss)(params)
    assert float(jnp.linalg.norm(parameter_grad["plan/delta_logits/kernel"])) > 0.0


def test_uniform_plan_is_exactly_inert_after_arbitrary_belief_changes() -> None:
    model = ActionConditionedMultiStepJEPA(
        4,
        0,
        5,
        (1, 2, 4),
        4,
        width=6,
        layers=1,
        units=7,
        name="multistep",
    )
    hidden = jax.random.normal(jax.random.key(20), (1, 2, 3, 8))
    actions = jnp.zeros((1, 2, 3, 4), jnp.int32)
    uniform_context = jnp.zeros((1, 2, 3, 3, 2, 4), jnp.float32)

    params = nj.init(lambda: model(hidden, actions, uniform_context))({}, seed=21)
    params = _set_nonzero_belief_path(params)
    without_plan = nj.pure(lambda: model(hidden, actions, None))(params, seed=22)[1]
    uniform = nj.pure(lambda: model(hidden, actions, uniform_context))(params, seed=22)[
        1
    ]
    for horizon in (1, 2, 4):
        np.testing.assert_array_equal(without_plan[horizon], uniform[horizon])

    evidence = uniform_context.at[..., 0, 0, 0].set(0.5)
    informed = nj.pure(lambda: model(hidden, actions, evidence))(params, seed=22)[1]
    np.testing.assert_array_equal(without_plan[1], informed[1])
    assert not np.allclose(
        np.asarray(without_plan[2], np.float32), np.asarray(informed[2], np.float32)
    )


def test_plan_pool_is_peer_permutation_invariant_and_team_normalized() -> None:
    model = ActionConditionedMultiStepJEPA(
        4,
        0,
        5,
        (1, 2, 4),
        4,
        width=6,
        layers=1,
        units=7,
        name="multistep",
    )
    hidden = jax.random.normal(jax.random.key(30), (1, 1, 3, 8))
    actions = jnp.zeros((1, 1, 3, 4), jnp.int32)
    peer = jax.random.normal(jax.random.key(31), (1, 1, 3, 3, 2, 4))
    params = nj.init(lambda: model(hidden, actions, peer))({}, seed=32)
    params = _set_nonzero_belief_path(params)
    factual = nj.pure(lambda: model(hidden, actions, peer))(params, seed=33)[1]
    permuted = nj.pure(lambda: model(hidden, actions, peer[..., ::-1, :]))(
        params, seed=33
    )[1]
    for horizon in (1, 2, 4):
        np.testing.assert_allclose(
            np.asarray(factual[horizon], np.float32),
            np.asarray(permuted[horizon], np.float32),
            atol=1e-6,
        )

    repeated = jnp.repeat(peer[..., :1, :], 5, axis=-2)
    singleton = peer[..., :1, :]
    one = nj.pure(lambda: model(hidden, actions, singleton))(params, seed=34)[1]
    five = nj.pure(lambda: model(hidden, actions, repeated))(params, seed=34)[1]
    for horizon in (1, 2, 4):
        np.testing.assert_allclose(
            np.asarray(one[horizon], np.float32),
            np.asarray(five[horizon], np.float32),
            atol=1e-6,
        )


def test_each_ms_head_sees_only_its_causal_plan_prefix() -> None:
    model = ActionConditionedMultiStepJEPA(
        4,
        0,
        5,
        (1, 2, 4, 8),
        8,
        width=6,
        layers=1,
        units=7,
        name="multistep",
    )
    hidden = jax.random.normal(jax.random.key(40), (1, 1, 3, 8))
    actions = jnp.zeros((1, 1, 3, 8), jnp.int32)
    plan = jax.random.normal(jax.random.key(41), (1, 1, 3, 7, 2, 4))
    params = nj.init(lambda: model(hidden, actions, plan))({}, seed=42)
    params = _set_nonzero_belief_path(params)
    base = nj.pure(lambda: model(hidden, actions, plan))(params, seed=43)[1]
    changed = plan.at[..., 1:, :, 0].add(10.0)
    intervened = nj.pure(lambda: model(hidden, actions, changed))(params, seed=43)[1]
    np.testing.assert_array_equal(base[1], intervened[1])
    np.testing.assert_array_equal(base[2], intervened[2])
    assert not np.allclose(
        np.asarray(base[4], np.float32), np.asarray(intervened[4], np.float32)
    )

    selected = nj.pure(lambda: model(hidden, actions, plan, selected_horizon=4))(
        params, seed=43
    )[1]
    assert set(selected) == {4}
    np.testing.assert_array_equal(selected[4], base[4])


def test_stopped_bounded_plan_context_blocks_ms_gradient() -> None:
    core = object.__new__(MARLCore)
    core.ctde_action_count = 4
    model = ActionConditionedMultiStepJEPA(
        4,
        0,
        5,
        (1, 2, 4),
        4,
        width=6,
        layers=1,
        units=7,
        name="multistep",
    )
    hidden = jax.random.normal(jax.random.key(50), (1, 1, 3, 8))
    actions = jnp.zeros((1, 1, 3, 4), jnp.int32)
    logits = jax.random.normal(jax.random.key(51), (1, 1, 3, 3, 2, 4))

    def objective(plan_logits):
        context = core._teammate_plan_context(plan_logits)
        return sum(model(hidden, actions, context).values()).sum()

    params = nj.init(lambda: objective(logits))({}, seed=52)

    def pure_objective(plan_logits):
        return nj.pure(lambda: objective(plan_logits))(params, seed=53)[1]

    gradient = jax.grad(pure_objective)(logits)
    np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))
    uniform = core._teammate_plan_context(jnp.ones_like(logits) * 1_000.0)
    np.testing.assert_array_equal(uniform, jnp.zeros_like(uniform))


def test_plan_ce_alignment_and_dead_peer_mask_boundary() -> None:
    agents, classes, max_horizon = 3, 5, 4
    core = object.__new__(MARLCore)
    core.team = TeamAxis(agents)
    core.ctde_action_count = classes
    core.ctde_action_low = 0
    core.ctde_multistep_jepa_max_horizon = max_horizon
    core.config = elements.Config(policy={"unimix": 0.01})

    length = max_horizon + 1
    grouped_action = jnp.zeros((1, length, agents), jnp.int32)
    for time in range(1, length):
        grouped_action = grouped_action.at[0, time].set(
            (time + jnp.arange(agents)) % classes
        )
    grouped_mask = jnp.ones((1, length, agents, classes), bool)
    grouped_present = jnp.ones((1, length, agents), bool)
    grouped_alive = jnp.ones_like(grouped_present)
    grouped_alive = grouped_alive.at[0, 1, 1].set(False)
    grouped_first = jnp.zeros((1, length), bool).at[0, 4].set(True)
    all_valid = {
        step: jnp.ones((1, 1, agents), bool) for step in range(1, max_horizon + 1)
    }
    # q1 must not inherit q2's invalidity.
    all_valid[2] = all_valid[2].at[0, 0, 2].set(False)
    peer_indices = np.asarray(core._teammate_peer_indices())
    plan_logits = -10.0 * jnp.ones(
        (1, 1, agents, max_horizon - 1, agents - 1, classes), jnp.float32
    )
    for step in range(1, max_horizon):
        for focal in range(agents):
            for peer_slot, peer in enumerate(peer_indices[focal]):
                target = int(grouped_action[0, step + 1, peer])
                plan_logits = plan_logits.at[
                    0, 0, focal, step - 1, peer_slot, target
                ].set(10.0)
    q0 = jnp.zeros((1, 1, agents, agents - 1, classes), jnp.float32)

    loss, metrics = core._ctde_teammate_plan_loss(
        plan_logits,
        q0,
        grouped_action,
        grouped_mask,
        grouped_present,
        grouped_alive,
        grouped_first,
        all_valid,
        roots=1,
        length=length,
    )

    assert loss.shape == (agents, length)
    assert np.isfinite(np.asarray(loss)).all()
    assert float(metrics["ctde/teammate_plan_q1_nll"]) < 1e-5
    assert float(metrics["ctde/teammate_plan_q1_count"]) == 6.0
    assert float(metrics["ctde/teammate_plan_q1_dead_count"]) == 2.0
    assert float(metrics["ctde/teammate_plan_q2_count"]) == 4.0
    assert float(metrics["ctde/teammate_plan_q3_count"]) == 0.0
