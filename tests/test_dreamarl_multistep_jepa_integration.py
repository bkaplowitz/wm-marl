"""Compiled dual-replay and exact-parity gates for multi-step JEPA."""

from __future__ import annotations

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from dreamarl.main import _load_configs, _resolve_config_profiles
from dreamarl.marl.core import MARLCore
from dreamarl.marl.spaces import add_agent_axis


def _spaces(team_size: int, action_count: int):
    local = {
        "vector": elements.Space(np.float32, (3,)),
        "reward": elements.Space(np.float32, ()),
        "agent_present": elements.Space(bool, ()),
        "agent_alive": elements.Space(bool, ()),
        "controllable_alive": elements.Space(bool, ()),
        "action_mask": elements.Space(bool, (action_count,)),
        "is_first": elements.Space(bool, ()),
        "is_last": elements.Space(bool, ()),
        "is_terminal": elements.Space(bool, ()),
    }
    global_fields = {"is_first", "is_last", "is_terminal"}
    observations = {
        key: space if key in global_fields else add_agent_axis(space, team_size)
        for key, space in local.items()
    }
    actions = {
        "action": add_agent_axis(
            elements.Space(np.int32, (), 0, action_count), team_size
        )
    }
    return observations, actions


def _config(
    team_size: int,
    *,
    enabled: bool,
    coupled: bool = False,
    uncoupled: bool = False,
    action0: bool = False,
):
    profiles = ["smac_vector", "ctde", "ctde_generalist"]
    if action0:
        profiles.append("ctde_tbv2_multistep_coupled_action0")
    elif coupled:
        profiles.append("ctde_tbv2_multistep_coupled")
    elif uncoupled:
        profiles.append("ctde_tbv2_multistep_uncoupled")
    elif enabled:
        profiles.append("ctde_multistep_jepa")
    profiles.append("debug")
    resolved = _resolve_config_profiles(_load_configs(), profiles).update(
        {
            "replay_context": 2,
            "agent.num_agents": team_size,
            "agent.imag_length": 1,
            "agent.imag_last": 2,
            "agent.enc.simple.units": 2,
            "agent.dyn.parallel_transformer.deter": 2,
            "agent.dyn.parallel_transformer.hidden": 2,
            "agent.dyn.parallel_transformer.stoch": 1,
            "agent.dyn.parallel_transformer.classes": 2,
            "agent.dyn.parallel_transformer.model": 2,
            "agent.dyn.parallel_transformer.heads": 1,
            "agent.dyn.parallel_transformer.context": 2,
            "agent.sigreg.knots": 3,
            "agent.sigreg.num_proj": 1,
            "agent.rewhead.units": 2,
            "agent.conhead.units": 2,
            "agent.maskhead.units": 2,
            "agent.policy.units": 2,
            "agent.value.units": 2,
            "agent.marl.ctde.joint.width": 2,
            "agent.marl.ctde.joint.heads": 1,
            "agent.marl.ctde.joint.context": 2,
            "agent.marl.ctde.head.units": 2,
            "agent.marl.ctde.critic.width": 2,
            "agent.marl.ctde.critic.heads": 1,
            "agent.marl.ctde.critic.value_units": 2,
            "agent.marl.ctde.multistep_jepa.width": 4,
            "agent.marl.ctde.multistep_jepa.layers": 1,
            "agent.marl.ctde.multistep_jepa.units": 4,
            "agent.marl.ctde.multistep_jepa.plan_units": 4,
            "agent.marl.ctde.teammate_belief.layers": 1,
            "agent.marl.ctde.teammate_belief.units": 4,
            "agent.marl.ctde.teammate_belief.adapter_layers": 1,
            "agent.marl.ctde.teammate_belief.adapter_units": 4,
            "agent.opt.warmup": 0,
            "agent.marl.ctde.opt.warmup": 0,
        }
    )
    return elements.Config(
        **resolved.agent,
        logdir="/tmp/dreamarl-multistep-jepa-test",
        seed=0,
        jax=resolved.jax,
        batch_size=1,
        batch_length=9,
        replay_context=2,
        replay_sampling="recent_world_uniform_behavior",
        report_length=1,
        replica=0,
        replicas=1,
    )


def _agent(
    team_size: int,
    action_count: int,
    *,
    enabled: bool,
    coupled: bool = False,
    uncoupled: bool = False,
    action0: bool = False,
):
    observations, actions = _spaces(team_size, action_count)
    agent = object.__new__(MARLCore)
    MARLCore.__init__(
        agent,
        observations,
        actions,
        _config(
            team_size,
            enabled=enabled,
            coupled=coupled,
            uncoupled=uncoupled,
            action0=action0,
        ),
    )
    return agent


def _replay_view(agent, team_size: int, action_count: int, seed: int):
    length = 11  # replay context 2 + optimized suffix 9; K=8 leaves one root.
    data = {
        "vector": jax.random.normal(
            jax.random.key(seed), (1, length, team_size, 3)
        ),
        "reward": jax.random.normal(
            jax.random.key(seed + 1), (1, length, team_size)
        ),
        "agent_present": jnp.ones((1, length, team_size), bool),
        "agent_alive": jnp.ones((1, length, team_size), bool),
        "controllable_alive": jnp.ones((1, length, team_size), bool),
        "action_mask": jnp.ones((1, length, team_size, action_count), bool),
        "is_first": jnp.zeros((1, length), bool).at[:, 0].set(True),
        "is_last": jnp.zeros((1, length), bool),
        "is_terminal": jnp.zeros((1, length), bool),
        "action": jax.random.randint(
            jax.random.key(seed + 2),
            (1, length, team_size),
            0,
            action_count,
        ),
        "stepid": jnp.zeros((1, length, 20), jnp.uint8),
        "consec": jnp.zeros((1, length), jnp.int32),
    }
    for key, space in agent.ext_space.items():
        if key.startswith("_behavior_replay/") or key in data:
            continue
        shape = (1, length, *space.shape)
        data[key] = (
            jnp.ones(shape, space.dtype)
            if space.dtype == np.bool_
            else jnp.zeros(shape, space.dtype)
        )
    data["dyn/reset"] = jnp.broadcast_to(
        data["is_first"][:, :, None], (1, length, team_size)
    )
    data["dyn/position"] = jnp.broadcast_to(
        jnp.arange(length, dtype=jnp.int32)[None, :, None],
        (1, length, team_size),
    )
    data["dyn/active"] = jnp.ones((1, length, team_size), bool)
    return data


def _paired(world, behavior):
    return {
        **world,
        **{f"_behavior_replay/{key}": value for key, value in behavior.items()},
    }


def _ready(tree):
    for value in jax.tree.leaves(tree):
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()


@pytest.mark.parametrize(
    ("team_size", "action_count"), ((3, 9), (5, 9), (8, 14))
)
def test_multistep_jepa_real_compiled_dual_update(
    team_size: int,
    action_count: int,
) -> None:
    agent = _agent(team_size, action_count, enabled=True)
    carry = agent.init_train(1)
    world = _replay_view(agent, team_size, action_count, 1)
    behavior_a = _replay_view(agent, team_size, action_count, 2)
    behavior_b = _replay_view(agent, team_size, action_count, 3)
    batch_a = _paired(world, behavior_a)
    batch_b = _paired(world, behavior_b)

    def step(batch):
        return agent.train(carry, batch)

    initial = nj.init(lambda: step(batch_a))({}, seed=10)
    pure = nj.pure(step)
    compiled = jax.jit(lambda state, batch: pure(state, batch, seed=11))
    state_a, result_a = compiled(initial, batch_a)
    state_b, _ = compiled(initial, batch_b)
    _ready((state_a, result_a, state_b))

    module_keys = [
        key for key in state_a if key.startswith("ctde_multistep_jepa/")
    ]
    assert module_keys
    for key in module_keys:
        np.testing.assert_array_equal(np.asarray(state_a[key]), np.asarray(state_b[key]))
    assert agent.ctde_multistep_jepa in agent.opt.groups["joint_world"][0]
    assert agent.ctde_multistep_jepa not in agent.opt.groups["actor"][0]
    assert agent.ctde_multistep_jepa not in agent.opt.groups["critic"][0]
    assert agent.policy_keys == "^(enc|dyn|pol)/"

    metrics = result_a[2]
    required = [
        "loss/ctde_multistep_jepa_cosine",
        "loss/ctde_multistep_jepa_action",
        "ctde/multistep_jepa_weighted_cosine_loss",
        "ctde/multistep_jepa_weighted_action_margin_loss",
        "opt/joint_world/grad_norm",
    ]
    for horizon in (1, 2, 4, 8):
        required.extend(
            [
                f"ctde/multistep_jepa_h{horizon}_cosine",
                f"ctde/multistep_jepa_h{horizon}_within_team_mean_rank",
                f"ctde/multistep_jepa_h{horizon}_action_counterfactual_cosine_drop",
                f"ctde/multistep_jepa_h{horizon}_action_distinct_legal_coverage",
                f"ctde/multistep_jepa_h{horizon}_valid_count",
            ]
        )
    for key in required:
        assert np.isfinite(float(metrics[key])), key
    for horizon in (1, 2, 4, 8):
        assert float(metrics[f"ctde/multistep_jepa_h{horizon}_valid_count"]) == (
            team_size
        )
    for horizon in (2, 4, 8):
        assert (
            float(
                metrics[
                    f"ctde/multistep_jepa_h{horizon}_action_distinct_legal_coverage"
                ]
            )
            > 0.0
        )


def test_multistep_jepa_preserves_base_initialization_and_first_update() -> None:
    team_size, action_count = 3, 9
    treatment = _agent(team_size, action_count, enabled=True)
    control = _agent(team_size, action_count, enabled=False)
    treatment_carry = treatment.init_train(1)
    control_carry = control.init_train(1)
    world = _replay_view(treatment, team_size, action_count, 20)
    behavior = _replay_view(treatment, team_size, action_count, 21)
    batch = _paired(world, behavior)

    def treatment_step():
        return treatment.train(treatment_carry, batch)

    def control_step():
        return control.train(control_carry, batch)

    treatment_initial = nj.init(treatment_step)({}, seed=30)
    control_initial = nj.init(control_step)({}, seed=30)

    def assert_common_equal(left, right, *, after_update):
        keys = sorted(set(left).intersection(right))
        keys = [
            key
            for key in keys
            if "ctde_multistep_jepa" not in key
            and np.shape(left[key]) == np.shape(right[key])
            and (
                not after_update
                or (
                    not key.startswith("ctde_joint/")
                    and not key.startswith("opt/joint_world")
                )
            )
        ]
        assert keys
        for key in keys:
            np.testing.assert_array_equal(
                np.asarray(left[key]), np.asarray(right[key]), err_msg=key
            )

    assert_common_equal(treatment_initial, control_initial, after_update=False)
    treatment_state, treatment_result = nj.pure(treatment_step)(
        treatment_initial, seed=31
    )
    control_state, control_result = nj.pure(control_step)(control_initial, seed=31)
    assert_common_equal(treatment_state, control_state, after_update=True)
    shared_joint = [
        key
        for key in control_state
        if key.startswith("ctde_joint/") and key in treatment_state
    ]
    assert shared_joint
    assert any(
        not np.array_equal(
            np.asarray(treatment_state[key]), np.asarray(control_state[key])
        )
        for key in shared_joint
    )
    for fragment in ("/agent_interaction/", "/temporal/"):
        selected = [key for key in shared_joint if fragment in key]
        assert selected, fragment
        assert any(
            not np.array_equal(
                np.asarray(treatment_state[key]), np.asarray(control_state[key])
            )
            for key in selected
        ), fragment

    treatment_metrics = treatment_result[2]
    control_metrics = control_result[2]
    for key in (
        "loss/posterior_jepa",
        "loss/dynamics_jepa",
        "loss/ctde_embedding",
        "loss/ctde_interface",
        "loss/ctde_reward",
        "loss/ctde_continuation",
        "loss/ctde_action_mask",
        "loss/ctde_alive",
        "loss/policy",
        "loss/value",
        "loss/repval",
    ):
        np.testing.assert_array_equal(
            np.asarray(treatment_metrics[key]),
            np.asarray(control_metrics[key]),
            err_msg=key,
        )


@pytest.mark.parametrize(
    ("team_size", "action_count"), ((3, 9), (5, 11), (8, 14))
)
def test_tbv2_coupled_multistep_real_compiled_dual_update(
    team_size: int,
    action_count: int,
) -> None:
    agent = _agent(
        team_size,
        action_count,
        enabled=True,
        coupled=True,
    )
    carry = agent.init_train(1)
    world = _replay_view(agent, team_size, action_count, 101)
    behavior_a = _replay_view(agent, team_size, action_count, 102)
    behavior_b = _replay_view(agent, team_size, action_count, 103)
    batch_a = _paired(world, behavior_a)
    batch_b = _paired(world, behavior_b)

    def step(batch):
        return agent.train(carry, batch)

    initial = nj.init(lambda: step(batch_a))({}, seed=110)
    pure = nj.pure(step)
    compiled = jax.jit(lambda state, batch: pure(state, batch, seed=111))
    state_a, result_a = compiled(initial, batch_a)
    state_b, _ = compiled(initial, batch_b)
    _ready((state_a, result_a, state_b))

    plan_keys = [key for key in state_a if key.startswith("ctde_teammate_plan/")]
    multistep_keys = [
        key for key in state_a if key.startswith("ctde_multistep_jepa/")
    ]
    assert plan_keys and multistep_keys
    assert any(
        not np.array_equal(np.asarray(initial[key]), np.asarray(state_a[key]))
        for key in plan_keys
        if key in initial
    )
    for key in plan_keys + multistep_keys:
        np.testing.assert_array_equal(np.asarray(state_a[key]), np.asarray(state_b[key]))

    assert agent.ctde_teammate_plan in agent.opt.groups["joint_world"][0]
    assert agent.ctde_multistep_jepa in agent.opt.groups["joint_world"][0]
    assert agent.ctde_teammate_plan not in agent.opt.groups["actor"][0]
    assert agent.ctde_multistep_jepa not in agent.opt.groups["actor"][0]
    assert agent.ctde_teammate_plan not in agent.opt.groups["critic"][0]
    assert agent.policy_keys == (
        "^(enc|dyn|pol|ctde_teammate_belief|ctde_teammate_actor)/"
    )

    metrics = result_a[2]
    required = [
        "loss/ctde_teammate_plan",
        "loss/ctde_multistep_jepa_cosine",
        "loss/ctde_multistep_jepa_action",
        "ctde/teammate_plan_recent_nll",
        "ctde/teammate_plan_recent_count",
        "ctde/multistep_jepa_plan_context_rms",
        "ctde/multistep_jepa_plan_context_nonzero_fraction",
        "ctde/multistep_jepa_action_counterfactual_enabled",
        "opt/joint_world/grad_norm",
        "opt/actor/grad_norm",
        "opt/critic/grad_norm",
    ]
    for step_index in range(1, 8):
        required.extend(
            [
                f"ctde/teammate_plan_q{step_index}_nll",
                f"ctde/teammate_plan_q{step_index}_count",
            ]
        )
    for key in required:
        assert np.isfinite(float(metrics[key])), key
    assert float(metrics["ctde/teammate_plan_recent_count"]) > 0.0
    assert float(metrics["ctde/multistep_jepa_plan_context_nonzero_fraction"]) >= 0.0
    assert float(metrics["ctde/multistep_jepa_action_counterfactual_enabled"]) == 1.0


def test_coupled_zero_residual_preserves_uncoupled_first_update() -> None:
    team_size, action_count = 3, 9
    coupled = _agent(
        team_size,
        action_count,
        enabled=True,
        coupled=True,
    )
    uncoupled = _agent(
        team_size,
        action_count,
        enabled=True,
        uncoupled=True,
    )
    coupled_carry = coupled.init_train(1)
    uncoupled_carry = uncoupled.init_train(1)
    world = _replay_view(coupled, team_size, action_count, 201)
    behavior = _replay_view(coupled, team_size, action_count, 202)
    batch = _paired(world, behavior)

    def coupled_step():
        return coupled.train(coupled_carry, batch)

    def uncoupled_step():
        return uncoupled.train(uncoupled_carry, batch)

    coupled_initial = nj.init(coupled_step)({}, seed=210)
    uncoupled_initial = nj.init(uncoupled_step)({}, seed=210)

    def common_keys(left, right):
        return [
            key
            for key in sorted(set(left).intersection(right))
            if np.shape(left[key]) == np.shape(right[key])
        ]

    initial_common = common_keys(coupled_initial, uncoupled_initial)
    assert initial_common
    for key in initial_common:
        np.testing.assert_array_equal(
            np.asarray(coupled_initial[key]),
            np.asarray(uncoupled_initial[key]),
            err_msg=key,
        )
    assert any(key.startswith("ctde_teammate_plan/") for key in coupled_initial)
    assert not any(key.startswith("ctde_teammate_plan/") for key in uncoupled_initial)

    coupled_state, coupled_result = nj.pure(coupled_step)(
        coupled_initial, seed=211
    )
    uncoupled_state, uncoupled_result = nj.pure(uncoupled_step)(
        uncoupled_initial, seed=211
    )
    update_common = common_keys(coupled_state, uncoupled_state)
    for key in update_common:
        np.testing.assert_array_equal(
            np.asarray(coupled_state[key]),
            np.asarray(uncoupled_state[key]),
            err_msg=key,
        )

    coupled_metrics = coupled_result[2]
    uncoupled_metrics = uncoupled_result[2]
    for key in (
        "loss/policy",
        "loss/value",
        "loss/ctde_teammate_belief",
        "loss/ctde_multistep_jepa_cosine",
        "loss/ctde_multistep_jepa_action",
        "ctde/multistep_jepa_weighted_cosine_loss",
        "ctde/multistep_jepa_weighted_action_margin_loss",
    ):
        np.testing.assert_array_equal(
            np.asarray(coupled_metrics[key]),
            np.asarray(uncoupled_metrics[key]),
            err_msg=key,
        )


def test_coupled_action0_compiled_path_skips_counterfactual_queries() -> None:
    team_size, action_count = 5, 11
    agent = _agent(
        team_size,
        action_count,
        enabled=True,
        action0=True,
    )
    carry = agent.init_train(1)
    world = _replay_view(agent, team_size, action_count, 301)
    behavior = _replay_view(agent, team_size, action_count, 302)
    batch = _paired(world, behavior)

    def step(data):
        return agent.train(carry, data)

    initial = nj.init(lambda: step(batch))({}, seed=310)
    compiled = jax.jit(
        lambda state, data: nj.pure(step)(state, data, seed=311)
    )
    state, result = compiled(initial, batch)
    _ready((state, result))
    metrics = result[2]
    assert float(metrics["ctde/multistep_jepa_action_counterfactual_enabled"]) == 0.0
    assert float(metrics["loss/ctde_multistep_jepa_action"]) == 0.0
    for horizon in (1, 2, 4, 8):
        assert (
            float(
                metrics[
                    f"ctde/multistep_jepa_h{horizon}_action_distinct_legal_count"
                ]
            )
            == 0.0
        )
