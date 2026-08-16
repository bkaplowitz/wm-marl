from __future__ import annotations

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamarl.agent import Agent as LocalAgent
from dreamarl.main import _load_configs
from dreamarl.marl.axes import TeamAxis
from dreamarl.marl.core import MARLCore
from dreamarl.marl.spaces import add_agent_axis
from dreamarl.training.learner import masked_mean
from dreamarl.world_model.transformer import CausalTransformer


GLOBAL_KEYS = {"is_first", "is_last", "is_terminal", "consec", "stepid"}


def _assert_tree_equal(actual, expected) -> None:
    actual_leaves, actual_tree = jax.tree.flatten(actual)
    expected_leaves, expected_tree = jax.tree.flatten(expected)
    assert actual_tree == expected_tree
    for left, right in zip(actual_leaves, expected_leaves):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def _local_model_state(state):
    return {key: value for key, value in state.items() if not key.startswith("opt/")}


def _assert_mapping_subset_equal(actual, expected) -> None:
    assert set(expected).issubset(actual)
    _assert_tree_equal(
        {key: actual[key] for key in expected},
        expected,
    )


def _agent_config(num_agents: int = 1):
    configs = _load_configs()
    resolved = elements.Config(configs["defaults"])
    resolved = resolved.update(configs["debug"])
    resolved = resolved.update(
        {
            "replay_context": 0,
            "agent.num_agents": num_agents,
            "agent.imag_length": 2,
            "agent.imag_last": 2,
        }
    )
    return elements.Config(
        **resolved.agent,
        logdir="/tmp/dreamarl-marl-core-test",
        seed=0,
        jax=resolved.jax,
        batch_size=resolved.batch_size,
        batch_length=resolved.batch_length,
        replay_context=0,
        report_length=resolved.report_length,
        replica=0,
        replicas=1,
    )


def _local_spaces(*, metadata: bool = True):
    observations = {
        "image": elements.Space(np.uint8, (64, 64, 3), 0, 256),
        "reward": elements.Space(np.float32, ()),
        "is_first": elements.Space(bool, ()),
        "is_last": elements.Space(bool, ()),
        "is_terminal": elements.Space(bool, ()),
    }
    if metadata:
        observations.update(
            agent_present=elements.Space(bool, ()),
            agent_alive=elements.Space(bool, ()),
            action_mask=elements.Space(bool, (4,)),
        )
    return observations, {"action": elements.Space(np.int32, (), 0, 4)}


def _team_spaces(team_size: int):
    observations, actions = _local_spaces(metadata=True)
    observations = {
        key: space if key in GLOBAL_KEYS else add_agent_axis(space, team_size)
        for key, space in observations.items()
    }
    actions = {key: add_agent_axis(space, team_size) for key, space in actions.items()}
    return observations, actions


def _local_batch(batch: int = 1, length: int = 4):
    return {
        "image": jax.random.randint(
            jax.random.key(1),
            (batch, length, 64, 64, 3),
            0,
            256,
            dtype=jnp.uint8,
        ),
        "reward": jax.random.normal(jax.random.key(2), (batch, length)),
        "agent_present": jnp.ones((batch, length), bool),
        "agent_alive": jnp.ones((batch, length), bool),
        "action_mask": jnp.ones((batch, length, 4), bool),
        "is_first": jnp.zeros((batch, length), bool).at[:, 0].set(True),
        "is_last": jnp.zeros((batch, length), bool),
        "is_terminal": jnp.zeros((batch, length), bool),
        "action": jax.random.randint(jax.random.key(3), (batch, length), 0, 4),
        "stepid": jnp.zeros((batch, length, 20), jnp.uint8),
        "consec": jnp.zeros((batch, length), jnp.int32),
    }


def _add_team_axis(data, team_size: int):
    return {
        key: (
            value
            if key in GLOBAL_KEYS
            else jnp.repeat(value[:, :, None, ...], team_size, axis=2)
        )
        for key, value in data.items()
    }


def test_team_axis_round_trip_preserves_identity() -> None:
    team = TeamAxis(3)
    values = np.arange(2 * 5 * 3 * 4).reshape(2, 5, 3, 4)
    np.testing.assert_array_equal(
        team.unfold_sequence(team.fold_sequence(values)), values
    )


def test_imagination_start_grouping_preserves_environment_and_agent_identity() -> None:
    team = TeamAxis(3)
    batch, starts = 2, 4
    logical = np.arange(batch * team.size * starts * 2).reshape(
        batch, team.size, starts, 2
    )
    folded = logical.reshape(batch * team.size * starts, 2)
    grouped = team.group_starts(folded, starts)
    expected = logical.transpose(0, 2, 1, 3).reshape(batch * starts, team.size, 2)
    np.testing.assert_array_equal(grouped, expected)
    np.testing.assert_array_equal(team.ungroup_starts(grouped, starts), folded)


def test_shared_local_mapping_is_permutation_equivariant() -> None:
    team = TeamAxis(4)
    values = np.arange(2 * 4 * 3).reshape(2, 4, 3)
    permutation = np.array([2, 0, 3, 1])
    output = team.unfold_batch(3 * team.fold_batch(values) - 7)
    permuted = team.unfold_batch(3 * team.fold_batch(values[:, permutation]) - 7)
    np.testing.assert_array_equal(permuted, output[:, permutation])


def test_joint_transition_is_equivariant_and_peer_action_sensitive() -> None:
    team_size, batch = 3, 2
    transition = CausalTransformer(
        12,
        units=16,
        output=8,
        layers=1,
        heads=4,
        context=4,
        ffup=2,
        team_size=team_size,
        name="transition",
    )
    pair = jax.random.normal(jax.random.key(4), (batch * team_size, 12))
    active = jnp.ones((batch * team_size,), bool)
    reset = jnp.zeros((batch * team_size,), bool)

    def forward(current_pair, current_active):
        carry = transition.initial(batch * team_size)
        return transition.step(carry, current_pair, reset, active=current_active)[1]

    state = nj.init(forward)({}, pair, active, seed=5)
    changed_pair = pair.at[1::team_size, -1].add(3.0)
    _, closed = nj.pure(forward)(state, pair, active, seed=6)
    _, changed_closed = nj.pure(forward)(state, changed_pair, active, seed=6)
    closed = closed.reshape((batch, team_size, -1))
    changed_closed = changed_closed.reshape((batch, team_size, -1))
    np.testing.assert_allclose(
        np.asarray(changed_closed[:, 0], np.float32),
        np.asarray(closed[:, 0], np.float32),
        rtol=0,
        atol=0,
    )

    gate_key = next(key for key in state if key.endswith("/peer_gate"))
    state = dict(state, **{gate_key: jnp.full_like(state[gate_key], 0.5)})
    _, output = nj.pure(forward)(state, pair, active, seed=6)
    output = output.reshape((batch, team_size, -1))

    permutation = np.array([2, 0, 1])
    permuted_pair = pair.reshape((batch, team_size, 12))[:, permutation].reshape(
        pair.shape
    )
    _, permuted = nj.pure(forward)(state, permuted_pair, active, seed=6)
    permuted = permuted.reshape((batch, team_size, -1))
    np.testing.assert_allclose(
        np.asarray(permuted, np.float32),
        np.asarray(output[:, permutation], np.float32),
        rtol=1e-5,
        atol=1e-5,
    )

    _, changed = nj.pure(forward)(state, changed_pair, active, seed=6)
    changed = changed.reshape((batch, team_size, -1))
    assert not np.allclose(
        np.asarray(changed[:, 0], np.float32),
        np.asarray(output[:, 0], np.float32),
    )


def test_joint_transition_parallel_and_recurrent_paths_match() -> None:
    team_size, batch, length = 3, 2, 4
    transition = CausalTransformer(
        12,
        units=16,
        output=8,
        layers=1,
        heads=4,
        context=4,
        ffup=2,
        team_size=team_size,
        name="transition",
    )
    pair = jax.random.normal(
        jax.random.key(40), (batch * team_size, length, 12)
    )
    active = jnp.ones((batch * team_size, length), bool)
    reset = jnp.zeros((batch * team_size, length), bool).at[:, 0].set(True)

    def forward(current_pair, current_active):
        cache = transition.initial(batch * team_size)
        _, parallel, _ = transition.sequence(
            cache, current_pair, reset, active=current_active
        )
        recurrent = []
        for index in range(length):
            cache, state = transition.step(
                cache,
                current_pair[:, index],
                reset[:, index],
                active=current_active[:, index],
            )
            recurrent.append(state)
        return parallel, jnp.stack(recurrent, axis=1)

    parameters = nj.init(forward)({}, pair, active, seed=41)
    gate_key = next(key for key in parameters if key.endswith("/peer_gate"))
    parameters = dict(
        parameters,
        **{gate_key: jnp.full_like(parameters[gate_key], 0.5)},
    )
    _, (parallel, recurrent) = nj.pure(forward)(parameters, pair, active, seed=42)
    np.testing.assert_allclose(
        np.asarray(parallel, np.float32),
        np.asarray(recurrent, np.float32),
        rtol=2e-5,
        atol=2e-5,
    )


def test_inactive_agents_are_excluded_from_loss() -> None:
    values = jnp.array([[1.0, 2.0], [1_000.0, 2_000.0]])
    valid = jnp.array([[True, True], [False, False]])
    assert float(masked_mean(values, valid)) == 1.5


def test_singleton_core_matches_local_training_exactly() -> None:
    local_obs_space, local_act_space = _local_spaces(metadata=False)
    team_obs_space, team_act_space = _team_spaces(1)
    config = _agent_config(1)
    local = object.__new__(LocalAgent)
    LocalAgent.__init__(local, local_obs_space, local_act_space, config)
    team = object.__new__(MARLCore)
    MARLCore.__init__(team, team_obs_space, team_act_space, config)

    full_data = _local_batch()
    local_data = {
        key: value
        for key, value in full_data.items()
        if key in local_obs_space or key in local_act_space or key in GLOBAL_KEYS
    }
    team_data = _add_team_axis(full_data, 1)
    local_carry = local.init_train(1)
    team_carry = team.init_train(1)

    def local_step():
        return local.train(local_carry, local_data)

    def team_step():
        return team.train(team_carry, team_data)

    local_state = nj.init(local_step)({}, seed=44)
    team_state = nj.init(team_step)({}, seed=44)
    _assert_tree_equal(_local_model_state(team_state), _local_model_state(local_state))
    local_next, local_output = nj.pure(local_step)(local_state, seed=45)
    team_next, team_output = nj.pure(team_step)(team_state, seed=45)
    _assert_tree_equal(_local_model_state(team_next), _local_model_state(local_next))
    _assert_tree_equal(team.team.fold_tree_batch(team_output[0]), local_output[0])
    _assert_tree_equal(team_output[1], local_output[1])
    local_metrics = {
        key: value
        for key, value in local_output[2].items()
        if not key.startswith("opt/")
    }
    _assert_mapping_subset_equal(team_output[2], local_metrics)


def test_multi_agent_core_completes_shared_local_update() -> None:
    observations, actions = _team_spaces(3)
    agent = object.__new__(MARLCore)
    config = _agent_config(3).update({"opt.warmup": 0})
    MARLCore.__init__(agent, observations, actions, config)
    data = _add_team_axis(_local_batch(), 3)
    carry = agent.init_train(1)

    def step():
        return agent.train(carry, data)

    state = nj.init(step)({}, seed=52)
    next_state, (_, _, metrics) = nj.pure(step)(state, seed=53)
    assert [module.name for module in agent.modules] == [
        "dyn",
        "enc",
        "rew",
        "con",
        "pol",
        "val",
    ]
    assert "loss/policy" in metrics
    assert "interaction/gate_mean" in metrics
    gate_key = next(key for key in state if key.endswith("/peer_gate"))
    assert np.max(np.abs(np.asarray(next_state[gate_key]))) > 0
