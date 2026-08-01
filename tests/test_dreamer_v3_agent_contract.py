from __future__ import annotations

import copy
import importlib
import inspect
from types import SimpleNamespace

import jax
import numpy as np
import numpy.typing as npt
import pytest

from world_marl.dreamer_v3_baseline.config import SequenceShapeConfig
from world_marl.dreamer_v3_baseline.networks import TensorSpace


Array = npt.NDArray[np.generic]


def _symbols():
    module = importlib.import_module("world_marl.dreamer_v3_baseline.agent")
    return (
        module.AgentCarry,
        module.DreamerAgent,
        module.validate_action_tree,
        module.validate_replay_row,
    )


def test_package_root_exports_public_agent_surface() -> None:
    package = importlib.import_module("world_marl.dreamer_v3_baseline")
    assert (
        package.AgentCarry,
        package.DreamerAgent,
        package.validate_action_tree,
        package.validate_replay_row,
    ) == _symbols()


def _agent(*, context: int):
    _carry, DreamerAgent, _action, _row = _symbols()
    config = SimpleNamespace(
        compute_dtype="float32",
        rssm=SimpleNamespace(deter=2, stoch=1, classes=2),
        sequence=SequenceShapeConfig(
            batch_size=2,
            sequence_length=3,
            context=context,
            consecutive=2,
            report_length=3,
            report_consecutive=2,
        ),
    )
    observation_spaces = {
        "is_first": TensorSpace((), "bool"),
        "is_last": TensorSpace((), "bool"),
        "is_terminal": TensorSpace((), "bool"),
        "obs": TensorSpace((1,), "float32"),
        "reward": TensorSpace((), "float32"),
    }
    action_spaces = {"action": TensorSpace((2,), "float32")}
    return DreamerAgent(observation_spaces, action_spaces, config)


def _tree_equal(left: object, right: object) -> None:
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _tree_equal(left[key], right[key])
    elif isinstance(left, list):
        assert isinstance(right, list)
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right, strict=True):
            _tree_equal(lhs, rhs)
    else:
        assert left == right


def _batch(*, context: int, consec: tuple[int, int]) -> dict[str, Array]:
    time = context + 3
    action = np.asarray(
        [
            [[10 + t, 20 + t] for t in range(time)],
            [[30 + t, 40 + t] for t in range(time)],
        ],
        np.float32,
    )
    ids = np.zeros((2, time, 20), np.uint8)
    ids[:, :, -1] = np.arange(time, dtype=np.uint8)
    return {
        "action": action,
        "consec": np.broadcast_to(np.asarray(consec, np.int32)[:, None], (2, time)),
        "dyn/deter": np.asarray(
            [
                [[100 + t, 200 + t] for t in range(time)],
                [[300 + t, 400 + t] for t in range(time)],
            ],
            np.float32,
        ),
        "dyn/stoch": np.asarray(
            [
                [[[500 + t, 600 + t]] for t in range(time)],
                [[[700 + t, 800 + t]] for t in range(time)],
            ],
            np.float32,
        ),
        "is_first": np.zeros((2, time), bool),
        "is_last": np.zeros((2, time), bool),
        "is_terminal": np.zeros((2, time), bool),
        "obs": np.asarray(
            [[[t] for t in range(time)], [[10 + t] for t in range(time)]],
            np.float32,
        ),
        "reward": np.zeros((2, time), np.float32),
        "stepid": ids,
    }


def test_agent_carry_inverse_signature_and_fresh_state() -> None:
    AgentCarry, _agent_cls, _action, _row = _symbols()
    assert list(inspect.signature(AgentCarry.from_state).parameters) == [
        "state",
        "agent",
        "expected_leading_shape",
    ]
    agent = _agent(context=0)
    carry = agent.initial(2)
    assert carry.rssm.deter.shape == (2, 2)
    assert carry.rssm.stoch.shape == (2, 1, 2)
    np.testing.assert_array_equal(carry.prev_action["action"], np.zeros((2, 2)))
    state = carry.state_dict()
    restored = AgentCarry.from_state(state, agent, (2,))
    _tree_equal(restored.state_dict(), state)
    state["rssm"]["deter"][0, 0] = 9
    assert np.asarray(restored.rssm.deter)[0, 0] == 0

    aliased = carry.state_dict()
    aliased["decoder"] = aliased["encoder"]
    with pytest.raises(ValueError, match="alias"):
        AgentCarry.from_state(aliased, agent, (2,))


def test_action_and_replay_row_validation_preserve_unbounded_values() -> None:
    _carry, _agent_cls, validate_action_tree, validate_replay_row = _symbols()
    spaces = {"action": TensorSpace((2,), "float32")}
    action = {"action": np.asarray([4.5, -7.0], np.float32)}
    action_before = copy.deepcopy(action)
    validated = validate_action_tree(action, spaces, ())
    np.testing.assert_array_equal(validated["action"], action["action"])
    _tree_equal(action, action_before)
    with pytest.raises(ValueError, match="finite"):
        validate_action_tree(
            {"action": np.asarray([np.inf, 0], np.float32)}, spaces, ()
        )

    row = {
        "action": np.asarray([0.0, 0.0], np.float32),
        "is_last": True,
        "is_terminal": False,
    }
    row_before = copy.deepcopy(row)
    validate_replay_row(row, spaces, ())
    _tree_equal(row, row_before)
    row["action"] = np.asarray([0.0, 1.0], np.float32)
    with pytest.raises(ValueError, match="final replay row action must be zero"):
        validate_replay_row(row, spaces, ())


@pytest.mark.parametrize("field", ["is_last", "is_terminal"])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint8, np.float32, object])
def test_replay_row_validation_rejects_non_boolean_boundary_flags(
    field: str, dtype: npt.DTypeLike
) -> None:
    _carry, _agent_cls, _action, validate_replay_row = _symbols()
    spaces = {"action": TensorSpace((2,), "float32")}
    row: dict[str, object] = {
        "action": np.zeros((2,), np.float32),
        "is_last": False,
        "is_terminal": False,
    }
    row[field] = np.asarray(0, dtype=dtype)

    with pytest.raises(ValueError, match="boundary flags.*bool"):
        validate_replay_row(row, spaces, ())


@pytest.mark.parametrize(
    ("leading_shape", "is_last", "is_terminal", "action"),
    [
        ((), True, np.bool_(False), np.zeros((2,), np.float32)),
        (
            (2,),
            np.asarray([False, True], bool),
            np.asarray([False, False], bool),
            np.zeros((2, 2), np.float32),
        ),
    ],
)
def test_replay_row_validation_accepts_exact_boolean_boundary_flags(
    leading_shape: tuple[int, ...],
    is_last: object,
    is_terminal: object,
    action: Array,
) -> None:
    _carry, _agent_cls, _action, validate_replay_row = _symbols()
    validate_replay_row(
        {
            "action": action,
            "is_last": is_last,
            "is_terminal": is_terminal,
        },
        {"action": TensorSpace((2,), "float32")},
        leading_shape,
    )


def test_zero_context_never_evaluates_replay_slice_and_uses_incoming_carry() -> None:
    agent = _agent(context=0)
    carry = agent.initial(2)
    carry = carry.replace(
        rssm=carry.rssm.replace(
            deter=np.asarray([[1, 2], [3, 4]], np.float32),
            stoch=np.asarray([[[5, 6]], [[7, 8]]], np.float32),
        ),
        prev_action={"action": np.asarray([[1, 2], [3, 4]], np.float32)},
    )
    data = _batch(context=0, consec=(0, 1))
    outgoing, observations, previous, step_ids = agent.apply_replay_context(carry, data)
    np.testing.assert_array_equal(outgoing.rssm.deter, carry.rssm.deter)
    np.testing.assert_array_equal(observations["obs"], data["obs"])
    np.testing.assert_array_equal(step_ids, data["stepid"])
    np.testing.assert_array_equal(previous["action"][:, 0], carry.prev_action["action"])
    np.testing.assert_array_equal(previous["action"][:, 1:], data["action"][:, :-1])
    np.testing.assert_array_equal(outgoing.prev_action["action"], data["action"][:, -1])


def test_nonzero_context_reconstructs_only_first_consecutive_rows() -> None:
    agent = _agent(context=2)
    carry = agent.initial(2)
    carry = carry.replace(
        rssm=carry.rssm.replace(
            deter=np.asarray([[1, 2], [3, 4]], np.float32),
            stoch=np.asarray([[[5, 6]], [[7, 8]]], np.float32),
        ),
        prev_action={"action": np.asarray([[1, 2], [3, 4]], np.float32)},
    )
    data = _batch(context=2, consec=(0, 1))
    outgoing, observations, previous, step_ids = agent.apply_replay_context(carry, data)
    np.testing.assert_array_equal(outgoing.rssm.deter[0], data["dyn/deter"][0, 1])
    np.testing.assert_array_equal(outgoing.rssm.stoch[0], data["dyn/stoch"][0, 1])
    np.testing.assert_array_equal(outgoing.rssm.deter[1], carry.rssm.deter[1])
    np.testing.assert_array_equal(observations["obs"], data["obs"][:, 2:])
    np.testing.assert_array_equal(step_ids, data["stepid"][:, 2:])
    np.testing.assert_array_equal(previous["action"][0], data["action"][0, 1:-1])
    np.testing.assert_array_equal(previous["action"][1], data["action"][1, 1:-1])
    np.testing.assert_array_equal(outgoing.prev_action["action"], data["action"][:, -1])


@pytest.mark.parametrize("context", [0, 2])
def test_replay_context_jit_matches_eager_for_mixed_rows(context: int) -> None:
    agent = _agent(context=context)
    carry = agent.initial(2)
    carry = carry.replace(
        rssm=carry.rssm.replace(
            deter=np.asarray([[1, 2], [3, 4]], np.float32),
            stoch=np.asarray([[[5, 6]], [[7, 8]]], np.float32),
        ),
        prev_action={"action": np.asarray([[1, 2], [3, 4]], np.float32)},
    )
    data = _batch(context=context, consec=(0, 1))
    data["is_first"][:, context] = np.asarray([True, False])

    eager = agent.apply_replay_context(carry, data)
    compiled = jax.jit(agent.apply_replay_context)(carry, data)

    eager_leaves, eager_tree = jax.tree.flatten(eager)
    compiled_leaves, compiled_tree = jax.tree.flatten(compiled)
    assert compiled_tree == eager_tree
    for eager_leaf, compiled_leaf in zip(eager_leaves, compiled_leaves, strict=True):
        np.testing.assert_array_equal(compiled_leaf, eager_leaf)
