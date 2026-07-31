from __future__ import annotations

import jax
import numpy as np
import pytest

from world_marl.dreamarl.collect import collect_coin_game_sequence
from world_marl.dreamarl.contracts import (
    MultiAgentSequenceBatch,
    sequence_batch_to_jax,
    stack_agent_actions,
)
from world_marl.dreamarl.foundation import verify_foundation


def _batch(**overrides) -> MultiAgentSequenceBatch:
    values = {
        "observations": np.zeros((4, 2, 3, 5), dtype=np.float32),
        "actions": np.zeros((4, 2, 3), dtype=np.int32),
        "rewards": np.zeros((4, 2, 3), dtype=np.float32),
        "team_rewards": np.zeros((4, 2), dtype=np.float32),
        "is_first": np.zeros((4, 2), dtype=bool),
        "is_last": np.zeros((4, 2), dtype=bool),
        "is_terminal": np.zeros((4, 2), dtype=bool),
        "agent_alive": np.ones((4, 2, 3), dtype=bool),
        "agent_ids": ("a", "b", "c"),
    }
    values.update(overrides)
    return MultiAgentSequenceBatch(**values)


def test_contract_keeps_explicit_agent_axis_through_jax() -> None:
    batch = _batch()
    jax_batch = sequence_batch_to_jax(batch)
    assert jax_batch.observations.shape == (4, 2, 3, 5)
    assert jax_batch.actions.shape == (4, 2, 3)
    assert jax.default_backend() in {"cpu", "gpu", "tpu"}


def test_terminal_transition_must_also_end_the_joint_sequence() -> None:
    terminal = np.zeros((4, 2), dtype=bool)
    terminal[2, 0] = True
    with pytest.raises(ValueError, match="terminal transition"):
        _batch(is_terminal=terminal)


def test_joint_actions_use_registered_agent_order() -> None:
    actions = {
        "blue": np.asarray([2, 3]),
        "red": np.asarray([0, 1]),
    }
    joint = stack_agent_actions(actions, ("red", "blue"))
    np.testing.assert_array_equal(joint, np.asarray([[0, 2], [1, 3]]))


def test_coin_game_collection_marks_cuts_but_not_terminals() -> None:
    batch = collect_coin_game_sequence(
        time_steps=12,
        num_envs=2,
        max_cycles=4,
        seed=0,
    )
    assert batch.observations.shape == (12, 2, 2, 36)
    assert batch.actions.shape == (12, 2, 2)
    assert batch.is_last[:, 0].tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    assert batch.is_first[:, 0].tolist() == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert not np.any(batch.is_terminal)
    assert np.all(batch.agent_alive)


def test_foundation_gate_runs_end_to_end(tmp_path) -> None:
    output = tmp_path / "foundation.json"
    result = verify_foundation(output=output)
    assert result["passed"] is True
    assert result["is_last_count"] == 32
    assert result["is_terminal_count"] == 0
    assert output.is_file()
