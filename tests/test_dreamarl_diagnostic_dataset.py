from __future__ import annotations

import numpy as np
import pytest

from world_marl.dreamarl.diagnostic_dataset import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    load_dataset,
    save_dataset,
    split_trajectories,
    trajectory_bootstrap_interval,
    valid_episode_starts,
    validate_dataset,
)


def _bundle(trajectories=10, time=4, agents=3):
    valid = np.ones((trajectories, time), bool)
    arrays = {
        "trajectory_id": np.arange(trajectories),
        "episode_id": np.arange(trajectories) + 100,
        "timestep": np.broadcast_to(
            np.arange(time)[None], (trajectories, time)
        ).copy(),
        "policy_checkpoint": np.asarray(
            ["early"] * (trajectories // 2) + ["late"] * (trajectories // 2)
        ),
        "belief": np.zeros((trajectories, time, agents, 7), np.float32),
        "action": np.zeros((trajectories, time, agents), np.int32),
        "next_target": np.ones((trajectories, time, agents, 5), np.float32),
        "reward": np.zeros((trajectories, time), np.float32),
        "is_last": np.zeros((trajectories, time), bool),
        "is_terminal": np.zeros((trajectories, time), bool),
        "valid": valid,
        "agent_valid": np.broadcast_to(valid[..., None], (trajectories, time, agents)),
        "action_available": np.broadcast_to(
            valid[..., None], (trajectories, time, agents)
        ),
        "track_id": np.broadcast_to(
            np.arange(agents)[None, None], (trajectories, time, agents)
        ),
    }
    manifest = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "temporal_contract": (
            "(belief_t,joint_action_t)->stopped_target_t_plus_1"
        ),
    }
    return arrays, manifest


def test_episode_windows_never_cross_reset_or_last():
    first = np.asarray([True, False, False, True, False, False, False])
    last = np.asarray([False, False, True, False, False, False, True])
    starts = valid_episode_starts(first, last, len(first), window=3)
    assert starts == [0, 3, 4]


def test_dataset_roundtrip_and_checkpoint_stratified_split(tmp_path):
    arrays, manifest = _bundle()
    path = tmp_path / "transitions.npz"
    save_dataset(path, arrays, manifest)
    loaded = load_dataset(path)
    np.testing.assert_array_equal(loaded.arrays["belief"], arrays["belief"])
    assert loaded.manifest["dataset_sha256"]
    split = split_trajectories(loaded.arrays, seed=3)
    for rows in (split.train, split.validation, split.test):
        labels = set(loaded.arrays["policy_checkpoint"][rows])
        assert labels == {"early", "late"}
    assert not set(split.train) & set(split.validation)
    assert not set(split.train) & set(split.test)


def test_invalid_terminal_and_timestep_contracts_are_rejected():
    arrays, manifest = _bundle()
    arrays["is_terminal"][0, 1] = True
    with pytest.raises(ValueError, match="terminal"):
        validate_dataset(arrays, manifest)
    arrays["is_last"][0, 1] = True
    arrays["timestep"][1, 2] = 99
    with pytest.raises(ValueError, match="consecutive"):
        validate_dataset(arrays, manifest)


def test_trajectory_bootstrap_uses_trajectory_means():
    values = np.asarray([0.0, 0.0, 2.0, 2.0])
    trajectory = np.asarray([10, 10, 11, 11])
    result = trajectory_bootstrap_interval(
        values, trajectory, seed=0, samples=1000
    )
    assert result["mean"] == 1.0
    assert result["low"] <= 1.0 <= result["high"]
