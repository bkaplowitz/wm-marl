from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from world_marl.dreamarl.collect import collect_coin_game_sequence
from world_marl.dreamarl.config import ReplayConfig
from world_marl.dreamarl.replay import JointSequenceReplay


def _chunk(seed: int = 0):
    return collect_coin_game_sequence(
        time_steps=12,
        num_envs=3,
        max_cycles=4,
        seed=seed,
    )


def test_replay_samples_contiguous_single_lane_windows() -> None:
    chunk = _chunk()
    replay = JointSequenceReplay(
        ReplayConfig(capacity=48, sequence_length=4, batch_size=5), seed=10
    )
    replay.append(chunk)
    sample = replay.sample()
    assert sample.observations.shape[:3] == (4, 5, 2)
    assert np.all(sample.is_first[0])
    assert np.all(sample.is_first[1:] == sample.is_last[:-1])
    np.testing.assert_allclose(
        sample.next_observations[:-1][~sample.is_last[:-1]],
        sample.observations[1:][~sample.is_last[:-1]],
    )


def test_replay_wraparound_and_snapshot_restore_are_exact(tmp_path) -> None:
    first = _chunk(1)
    second = _chunk(2)
    second.is_first[0] = True
    first.is_last[-1] = True
    replay = JointSequenceReplay(
        ReplayConfig(capacity=24, sequence_length=4, batch_size=3), seed=11
    )
    replay.append(first)
    replay.append(second)
    assert replay.size == 24

    state = replay.state_dict()
    expected = replay.sample()
    restored = JointSequenceReplay(state["config"], seed=999)
    restored.load_state_dict(state)
    actual = restored.sample()
    for name in state["storage"]:
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))

    replay.save(tmp_path)
    expected_disk = replay.sample()
    disk_restored = JointSequenceReplay(state["config"], seed=0)
    disk_restored.load(tmp_path)
    disk_actual = disk_restored.sample()
    for name in state["storage"]:
        np.testing.assert_array_equal(
            getattr(disk_actual, name), getattr(expected_disk, name)
        )


def test_replay_rejects_cross_chunk_lifecycle_mismatch() -> None:
    chunk = _chunk(3)
    config = ReplayConfig(capacity=48, sequence_length=4, batch_size=2)
    replay = JointSequenceReplay(config, seed=12)
    replay.append(chunk)
    bad = _chunk(4)
    bad.is_first[0] = False
    with pytest.raises(ValueError, match="cross-chunk lifecycle"):
        replay.append(bad)


def test_capacity_must_cover_sequence_in_each_environment_lane() -> None:
    with pytest.raises(ValueError, match="divided across env lanes"):
        JointSequenceReplay(
            replace(ReplayConfig(), capacity=12, sequence_length=8), seed=0
        ).append(_chunk())
