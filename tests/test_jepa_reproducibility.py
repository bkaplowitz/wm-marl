from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from world_marl.jepa.replay import SequenceReplayBuffer
from world_marl.jepa.reproducibility import (
    JaxRngStreams,
    NumpyRngStreams,
    fingerprint_pytree,
)


def test_isolated_jax_streams_do_not_depend_on_other_stream_consumption():
    interleaved = JaxRngStreams.create(7, isolated=True)
    model_first = interleaved.take("world_model")
    interleaved.take("evaluation")
    model_second = interleaved.take("world_model")

    model_only = JaxRngStreams.create(7, isolated=True)
    expected_first = model_only.take("world_model")
    expected_second = model_only.take("world_model")

    np.testing.assert_array_equal(model_first, expected_first)
    np.testing.assert_array_equal(model_second, expected_second)


def test_isolated_numpy_streams_do_not_depend_on_other_stream_consumption():
    interleaved = NumpyRngStreams.create(11, isolated=True)
    model_first = interleaved.get("world_model_replay").integers(0, 10_000)
    interleaved.get("online_collection").integers(0, 10_000, size=100)
    model_second = interleaved.get("world_model_replay").integers(0, 10_000)

    model_only = NumpyRngStreams.create(11, isolated=True)
    expected_first = model_only.get("world_model_replay").integers(0, 10_000)
    expected_second = model_only.get("world_model_replay").integers(0, 10_000)

    assert model_first == expected_first
    assert model_second == expected_second


def test_replay_fingerprint_survives_round_trip_and_detects_changes(tmp_path):
    replay = SequenceReplayBuffer(
        capacity=8,
        num_envs=2,
        observation_shape=(3,),
        action_shape=(1,),
        action_dtype=np.float32,
    )
    replay.add_step(
        observations=np.arange(6, dtype=np.float32).reshape(2, 3),
        actions=np.asarray([[0.25], [-0.5]], dtype=np.float32),
        rewards=np.asarray([1.0, 0.0], dtype=np.float32),
        is_last=np.asarray([0.0, 1.0], dtype=np.float32),
        is_terminal=np.asarray([0.0, 1.0], dtype=np.float32),
        cuts=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    expected = replay.fingerprint()
    path = tmp_path / "replay.npz"
    replay.save_npz(path)

    restored = SequenceReplayBuffer.load_npz(path)
    assert restored.fingerprint() == expected
    assert restored.cut_count == 1
    np.testing.assert_array_equal(restored.cuts[0], np.asarray([1.0, 0.0]))

    restored.add_step(
        observations=np.zeros((2, 3), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.float32),
        rewards=np.zeros(2, dtype=np.float32),
        is_last=np.zeros(2, dtype=np.float32),
        is_terminal=np.zeros(2, dtype=np.float32),
    )
    assert restored.fingerprint() != expected


def test_replay_can_materialize_pre_generated_sample_indices():
    replay = SequenceReplayBuffer(
        capacity=12,
        num_envs=2,
        observation_shape=(2,),
        action_shape=(1,),
        action_dtype=np.float32,
    )
    for step in range(10):
        replay.add_step(
            observations=np.asarray(
                [[step, 0], [step, 1]],
                dtype=np.float32,
            ),
            actions=np.asarray([[step], [-step]], dtype=np.float32),
            rewards=np.asarray([step, step + 1], dtype=np.float32),
            is_last=np.zeros(2, dtype=np.float32),
            is_terminal=np.zeros(2, dtype=np.float32),
        )

    first_rng = np.random.default_rng(23)
    starts, envs = replay.sample_indices(
        first_rng,
        batch_size=4,
        chunk_length=3,
        max_horizon=2,
    )
    indexed = replay.sample_from_indices(
        starts,
        envs,
        chunk_length=3,
        max_horizon=2,
    )

    direct = replay.sample(
        np.random.default_rng(23),
        batch_size=4,
        chunk_length=3,
        max_horizon=2,
    )
    for field in (
        "observations",
        "actions",
        "rewards",
        "is_last",
        "is_terminal",
    ):
        np.testing.assert_array_equal(
            getattr(indexed, field),
            getattr(direct, field),
        )


def test_legacy_replay_migrates_done_to_explicit_boundary_fields(tmp_path):
    path = tmp_path / "legacy_replay.npz"
    observations = np.arange(12, dtype=np.float32).reshape(4, 1, 3)
    dones = np.asarray([[0.0], [1.0], [0.0], [0.0]], dtype=np.float32)
    np.savez_compressed(
        path,
        observations=observations,
        actions=np.zeros((4, 1, 1), dtype=np.float32),
        rewards=np.zeros((4, 1), dtype=np.float32),
        dones=dones,
        capacity=np.asarray(8, dtype=np.int64),
        num_envs=np.asarray(1, dtype=np.int64),
        observation_shape=np.asarray([3], dtype=np.int64),
        action_shape=np.asarray([1], dtype=np.int64),
        action_dtype=np.asarray(np.dtype(np.float32).str),
    )

    replay = SequenceReplayBuffer.load_npz(path)

    np.testing.assert_array_equal(replay.is_last[:4], dones)
    np.testing.assert_array_equal(replay.is_terminal[:4], dones)


def test_pytree_fingerprint_is_stable_and_value_sensitive():
    tree = {"a": jnp.asarray([1.0, 2.0]), "b": {"c": jnp.asarray(3)}}
    same = {"a": jnp.asarray([1.0, 2.0]), "b": {"c": jnp.asarray(3)}}
    changed = {"a": jnp.asarray([1.0, 2.1]), "b": {"c": jnp.asarray(3)}}

    assert fingerprint_pytree(tree) == fingerprint_pytree(same)
    assert fingerprint_pytree(tree) != fingerprint_pytree(changed)
