from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
from flax import serialization

from world_marl.algs.ippo import IPPOConfig, create_train_state
from world_marl.jepa.replay import SequenceReplayBuffer
from world_marl.jepa.reproducibility import JaxRngStreams, NumpyRngStreams
from world_marl.jepa.training_snapshot import (
    load_training_snapshot,
    save_training_snapshot,
)


class _SnapshotAdapter:
    def __init__(self, value: float):
        self.value = float(value)

    def save_state_npz(self, path: str | Path) -> Path:
        path = Path(path)
        with path.open("wb") as handle:
            np.savez_compressed(handle, value=np.asarray(self.value))
        return path

    def load_state_npz(self, path: str | Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            self.value = float(np.asarray(data["value"]).item())


def _replay() -> SequenceReplayBuffer:
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
    )
    return replay


def test_complete_training_snapshot_round_trip(tmp_path):
    state = create_train_state(jax.random.PRNGKey(3), (8, 8, 3), 3, IPPOConfig())
    replay = _replay()
    adapter = _SnapshotAdapter(17.0)
    train_jax = JaxRngStreams.create(5, isolated=True)
    validation_jax = JaxRngStreams.create(7, isolated=True)
    train_numpy = NumpyRngStreams.create(11, isolated=True)
    validation_numpy = NumpyRngStreams.create(13, isolated=True)
    train_jax.take("world_model")
    train_numpy.get("policy_replay").normal(size=5)
    snapshot_dir = tmp_path / "snapshot"

    save_training_snapshot(
        snapshot_dir,
        train_state=state,
        policy_bundle_ema=state.params,
        replays={"main": replay, "recent": None},
        observations=np.arange(6, dtype=np.float32).reshape(2, 1, 3),
        arrays={"validation": np.asarray([3.0, 5.0], dtype=np.float32)},
        adapter=adapter,
        jax_rng_streams={"train": train_jax, "validation": validation_jax},
        numpy_rng_streams={
            "train": train_numpy,
            "validation": validation_numpy,
        },
        metadata={"train_env_steps": 150_528, "metric": np.float32(2.5)},
    )
    expected_jax = train_jax.take("world_model")
    expected_numpy = train_numpy.get("policy_replay").normal(size=4)

    target_state = create_train_state(
        jax.random.PRNGKey(99),
        (8, 8, 3),
        3,
        IPPOConfig(),
    )
    restored_adapter = _SnapshotAdapter(-1.0)
    restored_train_jax = JaxRngStreams.create(5, isolated=True)
    restored_validation_jax = JaxRngStreams.create(7, isolated=True)
    restored_train_numpy = NumpyRngStreams.create(11, isolated=True)
    restored_validation_numpy = NumpyRngStreams.create(13, isolated=True)
    loaded = load_training_snapshot(
        snapshot_dir,
        target_train_state=target_state,
        target_policy_bundle_ema=target_state.params,
        adapter=restored_adapter,
        jax_rng_streams={
            "train": restored_train_jax,
            "validation": restored_validation_jax,
        },
        numpy_rng_streams={
            "train": restored_train_numpy,
            "validation": restored_validation_numpy,
        },
    )

    assert serialization.to_bytes(loaded.train_state) == serialization.to_bytes(state)
    assert serialization.to_bytes(loaded.policy_bundle_ema) == serialization.to_bytes(
        state.params
    )
    assert loaded.replays["main"].fingerprint() == replay.fingerprint()
    assert loaded.replays["recent"] is None
    np.testing.assert_array_equal(
        loaded.observations,
        np.arange(6, dtype=np.float32).reshape(2, 1, 3),
    )
    assert loaded.metadata == {"metric": 2.5, "train_env_steps": 150_528}
    np.testing.assert_array_equal(
        loaded.arrays["validation"],
        np.asarray([3.0, 5.0], dtype=np.float32),
    )
    assert restored_adapter.value == 17.0
    np.testing.assert_array_equal(
        restored_train_jax.take("world_model"),
        expected_jax,
    )
    np.testing.assert_array_equal(
        restored_train_numpy.get("policy_replay").normal(size=4),
        expected_numpy,
    )
