"""Complete phase-boundary snapshots for matched JEPA training branches."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from world_marl.checkpointing import (
    load_pytree,
    load_train_state,
    save_pytree,
    save_train_state,
)
from world_marl.jepa.replay import SequenceReplayBuffer
from world_marl.jepa.reproducibility import JaxRngStreams, NumpyRngStreams


@dataclass(frozen=True)
class LoadedTrainingSnapshot:
    """Objects restored from a complete training snapshot."""

    train_state: Any
    policy_bundle_ema: Any | None
    replays: dict[str, SequenceReplayBuffer | None]
    observations: np.ndarray
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


def save_training_snapshot(
    snapshot_dir: str | Path,
    *,
    train_state: Any,
    policy_bundle_ema: Any | None,
    replays: Mapping[str, SequenceReplayBuffer | None],
    observations: np.ndarray,
    arrays: Mapping[str, Any] | None,
    adapter: Any,
    jax_rng_streams: Mapping[str, JaxRngStreams],
    numpy_rng_streams: Mapping[str, NumpyRngStreams],
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically persist all state required to continue a matched DMC branch."""

    destination = Path(snapshot_dir)
    if destination.exists():
        raise FileExistsError(f"training snapshot already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.incomplete")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        save_train_state(temporary / "train_state.msgpack", train_state)
        has_policy_bundle_ema = policy_bundle_ema is not None
        if has_policy_bundle_ema:
            save_pytree(temporary / "policy_bundle_ema.msgpack", policy_bundle_ema)

        replay_names = []
        absent_replay_names = []
        for name, replay in replays.items():
            if replay is None:
                absent_replay_names.append(name)
                continue
            replay.save_npz(temporary / f"replay_{name}.npz")
            replay_names.append(name)

        np.save(temporary / "observations.npy", np.asarray(observations))
        saved_arrays = {} if arrays is None else {
            name: np.asarray(value) for name, value in arrays.items()
        }
        with (temporary / "arrays.npz").open("wb") as handle:
            np.savez_compressed(handle, **saved_arrays)
        if not hasattr(adapter, "save_state_npz"):
            raise TypeError("training snapshots require a stateful vector adapter")
        adapter.save_state_npz(temporary / "environment_state.npz")

        payload = {
            "format_version": 1,
            "has_policy_bundle_ema": has_policy_bundle_ema,
            "replay_names": sorted(replay_names),
            "absent_replay_names": sorted(absent_replay_names),
            "jax_rng_streams": {
                name: streams.state_dict()
                for name, streams in jax_rng_streams.items()
            },
            "numpy_rng_streams": {
                name: streams.state_dict()
                for name, streams in numpy_rng_streams.items()
            },
            "metadata": _json_compatible(metadata),
        }
        (temporary / "snapshot.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (temporary / "done").touch()
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_training_snapshot(
    snapshot_dir: str | Path,
    *,
    target_train_state: Any,
    target_policy_bundle_ema: Any,
    adapter: Any,
    jax_rng_streams: Mapping[str, JaxRngStreams],
    numpy_rng_streams: Mapping[str, NumpyRngStreams],
) -> LoadedTrainingSnapshot:
    """Restore a complete snapshot into initialized learner and RNG templates."""

    source = Path(snapshot_dir)
    if not (source / "done").is_file():
        raise FileNotFoundError(f"training snapshot is incomplete: {source}")
    payload = json.loads((source / "snapshot.json").read_text(encoding="utf-8"))
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError(
            f"unsupported training snapshot format: {payload.get('format_version')}"
        )

    expected_jax_streams = set(jax_rng_streams)
    saved_jax_streams = set(payload["jax_rng_streams"])
    expected_numpy_streams = set(numpy_rng_streams)
    saved_numpy_streams = set(payload["numpy_rng_streams"])
    if saved_jax_streams != expected_jax_streams:
        raise ValueError("training snapshot JAX stream names do not match")
    if saved_numpy_streams != expected_numpy_streams:
        raise ValueError("training snapshot NumPy stream names do not match")
    for name, streams in jax_rng_streams.items():
        streams.restore_state_dict(payload["jax_rng_streams"][name])
    for name, streams in numpy_rng_streams.items():
        streams.restore_state_dict(payload["numpy_rng_streams"][name])

    train_state = load_train_state(source / "train_state.msgpack", target_train_state)
    policy_bundle_ema = None
    if payload["has_policy_bundle_ema"]:
        policy_bundle_ema = load_pytree(
            source / "policy_bundle_ema.msgpack",
            target_policy_bundle_ema,
        )

    replays: dict[str, SequenceReplayBuffer | None] = {
        name: SequenceReplayBuffer.load_npz(source / f"replay_{name}.npz")
        for name in payload["replay_names"]
    }
    replays.update({name: None for name in payload["absent_replay_names"]})
    observations = np.load(source / "observations.npy", allow_pickle=False)
    with np.load(source / "arrays.npz", allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    if not hasattr(adapter, "load_state_npz"):
        raise TypeError("training snapshots require a stateful vector adapter")
    adapter.load_state_npz(source / "environment_state.npz")
    return LoadedTrainingSnapshot(
        train_state=train_state,
        policy_bundle_ema=policy_bundle_ema,
        replays=replays,
        observations=np.asarray(observations),
        arrays=arrays,
        metadata=dict(payload["metadata"]),
    )


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        import jax

        if isinstance(value, jax.Array):
            return np.asarray(jax.device_get(value)).tolist()
    except ImportError:
        pass
    return value
