"""Versioned transition datasets for DreaMARL mechanism discovery.

The contract is intentionally independent of any proposed multi-agent model.
Every row describes the causal transition ``(belief_t, joint_action_t)`` to a
stopped representation target at ``t + 1``.  Complete trajectories are the
smallest unit that may be assigned to train, validation, or test.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


SCHEMA_NAME = "dreamarl_transition_ladder"
SCHEMA_VERSION = 1

REQUIRED_ARRAYS = frozenset(
    {
        "trajectory_id",
        "episode_id",
        "timestep",
        "belief",
        "action",
        "next_target",
        "reward",
        "is_last",
        "is_terminal",
        "valid",
        "agent_valid",
        "action_available",
        "track_id",
    }
)


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    arrays: dict[str, np.ndarray]
    manifest: dict[str, object]


@dataclass(frozen=True, slots=True)
class TrajectorySplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def manifest_path(dataset_path: Path) -> Path:
    return dataset_path.with_suffix(dataset_path.suffix + ".json")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_dataset(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, object],
) -> None:
    path = path.expanduser().resolve()
    normalized = {key: np.asarray(value) for key, value in arrays.items()}
    normalized_manifest = dict(manifest)
    normalized_manifest.update(
        schema=SCHEMA_NAME,
        schema_version=SCHEMA_VERSION,
        temporal_contract="(belief_t,joint_action_t)->stopped_target_t_plus_1",
    )
    validate_dataset(normalized, normalized_manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **normalized)
    normalized_manifest["dataset_sha256"] = sha256_file(path)
    manifest_path(path).write_text(
        json.dumps(normalized_manifest, indent=2, sort_keys=True) + "\n"
    )


def load_dataset(path: Path) -> DatasetBundle:
    path = path.expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    sidecar = manifest_path(path)
    if not sidecar.exists():
        raise FileNotFoundError(f"missing dataset manifest: {sidecar}")
    manifest = json.loads(sidecar.read_text())
    expected = manifest.get("dataset_sha256")
    if expected and expected != sha256_file(path):
        raise ValueError(f"dataset hash does not match {sidecar}")
    validate_dataset(arrays, manifest)
    return DatasetBundle(arrays, manifest)


def validate_dataset(
    arrays: Mapping[str, np.ndarray], manifest: Mapping[str, object]
) -> None:
    missing = REQUIRED_ARRAYS - arrays.keys()
    if missing:
        raise ValueError(f"dataset is missing arrays: {sorted(missing)}")
    if manifest.get("schema") != SCHEMA_NAME:
        raise ValueError(f"unexpected schema: {manifest.get('schema')!r}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema version: {manifest.get('schema_version')!r}"
        )
    if manifest.get("temporal_contract") != (
        "(belief_t,joint_action_t)->stopped_target_t_plus_1"
    ):
        raise ValueError("dataset does not declare the required temporal contract")

    trajectory = np.asarray(arrays["trajectory_id"])
    if trajectory.ndim != 1 or len(np.unique(trajectory)) != trajectory.size:
        raise ValueError("trajectory_id must contain one unique ID per trajectory")
    trajectories = trajectory.size
    time = np.asarray(arrays["timestep"]).shape[1]
    leading_2d = {
        "timestep",
        "reward",
        "is_last",
        "is_terminal",
        "valid",
    }
    for key in leading_2d:
        value = np.asarray(arrays[key])
        if value.shape[:2] != (trajectories, time):
            raise ValueError(f"{key} has invalid leading axes: {value.shape}")
    episode = np.asarray(arrays["episode_id"])
    if episode.shape != (trajectories,):
        raise ValueError(f"episode_id has invalid shape: {episode.shape}")

    belief = np.asarray(arrays["belief"])
    action = np.asarray(arrays["action"])
    target = np.asarray(arrays["next_target"])
    if belief.ndim != 4 or target.ndim != 4:
        raise ValueError((belief.shape, target.shape))
    if belief.shape[:3] != target.shape[:3]:
        raise ValueError((belief.shape, target.shape))
    if action.shape[:3] != belief.shape[:3]:
        raise ValueError((action.shape, belief.shape))
    agents = belief.shape[2]
    for key in ("agent_valid", "action_available", "track_id"):
        value = np.asarray(arrays[key])
        if value.shape != (trajectories, time, agents):
            raise ValueError(f"{key} has invalid shape: {value.shape}")
    if "observation" in arrays:
        observation = np.asarray(arrays["observation"])
        if observation.shape[:3] != belief.shape[:3]:
            raise ValueError((observation.shape, belief.shape))
    if "oracle_state" in arrays:
        oracle = np.asarray(arrays["oracle_state"])
        if oracle.shape[:2] != (trajectories, time):
            raise ValueError((oracle.shape, (trajectories, time)))

    valid = np.asarray(arrays["valid"], bool)
    agent_valid = np.asarray(arrays["agent_valid"], bool)
    action_available = np.asarray(arrays["action_available"], bool)
    if np.any(agent_valid & ~valid[..., None]):
        raise ValueError("agent_valid cannot be true on an invalid transition")
    if np.any(action_available & ~agent_valid):
        raise ValueError("actions cannot be available for invalid agents")
    is_terminal = np.asarray(arrays["is_terminal"], bool)
    is_last = np.asarray(arrays["is_last"], bool)
    if np.any(is_terminal & ~is_last):
        raise ValueError("terminal transitions must also be last transitions")

    timestep = np.asarray(arrays["timestep"])
    differences = np.diff(timestep, axis=1)
    consecutive = valid[:, 1:] & valid[:, :-1]
    if np.any(differences[consecutive] != 1):
        raise ValueError("valid adjacent transitions must have consecutive timesteps")


def split_trajectories(
    arrays: Mapping[str, np.ndarray],
    *,
    seed: int,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> TrajectorySplit:
    """Split complete trajectories, stratified by policy checkpoint when present."""

    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")
    trajectory_ids = np.asarray(arrays["trajectory_id"])
    checkpoint = np.asarray(
        arrays.get("policy_checkpoint", np.zeros_like(trajectory_ids))
    )
    if checkpoint.shape != trajectory_ids.shape:
        raise ValueError((checkpoint.shape, trajectory_ids.shape))
    generator = np.random.default_rng(seed)
    partitions = {"train": [], "validation": [], "test": []}
    for label in np.unique(checkpoint):
        rows = np.flatnonzero(checkpoint == label)
        if rows.size < 5:
            raise ValueError(
                "each policy checkpoint requires at least five trajectories "
                f"for a three-way split; {label!r} has {rows.size}"
            )
        rows = generator.permutation(rows)
        validation_size = max(1, round(rows.size * validation_fraction))
        test_size = max(1, round(rows.size * test_fraction))
        partitions["validation"].extend(rows[:validation_size])
        partitions["test"].extend(rows[validation_size : validation_size + test_size])
        partitions["train"].extend(rows[validation_size + test_size :])
    result = TrajectorySplit(
        train=np.asarray(sorted(partitions["train"]), np.int32),
        validation=np.asarray(sorted(partitions["validation"]), np.int32),
        test=np.asarray(sorted(partitions["test"]), np.int32),
    )
    assigned = np.concatenate([result.train, result.validation, result.test])
    if sorted(assigned.tolist()) != list(range(trajectory_ids.size)):
        raise AssertionError("trajectory split is not exhaustive and disjoint")
    return result


def trajectory_bootstrap_interval(
    values: np.ndarray,
    trajectory: np.ndarray,
    *,
    seed: int,
    samples: int = 2_000,
    confidence: float = 0.95,
) -> dict[str, float]:
    values = np.asarray(values, np.float64)
    trajectory = np.asarray(trajectory)
    if values.shape != trajectory.shape:
        raise ValueError((values.shape, trajectory.shape))
    unique = np.unique(trajectory)
    if unique.size < 2:
        raise ValueError("trajectory bootstrap requires at least two trajectories")
    means = np.asarray([values[trajectory == item].mean() for item in unique])
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, means.size, (samples, means.size))
    estimates = means[indices].mean(1)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": float(means.mean()),
        "low": float(np.quantile(estimates, tail)),
        "high": float(np.quantile(estimates, 1.0 - tail)),
        "trajectories": int(unique.size),
    }


def valid_episode_starts(
    first: np.ndarray, last: np.ndarray, length: int, window: int
) -> list[int]:
    """Return replay windows that never cross an episode boundary."""

    first = np.asarray(first, bool)
    last = np.asarray(last, bool)
    if first.shape != (length,) or last.shape != (length,):
        raise ValueError((first.shape, last.shape, length))
    boundaries = np.flatnonzero(first)
    if boundaries.size == 0 or boundaries[0] != 0:
        boundaries = np.concatenate([np.asarray([0]), boundaries])
    starts = []
    for begin in boundaries:
        following = boundaries[boundaries > begin]
        stop = int(following[0]) if following.size else length
        episode_last = np.flatnonzero(last[begin:stop])
        if episode_last.size:
            stop = min(stop, int(begin + episode_last[0] + 1))
        if stop - begin >= window:
            starts.extend(range(int(begin), stop - window + 1))
    return starts
