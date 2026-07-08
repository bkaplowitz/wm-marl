from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from world_marl.world_model_foundation.replay import (
    WorldModelSequenceBatch,
    synthetic_observation_batch,
)


def synthetic_sequence_collector(
    *,
    env_name: str,
    time_steps: int,
    batch_size: int,
    observation_shape: tuple[int, ...],
    action_dim: int,
) -> WorldModelSequenceBatch:
    batch = synthetic_observation_batch(
        time_steps=time_steps,
        batch_size=batch_size,
        observation_shape=observation_shape,
        action_dim=action_dim,
    )
    metadata = dict(batch.metadata)
    metadata.update({"collector": "synthetic_sequence_collector", "env": env_name})
    return WorldModelSequenceBatch(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        continues=batch.continues,
        is_first=batch.is_first,
        is_terminal=batch.is_terminal,
        metadata=metadata,
    )


def write_json_artifact(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    return path


def write_jsonl_metrics(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return path
