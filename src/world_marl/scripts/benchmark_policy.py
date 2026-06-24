"""Compare model-free and world-model policy training runs."""

from __future__ import annotations

from typing import Any


LOSS_KEYS = (
    "ppo/total_loss",
    "ppo/actor_loss",
    "ppo/value_loss",
    "ppo/entropy",
)


def loss_at_episode_checkpoints(
    rows: list[dict[str, Any]],
    checkpoints: list[int],
) -> dict[str, dict[str, Any] | None]:
    result: dict[str, dict[str, Any] | None] = {}
    for checkpoint in checkpoints:
        selected = None
        for row in rows:
            if int(row.get("cumulative_real_episodes", 0)) >= checkpoint:
                selected = row
                break
        if selected is None:
            result[str(checkpoint)] = None
            continue
        result[str(checkpoint)] = {
            "checkpoint": checkpoint,
            "actual_real_episodes": int(selected["cumulative_real_episodes"]),
            "update": int(selected["update"]),
            **{key: selected.get(key) for key in LOSS_KEYS},
        }
    return result
