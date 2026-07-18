"""Pure reporting and environment-step accounting for JEPA experiments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def return_tail_metrics(
    returns: list[float],
    *,
    failure_threshold: float,
    success_threshold: float,
) -> dict[str, Any]:
    """Summarize lower-tail failures and solved episodes."""

    if not returns:
        return {
            "failure_return_threshold": float(failure_threshold),
            "success_return_threshold": float(success_threshold),
            "failure_count": 0,
            "failure_rate": None,
            "success_count": 0,
            "success_rate": None,
            "return_min": None,
            "return_max": None,
            "return_p05": None,
            "return_p10": None,
            "return_p25": None,
            "return_cvar10": None,
            "nonfailure_mean_return": None,
        }
    values = np.asarray(returns, dtype=np.float32)
    failures = values < float(failure_threshold)
    successes = values >= float(success_threshold)
    tail_count = max(1, int(math.ceil(0.10 * values.size)))
    nonfailures = values[~failures]
    return {
        "failure_return_threshold": float(failure_threshold),
        "success_return_threshold": float(success_threshold),
        "failure_count": int(np.sum(failures)),
        "failure_rate": float(np.mean(failures)),
        "success_count": int(np.sum(successes)),
        "success_rate": float(np.mean(successes)),
        "return_min": float(np.min(values)),
        "return_max": float(np.max(values)),
        "return_p05": float(np.quantile(values, 0.05)),
        "return_p10": float(np.quantile(values, 0.10)),
        "return_p25": float(np.quantile(values, 0.25)),
        "return_cvar10": float(np.mean(np.sort(values)[:tail_count])),
        "nonfailure_mean_return": (
            float(np.mean(nonfailures)) if nonfailures.size else None
        ),
    }


def collection_report_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "return_mean": metrics.get("mean_return"),
        "return_std": metrics.get("std_return"),
        "return_p10": metrics.get("return_p10"),
        "return_cvar10": metrics.get("return_cvar10"),
        "failure_rate": metrics.get("failure_rate"),
        "success_rate": metrics.get("success_rate"),
        "completed_episodes": metrics.get("completed_episodes", 0),
    }


def optional_value(payload: dict[str, Any] | None, key: str) -> Any:
    """Read a metric from an optional report payload."""

    return None if payload is None else payload.get(key)


def dreamer_style_training_score(
    online_history: list[dict[str, Any]],
    *,
    window_env_steps: int,
    budget_env_steps: int,
) -> dict[str, Any]:
    """Aggregate naturally completed training episodes in a trailing window."""

    enabled = window_env_steps > 0 and budget_env_steps > 0
    episodes: list[dict[str, Any]] = []
    for item in online_history:
        replay = item.get("actor_replay", {})
        returns = replay.get("returns") or []
        lengths = replay.get("lengths") or []
        finish_steps = replay.get("episode_finish_train_env_steps") or []
        if len(finish_steps) != len(returns):
            continue
        for index, (value, finish_step) in enumerate(zip(returns, finish_steps)):
            episodes.append(
                {
                    "online_iteration": item.get("iteration"),
                    "return": float(value),
                    "length": int(lengths[index]) if index < len(lengths) else None,
                    "finish_train_env_step": int(finish_step),
                }
            )

    final_step = max(
        (item["finish_train_env_step"] for item in episodes),
        default=None,
    )
    if not enabled or final_step is None:
        return {
            "enabled": enabled,
            "budget_env_steps": int(budget_env_steps),
            "window_env_steps": int(window_env_steps),
            "budget_reached": False,
            "final_train_env_step": final_step,
            "window_start_env_step": None,
            "window_end_env_step": None,
            "episodes": 0,
            "mean_return": None,
            "std_return": None,
            "returns": [],
            "episode_finish_train_env_steps": [],
        }
    budget_reached = final_step >= budget_env_steps
    window_end = budget_env_steps if budget_reached else final_step
    window_start = max(0, window_end - window_env_steps)
    selected = [
        item
        for item in episodes
        if window_start < item["finish_train_env_step"] <= window_end
    ]
    selected_returns = [item["return"] for item in selected]
    return {
        "enabled": True,
        "budget_env_steps": int(budget_env_steps),
        "window_env_steps": int(window_env_steps),
        "budget_reached": bool(budget_reached),
        "final_train_env_step": int(final_step),
        "window_start_env_step": int(window_start),
        "window_end_env_step": int(window_end),
        "episodes": len(selected_returns),
        "mean_return": (float(np.mean(selected_returns)) if selected_returns else None),
        "std_return": float(np.std(selected_returns)) if selected_returns else None,
        "returns": selected_returns,
        "episode_finish_train_env_steps": [
            item["finish_train_env_step"] for item in selected
        ],
        "episode_records": selected,
    }


def real_step_accounting(
    *,
    initial_train_env_steps: int,
    validation_env_steps: int,
    online_history: list[dict[str, Any]],
    final_policy_eval: dict[str, Any] | None,
) -> dict[str, int]:
    """Separate learning, validation, and measurement-only interactions."""

    online_env_steps = sum(
        int(item["actor_replay"]["env_steps"]) for item in online_history
    )
    curve_eval_env_steps = sum(
        int(optional_value(item.get("policy_evaluation"), "env_steps") or 0)
        for item in online_history
    )
    curve_completed_eval_steps = sum(
        int(
            optional_value(
                item.get("policy_evaluation"),
                "completed_episode_steps",
            )
            or 0
        )
        for item in online_history
    )
    policy_eval_env_steps = curve_eval_env_steps + int(
        optional_value(final_policy_eval, "env_steps") or 0
    )
    completed_eval_steps = (
        int(optional_value(final_policy_eval, "completed_episode_steps") or 0)
        + curve_completed_eval_steps
    )
    train_env_steps = initial_train_env_steps + online_env_steps
    return {
        "real_initial_train_replay_env_steps": int(initial_train_env_steps),
        "real_online_actor_replay_env_steps": int(online_env_steps),
        "real_train_replay_env_steps": int(train_env_steps),
        "real_validation_replay_env_steps": int(validation_env_steps),
        "real_train_plus_validation_env_steps": int(
            train_env_steps + validation_env_steps
        ),
        "real_policy_eval_env_steps": policy_eval_env_steps,
        "real_policy_eval_completed_episode_steps": completed_eval_steps,
        "real_total_env_steps": int(
            train_env_steps + validation_env_steps + policy_eval_env_steps
        ),
    }
