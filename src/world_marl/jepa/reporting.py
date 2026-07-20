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
    final_bins: int = 3,
) -> dict[str, Any]:
    """Build the paper-style online score curve and final per-seed score.

    Training episodes are assigned to disjoint ``window_env_steps`` bins by
    their completion step. The per-seed final score is the unweighted mean of
    the final populated bins, matching the DreamerV3 DMC table aggregation.
    """

    enabled = window_env_steps > 0 and budget_env_steps > 0 and final_bins > 0
    episodes: list[dict[str, Any]] = []
    train_env_steps = 0
    for item in online_history:
        replay = item.get("actor_replay", {})
        train_env_steps = max(
            train_env_steps,
            int(replay.get("train_replay_total_env_steps") or 0),
        )
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

    final_episode_step = max(
        (item["finish_train_env_step"] for item in episodes),
        default=0,
    )
    train_env_steps = max(train_env_steps, final_episode_step)
    if not enabled or train_env_steps <= 0:
        return {
            "enabled": enabled,
            "budget_env_steps": int(budget_env_steps),
            "window_env_steps": int(window_env_steps),
            "final_bins": int(final_bins),
            "budget_reached": False,
            "final_train_env_step": train_env_steps or None,
            "final_episode_finish_env_step": final_episode_step or None,
            "window_start_env_step": None,
            "window_end_env_step": None,
            "episodes": 0,
            "mean_return": None,
            "std_return": None,
            "episode_mean_return": None,
            "episode_std_return": None,
            "returns": [],
            "episode_finish_train_env_steps": [],
            "selected_bin_end_env_steps": [],
            "selected_bin_means": [],
            "curve": [],
        }

    budget_reached = train_env_steps >= budget_env_steps
    curve_end = min(train_env_steps, budget_env_steps)
    bins: dict[int, list[dict[str, Any]]] = {}
    for episode in episodes:
        finish_step = episode["finish_train_env_step"]
        if finish_step <= 0 or finish_step > curve_end:
            continue
        bin_index = (finish_step - 1) // window_env_steps
        bins.setdefault(bin_index, []).append(episode)

    curve = []
    for bin_index in sorted(bins):
        records = bins[bin_index]
        values = [record["return"] for record in records]
        bin_start = bin_index * window_env_steps
        bin_end = min((bin_index + 1) * window_env_steps, budget_env_steps)
        curve.append(
            {
                "bin_index": int(bin_index),
                "bin_start_env_step": int(bin_start),
                "bin_end_env_step": int(bin_end),
                "episodes": len(values),
                "mean_return": float(np.mean(values)),
                "std_return": float(np.std(values)),
                "returns": values,
                "episode_finish_train_env_steps": [
                    record["finish_train_env_step"] for record in records
                ],
            }
        )

    selected_bins = curve[-final_bins:]
    selected_bin_means = [item["mean_return"] for item in selected_bins]
    selected_bin_ends = [item["bin_end_env_step"] for item in selected_bins]
    selected_bin_indices = {item["bin_index"] for item in selected_bins}
    selected = [
        episode
        for episode in episodes
        if (episode["finish_train_env_step"] - 1) // window_env_steps
        in selected_bin_indices
        and episode["finish_train_env_step"] <= curve_end
    ]
    selected_returns = [episode["return"] for episode in selected]
    window_start = (
        selected_bins[0]["bin_start_env_step"] if selected_bins else None
    )
    window_end = selected_bins[-1]["bin_end_env_step"] if selected_bins else None
    return {
        "enabled": True,
        "budget_env_steps": int(budget_env_steps),
        "window_env_steps": int(window_env_steps),
        "final_bins": int(final_bins),
        "budget_reached": bool(budget_reached),
        "final_train_env_step": int(train_env_steps),
        "final_episode_finish_env_step": (
            int(final_episode_step) if final_episode_step else None
        ),
        "window_start_env_step": (
            int(window_start) if window_start is not None else None
        ),
        "window_end_env_step": int(window_end) if window_end is not None else None,
        "episodes": len(selected_returns),
        "mean_return": (
            float(np.mean(selected_bin_means)) if selected_bin_means else None
        ),
        "std_return": (
            float(np.std(selected_bin_means)) if selected_bin_means else None
        ),
        "episode_mean_return": (
            float(np.mean(selected_returns)) if selected_returns else None
        ),
        "episode_std_return": (
            float(np.std(selected_returns)) if selected_returns else None
        ),
        "returns": selected_returns,
        "episode_finish_train_env_steps": [
            item["finish_train_env_step"] for item in selected
        ],
        "episode_records": selected,
        "selected_bin_end_env_steps": selected_bin_ends,
        "selected_bin_means": selected_bin_means,
        "curve": curve,
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
