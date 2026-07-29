"""Normalize upstream NE-Dreamer metrics into shared benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from world_marl.baselines.dreamerv3.artifacts import summarize_returns


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise
        if isinstance(value, dict):
            records.append(value)
    return records


def _series(
    records: Iterable[dict[str, Any]], key: str, *, max_steps: int
) -> list[dict[str, int | float]]:
    result = []
    for record in records:
        if key not in record or "step" not in record:
            continue
        step = int(record["step"])
        if step <= max_steps:
            result.append(
                {
                    "real_environment_transitions": step,
                    "episode_return": float(record[key]),
                }
            )
    return result


def _bin_curve(
    episodes: Iterable[dict[str, int | float]], *, bin_size: int
) -> list[dict[str, int | float]]:
    buckets: dict[int, list[float]] = {}
    for episode in episodes:
        step = int(episode["real_environment_transitions"])
        end = ((max(step, 1) - 1) // bin_size + 1) * bin_size
        buckets.setdefault(end, []).append(float(episode["episode_return"]))
    return [
        {
            "real_environment_transitions": end,
            "episode_return_mean": float(np.mean(buckets[end])),
            "episode_return_std": float(np.std(buckets[end])),
            "episodes": len(buckets[end]),
        }
        for end in sorted(buckets)
    ]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def normalize_training_artifacts(
    experiment_dir: str | Path,
    *,
    upstream_logdir: str | Path,
    task: str,
    seed: int,
    train_steps_budget: int,
    bin_size: int = 10_000,
) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir)
    normalized_dir = experiment_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    upstream_logdir = Path(upstream_logdir)
    metrics_path = upstream_logdir / "metrics.jsonl"
    records = read_jsonl(metrics_path)
    training = _series(records, "episode/score", max_steps=train_steps_budget)
    evaluation = _series(records, "episode/eval_score", max_steps=train_steps_budget)
    curve = _bin_curve(training, bin_size=bin_size)
    training_returns = [float(item["episode_return"]) for item in training]
    evaluation_returns = [float(item["episode_return"]) for item in evaluation]
    summary = {
        "implementation": "corl-team/nedreamer",
        "task": task,
        "seed": seed,
        "train_real_transition_budget": train_steps_budget,
        "max_logged_train_real_transitions": max(
            (int(item["real_environment_transitions"]) for item in training),
            default=0,
        ),
        "native_action_repeat": 2,
        "online_training_episodes": summarize_returns(training_returns),
        "last_20_online_training_episodes": summarize_returns(training_returns[-20:]),
        "periodic_deterministic_evaluations": summarize_returns(evaluation_returns),
        "latest_periodic_deterministic_evaluation": (
            evaluation[-1] if evaluation else None
        ),
        "held_out_final_evaluation": None,
        "latest_checkpoint": str(upstream_logdir / "latest.pt"),
        "score_source": str(metrics_path),
        "curve_bin_size": bin_size,
    }
    _write_jsonl(normalized_dir / "training_episodes.jsonl", training)
    _write_jsonl(normalized_dir / "periodic_evaluations.jsonl", evaluation)
    _write_json(normalized_dir / "training_curve.json", curve)
    _write_json(normalized_dir / "training_summary.json", summary)
    return summary
