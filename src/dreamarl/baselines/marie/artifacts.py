"""Normalize official MARIE evaluation artifacts without modifying upstream."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .config import PAPER_GATE_MAPS


def read_official_result(path: str | Path) -> list[dict[str, float | int]]:
    """Read the trusted pickle emitted by the pinned official source."""
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    required = {"steps", "eval_win_rates", "eval_returns"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("invalid MARIE result artifact")
    steps = np.asarray(payload["steps"]).reshape(-1)
    win_rates = np.asarray(payload["eval_win_rates"]).reshape(-1)
    returns = np.asarray(payload["eval_returns"]).reshape(-1)
    if not (len(steps) == len(win_rates) == len(returns)):
        raise ValueError("MARIE result arrays have inconsistent lengths")
    if len(steps) and np.any(np.diff(steps) <= 0):
        raise ValueError("MARIE evaluation steps must be strictly increasing")
    return [
        {
            "env_steps": int(step),
            "eval_win_rate": float(win_rate),
            "eval_return": float(episode_return),
        }
        for step, win_rate, episode_return in zip(
            steps, win_rates, returns, strict=True
        )
    ]


def _normalized_auc(rows: list[dict[str, float | int]], budget: int) -> float | None:
    if not rows:
        return None
    xs = np.asarray([0, *[row["env_steps"] for row in rows]], dtype=np.float64)
    ys = np.asarray([0.0, *[row["eval_win_rate"] for row in rows]], dtype=np.float64)
    if xs[-1] < budget:
        xs = np.append(xs, budget)
        ys = np.append(ys, ys[-1])
    within = xs <= budget
    xs = xs[within]
    ys = ys[within]
    if xs[-1] < budget:
        xs = np.append(xs, budget)
        ys = np.append(ys, ys[-1])
    return float(np.trapz(ys, xs) / budget)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def normalize_training_artifacts(
    experiment_dir: str | Path,
    *,
    result_path: str | Path,
    map_name: str,
    cli_seed: int,
    steps_budget: int,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Copy upstream outputs and emit a stable, source-independent summary."""
    experiment_dir = Path(experiment_dir)
    result_path = Path(result_path)
    upstream_dir = experiment_dir / "upstream"
    normalized_dir = experiment_dir / "normalized"
    upstream_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    preserved_result = upstream_dir / "marie_results.pkl"
    shutil.copy2(result_path, preserved_result)
    preserved_checkpoint = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        preserved_checkpoint = upstream_dir / "model_final.pth"
        shutil.copy2(checkpoint_path, preserved_checkpoint)

    rows = read_official_result(preserved_result)
    latest = rows[-1] if rows else None
    best = max(rows, key=lambda row: row["eval_win_rate"]) if rows else None
    summary = {
        "implementation": "breez3young/MARIE",
        "map_name": map_name,
        "cli_seed": cli_seed,
        "effective_internal_seed": 23 + 100 * cli_seed,
        "real_environment_steps_budget": steps_budget,
        "last_recorded_environment_steps": latest["env_steps"] if latest else 0,
        "evaluation_points": len(rows),
        "latest_evaluation": latest,
        "best_evaluation": best,
        "normalized_win_rate_auc": _normalized_auc(rows, steps_budget),
        "paper_four_seed_mean_win_rate": PAPER_GATE_MAPS[map_name]["mean_win_rate"],
        "upstream_result": str(preserved_result),
        "upstream_checkpoint": (
            str(preserved_checkpoint) if preserved_checkpoint else None
        ),
    }
    _write_jsonl(normalized_dir / "periodic_evaluations.jsonl", rows)
    _write_json(normalized_dir / "training_summary.json", summary)
    return summary
