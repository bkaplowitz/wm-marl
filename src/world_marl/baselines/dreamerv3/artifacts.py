"""Normalize upstream DreamerV3 JSONL into shared benchmark artifacts."""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from world_marl.baselines.dreamerv3.checkpoints import latest_checkpoint


@dataclass(frozen=True)
class EpisodeScore:
  env_steps: int
  episode_return: float

  def to_dict(self) -> dict[str, int | float]:
    return {
      "env_steps": self.env_steps,
      "episode_return": self.episode_return,
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
  """Read JSONL while tolerating a partially written final line."""
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


def extract_episode_scores(records: Iterable[dict[str, Any]]) -> list[EpisodeScore]:
  scores = []
  for record in records:
    if "episode/score" not in record:
      continue
    step = record.get("step", record.get("env_step", record.get("env_steps")))
    if step is None:
      continue
    scores.append(EpisodeScore(int(step), float(record["episode/score"])))
  return scores


def summarize_returns(values: Iterable[float]) -> dict[str, Any]:
  returns = np.asarray(list(values), dtype=np.float64)
  if not len(returns):
    return {
      "episodes": 0,
      "mean": None,
      "std": None,
      "median": None,
      "p10": None,
      "p90": None,
      "cvar10": None,
      "min": None,
      "max": None,
    }
  tail_size = max(1, math.ceil(0.1 * len(returns)))
  return {
    "episodes": int(len(returns)),
    "mean": float(returns.mean()),
    "std": float(returns.std()),
    "median": float(np.median(returns)),
    "p10": float(np.percentile(returns, 10)),
    "p90": float(np.percentile(returns, 90)),
    "cvar10": float(np.sort(returns)[:tail_size].mean()),
    "min": float(returns.min()),
    "max": float(returns.max()),
  }


def bin_episode_scores(
  scores: Iterable[EpisodeScore],
  *,
  bin_size: int = 10_000,
  max_steps: int | None = None,
) -> list[dict[str, Any]]:
  if bin_size < 1:
    raise ValueError("bin_size must be >= 1")
  buckets: dict[int, list[float]] = {}
  for score in scores:
    if max_steps is not None and score.env_steps > max_steps:
      continue
    end = ((max(score.env_steps, 1) - 1) // bin_size + 1) * bin_size
    buckets.setdefault(end, []).append(score.episode_return)
  return [
    {
      "env_steps": end,
      "episode_return_mean": float(np.mean(buckets[end])),
      "episode_return_std": float(np.std(buckets[end])),
      "episodes": len(buckets[end]),
    }
    for end in sorted(buckets)
  ]


def load_official_reference(
  upstream_root: str | Path,
  *,
  task: str,
) -> dict[str, Any] | None:
  path = Path(upstream_root) / "scores" / "dmc_proprio-dreamerv3.json.gz"
  if not path.exists():
    return None
  with gzip.open(path, "rt", encoding="utf-8") as handle:
    rows = [row for row in json.load(handle) if row.get("task") == task]
  if not rows:
    return None
  xs = np.asarray(rows[0]["xs"], dtype=np.float64)
  ys = np.asarray([row["ys"] for row in rows], dtype=np.float64)
  return {
    "task": task,
    "source": "danijar/dreamerv3 scores/dmc_proprio-dreamerv3.json.gz",
    "seeds": [int(row["seed"]) for row in rows],
    "env_steps": xs.astype(int).tolist(),
    "mean": np.mean(ys, axis=0).tolist(),
    "std": np.std(ys, axis=0).tolist(),
    "per_seed": {
      str(row["seed"]): [float(value) for value in row["ys"]]
      for row in rows
    },
  }


def _write_json(path: Path, payload: Any) -> None:
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, sort_keys=True) + "\n")


def _plot_training_curve(
  path: Path,
  curve: list[dict[str, Any]],
  reference: dict[str, Any] | None,
) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(figsize=(8, 4.5))
  if reference:
    xref = np.asarray(reference["env_steps"])
    mean = np.asarray(reference["mean"])
    std = np.asarray(reference["std"])
    ax.plot(xref, mean, label="Official DreamerV3 (5-seed mean)", color="#555555")
    ax.fill_between(xref, mean - std, mean + std, color="#999999", alpha=0.2)
  if curve:
    ax.plot(
      [row["env_steps"] for row in curve],
      [row["episode_return_mean"] for row in curve],
      label="This run (online episodes)",
      color="#0066cc",
      linewidth=2,
    )
  ax.set_xlabel("Training environment transitions")
  ax.set_ylabel("Episode return")
  ax.set_ylim(bottom=0)
  ax.grid(True, alpha=0.25)
  ax.legend()
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)


def normalize_training_artifacts(
  experiment_dir: str | Path,
  *,
  upstream_root: str | Path,
  task: str,
  seed: int,
  train_steps_budget: int,
  bin_size: int = 10_000,
) -> dict[str, Any]:
  experiment_dir = Path(experiment_dir)
  normalized_dir = experiment_dir / "normalized"
  normalized_dir.mkdir(parents=True, exist_ok=True)
  scores_path = experiment_dir / "upstream" / "scores.jsonl"
  scores = extract_episode_scores(read_jsonl(scores_path))
  curve = bin_episode_scores(scores, bin_size=bin_size, max_steps=train_steps_budget)
  reference = load_official_reference(upstream_root, task=task)
  final_scores = [score.episode_return for score in scores[-20:]]
  checkpoint_path = None
  checkpoint_env_steps = None
  try:
    checkpoint = latest_checkpoint(experiment_dir / "upstream" / "ckpt")
    checkpoint_path = str(checkpoint.path)
    checkpoint_env_steps = checkpoint.env_steps
  except FileNotFoundError:
    pass
  summary = {
    "implementation": "danijar/dreamerv3",
    "task": task,
    "seed": seed,
    "train_env_steps_budget": train_steps_budget,
    "max_logged_train_env_steps": max(
      (score.env_steps for score in scores), default=0
    ),
    "latest_checkpoint": checkpoint_path,
    "latest_checkpoint_train_env_steps": checkpoint_env_steps,
    "online_training_episodes": summarize_returns(
      score.episode_return for score in scores
    ),
    "last_20_online_training_episodes": summarize_returns(final_scores),
    "held_out_evaluation": None,
    "score_source": str(scores_path),
    "curve_bin_size": bin_size,
  }
  _write_jsonl(
    normalized_dir / "training_episodes.jsonl",
    [score.to_dict() for score in scores],
  )
  _write_json(normalized_dir / "training_curve.json", curve)
  _write_json(normalized_dir / "training_summary.json", summary)
  if reference:
    _write_json(normalized_dir / "official_reference.json", reference)
  _plot_training_curve(normalized_dir / "training_curve.png", curve, reference)
  return summary


def normalize_evaluation_artifacts(
  eval_dir: str | Path,
  *,
  requested_episodes: int,
  train_env_steps: int,
  success_threshold: float | None = None,
) -> dict[str, Any]:
  eval_dir = Path(eval_dir)
  scores = extract_episode_scores(read_jsonl(eval_dir / "upstream" / "scores.jsonl"))
  scores = scores[:requested_episodes]
  returns = [score.episode_return for score in scores]
  observed_eval_steps = max((score.env_steps for score in scores), default=0)
  summary = {
    "requested_episodes": requested_episodes,
    "completed_episodes": len(scores),
    "returns": returns,
    "statistics": summarize_returns(returns),
    "success_threshold": success_threshold,
    "success_rate": (
      float(np.mean(np.asarray(returns) >= success_threshold))
      if returns and success_threshold is not None
      else None
    ),
    "train_env_steps": train_env_steps,
    "eval_env_steps": observed_eval_steps,
    "total_real_env_steps": train_env_steps + observed_eval_steps,
  }
  _write_json(eval_dir / "evaluation_summary.json", summary)
  _write_jsonl(
    eval_dir / "evaluation_episodes.jsonl",
    [score.to_dict() for score in scores],
  )
  return summary
