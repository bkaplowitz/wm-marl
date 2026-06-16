"""Artifact logging utilities."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import numpy as np


TRACKED_DISTRIBUTIONS = (
    "world-marl",
    "jax",
    "jaxlib",
    "jaxmarl",
    "dm-meltingpot",
    "dmlab2d",
    "shimmy",
    "flax",
    "distrax",
    "optax",
    "numpy",
)


def timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def dependency_versions() -> dict[str, str]:
    versions = {}
    for name in TRACKED_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    versions["jax_devices"] = ", ".join(str(device) for device in jax.devices())
    return versions


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


class RunLogger:
    """Writes JSON artifacts and metrics rows to a run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.run_dir / name
        path.write_text(
            json.dumps(to_jsonable(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def append_metrics(self, row: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")

    def plot_returns(
        self, rows: list[dict[str, Any]], filename: str = "returns.png"
    ) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = self.run_dir / filename
        if not rows:
            return path

        updates = [row["update"] for row in rows]
        rollout_rewards = [row.get("rollout_mean_reward", np.nan) for row in rows]
        episode_returns = [
            row.get("episode_return_mean", np.nan)
            if row.get("episode_return_mean") is not None
            else np.nan
            for row in rows
        ]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(updates, rollout_rewards, label="rollout mean reward")
        if not np.isnan(np.asarray(episode_returns, dtype=float)).all():
            ax.plot(updates, episode_returns, label="completed episode return")
        ax.set_xlabel("update")
        ax.set_ylabel("return / reward")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path
