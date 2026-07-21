"""Artifact logging utilities."""

from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import json
import math
import warnings
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
    "gymnax",
    "dm-control",
    "dm-meltingpot",
    "dmlab2d",
    "shimmy",
    "flax",
    "distrax",
    "optax",
    "numpy",
    "wandb",
    "imageio",
    "imageio-ffmpeg",
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
            if hasattr(value, "__array__"):
                return np.asarray(value).tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


@dataclasses.dataclass(frozen=True)
class WandbConfig:
    """Optional Weights & Biases run configuration."""

    project: str
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    mode: str = "online"
    config: dict[str, Any] = dataclasses.field(default_factory=dict)


def _write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    imageio = importlib.import_module("imageio.v3")
    imageio.imwrite(path, frames, fps=fps, codec="libx264", quality=8)


class RunLogger:
    """Writes JSON artifacts and metrics rows to a run directory."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        wandb_config: WandbConfig | None = None,
        wandb_run: Any = None,
    ) -> None:
        if wandb_config is not None and wandb_run is not None:
            raise ValueError("pass either wandb_config or wandb_run, not both")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._wandb = None
        self._wandb_run = None
        self._external_wandb_run = wandb_run is not None
        self._wandb_failed = False
        self._local_logging_failed = False
        self._row_index = 0
        self._train_env_steps: int | None = None
        if wandb_config is not None:
            self._init_wandb(wandb_config)
        elif wandb_run is not None:
            self._wandb_run = wandb_run

    @property
    def wandb_enabled(self) -> bool:
        return self._wandb_run is not None

    def _warn_wandb(self, action: str, error: Exception) -> None:
        if not self._wandb_failed:
            warnings.warn(
                f"W&B {action} failed; local logging will continue: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
        self._wandb_failed = True

    def _warn_local_logging(self, action: str, error: OSError) -> None:
        if not self._local_logging_failed:
            warnings.warn(
                f"Local artifact {action} failed; training and W&B logging will "
                f"continue: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
        self._local_logging_failed = True

    def _init_wandb(self, config: WandbConfig) -> None:
        try:
            self._wandb = importlib.import_module("wandb")
            self._wandb_run = self._wandb.init(
                project=config.project,
                entity=config.entity,
                name=config.name,
                group=config.group,
                tags=list(config.tags),
                mode=config.mode,
                dir=str(self.run_dir),
                config=to_jsonable(config.config),
            )
            define_metric = getattr(self._wandb_run, "define_metric", None)
            if define_metric is not None:
                define_metric("budget/train_env_steps")
                define_metric(
                    "*",
                    step_metric="budget/train_env_steps",
                    step_sync=True,
                )
        except Exception as error:  # W&B must never take down an experiment.
            self._wandb_run = None
            self._warn_wandb("initialization", error)

    @staticmethod
    def _flatten_scalars(
        payload: dict[str, Any],
        *,
        prefix: str = "",
    ) -> dict[str, int | float | str | bool]:
        flattened: dict[str, int | float | str | bool] = {}
        for key, value in payload.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(value, dict):
                flattened.update(RunLogger._flatten_scalars(value, prefix=name))
                continue
            value = to_jsonable(value)
            if isinstance(value, bool | int | str):
                flattened[name] = value
            elif isinstance(value, float) and math.isfinite(value):
                flattened[name] = value
        return flattened

    def _log_wandb(self, payload: dict[str, Any]) -> None:
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.log(payload)
        except Exception as error:  # W&B must never take down an experiment.
            self._warn_wandb("metric logging", error)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.run_dir / name
        try:
            path.write_text(
                json.dumps(to_jsonable(payload), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as error:
            self._warn_local_logging("JSON write", error)
        return path

    def append_metrics(self, row: dict[str, Any]) -> None:
        try:
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
        except OSError as error:
            self._warn_local_logging("metric write", error)
        if self._external_wandb_run:
            self._log_wandb(to_jsonable(row))
            return
        metrics = self._flatten_scalars(row)
        metrics["logger/row_index"] = self._row_index
        self._row_index += 1
        if self._train_env_steps is not None:
            metrics.setdefault("budget/train_env_steps", self._train_env_steps)
        elif "env_steps" in metrics:
            metrics["budget/train_env_steps"] = metrics["env_steps"]
        self._log_wandb(metrics)

    def set_train_env_steps(self, env_steps: int) -> None:
        """Set the real training-replay budget used by subsequent metric rows."""

        if env_steps < 0:
            raise ValueError("env_steps must be >= 0")
        self._train_env_steps = int(env_steps)

    def update_summary(self, payload: dict[str, Any]) -> None:
        """Mirror final scalar values to the W&B run summary when enabled."""

        if self._wandb_run is None:
            return
        try:
            self._wandb_run.summary.update(self._flatten_scalars(payload))
        except Exception as error:  # W&B must never take down an experiment.
            self._warn_wandb("summary update", error)

    def update_config(self, payload: dict[str, Any]) -> None:
        """Add resolved runtime values to the W&B run configuration."""

        if self._wandb_run is None:
            return
        try:
            self._wandb_run.config.update(
                to_jsonable(payload),
                allow_val_change=True,
            )
        except Exception as error:  # W&B must never take down an experiment.
            self._warn_wandb("config update", error)

    def log_image(self, key: str, path: str | Path, *, caption: str = "") -> None:
        if self._wandb_run is None or self._wandb is None:
            return
        try:
            self._wandb_run.log(
                {key: self._wandb.Image(str(path), caption=caption or None)}
            )
        except Exception as error:  # W&B must never take down an experiment.
            self._warn_wandb("image logging", error)

    def write_video(
        self,
        filename: str,
        frames: list[np.ndarray],
        *,
        fps: int,
        key: str,
        caption: str = "",
    ) -> Path | None:
        """Encode an MP4 locally and optionally mirror it to W&B."""

        if not frames:
            return None
        path = self.run_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            array = np.stack(frames).astype(np.uint8, copy=False)
            _write_mp4(path, array, fps)
        except Exception as error:
            warnings.warn(
                f"Video encoding failed; training will continue: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        if self._wandb_run is not None and self._wandb is not None:
            try:
                video = self._wandb.Video(
                    str(path),
                    format="mp4",
                    caption=caption or None,
                )
                self._wandb_run.log({key: video})
            except Exception as error:  # W&B must never take down an experiment.
                self._warn_wandb("video logging", error)
        return path

    def close(self, *, exit_code: int = 0) -> None:
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.finish(exit_code=exit_code)
        except Exception as error:  # W&B must never take down an experiment.
            self._warn_wandb("shutdown", error)
        finally:
            self._wandb_run = None

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
        if self._save_figure(fig, path, plt=plt):
            self.log_image(f"plots/{path.stem}", path, caption="Training returns")
        return path

    def plot_world_model_loss(
        self, loss_history: list[float], filename: str = "world_model_loss.png"
    ) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = self.run_dir / filename
        if not loss_history:
            return path

        steps = list(range(1, len(loss_history) + 1))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(steps, loss_history, label="world-model fit loss")
        if all(value > 0.0 for value in loss_history):
            ax.set_yscale("log")
        ax.set_xlabel("fit step")
        ax.set_ylabel("loss")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        if self._save_figure(fig, path, plt=plt):
            self.log_image(
                f"plots/{path.stem}",
                path,
                caption="World-model fit loss",
            )
        return path

    @staticmethod
    def _save_figure(fig, path: Path, *, plt) -> bool:
        """Save a diagnostic plot without letting auxiliary I/O stop training."""

        try:
            fig.savefig(path)
        except Exception as error:
            warnings.warn(
                f"Plot saving failed; training will continue: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return False
        finally:
            plt.close(fig)
        return True
