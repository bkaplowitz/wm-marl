"""Configuration for launching the pinned upstream DreamerV3 code."""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path


OFFICIAL_DREAMERV3_REPOSITORY = "https://github.com/danijar/dreamerv3.git"
OFFICIAL_DREAMERV3_COMMIT = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"
OFFICIAL_DMC_CONFIG = "dmc_proprio"
OFFICIAL_DMC_TRAIN_STEPS = 1_100_000
COMPARISON_DMC_TRAIN_STEPS = 500_000


def repository_root() -> Path:
    """Return the world_marl source checkout root."""
    return Path(__file__).resolve().parents[4]


def default_upstream_root() -> Path:
    return repository_root() / "external" / "dreamerv3"


def default_dreamerv3_python() -> Path:
    configured = os.environ.get("DREAMERV3_PYTHON")
    if configured:
        return Path(configured).expanduser()
    candidate = repository_root() / ".venv-dreamerv3" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def absolute_path(path: str | Path) -> Path:
    """Make a path absolute without dereferencing virtualenv symlinks."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


@dataclasses.dataclass(frozen=True)
class DreamerV3RunSpec:
    """A reproducible invocation of the upstream DreamerV3 trainer."""

    experiment_dir: Path
    task: str = "dmc_reacher_easy"
    seed: int = 0
    train_steps: int = COMPARISON_DMC_TRAIN_STEPS
    platform: str = "cuda"
    configs: tuple[str, ...] = (OFFICIAL_DMC_CONFIG,)
    upstream_root: Path = dataclasses.field(default_factory=default_upstream_root)
    python: Path = dataclasses.field(default_factory=default_dreamerv3_python)
    save_every_seconds: int | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "experiment_dir", Path(self.experiment_dir).expanduser().resolve()
        )
        object.__setattr__(
            self, "upstream_root", Path(self.upstream_root).expanduser().resolve()
        )
        object.__setattr__(self, "python", absolute_path(self.python))
        if not self.task.startswith("dmc_"):
            raise ValueError("DreamerV3 DMC tasks must start with 'dmc_'")
        if self.train_steps < 1:
            raise ValueError("train_steps must be >= 1")
        if self.platform not in {"cpu", "cuda", "tpu"}:
            raise ValueError("platform must be one of: cpu, cuda, tpu")
        if not self.configs:
            raise ValueError("at least one upstream config is required")
        if self.save_every_seconds is not None and self.save_every_seconds < 1:
            raise ValueError("save_every_seconds must be >= 1")

    @property
    def upstream_logdir(self) -> Path:
        return self.experiment_dir / "upstream"

    @property
    def command(self) -> list[str]:
        command = [
            str(self.python),
            str(self.upstream_root / "dreamerv3" / "main.py"),
            "--logdir",
            str(self.upstream_logdir),
            "--configs",
            *self.configs,
            "--task",
            self.task,
            "--seed",
            str(self.seed),
            "--run.steps",
            str(self.train_steps),
            "--jax.platform",
            self.platform,
        ]
        if self.save_every_seconds is not None:
            command.extend(["--run.save_every", str(self.save_every_seconds)])
        if self.wandb_project:
            command.extend(["--logger.outputs", "jsonl", "scope", "wandb"])
        command.extend(self.extra_args)
        return command

    def to_dict(self) -> dict[str, object]:
        return {
            "implementation": "danijar/dreamerv3",
            "upstream_repository": OFFICIAL_DREAMERV3_REPOSITORY,
            "upstream_commit": OFFICIAL_DREAMERV3_COMMIT,
            "experiment_dir": str(self.experiment_dir.resolve()),
            "upstream_logdir": str(self.upstream_logdir.resolve()),
            "upstream_root": str(self.upstream_root.resolve()),
            "python": str(self.python),
            "task": self.task,
            "seed": self.seed,
            "train_env_steps_budget": self.train_steps,
            "eval_env_steps_budget": 0,
            "total_real_env_steps_budget": self.train_steps,
            "platform": self.platform,
            "configs": list(self.configs),
            "save_every_seconds": self.save_every_seconds,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "extra_args": list(self.extra_args),
            "command": self.command,
        }
