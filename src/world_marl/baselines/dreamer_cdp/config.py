"""Configuration for the pinned official Dreamer-CDP implementation."""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

from world_marl.baselines.dreamerv3.config import absolute_path, repository_root


OFFICIAL_DREAMER_CDP_REPOSITORY = "https://github.com/fmi-basel/Dreamer-CDP.git"
OFFICIAL_DREAMER_CDP_COMMIT = "a851fa3e3d70b624b094ee1810ad4bb602346092"
OFFICIAL_DMC_VISION_CONFIG = "dmc_vision"
M2_TRAIN_STEPS = 250_000


def default_upstream_root() -> Path:
    return repository_root() / "external" / "dreamer-cdp"


def default_dreamer_cdp_python() -> Path:
    configured = os.environ.get("DREAMER_CDP_PYTHON")
    if configured:
        return Path(configured).expanduser()
    candidate = repository_root() / ".venv-dreamer-cdp" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


@dataclasses.dataclass(frozen=True)
class DreamerCDPRunSpec:
    """A reproducible visual-DMC invocation of official Dreamer-CDP."""

    experiment_dir: Path
    task: str = "dmc_reacher_easy"
    seed: int = 0
    train_steps: int = M2_TRAIN_STEPS
    platform: str = "cuda"
    upstream_root: Path = dataclasses.field(default_factory=default_upstream_root)
    python: Path = dataclasses.field(default_factory=default_dreamer_cdp_python)
    save_every_seconds: int | None = 1_800
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
            raise ValueError("Dreamer-CDP DMC tasks must start with 'dmc_'")
        if self.train_steps < 1:
            raise ValueError("train_steps must be >= 1")
        if self.platform not in {"cpu", "cuda", "tpu"}:
            raise ValueError("platform must be one of: cpu, cuda, tpu")
        if self.save_every_seconds is not None and self.save_every_seconds < 1:
            raise ValueError("save_every_seconds must be >= 1")

    @property
    def upstream_logdir(self) -> Path:
        return self.experiment_dir / "upstream"

    @property
    def command(self) -> list[str]:
        outputs = ["jsonl", "scope"]
        if self.wandb_project:
            outputs.append("wandb")
        command = [
            str(self.python),
            str(self.upstream_root / "dreamerv3" / "main.py"),
            "--logdir",
            str(self.upstream_logdir),
            "--configs",
            OFFICIAL_DMC_VISION_CONFIG,
            "--task",
            self.task,
            "--seed",
            str(self.seed),
            "--run.steps",
            str(self.train_steps),
            "--jax.platform",
            self.platform,
            "--logger.outputs",
            *outputs,
            "--logger.filter",
            (
                "score|length|fps|ratio|train/loss/|train/rand/|"
                "train/dyn_ent|train/rep_ent"
            ),
        ]
        if self.save_every_seconds is not None:
            command.extend(["--run.save_every", str(self.save_every_seconds)])
        command.extend(self.extra_args)
        return command

    def to_dict(self) -> dict[str, object]:
        return {
            "implementation": "fmi-basel/Dreamer-CDP",
            "upstream_repository": OFFICIAL_DREAMER_CDP_REPOSITORY,
            "upstream_commit": OFFICIAL_DREAMER_CDP_COMMIT,
            "experiment_dir": str(self.experiment_dir),
            "upstream_logdir": str(self.upstream_logdir),
            "upstream_root": str(self.upstream_root),
            "python": str(self.python),
            "task": self.task,
            "seed": self.seed,
            "train_env_steps_budget": self.train_steps,
            "eval_env_steps_budget": 0,
            "total_real_env_steps_budget": self.train_steps,
            "platform": self.platform,
            "observation_mode": "vision",
            "configs": [OFFICIAL_DMC_VISION_CONFIG],
            "save_every_seconds": self.save_every_seconds,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "extra_args": list(self.extra_args),
            "command": self.command,
        }
