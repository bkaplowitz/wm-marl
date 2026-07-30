"""Experiment configuration for the Milestone 3 JEPA Transformer."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from world_marl.baselines.dreamer_cdp.config import default_dreamer_cdp_python
from world_marl.baselines.dreamerv3.config import absolute_path
from world_marl.jepa_transformer.runtime import (
    default_runtime_root,
    runtime_fingerprint,
)


M3_CONTROL_GATE_STEPS = 25_000


@dataclasses.dataclass(frozen=True)
class JEPATransformerRunSpec:
    """A reproducible visual-DMC invocation of the registered M3 system."""

    experiment_dir: Path
    task: str = "dmc_reacher_easy"
    seed: int = 0
    train_steps: int = M3_CONTROL_GATE_STEPS
    platform: str = "cuda"
    runtime_root: Path = dataclasses.field(default_factory=default_runtime_root)
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
            self, "runtime_root", Path(self.runtime_root).expanduser().resolve()
        )
        object.__setattr__(self, "python", absolute_path(self.python))
        if not self.task.startswith("dmc_"):
            raise ValueError("JEPA Transformer DMC tasks must start with 'dmc_'")
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
            str(self.runtime_root / "dreamerv3" / "main.py"),
            "--logdir",
            str(self.upstream_logdir),
            "--configs",
            "dmc_vision",
            "jepa_transformer",
            "--task",
            self.task,
            "--seed",
            str(self.seed),
            "--run.steps",
            str(self.train_steps),
            "--jax.platform",
            self.platform,
            "--jax.profiler",
            "False",
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
            "implementation": "JEPA-Transformer M3 on fmi-basel/Dreamer-CDP",
            "overlay_fingerprint": runtime_fingerprint(),
            "experiment_dir": str(self.experiment_dir),
            "upstream_logdir": str(self.upstream_logdir),
            "runtime_root": str(self.runtime_root),
            "python": str(self.python),
            "task": self.task,
            "seed": self.seed,
            "train_env_steps_budget": self.train_steps,
            "eval_env_steps_budget": 0,
            "total_real_env_steps_budget": self.train_steps,
            "platform": self.platform,
            "observation_mode": "vision",
            "configs": ["dmc_vision", "jepa_transformer"],
            "save_every_seconds": self.save_every_seconds,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "extra_args": list(self.extra_args),
            "command": self.command,
        }
