"""Configuration for the first-party DreaMARL executable."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from world_marl.baselines.dreamer_cdp.config import (
    default_dreamer_cdp_python,
    default_upstream_root,
)
from world_marl.baselines.dreamerv3.config import absolute_path
from world_marl.dreamarl.runtime import runtime_fingerprint


@dataclass(frozen=True, slots=True)
class DreaMARLRunSpec:
    """One reproducible invocation of the agent-axis-native algorithm."""

    experiment_dir: Path
    task: str
    num_agents: int
    seed: int = 0
    train_steps: int = 50_000
    platform: str = "cuda"
    infrastructure_root: Path = field(default_factory=default_upstream_root)
    python: Path = field(default_factory=default_dreamer_cdp_python)
    save_every_seconds: int | None = 1_800
    wandb_project: str | None = None
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "experiment_dir",
            Path(self.experiment_dir).expanduser().resolve(),
        )
        object.__setattr__(
            self,
            "infrastructure_root",
            Path(self.infrastructure_root).expanduser().resolve(),
        )
        object.__setattr__(self, "python", absolute_path(self.python))
        if self.num_agents < 1:
            raise ValueError("num_agents must be positive")
        if not self.task:
            raise ValueError("task must be non-empty")
        if self.train_steps < 1:
            raise ValueError("train_steps must be positive")
        if self.platform not in {"cpu", "cuda", "tpu"}:
            raise ValueError("platform must be one of: cpu, cuda, tpu")
        if self.save_every_seconds is not None and self.save_every_seconds < 1:
            raise ValueError("save_every_seconds must be positive")

    @property
    def logdir(self) -> Path:
        return self.experiment_dir / "run"

    @property
    def configs(self) -> list[str]:
        if not self.task.startswith("meltingpot_"):
            raise ValueError(
                "DreaMARL is a multi-agent algorithm; maintained launches require "
                "a Melting Pot task"
            )
        return ["meltingpot_vision", "joint_world"]

    @property
    def command(self) -> list[str]:
        outputs = ["jsonl", "scope"]
        if self.wandb_project:
            outputs.append("wandb")
        command = [
            str(self.python),
            "-m",
            "world_marl.dreamarl.main",
            "--logdir",
            str(self.logdir),
            "--configs",
            *self.configs,
            "--task",
            self.task,
            "--seed",
            str(self.seed),
            "--agent.num_agents",
            str(self.num_agents),
            "--run.steps",
            str(self.train_steps),
            "--jax.platform",
            self.platform,
            "--logger.outputs",
            *outputs,
            "--logger.filter",
            (
                "score|length|fps|ratio|train/loss/|train/rand/|"
                "train/dyn_ent|train/rep_ent|train/world_model/|"
                "report/world_model/"
            ),
        ]
        if self.save_every_seconds is not None:
            command.extend(["--run.save_every", str(self.save_every_seconds)])
        return command

    def to_dict(self) -> dict[str, object]:
        return {
            "implementation": "first-party DreaMARL",
            "experiment_dir": str(self.experiment_dir),
            "logdir": str(self.logdir),
            "infrastructure_root": str(self.infrastructure_root),
            "python": str(self.python),
            "task": self.task,
            "seed": self.seed,
            "train_env_steps_budget": self.train_steps,
            "num_agents": self.num_agents,
            "agent_axis_native": True,
            "execution": "decentralized shared local-belief actor",
            "training_state": (
                "joint posterior and joint-action-conditioned prior with directly "
                "predicted local control beliefs"
            ),
            "critic": "centralized team value over the joint latent state",
            "algorithm_overrides": [],
            "platform": self.platform,
            "observation_mode": "local RGB vision",
            "accelerator_memory_preallocation": False,
            "configs": self.configs,
            "source_fingerprint": runtime_fingerprint(),
            "save_every_seconds": self.save_every_seconds,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "command": self.command,
        }
