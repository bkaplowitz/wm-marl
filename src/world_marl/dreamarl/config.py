"""Configuration for the first-party DreaMARL executable."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
    task: str = "dmc_reacher_easy"
    seed: int = 0
    train_steps: int = 250_000
    num_agents: int = 1
    interaction_context: Literal["none", "aligned", "shuffled"] = "none"
    local_memory: bool = False
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
        if self.interaction_context not in {"none", "aligned", "shuffled"}:
            raise ValueError("interaction_context must be none, aligned, or shuffled")
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
        suite = (
            "meltingpot_vision" if self.task.startswith("meltingpot_") else "dmc_vision"
        )
        configs = [suite, "jepa_transformer"]
        if self.interaction_context == "aligned":
            configs.append("interaction_jepa")
        elif self.interaction_context == "shuffled":
            configs.append("interaction_jepa_shuffled")
        if self.local_memory:
            configs.append("local_memory_sidecar")
        return configs

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
                "train/dyn_ent|train/rep_ent"
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
            "interaction_context": self.interaction_context,
            "local_memory": self.local_memory,
            "agent_axis_native": True,
            "agent_count_dependent_modules": [],
            "algorithm_overrides": [
                *(
                    []
                    if self.interaction_context == "none"
                    else [f"interaction_context={self.interaction_context}"]
                ),
                *(["local_memory_sidecar"] if self.local_memory else []),
            ],
            "platform": self.platform,
            "observation_mode": "vision",
            "accelerator_memory_preallocation": not self.task.startswith("meltingpot_"),
            "configs": self.configs,
            "source_fingerprint": runtime_fingerprint(),
            "save_every_seconds": self.save_every_seconds,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "command": self.command,
        }
