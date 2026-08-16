"""Configuration for launching the pinned upstream NE-Dreamer code."""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path


OFFICIAL_NEDREAMER_REPOSITORY = "https://github.com/corl-team/nedreamer.git"
OFFICIAL_NEDREAMER_COMMIT = "11cd3a978b83743f795cbfa81c2e095344912c17"
OFFICIAL_NEDREAMER_METHOD = "ne_dreamer"
OFFICIAL_NEDREAMER_MODEL_CONFIG = "size200M"
OFFICIAL_NEDREAMER_ACTION_REPEAT = 2
PHASE_1_TRAIN_TRANSITIONS = 1_100_000


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_upstream_root() -> Path:
    return repository_root() / "external" / "nedreamer"


def default_nedreamer_python() -> Path:
    configured = os.environ.get("NEDREAMER_PYTHON")
    if configured:
        return Path(configured).expanduser()
    candidate = repository_root() / ".venv-nedreamer" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def absolute_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


@dataclasses.dataclass(frozen=True)
class NEDreamerRunSpec:
    """A reproducible invocation of the official NE-Dreamer trainer."""

    experiment_dir: Path
    task: str = "dmc_walker_walk"
    seed: int = 0
    train_steps: int = PHASE_1_TRAIN_TRANSITIONS
    device: str = "cuda:0"
    upstream_root: Path = dataclasses.field(default_factory=default_upstream_root)
    python: Path = dataclasses.field(default_factory=default_nedreamer_python)
    wandb_project: str | None = None
    wandb_entity: str | None = None
    extra_overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "experiment_dir", Path(self.experiment_dir).expanduser().resolve()
        )
        object.__setattr__(
            self, "upstream_root", Path(self.upstream_root).expanduser().resolve()
        )
        object.__setattr__(self, "python", absolute_path(self.python))
        if not self.task.startswith("dmc_"):
            raise ValueError("NE-Dreamer DMC tasks must start with 'dmc_'")
        if self.train_steps < 1:
            raise ValueError("train_steps must be >= 1")
        if not self.device.startswith(("cuda", "cpu")):
            raise ValueError("device must start with 'cuda' or 'cpu'")
        for override in self.extra_overrides:
            if "=" not in override:
                raise ValueError(f"Hydra override must be KEY=VALUE, got: {override}")

    @property
    def upstream_logdir(self) -> Path:
        # Upstream explicitly derives the W&B run name from this final path part.
        return self.experiment_dir / f"nedreamer_{self.task}_seed{self.seed}"

    @property
    def command(self) -> list[str]:
        return [
            str(self.python),
            str(self.upstream_root / "train.py"),
            f"env.task={self.task}",
            f"model.rep_loss={OFFICIAL_NEDREAMER_METHOD}",
            f"device={self.device}",
            f"seed={self.seed}",
            f"logdir={self.upstream_logdir}",
            f"env.steps={self.train_steps}",
            f"trainer.steps={self.train_steps}",
            "trainer.eval_video_every=0",
            "trainer.s3_bucket=null",
            "model.imagination_decoding.enabled=false",
            *self.extra_overrides,
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "implementation": "corl-team/nedreamer",
            "upstream_repository": OFFICIAL_NEDREAMER_REPOSITORY,
            "upstream_commit": OFFICIAL_NEDREAMER_COMMIT,
            "upstream_model_config": OFFICIAL_NEDREAMER_MODEL_CONFIG,
            "method": OFFICIAL_NEDREAMER_METHOD,
            "experiment_dir": str(self.experiment_dir),
            "upstream_logdir": str(self.upstream_logdir),
            "upstream_root": str(self.upstream_root),
            "python": str(self.python),
            "task": self.task,
            "seed": self.seed,
            "train_real_transition_budget": self.train_steps,
            "evaluation_real_transition_budget": 0,
            "native_action_repeat": OFFICIAL_NEDREAMER_ACTION_REPEAT,
            "environment_decision_budget": self.train_steps
            // OFFICIAL_NEDREAMER_ACTION_REPEAT,
            "observation_mode": "vision",
            "observation_shape": [64, 64, 3],
            "device": self.device,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "extra_overrides": list(self.extra_overrides),
            "command": self.command,
        }
