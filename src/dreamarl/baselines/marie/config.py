"""Configuration for the pinned paper-era MARIE implementation."""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

from dreamarl.baselines.dreamerv3.config import absolute_path, repository_root


OFFICIAL_MARIE_REPOSITORY = "https://github.com/breez3young/MARIE.git"
OFFICIAL_MARIE_COMMIT = "5dc114f78e9f35389b843e05f01c455988451d0e"
OFFICIAL_SMAC_COMMIT = "d6aab33f76abc3849c50463a8592a84f59a5ef84"

# These gates use the released defaults without undocumented source edits.
PAPER_GATE_MAPS = {
    "3m": {"steps": 100_000, "temperature": 1.0, "mean_win_rate": 99.5},
    "3s_vs_4z": {"steps": 100_000, "temperature": 1.0, "mean_win_rate": 73.0},
}


def default_upstream_root() -> Path:
    return repository_root() / "external" / "marie"


def default_marie_python() -> Path:
    configured = os.environ.get("MARIE_PYTHON")
    if configured:
        return Path(configured).expanduser()
    candidate = repository_root() / ".venv-marie" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


@dataclasses.dataclass(frozen=True)
class MARIERunSpec:
    """An immutable invocation of official MARIE on its paper SMAC protocol."""

    experiment_dir: Path
    map_name: str = "3m"
    seed: int = 1
    steps: int | None = None
    temperature: float | None = None
    mode: str = "online"
    upstream_root: Path = dataclasses.field(default_factory=default_upstream_root)
    python: Path = dataclasses.field(default_factory=default_marie_python)
    n_workers: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "experiment_dir", Path(self.experiment_dir).expanduser().resolve()
        )
        object.__setattr__(
            self, "upstream_root", Path(self.upstream_root).expanduser().resolve()
        )
        object.__setattr__(self, "python", absolute_path(self.python))
        if self.map_name not in PAPER_GATE_MAPS:
            raise ValueError(
                f"unsupported paper gate {self.map_name!r}; "
                f"choose one of {sorted(PAPER_GATE_MAPS)}"
            )
        defaults = PAPER_GATE_MAPS[self.map_name]
        object.__setattr__(self, "steps", self.steps or defaults["steps"])
        object.__setattr__(
            self,
            "temperature",
            self.temperature
            if self.temperature is not None
            else defaults["temperature"],
        )
        if self.seed < 1:
            raise ValueError("paper reproduction seeds start at 1")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.n_workers != 1:
            raise ValueError("the paper reproduction command uses one worker")
        if self.mode not in {"disabled", "offline", "online"}:
            raise ValueError("mode must be disabled, offline, or online")

    @property
    def command(self) -> list[str]:
        return [
            str(self.python),
            str(self.upstream_root / "train.py"),
            "--n_workers",
            "1",
            "--env",
            "starcraft",
            "--env_name",
            self.map_name,
            "--seed",
            str(self.seed),
            "--steps",
            str(self.steps),
            "--mode",
            self.mode,
            "--tokenizer",
            "vq",
            "--decay",
            "0.8",
            "--temperature",
            str(self.temperature),
            "--sample_temp",
            "inf",
            "--ce_for_av",
        ]

    def to_dict(self) -> dict[str, object]:
        reference = PAPER_GATE_MAPS[self.map_name]
        return {
            "implementation": "breez3young/MARIE",
            "upstream_repository": OFFICIAL_MARIE_REPOSITORY,
            "upstream_commit": OFFICIAL_MARIE_COMMIT,
            "smac_commit": OFFICIAL_SMAC_COMMIT,
            "experiment_dir": str(self.experiment_dir),
            "upstream_root": str(self.upstream_root),
            "python": str(self.python),
            "environment": "SMAC v1 / StarCraft II 4.10",
            "map_name": self.map_name,
            "cli_seed": self.seed,
            "effective_internal_seed": 23 + 100 * self.seed,
            "real_environment_steps": self.steps,
            "temperature": self.temperature,
            "n_workers": self.n_workers,
            "tokenizer": "vq",
            "ema_decay": 0.8,
            "sample_temperature": "inf",
            "available_action_loss": "cross_entropy",
            "wandb_mode": self.mode,
            "upstream_wandb_project": "starcraft",
            "paper_four_seed_mean_win_rate": reference["mean_win_rate"],
            "command": self.command,
        }
