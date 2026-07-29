"""Latest-policy held-out evaluation for official NE-Dreamer runs."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from world_marl.baselines.dreamerv3.artifacts import summarize_returns
from world_marl.baselines.nedreamer.artifacts import read_jsonl
from world_marl.baselines.nedreamer.config import default_nedreamer_python
from world_marl.baselines.nedreamer.launcher import timestamp, verify_upstream


@dataclasses.dataclass(frozen=True)
class NEDreamerEvaluationSpec:
    experiment_dir: Path
    episodes: int = 20
    eval_seed: int = 10_000
    device: str = "cuda:0"
    python: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "experiment_dir", Path(self.experiment_dir).expanduser().resolve()
        )
        if self.python is not None:
            object.__setattr__(self, "python", Path(self.python).expanduser().resolve())
        if self.episodes < 1:
            raise ValueError("episodes must be >= 1")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def evaluation_command(
    spec: NEDreamerEvaluationSpec, *, eval_dir: Path
) -> tuple[list[str], dict[str, Any]]:
    launch = _read_json(spec.experiment_dir / "launch.json")
    upstream_root = Path(launch["upstream_root"])
    upstream_logdir = Path(launch["upstream_logdir"])
    checkpoint = upstream_logdir / "latest.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"latest NE-Dreamer checkpoint missing: {checkpoint}")
    python = spec.python or Path(launch.get("python", default_nedreamer_python()))
    driver = Path(__file__).with_name("eval_driver.py")
    command = [
        str(python),
        str(driver),
        "--upstream-root",
        str(upstream_root),
        "--checkpoint",
        str(checkpoint),
        "--task",
        str(launch["task"]),
        "--training-seed",
        str(launch["seed"]),
        "--eval-seed",
        str(spec.eval_seed),
        "--episodes",
        str(spec.episodes),
        "--device",
        spec.device,
        "--output",
        str(eval_dir / "episodes.jsonl"),
    ]
    return command, launch


def run_evaluation(
    spec: NEDreamerEvaluationSpec, *, dry_run: bool = False
) -> tuple[int, Path]:
    eval_dir = (
        spec.experiment_dir
        / "evaluation"
        / f"latest_{spec.episodes}eps_seed{spec.eval_seed}"
    )
    if eval_dir.exists():
        raise FileExistsError(f"evaluation directory exists: {eval_dir}")
    command, launch = evaluation_command(spec, eval_dir=eval_dir)
    verify_upstream(launch["upstream_root"])
    eval_dir.mkdir(parents=True)
    _write_json(
        eval_dir / "evaluation_launch.json",
        {
            "created_at": timestamp(),
            "checkpoint_policy": "latest final upstream checkpoint",
            "training_experiment": str(spec.experiment_dir),
            "episodes": spec.episodes,
            "eval_seed": spec.eval_seed,
            "device": spec.device,
            "command": command,
            "dry_run": dry_run,
        },
    )
    if dry_run:
        print(" ".join(command))
        return 0, eval_dir
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("WANDB_MODE", "disabled")
    with (eval_dir / "process.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=launch["upstream_root"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        returncode = process.wait()
    rows = read_jsonl(eval_dir / "episodes.jsonl")
    returns = [float(row["episode_return"]) for row in rows]
    summary = {
        "requested_episodes": spec.episodes,
        "completed_episodes": len(rows),
        "returns": returns,
        "statistics": summarize_returns(returns),
        "eval_real_environment_transitions": sum(
            int(row["real_environment_transitions"]) for row in rows
        ),
        "training_real_environment_transitions": int(
            launch["train_real_transition_budget"]
        ),
    }
    _write_json(eval_dir / "summary.json", summary)
    completed = returncode == 0 and len(rows) == spec.episodes
    _write_json(
        eval_dir / "outcome.json",
        {"returncode": returncode, "completed": completed, "summary": summary},
    )
    return (returncode if completed else returncode or 2), eval_dir
