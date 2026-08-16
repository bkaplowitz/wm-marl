"""Subprocess launcher for the pinned official NE-Dreamer implementation."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dreamarl.baselines.nedreamer.artifacts import normalize_training_artifacts
from dreamarl.baselines.nedreamer.config import (
    OFFICIAL_NEDREAMER_COMMIT,
    NEDreamerRunSpec,
)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def verify_upstream(upstream_root: str | Path) -> str:
    upstream_root = Path(upstream_root)
    if not (upstream_root / "train.py").is_file():
        raise FileNotFoundError(
            f"NE-Dreamer checkout is missing at {upstream_root}; initialize submodules"
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != OFFICIAL_NEDREAMER_COMMIT:
        raise RuntimeError(
            f"NE-Dreamer revision mismatch: expected {OFFICIAL_NEDREAMER_COMMIT}, "
            f"found {revision}"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=upstream_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            f"NE-Dreamer checkout has local modifications; refusing to run:\n{status}"
        )
    return revision


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _environment(spec: NEDreamerRunSpec) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("WANDB_DIR", str(spec.experiment_dir))
    if spec.wandb_project:
        env["WANDB_PROJECT"] = spec.wandb_project
    if spec.wandb_entity:
        env["WANDB_ENTITY"] = spec.wandb_entity
    return env


def run_training(
    spec: NEDreamerRunSpec, *, resume: bool = False, dry_run: bool = False
) -> int:
    revision = verify_upstream(spec.upstream_root)
    if spec.upstream_logdir.exists() and not resume:
        raise FileExistsError(
            f"upstream logdir exists: {spec.upstream_logdir}; use --resume intentionally"
        )
    spec.experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        spec.experiment_dir / "launch.json",
        {
            **spec.to_dict(),
            "created_at": timestamp(),
            "verified_upstream_commit": revision,
            "host_platform": platform.platform(),
            "resume": resume,
            "dry_run": dry_run,
        },
    )
    if dry_run:
        print(" ".join(spec.command))
        return 0
    if not spec.python.exists():
        raise FileNotFoundError(
            f"NE-Dreamer Python not found: {spec.python}; run setup first"
        )

    log_path = spec.experiment_dir / "process.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            spec.command,
            cwd=spec.upstream_root,
            env=_environment(spec),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
            returncode = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            returncode = process.wait()

    summary = normalize_training_artifacts(
        spec.experiment_dir,
        upstream_logdir=spec.upstream_logdir,
        task=spec.task,
        seed=spec.seed,
        train_steps_budget=spec.train_steps,
    )
    _write_json(
        spec.experiment_dir / "outcome.json",
        {
            "returncode": returncode,
            "completed": returncode == 0,
            "training_summary": summary,
        },
    )
    return returncode
