"""Launch the immutable official Dreamer-CDP source checkout."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from world_marl.baselines.dreamer_cdp.config import (
    OFFICIAL_DREAMER_CDP_COMMIT,
    DreamerCDPRunSpec,
)
from world_marl.baselines.dreamerv3.artifacts import normalize_training_artifacts
from world_marl.baselines.dreamerv3.config import default_upstream_root as dreamer_root


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def upstream_revision(upstream_root: str | Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(upstream_root),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_upstream(upstream_root: str | Path) -> str:
    upstream_root = Path(upstream_root)
    if not (upstream_root / "dreamerv3" / "main.py").is_file():
        raise FileNotFoundError(f"Dreamer-CDP checkout is missing: {upstream_root}")
    revision = upstream_revision(upstream_root)
    if revision != OFFICIAL_DREAMER_CDP_COMMIT:
        raise RuntimeError(
            "Dreamer-CDP revision mismatch: expected "
            f"{OFFICIAL_DREAMER_CDP_COMMIT}, found {revision}"
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
            "Dreamer-CDP checkout has tracked modifications; refusing to run:\n"
            f"{status}"
        )
    return revision


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_training(
    spec: DreamerCDPRunSpec,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> int:
    revision = verify_upstream(spec.upstream_root)
    if spec.upstream_logdir.exists() and not resume:
        raise FileExistsError(
            f"upstream logdir already exists: {spec.upstream_logdir}"
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
            "representation_contract": (
                "official CDP predictor loss; diagnostic decoder is detached "
                "from the representation gradient"
            ),
        },
    )
    if dry_run:
        print(" ".join(spec.command))
        return 0
    if not spec.python.exists():
        raise FileNotFoundError(f"Dreamer-CDP Python is missing: {spec.python}")

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Darwin" else "egl")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("WANDB_DIR", str(spec.experiment_dir))
    if spec.wandb_project:
        env["WANDB_PROJECT"] = spec.wandb_project
    if spec.wandb_entity:
        env["WANDB_ENTITY"] = spec.wandb_entity
    with (spec.experiment_dir / "process.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            spec.command,
            cwd=spec.upstream_root,
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

    summary = normalize_training_artifacts(
        spec.experiment_dir,
        upstream_root=dreamer_root(),
        task=spec.task,
        seed=spec.seed,
        train_steps_budget=spec.train_steps,
        implementation="fmi-basel/Dreamer-CDP",
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
