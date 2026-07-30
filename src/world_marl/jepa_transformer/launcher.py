"""Launch the registered M3 overlay on an immutable Dreamer-CDP snapshot."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from world_marl.baselines.dreamer_cdp.config import default_upstream_root
from world_marl.baselines.dreamer_cdp.launcher import verify_upstream
from world_marl.baselines.dreamerv3.artifacts import normalize_training_artifacts
from world_marl.baselines.dreamerv3.config import default_upstream_root as dreamer_root
from world_marl.jepa_transformer.config import JEPATransformerRunSpec
from world_marl.jepa_transformer.runtime import prepare_runtime, runtime_fingerprint


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_training(
    spec: JEPATransformerRunSpec,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> int:
    official_revision = verify_upstream(default_upstream_root())
    runtime = prepare_runtime(spec.runtime_root)
    if runtime != spec.runtime_root:
        raise RuntimeError(f"prepared unexpected runtime: {runtime}")
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
            "verified_official_commit": official_revision,
            "verified_overlay_fingerprint": runtime_fingerprint(),
            "host_platform": platform.platform(),
            "resume": resume,
            "dry_run": dry_run,
            "causal_delta": (
                "official RSSM deterministic transition replaced by a bounded "
                "causal Transformer; all M2 losses, heads, and schedules fixed"
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
            cwd=spec.runtime_root,
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
        implementation="JEPA-Transformer M3",
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
