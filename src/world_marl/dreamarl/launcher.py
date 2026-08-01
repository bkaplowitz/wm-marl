"""Train the first-party DreaMARL implementation."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from world_marl.baselines.dreamerv3.artifacts import normalize_training_artifacts
from world_marl.baselines.dreamerv3.config import default_upstream_root as dreamer_root
from world_marl.dreamarl.config import DreaMARLRunSpec
from world_marl.dreamarl.parity import verify_m3_reduction_contract
from world_marl.dreamarl.runtime import verify_first_party_source
from world_marl.jepa_transformer.foundation import repository_root


MIN_FREE_DISK_BYTES = 10 * 1024**3


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_training(
    spec: DreaMARLRunSpec,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> int:
    """Run DreaMARL source directly with pinned Embodied infrastructure."""

    verification = verify_first_party_source()
    verification.update(verify_m3_reduction_contract(spec))
    if spec.logdir.exists() and not resume:
        raise FileExistsError(f"run already exists: {spec.logdir}")
    spec.experiment_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **spec.to_dict(),
        **verification,
        "created_at": timestamp(),
        "host_platform": platform.platform(),
        "resume": resume,
        "dry_run": dry_run,
    }
    _write_json(spec.experiment_dir / "launch.json", manifest)
    if dry_run:
        print(" ".join(spec.command))
        return 0
    if shutil.disk_usage(spec.experiment_dir).free < MIN_FREE_DISK_BYTES:
        raise RuntimeError("insufficient free disk for DreaMARL training")
    if not spec.python.exists():
        raise FileNotFoundError(f"DreaMARL Python is missing: {spec.python}")

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "glfw" if platform.system() == "Darwin" else "egl")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("WANDB_DIR", str(spec.experiment_dir))
    if spec.wandb_project:
        env["WANDB_PROJECT"] = spec.wandb_project
    if spec.wandb_entity:
        env["WANDB_ENTITY"] = spec.wandb_entity
    env.setdefault("WANDB_NAME", spec.experiment_dir.name)
    pythonpath = [str(spec.infrastructure_root), str(repository_root() / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    with (spec.experiment_dir / "process.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            spec.command,
            cwd=repository_root(),
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
        implementation="first-party DreaMARL",
        artifact_subdir="run",
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
