"""Train the first-party DreaMARL implementation."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any

from dreamarl.baselines.dreamerv3.artifacts import normalize_training_artifacts
from dreamarl.config import DreaMARLRunSpec
from dreamarl.contracts import verify_run_contract
from dreamarl.runtime import repository_root


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _stage_continuation_replay(source, destination) -> None:
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"continuation replay directory is missing: {source}")
    chunks = sorted(source.glob("*.npz"))
    if not chunks:
        raise FileNotFoundError(f"continuation replay has no chunks: {source}")
    destination.mkdir(parents=True, exist_ok=False)
    for chunk in chunks:
        target = destination / chunk.name
        try:
            os.link(chunk, target)
        except OSError:
            shutil.copy2(chunk, target)


def run_training(
    spec: DreaMARLRunSpec,
    *,
    resume: bool = False,
    dry_run: bool = False,
    contract_verifier: Callable[[Any], dict[str, object]] | None = None,
) -> int:
    """Run DreaMARL source directly with pinned Embodied infrastructure."""

    verification = (contract_verifier or verify_run_contract)(spec)
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
    if not spec.python.exists():
        raise FileNotFoundError(f"DreaMARL Python is missing: {spec.python}")
    if spec.load_replay:
        _stage_continuation_replay(
            spec.replay_source,
            spec.logdir / "replay",
        )

    env = os.environ.copy()
    if spec.task.startswith("smac_"):
        if not env.get("SC2PATH"):
            raise RuntimeError("SMAC runs require SC2PATH to point to StarCraft II 4.10")
        # SMAC-v1's pinned s2clientprotocol ships legacy generated descriptors.
        # The Python protobuf backend keeps them compatible with the modern
        # protobuf required by W&B in the shared training runtime.
        env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
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
        upstream_root=spec.infrastructure_root,
        task=spec.task,
        seed=spec.seed,
        train_steps_budget=spec.train_steps,
        observation_mode="vision" if spec.task.startswith("dmc_") else None,
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
