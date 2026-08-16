"""Launch the immutable official MARIE source checkout."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import normalize_training_artifacts
from .config import OFFICIAL_MARIE_COMMIT, MARIERunSpec


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
    if not (upstream_root / "train.py").is_file():
        raise FileNotFoundError(f"MARIE checkout is missing: {upstream_root}")
    revision = upstream_revision(upstream_root)
    if revision != OFFICIAL_MARIE_COMMIT:
        raise RuntimeError(
            f"MARIE revision mismatch: expected {OFFICIAL_MARIE_COMMIT}, "
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
            "MARIE checkout has tracked modifications; refusing to run:\n" + status
        )
    return revision


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _latest_output(
    upstream_root: Path,
    pattern: str,
    *,
    started_ns: int,
) -> Path | None:
    candidates = [
        path
        for path in upstream_root.glob(pattern)
        if path.stat().st_mtime_ns >= started_ns
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime_ns, default=None)


def run_training(spec: MARIERunSpec, *, dry_run: bool = False) -> int:
    revision = verify_upstream(spec.upstream_root)
    if spec.experiment_dir.exists():
        raise FileExistsError(
            f"experiment directory already exists: {spec.experiment_dir}"
        )
    spec.experiment_dir.mkdir(parents=True)
    launch = {
        **spec.to_dict(),
        "created_at": timestamp(),
        "verified_upstream_commit": revision,
        "host_platform": platform.platform(),
        "dry_run": dry_run,
        "source_policy": "unmodified official model and training code",
    }
    _write_json(spec.experiment_dir / "launch.json", launch)
    if dry_run:
        print(" ".join(spec.command))
        return 0
    if not spec.python.exists():
        raise FileNotFoundError(f"MARIE Python is missing: {spec.python}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("SC2PATH", str(Path.home() / "StarCraftII"))
    env.setdefault("WANDB_DIR", str(spec.experiment_dir))
    started_ns = time.time_ns()
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
    outcome: dict[str, Any] = {
        "returncode": returncode,
        "completed": returncode == 0,
    }
    if returncode == 0:
        effective_seed = 23 + 100 * spec.seed
        result_path = _latest_output(
            spec.upstream_root,
            f"*_results/starcraft/{spec.map_name}-vq/"
            f"marie_{spec.map_name}_seed{effective_seed}.pkl",
            started_ns=started_ns,
        )
        checkpoint_path = _latest_output(
            spec.upstream_root,
            f"*_results/starcraft/{spec.map_name}-vq/run*/ckpt/model_final.pth",
            started_ns=started_ns,
        )
        if result_path is None:
            outcome["completed"] = False
            outcome["artifact_error"] = "official MARIE result pickle was not found"
            returncode = 2
            outcome["returncode"] = returncode
        else:
            outcome["summary"] = normalize_training_artifacts(
                spec.experiment_dir,
                result_path=result_path,
                checkpoint_path=checkpoint_path,
                map_name=spec.map_name,
                cli_seed=spec.seed,
                steps_budget=spec.steps,
            )
            outcome["upstream_output_directory"] = str(result_path.parent)
    _write_json(spec.experiment_dir / "outcome.json", outcome)
    return returncode
