"""Create the dependency-isolated runtime for official NE-Dreamer."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from dreamarl.baselines.nedreamer.config import (
    OFFICIAL_NEDREAMER_COMMIT,
    default_upstream_root,
    repository_root,
)
from dreamarl.baselines.nedreamer.launcher import verify_upstream


def resolved_requirements(upstream_root: str | Path) -> list[str]:
    lines = (
        (Path(upstream_root) / "requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def prepare_environment(
    *,
    venv_dir: str | Path,
    upstream_root: str | Path | None = None,
    recreate: bool = False,
) -> Path:
    venv_dir = Path(venv_dir).expanduser().resolve()
    upstream_root = Path(upstream_root or default_upstream_root()).resolve()
    revision = verify_upstream(upstream_root)
    if revision != OFFICIAL_NEDREAMER_COMMIT:
        raise RuntimeError(f"refusing to install unpinned revision {revision}")
    if venv_dir.exists():
        if not recreate:
            raise FileExistsError(
                f"environment exists: {venv_dir}; pass --recreate to replace it"
            )
        shutil.rmtree(venv_dir)
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required to create the NE-Dreamer environment")
    subprocess.run(
        [uv, "venv", "--python", "3.11", "--seed", "--no-project", str(venv_dir)],
        check=True,
    )
    python = venv_dir / "bin" / "python"
    requirements = resolved_requirements(upstream_root)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(requirements) + "\n")
        path = Path(handle.name)
    try:
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(path)], check=True
        )
    finally:
        path.unlink(missing_ok=True)
    subprocess.run(
        [
            str(python),
            "-c",
            "import dm_control, hydra, torch, torchrl, wandb; print(torch.cuda.is_available())",
        ],
        cwd=upstream_root,
        check=True,
    )
    frozen = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    metadata = {
        "upstream_commit": revision,
        "python": str(python),
        "requirements": requirements,
        "installed_packages": frozen,
    }
    (repository_root() / "nedreamer-environment.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return python
