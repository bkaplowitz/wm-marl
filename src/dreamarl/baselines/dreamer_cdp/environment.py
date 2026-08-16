"""Create the isolated runtime used by official Dreamer-CDP."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from dreamarl.baselines.dreamer_cdp.config import (
    OFFICIAL_DREAMER_CDP_COMMIT,
    default_upstream_root,
)
from dreamarl.baselines.dreamer_cdp.launcher import upstream_revision
from dreamarl.baselines.dreamerv3.config import repository_root


def resolved_requirements(
    upstream_root: str | Path,
    *,
    accelerator: str,
) -> list[str]:
    if accelerator not in {"cpu", "cuda12"}:
        raise ValueError("accelerator must be 'cpu' or 'cuda12'")
    requirements = []
    for line in (Path(upstream_root) / "requirements.txt").read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("jax[cuda12]") and accelerator == "cpu":
            requirements.append(re.sub(r"jax\[cuda12\]", "jax", stripped))
            continue
        if stripped.startswith("nvidia-cuda-") and accelerator == "cpu":
            continue
        requirements.append(stripped)
    requirements.extend(["dm_control", "wandb[media]"])
    return requirements


def prepare_environment(
    *,
    venv_dir: str | Path,
    upstream_root: str | Path | None = None,
    accelerator: str = "cuda12",
    recreate: bool = False,
) -> Path:
    venv_dir = Path(venv_dir).expanduser().resolve()
    upstream_root = Path(upstream_root or default_upstream_root()).resolve()
    if upstream_revision(upstream_root) != OFFICIAL_DREAMER_CDP_COMMIT:
        raise RuntimeError("refusing to install an unpinned Dreamer-CDP checkout")
    if venv_dir.exists():
        if not recreate:
            raise FileExistsError(f"environment already exists: {venv_dir}")
        shutil.rmtree(venv_dir)
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required to create the Dreamer-CDP environment")
    subprocess.run(
        [uv, "venv", "--python", "3.11", "--seed", "--no-project", str(venv_dir)],
        check=True,
    )
    python = venv_dir / "bin" / "python"
    requirements = resolved_requirements(upstream_root, accelerator=accelerator)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(requirements) + "\n")
        requirements_path = Path(handle.name)
    try:
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(requirements_path)],
            check=True,
        )
    finally:
        requirements_path.unlink(missing_ok=True)
    subprocess.run(
        [
            str(python),
            "-c",
            "import dm_control, elements, jax, moviepy, ninjax, optax, wandb; "
            "print(jax.devices())",
        ],
        cwd=upstream_root,
        check=True,
    )
    installed = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    metadata = {
        "upstream_commit": OFFICIAL_DREAMER_CDP_COMMIT,
        "accelerator": accelerator,
        "python": str(python),
        "requirements": requirements,
        "installed_packages": installed,
    }
    (repository_root() / "dreamer-cdp-environment.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return python
