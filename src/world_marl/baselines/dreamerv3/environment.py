"""Create the dependency-isolated environment used by upstream DreamerV3."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from world_marl.baselines.dreamerv3.config import (
  OFFICIAL_DREAMERV3_COMMIT,
  default_upstream_root,
  repository_root,
)
from world_marl.baselines.dreamerv3.launcher import upstream_revision


def resolved_requirements(
  upstream_root: str | Path,
  *,
  accelerator: str,
) -> list[str]:
  """Return upstream requirements with only the JAX platform line adapted."""
  if accelerator not in {"cpu", "cuda12"}:
    raise ValueError("accelerator must be 'cpu' or 'cuda12'")
  lines = (Path(upstream_root) / "requirements.txt").read_text(
    encoding="utf-8"
  ).splitlines()
  requirements = []
  for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
      continue
    if stripped.startswith("jax[cuda12]") and accelerator == "cpu":
      requirements.append(re.sub(r"jax\[cuda12\]", "jax", stripped))
      continue
    if stripped.startswith("nvidia-cuda-") and accelerator == "cpu":
      continue
    requirements.append(stripped)
  requirements.append("dm_control")
  requirements.append("wandb")
  return requirements


def environment_python(venv_dir: str | Path) -> Path:
  return Path(venv_dir) / "bin" / "python"


def prepare_environment(
  *,
  venv_dir: str | Path,
  upstream_root: str | Path | None = None,
  accelerator: str = "cuda12",
  recreate: bool = False,
) -> Path:
  """Build a Python 3.11 environment from the pinned upstream requirements."""
  venv_dir = Path(venv_dir).expanduser().resolve()
  upstream_root = Path(upstream_root or default_upstream_root()).resolve()
  revision = upstream_revision(upstream_root)
  if revision != OFFICIAL_DREAMERV3_COMMIT:
    raise RuntimeError(
      f"refusing to install unpinned DreamerV3 revision {revision}"
    )
  if venv_dir.exists():
    if not recreate:
      raise FileExistsError(
        f"environment already exists: {venv_dir}; pass --recreate to replace it"
      )
    shutil.rmtree(venv_dir)
  uv = shutil.which("uv")
  if not uv:
    raise RuntimeError(
      "uv is required to create the isolated DreamerV3 environment"
    )
  subprocess.run(
    [
      uv,
      "venv",
      "--python",
      "3.11",
      "--seed",
      "--no-project",
      str(venv_dir),
    ],
    check=True,
  )
  python = environment_python(venv_dir)
  subprocess.run(
    [str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools"],
    check=True,
  )
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
      "import dm_control, embodied, elements, jax, wandb; print(jax.devices())",
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
    "upstream_commit": revision,
    "accelerator": accelerator,
    "python": str(python),
    "requirements": requirements,
    "installed_packages": installed,
  }
  metadata_path = repository_root() / "dreamerv3-environment.json"
  metadata_path.write_text(
    json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
  )
  return python
