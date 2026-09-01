"""Paths used by the MA-JEPA command-line tools."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ALGORITHM_ROOT = Path(__file__).resolve().parent


def algorithm_root() -> Path:
    """Return the installed MA-JEPA package directory."""

    return _ALGORITHM_ROOT


def repository_root() -> Path:
    """Return the repository checkout containing MA-JEPA."""

    return algorithm_root().parents[1]


def infrastructure_root() -> Path:
    """Return the pinned Embodied/DreamerV3 infrastructure checkout."""

    return repository_root() / "external" / "dreamerv3"


def runtime_python() -> Path:
    """Return the Python interpreter used for training subprocesses."""

    configured = os.environ.get("MAJEPA_PYTHON")
    return absolute_path(configured) if configured else Path(sys.executable)


def absolute_path(path: str | Path) -> Path:
    """Make a path absolute without dereferencing virtualenv symlinks."""

    path = Path(path).expanduser()
    return path if path.is_absolute() else Path.cwd() / path
