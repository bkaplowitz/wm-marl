"""Repository paths for first-party DreaMARL."""

from __future__ import annotations

from pathlib import Path

_ALGORITHM_ROOT = Path(__file__).resolve().parent


def algorithm_root() -> Path:
    """Return the source directory executed by the DreaMARL launcher."""

    return _ALGORITHM_ROOT


def repository_root() -> Path:
    """Return the checkout containing the first-party DreaMARL package."""

    return algorithm_root().parents[1]
