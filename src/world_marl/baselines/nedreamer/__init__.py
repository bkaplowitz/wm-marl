"""Pinned integration for the official NE-Dreamer implementation."""

from world_marl.baselines.nedreamer.config import (
    OFFICIAL_NEDREAMER_COMMIT,
    OFFICIAL_NEDREAMER_REPOSITORY,
    NEDreamerRunSpec,
)

__all__ = [
    "NEDreamerRunSpec",
    "OFFICIAL_NEDREAMER_COMMIT",
    "OFFICIAL_NEDREAMER_REPOSITORY",
]
