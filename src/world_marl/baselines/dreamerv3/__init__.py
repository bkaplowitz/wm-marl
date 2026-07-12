"""Pinned integration for the upstream DreamerV3 implementation."""

from world_marl.baselines.dreamerv3.config import (
  OFFICIAL_DREAMERV3_COMMIT,
  OFFICIAL_DREAMERV3_REPOSITORY,
  DreamerV3RunSpec,
)

__all__ = [
  "DreamerV3RunSpec",
  "OFFICIAL_DREAMERV3_COMMIT",
  "OFFICIAL_DREAMERV3_REPOSITORY",
]
