"""Pinned official MARIE benchmark integration."""

from .config import MARIERunSpec
from .launcher import run_training, verify_upstream

__all__ = ["MARIERunSpec", "run_training", "verify_upstream"]
