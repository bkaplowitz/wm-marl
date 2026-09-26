"""Observation and latent representation modules."""

from .latent import CategoricalLatent
from .visual import Encoder

__all__ = [
    "CategoricalLatent",
    "Encoder",
]
