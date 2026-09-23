"""Observation and latent representation modules."""

from .encoder import Encoder
from .latent import CategoricalLatent

__all__ = [
    "CategoricalLatent",
    "Encoder",
]
