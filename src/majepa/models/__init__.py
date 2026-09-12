"""Observation and latent representation modules."""

from .latent import CategoricalLatent
from .encoder import Encoder

__all__ = [
    "CategoricalLatent",
    "Encoder",
]
