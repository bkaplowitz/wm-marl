"""Maintained MA-JEPA world-model boundary."""

from .backend import WorldModelBackend
from .transformer import ParallelTransformerDynamics, parallel_backend


def world_model_backend() -> WorldModelBackend:
    """Return canonical MA-JEPA's causal Transformer backend."""

    return parallel_backend()


__all__ = [
    "ParallelTransformerDynamics",
    "WorldModelBackend",
    "world_model_backend",
]
