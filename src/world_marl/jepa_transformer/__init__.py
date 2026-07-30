"""Components for the visual JEPA Transformer research programme."""

from world_marl.jepa_transformer.foundation import verify_foundation
from world_marl.jepa_transformer.temporal import (
    CausalTemporalTransformer,
    TemporalCache,
    TemporalConfig,
)

__all__ = [
    "CausalTemporalTransformer",
    "TemporalCache",
    "TemporalConfig",
    "verify_foundation",
]
