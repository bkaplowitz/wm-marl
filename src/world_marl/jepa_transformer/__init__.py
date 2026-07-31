"""Components for the visual JEPA Transformer research programme.

The independent Flax temporal kernel is loaded lazily so that the pinned
Dreamer-CDP runtime does not need Flax merely to import its launcher.
"""

from world_marl.jepa_transformer.foundation import verify_foundation

__all__ = [
    "CausalTemporalTransformer",
    "TemporalCache",
    "TemporalConfig",
    "verify_foundation",
]


def __getattr__(name: str):
    if name in {"CausalTemporalTransformer", "TemporalCache", "TemporalConfig"}:
        from world_marl.jepa_transformer import temporal

        return getattr(temporal, name)
    raise AttributeError(name)
