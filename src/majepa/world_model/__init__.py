"""Strict-causal local Transformer dynamics and their replay representation."""

from .transformer import ParallelTransformerDynamics, feature_tensor, replay_entries

__all__ = ["ParallelTransformerDynamics", "feature_tensor", "replay_entries"]
