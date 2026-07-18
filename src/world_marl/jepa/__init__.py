"""Representation-space SIGReg/JEPA world-model components."""

from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.replay import ReplayBatch, SequenceReplayBuffer
from world_marl.jepa.training import (
    JepaTrainState,
    actor_value_from_latent,
    actor_value_stats_from_latent,
    continuous_policy_train_step,
    create_jepa_train_state,
    evaluate_open_loop,
    lambda_returns,
    latent_collapse_metrics,
    reset_policy_heads,
    select_continuous_actions,
    sigreg_loss,
    train_model_step,
)

__all__ = [
    "JepaConfig",
    "JepaTrainState",
    "JepaWorldModel",
    "ReplayBatch",
    "SequenceReplayBuffer",
    "actor_value_from_latent",
    "actor_value_stats_from_latent",
    "continuous_policy_train_step",
    "create_jepa_train_state",
    "evaluate_open_loop",
    "lambda_returns",
    "latent_collapse_metrics",
    "reset_policy_heads",
    "select_continuous_actions",
    "sigreg_loss",
    "train_model_step",
]
