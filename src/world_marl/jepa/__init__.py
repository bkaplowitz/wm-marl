"""Representation-space SIGReg/JEPA world-model components."""

from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.replay import ReplayBatch, SequenceReplayBuffer
from world_marl.jepa.training import (
    JepaTrainState,
    actor_value_from_control_latent,
    apply_control_alignment,
    continuous_candidate_distill_step,
    continuous_critic_warmup_step,
    continuous_policy_train_step,
    create_jepa_train_state,
    evaluate_open_loop,
    lambda_returns,
    latent_collapse_metrics,
    procrustes_control_alignment,
    reset_policy_heads,
    reward_only_returns,
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
    "actor_value_from_control_latent",
    "apply_control_alignment",
    "continuous_candidate_distill_step",
    "continuous_critic_warmup_step",
    "continuous_policy_train_step",
    "create_jepa_train_state",
    "evaluate_open_loop",
    "lambda_returns",
    "latent_collapse_metrics",
    "procrustes_control_alignment",
    "reset_policy_heads",
    "reward_only_returns",
    "select_continuous_actions",
    "sigreg_loss",
    "train_model_step",
]
