"""Typed training configuration for the ``train_e2e`` core.

``TrainConfig`` mirrors the ``train_e2e`` argparse dests one-to-one so the pure
training core consumes a structured object instead of ``argparse.Namespace``.
It doubles as the schema for a future Hydra structured config.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    config: str | None = None
    wandb: bool = False
    wandb_project: str = "world-marl"
    algorithm: str = "ippo"
    substrate: str = "coins"
    num_envs: int = 4
    rollout_steps: int = 128
    total_env_steps: int = 100_000
    eval_episodes: int = 50
    num_runs: int = 3
    seed: int = 0
    max_cycles: int = 1000
    observation_size: int | None = None
    append_agent_id: bool = False
    include_observation_scalars: bool = False
    stochastic_eval: bool = False
    eval_max_steps: int | None = None
    out_dir: str = "runs"
    min_improvement: float = 0.2
    negative_control: str = "freeze-policy"
    prefit_world_model: bool = False
    wm_random_rollouts: int = 1
    wm_initial_rollouts: int = 1
    wm_fit_steps: int = 10_000
    wm_learning_rate: float = 3e-4
    wm_hidden_dim: int = 256
    wm_integration_steps: int = 10
    wm_policy_warmup_updates: int = 0
    wm_flow_type: str = "linear"
    wm_num_categories: int = 9
    learning_rate: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    activation: str = "relu"
    eval_checkpoint: str | None = None

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "TrainConfig":
        return cls(**vars(namespace))
