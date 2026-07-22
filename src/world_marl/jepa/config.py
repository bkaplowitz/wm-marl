"""Canonical configuration for the maintained JEPA agent."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any


# Algorithm, optimization, data, and measurement settings shared by every
# maintained budget. Environment, seed, output path, and W&B destination are
# intentionally launcher concerns and do not belong here.
_CANONICAL_BASE = MappingProxyType(
    {
        "num_runs": 1,
        "num_envs": 16,
        "env_workers": 16,
        "brax_backend": None,
        "max_cycles": 1_000,
        "isolated_rng_streams": True,
        "deterministic_compute": True,
        "collect_steps": 320,
        "initial_reset_interval": 80,
        "initial_random_action_hold_steps": 1,
        "initial_random_action_hold_schedule": (),
        "validation_steps": 80,
        "validation_seed": 1_000_042,
        "replay_capacity": 1_000_000,
        "batch_size": 16,
        "chunk_length": 64,
        "context_window": 8,
        "model_horizon": 8,
        "open_loop_horizon": 8,
        "latent_dim": 144,
        "model_dim": 144,
        "num_layers": 2,
        "num_heads": 4,
        "mlp_ratio": 4,
        "train_steps": 1_280,
        "eval_interval": 250,
        "policy_train_steps": 1_280,
        "online_collect_steps": 64,
        "online_train_steps": 1_024,
        "online_policy_train_steps": 512,
        "online_iterations": 483,
        "online_policy_actor_update_interval": 2,
        "online_policy_actor_update_interval_start_env_steps": 50_000,
        "online_policy_critic_first_steps": 0,
        "online_freeze_encoder_after_env_steps": 101_376,
        "online_checkpoint_interval": 16,
        "online_recent_replay_steps": 320,
        "online_recent_world_model_fraction": 0.5,
        "online_recent_world_model_until_env_steps": 50_000,
        "online_recent_replay_max_oversample": 10.0,
        "policy_batch_size": 1_024,
        "policy_recent_start_fraction": 0.0,
        "policy_reset_start_fraction": 0.1,
        "policy_reset_start_fraction_start_env_steps": 201_728,
        "policy_reset_start_max_age": 63,
        "imag_horizon": 15,
        "critic_horizon": 64,
        "policy_return_ema_decay": 0.99,
        "value_clip": 100.0,
        "value_clip_final": 333.0,
        "value_clip_schedule_start_env_steps": 150_528,
        "value_clip_schedule_end_env_steps": 250_880,
        "policy_actor_kl_coef": 1.0,
        "policy_actor_kl_target_per_dim": 0.1,
        "policy_actor_kl_reference_interval": 512,
        "policy_replay_critic_loss_coef": 0.3,
        "policy_replay_critic_batch_size": 16,
        "policy_replay_critic_horizon": 64,
        "policy_slow_value_regularization_coef": 1.0,
        "target_critic_ema_decay": 0.98,
        "actor_hidden_dim": 64,
        "critic_hidden_dim": 64,
        "actor_num_layers": 3,
        "critic_num_layers": 3,
        "actor_layer_norm": True,
        "critic_layer_norm": True,
        "actor_entropy_coef": 3e-3,
        "policy_pathwise_reward_coef": 0.0,
        "policy_pathwise_horizon": 4,
        "actor_entropy_coef_final": None,
        "actor_entropy_schedule_start_env_steps": None,
        "actor_entropy_schedule_end_env_steps": None,
        "actor_log_std_min": -2.302585092994046,
        "actor_log_std_max": 0.0,
        "actor_output_scale": 0.01,
        "value_output_scale": 0.0,
        "reward_output_scale": 0.0,
        "twohot_bins": 255,
        "twohot_min": -20.0,
        "twohot_max": 20.0,
        "regularizer_weight": 0.05,
        "sigreg_knots": 17,
        "sigreg_num_proj": 256,
        "reward_weight": 1.0,
        "continue_weight": 1.0,
        "learning_rate": 4e-5,
        "actor_learning_rate": 4e-5,
        "model_grad_clip_norm": 0.0,
        "actor_grad_clip_norm": 10.0,
        "critic_grad_clip_norm": 100.0,
        "optimizer_warmup_steps": 1_000,
        "adaptive_grad_clip": 0.3,
        "optimizer_epsilon": 1e-8,
        "gamma": 1.0 - 1.0 / 333.0,
        "lambda_return": 0.95,
        "final_policy_eval_episodes": 100,
        "final_policy_eval_num_envs": None,
        "final_policy_eval_seed": 9_000_000,
        "dreamer_report_window_env_steps": 10_000,
        # The fixed vectorized schedule collects exactly 499,712 learning
        # transitions. Use that exact endpoint so the final 10k bin is neither
        # mislabeled as 500k nor omitted from the logged curve.
        "dreamer_report_budget_env_steps": 499_712,
        "dreamer_report_final_bins": 3,
        "curve_eval_interval_env_steps": 0,
        "curve_eval_episodes": 0,
        "curve_eval_num_envs": 16,
        "curve_eval_seed": 9_000_000,
        "failure_return_threshold": 100.0,
        "success_return_threshold": 900.0,
        "training_snapshot_env_steps": (),
        "resume_training_snapshot": None,
    }
)


def canonical_jepa_config() -> dict[str, Any]:
    """Return the mutable resolved configuration for the fixed 500k agent."""

    return dict(_CANONICAL_BASE)


def smoke_jepa_config() -> dict[str, Any]:
    """Return a cheap execution check of the canonical code path."""

    return {
        **canonical_jepa_config(),
        "num_envs": 2,
        "env_workers": 2,
        "collect_steps": 80,
        "validation_steps": 80,
        "batch_size": 2,
        "policy_batch_size": 8,
        "train_steps": 2,
        "policy_train_steps": 2,
        "online_iterations": 1,
        "online_train_steps": 2,
        "online_policy_train_steps": 2,
        "online_checkpoint_interval": 1,
        "final_policy_eval_episodes": 0,
        "dreamer_report_budget_env_steps": 0,
        "curve_eval_interval_env_steps": 0,
        "curve_eval_episodes": 0,
    }
