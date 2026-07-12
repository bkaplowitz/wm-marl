"""Write launcher scripts for the DMC vector JEPA benchmark track."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Any


DEFAULT_TASKS = (
    "reacher/easy",
    "cartpole/swingup",
    "finger/spin",
    "cheetah/run",
    "walker/walk",
)

_DREAMER_AC_BASE: dict[str, Any] = {
    "num_envs": 16,
    "env_workers": 16,
    "collect_steps": 8192,
    "validation_steps": 2048,
    "train_steps": 12000,
    "policy_train_steps": 3000,
    "online_iterations": 8,
    "online_collect_steps": 2048,
    "online_validation_steps": 1024,
    "online_train_steps": 3000,
    "online_policy_train_steps": 750,
    "policy_selection_interval": 250,
    "policy_selection_episodes": 32,
    "policy_eval_episodes": 64,
    "policy_confirmation_episodes": 64,
    "final_policy_eval_episodes": 256,
    "policy_return_mode": "lambda",
    "policy_actor_baseline": "value",
    "policy_return_normalization": "percentile",
    "learning_rate": 1e-4,
    "actor_learning_rate": 3e-5,
    "latent_dim": 512,
    "model_dim": 512,
    "num_layers": 2,
    "num_heads": 8,
    "batch_size": 16,
    "chunk_length": 64,
    "policy_batch_size": 1024,
    "context_window": 8,
    "imag_horizon": 16,
    "critic_horizon": 32,
    "actor_hidden_dim": 512,
    "critic_hidden_dim": 512,
    "actor_num_layers": 3,
    "critic_num_layers": 3,
    "actor_layer_norm": True,
    "critic_layer_norm": True,
    "stochastic_actor": True,
    "stochastic_collection": True,
    "policy_replay_critic_loss_coef": 0.1,
    "policy_replay_critic_batch_size": 1024,
    "policy_replay_critic_horizon": 32,
    "reward_prediction_mode": "symlog-twohot",
    "value_prediction_mode": "symlog-twohot",
    "replay_capacity": 1_000_000,
    "clip_imagined_rewards": True,
    "imagined_reward_min": 0.0,
    "imagined_reward_max": 1.0,
    "actor_log_std_max": 0.0,
    "actor_entropy_coef": 1e-4,
    "model_grad_clip_norm": 300.0,
    "actor_grad_clip_norm": 10.0,
    "critic_grad_clip_norm": 30.0,
    "target_critic_ema_decay": 0.98,
    "policy_selection_std_penalty": 0.25,
    "online_policy_std_penalty": 0.25,
    "policy_failure_return_threshold": 100.0,
    "policy_success_return_threshold": 900.0,
    "policy_selection_failure_penalty": 400.0,
    "online_policy_failure_penalty": 400.0,
    "policy_soft_failure_return_threshold": 700.0,
    "policy_soft_failure_penalty": 250.0,
}

_DREAMER_AC_HARD_START_BASE: dict[str, Any] = {
    **_DREAMER_AC_BASE,
    "validation_steps": 256,
    "online_validation_steps": 256,
    "online_candidate_max_anchor_degradation": 0.08,
    "online_candidate_anchor_penalty": 2.0,
    "online_anchor_batch_fraction": 0.5,
    "policy_action_bound_coef": 2.0,
    "policy_action_bound_limit": 0.85,
    "policy_hard_start_max_steps": 65_536,
    "policy_hard_start_fraction": 0.5,
    "policy_hard_critic_fraction": 0.5,
    "policy_hard_start_return_percentile": 0.0,
    "policy_hard_start_absolute_threshold": 700.0,
    "policy_hard_start_prefix_steps": 64,
    "policy_hard_start_recovery_windows": 4,
    "policy_hard_start_mode_buckets": 16,
    "policy_hard_start_balance_modes": True,
    "policy_actor_cvar_fraction": 0.25,
    "policy_actor_cvar_coef": 0.5,
    "online_policy_trust_coef": 3.0,
}

_DREAMER_AC_500K_BASE: dict[str, Any] = {
    **_DREAMER_AC_HARD_START_BASE,
    # Keep evaluation/validation light enough that strict step accounting remains
    # close to the 500k training-replay target.
    "validation_steps": 128,
    "online_iterations": 12,
    "online_validation_steps": 128,
    "final_policy_eval_episodes": 20,
}

_JEPA_DREAMER_PARITY_BASE: dict[str, Any] = {
    # Keep the JEPA latent predictor, but match the control and optimization
    # mechanics of the official DreamerV3 DMC setup as closely as this staged
    # trainer permits.
    "num_envs": 16,
    "env_workers": 16,
    "isolated_rng_streams": True,
    "deterministic_compute": True,
    "collect_steps": 80,
    "validation_steps": 80,
    "replay_capacity": 1_000_000,
    "batch_size": 16,
    "chunk_length": 64,
    "context_window": 8,
    "model_horizon": 5,
    "open_loop_horizon": 5,
    "latent_dim": 128,
    "model_dim": 128,
    "num_layers": 2,
    "num_heads": 4,
    "mlp_ratio": 4,
    "dynamics_ensemble_size": 1,
    "train_steps": 1280,
    "policy_train_steps": 1280,
    "online_collect_steps": 256,
    "online_validation_steps": None,
    "online_train_steps": 4096,
    "online_policy_train_steps": 4096,
    "online_checkpoint_interval": 5,
    "online_candidate_refit": False,
    "online_freeze_encoder": False,
    "online_control_value_weight": 0.0,
    "online_reset_replay_env": False,
    "policy_batch_size": 1024,
    "imag_horizon": 15,
    "critic_warmup_steps": 0,
    "critic_horizon": 64,
    "policy_return_mode": "lambda",
    "policy_actor_baseline": "value",
    "policy_return_normalization": "ema-percentile",
    "policy_return_ema_decay": 0.99,
    "value_clip": 100.0,
    "policy_gradient_mode": "reinforce",
    "policy_actor_cvar_fraction": 1.0,
    "policy_actor_cvar_coef": 0.0,
    "policy_replay_critic_loss_coef": 0.3,
    "policy_replay_critic_batch_size": 16,
    "policy_replay_critic_horizon": 64,
    "policy_replay_critic_return_mode": "lambda",
    "policy_replay_critic_all_steps": True,
    "policy_slow_value_regularization_coef": 1.0,
    "target_critic_ema_decay": 0.98,
    "actor_hidden_dim": 64,
    "critic_hidden_dim": 64,
    "actor_num_layers": 3,
    "critic_num_layers": 3,
    "actor_layer_norm": True,
    "critic_layer_norm": True,
    "stochastic_actor": True,
    "stochastic_collection": True,
    "actor_entropy_coef": 3e-4,
    "actor_log_std_min": -2.302585092994046,
    "actor_log_std_max": 0.0,
    "input_symlog": True,
    "activation": "silu",
    "normalization": "rms",
    "actor_output_scale": 0.01,
    "value_output_scale": 0.0,
    "reward_output_scale": 0.0,
    "reward_prediction_mode": "symlog-twohot",
    "value_prediction_mode": "symlog-twohot",
    "twohot_bins": 255,
    "twohot_min": -20.0,
    "twohot_max": 20.0,
    "learning_rate": 4e-5,
    "actor_learning_rate": 4e-5,
    "model_grad_clip_norm": 0.0,
    "actor_grad_clip_norm": 0.0,
    "critic_grad_clip_norm": 0.0,
    "optimizer_warmup_steps": 1000,
    "adaptive_grad_clip": 0.3,
    "optimizer_epsilon": 1e-8,
    "gamma": 1.0 - 1.0 / 333.0,
    "lambda_return": 0.95,
    "uncertainty_penalty": 0.0,
    "policy_uncertainty_coef": 0.0,
    "policy_action_bound_coef": 0.0,
    "policy_hard_action_bound_coef": 0.0,
    "policy_hard_start_max_steps": 0,
    "policy_hard_start_fraction": 0.0,
    "policy_hard_critic_fraction": 0.0,
    "policy_hard_start_return_percentile": 0.0,
    "policy_trust_coef": 0.0,
    "online_policy_trust_coef": 0.0,
    "clip_imagined_rewards": False,
    "policy_eval_during_training": False,
    "policy_selection_interval": 0,
    "policy_model_selection_interval": 0,
    "policy_confirmation_episodes": 0,
    "online_policy_champion": False,
    "final_policy_eval_episodes": 20,
    "final_policy_eval_seed": 9_000_000,
    "wandb_video_every_phases": 10,
}

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "num_envs": 8,
        "env_workers": 8,
        "collect_steps": 1024,
        "validation_steps": 256,
        "train_steps": 1500,
        "policy_train_steps": 750,
        "online_iterations": 1,
        "online_collect_steps": 512,
        "online_validation_steps": 256,
        "online_train_steps": 750,
        "online_policy_train_steps": 500,
        "policy_selection_interval": 250,
        "policy_selection_episodes": 8,
        "policy_eval_episodes": 16,
        "policy_confirmation_episodes": 16,
        "final_policy_eval_episodes": 0,
    },
    "dreamer_ac": {
        **_DREAMER_AC_BASE,
    },
    "dreamer_ac_online_adaptive": {
        **_DREAMER_AC_BASE,
        # Smaller bootstrap and more online collection rounds, useful for
        # sample-efficiency experiments closer to Dreamer/STORM reporting.
        "collect_steps": 1024,
        "validation_steps": 256,
        "online_iterations": 16,
        "online_collect_steps": 1536,
        "online_validation_steps": 256,
        "online_candidate_max_anchor_degradation": 0.08,
        "online_candidate_anchor_penalty": 2.0,
        "online_anchor_batch_fraction": 0.5,
        "policy_action_bound_coef": 2.0,
        "policy_action_bound_limit": 0.85,
    },
    "dreamer_ac_online_adaptive_hard_start": {
        **_DREAMER_AC_HARD_START_BASE,
        # Current best Reacher/easy setup: data-rich random bootstrap, larger
        # online replay chunks, and actor/critic starts mixed with failed prefixes.
        "online_collect_steps": 6144,
    },
    "dreamer_ac_500k_hard_start_lean": {
        **_DREAMER_AC_500K_BASE,
        # 3.3% initial random data, then almost all learning from online replay.
        "collect_steps": 1024,
        "online_collect_steps": 2496,
    },
    "dreamer_ac_500k_hard_start_balanced": {
        **_DREAMER_AC_500K_BASE,
        # 6.6% initial random data; the first config to try for 920+ by 500k.
        "collect_steps": 2048,
        "online_collect_steps": 2432,
    },
    "dreamer_ac_500k_hard_start_coverage": {
        **_DREAMER_AC_500K_BASE,
        # 13.1% initial random data, still far below the current >920 preset.
        "collect_steps": 4096,
        "online_collect_steps": 2256,
    },
    "jepa_dreamer_parity_100k": {
        **_JEPA_DREAMER_PARITY_BASE,
        # 99,584 train-replay transitions; 100,864 including held-out replay.
        "online_iterations": 24,
    },
    "jepa_dreamer_parity_500k": {
        **_JEPA_DREAMER_PARITY_BASE,
        # 496,896 train-replay transitions; 498,176 including held-out replay.
        "online_iterations": 121,
    },
}

COMMON_PARAMS: dict[str, Any] = {
    "num_runs": 1,
    "isolated_rng_streams": False,
    "deterministic_compute": False,
    "critic_warmup_steps": 1000,
    "critic_horizon": 32,
    "policy_batch_size": 512,
    "policy_return_mode": "reward-only",
    "policy_actor_baseline": "none",
    "policy_return_normalization": "none",
    "policy_return_ema_decay": 0.99,
    "policy_gradient_mode": "dynamics",
    "imag_horizon": 15,
    "final_policy_eval_episodes": 0,
    "final_policy_eval_seed": None,
    "policy_model_selection_interval": 0,
    "policy_model_selection_metric": "policy/imagined_return",
    "policy_model_selection_source": "policy-starts",
    "policy_model_selection_batch_size": None,
    "policy_model_selection_cvar_coef": 0.5,
    "policy_model_selection_uncertainty_penalty": 0.0,
    "policy_model_selection_action_saturation_penalty": 0.0,
    "policy_model_selection_diagnostics": False,
    "policy_selection_std_penalty": 0.0,
    "online_policy_std_penalty": 0.0,
    "online_checkpoint_interval": 0,
    "model_grad_clip_norm": 100.0,
    "actor_grad_clip_norm": 10.0,
    "critic_grad_clip_norm": 100.0,
    "optimizer_warmup_steps": 0,
    "adaptive_grad_clip": 0.0,
    "optimizer_epsilon": 1e-5,
    "learning_rate": None,
    "actor_learning_rate": None,
    "target_critic_ema_decay": 0.0,
    "actor_hidden_dim": 0,
    "critic_hidden_dim": 0,
    "actor_num_layers": 1,
    "critic_num_layers": 1,
    "actor_layer_norm": False,
    "critic_layer_norm": False,
    "stochastic_actor": False,
    "stochastic_collection": False,
    "actor_entropy_coef": 0.0,
    "actor_log_std_min": -5.0,
    "actor_log_std_max": 2.0,
    "input_symlog": False,
    "activation": "gelu",
    "normalization": "layer",
    "actor_output_scale": 1.0,
    "value_output_scale": 1.0,
    "reward_output_scale": 1.0,
    "policy_real_critic_interval": 0,
    "policy_real_critic_updates": 1,
    "policy_real_critic_batch_size": None,
    "policy_replay_critic_loss_coef": 0.0,
    "policy_replay_critic_batch_size": None,
    "policy_replay_critic_horizon": None,
    "policy_replay_critic_return_mode": "reward-only",
    "policy_replay_critic_all_steps": False,
    "policy_slow_value_regularization_coef": 0.0,
    "value_clip": 100.0,
    "policy_hard_start_max_steps": 0,
    "policy_hard_start_fraction": 0.0,
    "policy_hard_critic_fraction": 0.0,
    "policy_hard_start_return_percentile": 30.0,
    "policy_hard_start_absolute_threshold": None,
    "policy_hard_start_prefix_steps": 64,
    "policy_hard_start_recovery_windows": 1,
    "policy_hard_start_recovery_stride": 8,
    "policy_hard_start_mode_buckets": 0,
    "policy_hard_start_balance_modes": False,
    "policy_hard_action_bound_coef": 0.0,
    "online_policy_trust_coef": 1.0,
    "online_candidate_refit": True,
    "online_freeze_encoder": True,
    "online_reset_replay_env": True,
    "online_candidate_eval_interval": 250,
    "online_candidate_min_recent_improvement": 0.0,
    "online_candidate_max_anchor_degradation": 0.03,
    "online_anchor_batch_fraction": 0.5,
    "online_control_value_weight": 0.1,
    "batch_size": 64,
    "chunk_length": 32,
    "open_loop_horizon": 15,
    "model_horizon": 5,
    "context_window": 4,
    "latent_dim": 128,
    "model_dim": 128,
    "num_layers": 2,
    "num_heads": 4,
    "mlp_ratio": 4,
    "dynamics_ensemble_size": 5,
    "uncertainty_penalty": 0.1,
    "policy_uncertainty_coef": 0.0,
    "regularizer": "sigreg",
    "regularizer_weight": 0.05,
    "reward_prediction_mode": "mse",
    "value_prediction_mode": "mse",
    "twohot_bins": 41,
    "twohot_min": -20.0,
    "twohot_max": 20.0,
    "clip_imagined_rewards": False,
    "imagined_reward_min": 0.0,
    "imagined_reward_max": 1.0,
    "controls": ("none",),
    "allow_fail": True,
}

OVERRIDABLE_PARAMS = (
    "num_envs",
    "env_workers",
    "isolated_rng_streams",
    "deterministic_compute",
    "collect_steps",
    "validation_steps",
    "replay_capacity",
    "save_initial_replay",
    "load_initial_replay",
    "online_iterations",
    "online_collect_steps",
    "online_validation_steps",
    "train_steps",
    "online_train_steps",
    "policy_train_steps",
    "online_policy_train_steps",
    "online_checkpoint_interval",
    "policy_return_mode",
    "policy_actor_baseline",
    "policy_return_normalization",
    "policy_return_ema_decay",
    "value_clip",
    "policy_gradient_mode",
    "policy_actor_cvar_fraction",
    "policy_actor_cvar_coef",
    "policy_eval_during_training",
    "policy_selection_interval",
    "policy_model_selection_interval",
    "policy_model_selection_metric",
    "policy_model_selection_source",
    "policy_model_selection_batch_size",
    "policy_model_selection_cvar_coef",
    "policy_model_selection_uncertainty_penalty",
    "policy_model_selection_action_saturation_penalty",
    "policy_model_selection_diagnostics",
    "policy_selection_episodes",
    "policy_eval_episodes",
    "policy_confirmation_episodes",
    "final_policy_eval_episodes",
    "final_policy_eval_seed",
    "policy_selection_std_penalty",
    "policy_selection_failure_penalty",
    "policy_failure_return_threshold",
    "policy_success_return_threshold",
    "policy_soft_failure_return_threshold",
    "policy_soft_failure_penalty",
    "online_policy_std_penalty",
    "online_policy_failure_penalty",
    "online_policy_champion",
    "reward_prediction_mode",
    "value_prediction_mode",
    "twohot_bins",
    "twohot_min",
    "twohot_max",
    "clip_imagined_rewards",
    "imagined_reward_min",
    "imagined_reward_max",
    "model_grad_clip_norm",
    "actor_grad_clip_norm",
    "critic_grad_clip_norm",
    "optimizer_warmup_steps",
    "adaptive_grad_clip",
    "optimizer_epsilon",
    "policy_uncertainty_coef",
    "policy_action_bound_coef",
    "policy_action_bound_limit",
    "learning_rate",
    "actor_learning_rate",
    "target_critic_ema_decay",
    "actor_hidden_dim",
    "critic_hidden_dim",
    "actor_num_layers",
    "critic_num_layers",
    "actor_layer_norm",
    "critic_layer_norm",
    "stochastic_actor",
    "stochastic_collection",
    "actor_entropy_coef",
    "actor_log_std_min",
    "actor_log_std_max",
    "input_symlog",
    "activation",
    "normalization",
    "actor_output_scale",
    "value_output_scale",
    "reward_output_scale",
    "policy_real_critic_interval",
    "policy_real_critic_updates",
    "policy_real_critic_batch_size",
    "policy_replay_critic_loss_coef",
    "policy_replay_critic_batch_size",
    "policy_replay_critic_horizon",
    "policy_replay_critic_return_mode",
    "policy_replay_critic_all_steps",
    "policy_slow_value_regularization_coef",
    "policy_hard_start_max_steps",
    "policy_hard_start_fraction",
    "policy_hard_critic_fraction",
    "policy_hard_start_return_percentile",
    "policy_hard_start_absolute_threshold",
    "policy_hard_start_prefix_steps",
    "policy_hard_start_recovery_windows",
    "policy_hard_start_recovery_stride",
    "policy_hard_start_mode_buckets",
    "policy_hard_start_balance_modes",
    "policy_hard_action_bound_coef",
    "online_candidate_refit",
    "online_freeze_encoder",
    "online_reset_replay_env",
    "online_candidate_eval_interval",
    "online_candidate_min_recent_improvement",
    "online_candidate_max_anchor_degradation",
    "online_candidate_anchor_penalty",
    "online_anchor_batch_fraction",
    "online_control_value_weight",
    "batch_size",
    "policy_batch_size",
    "imag_horizon",
    "gamma",
    "lambda_return",
    "latent_dim",
    "model_dim",
    "num_layers",
    "num_heads",
    "wandb_project",
    "wandb_entity",
    "wandb_name",
    "wandb_group",
    "wandb_tags",
    "wandb_mode",
    "wandb_videos",
    "wandb_video_every_phases",
    "wandb_video_frame_stride",
    "wandb_video_size",
    "wandb_video_fps",
    "wandb_video_camera",
)

NEGATED_BOOL_PARAMS = {
    "online_freeze_encoder",
    "online_policy_champion",
    "online_reset_replay_env",
    "policy_eval_during_training",
}


def main() -> None:
    args = parse_args()
    params = {**COMMON_PARAMS, **PRESETS[args.preset]}
    apply_optional_overrides(args, params)

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = [
        {
            "task": task,
            "seed": int(seed),
            "short": f"{task_short_name(task)}_seed{seed}",
        }
        for task in args.tasks
        for seed in args.seeds
    ]
    manifest = {
        "preset": args.preset,
        "tasks": args.tasks,
        "seeds": args.seeds,
        "gpus": args.gpus,
        "out_root": str(out_root),
        "params": params,
        "step_accounting": step_accounting(params),
        "jobs": jobs,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    write_run_one(out_root, params)
    write_launcher(
        out_root,
        jobs,
        args.gpus,
        sync=args.sync,
        tracking=bool(params.get("wandb_project")),
    )
    write_tail(out_root)
    write_summarize(out_root)

    print(f"Wrote DMC vector launcher to {out_root}")
    print(f"- {out_root / 'manifest.json'}")
    print(f"- {out_root / 'run_one.sh'}")
    print(f"- {out_root / 'launcher.sh'}")
    print(f"- {out_root / 'tail.sh'}")
    print(f"- {out_root / 'summarize.sh'}")
    print()
    print(
        f"Start with: nohup bash {shlex.quote(str(out_root / 'launcher.sh'))} "
        f"> {shlex.quote(str(out_root / 'launcher.nohup.log'))} 2>&1 &"
    )
    print(f"Watch with: bash {shlex.quote(str(out_root / 'tail.sh'))}")

    if args.start:
        log = (out_root / "launcher.nohup.log").open("wb")
        process = subprocess.Popen(
            ["bash", str(out_root / "launcher.sh")],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"launcher pid: {process.pid}")


def step_accounting(params: dict[str, Any]) -> dict[str, int | float | None]:
    num_envs = _int_param(params, "num_envs")
    collect_steps = _int_param(params, "collect_steps")
    validation_steps = _int_param(params, "validation_steps")
    online_iterations = _int_param(params, "online_iterations")
    online_collect_steps = _int_param(params, "online_collect_steps")
    online_validation_steps = _int_param(params, "online_validation_steps")
    train_steps = _int_param(params, "train_steps")
    online_train_steps = _int_param(params, "online_train_steps")
    policy_train_steps = _int_param(params, "policy_train_steps")
    online_policy_train_steps = _int_param(params, "online_policy_train_steps")
    train_replay_vector_steps = (
        None
        if None in (collect_steps, online_iterations, online_collect_steps)
        else collect_steps + online_iterations * online_collect_steps
    )
    if not params.get("online_candidate_refit", False):
        validation_replay_vector_steps = validation_steps
    else:
        resolved_online_validation_steps = online_validation_steps
        if resolved_online_validation_steps is None and None not in (
            validation_steps,
            online_collect_steps,
        ):
            resolved_online_validation_steps = min(
                validation_steps,
                online_collect_steps,
            )
        validation_replay_vector_steps = (
            None
            if None
            in (validation_steps, online_iterations, resolved_online_validation_steps)
            else validation_steps + online_iterations * resolved_online_validation_steps
        )
    train_replay_env_steps = (
        None
        if None in (num_envs, train_replay_vector_steps)
        else num_envs * train_replay_vector_steps
    )
    validation_replay_env_steps = (
        None
        if None in (num_envs, validation_replay_vector_steps)
        else num_envs * validation_replay_vector_steps
    )
    world_model_updates = _phase_total(
        train_steps,
        online_iterations,
        online_train_steps,
    )
    policy_updates = _phase_total(
        policy_train_steps,
        online_iterations,
        online_policy_train_steps,
    )
    batch_size = _int_param(params, "batch_size")
    chunk_length = _int_param(params, "chunk_length")
    world_model_sampled_transitions = _product_optional(
        world_model_updates,
        batch_size,
        chunk_length,
    )
    world_model_replay_ratio = (
        None
        if world_model_sampled_transitions is None
        or train_replay_env_steps in (None, 0)
        else world_model_sampled_transitions / train_replay_env_steps
    )
    return {
        "num_envs": num_envs,
        "train_replay_vector_steps": train_replay_vector_steps,
        "train_replay_env_steps": train_replay_env_steps,
        "validation_replay_vector_steps": validation_replay_vector_steps,
        "validation_replay_env_steps": validation_replay_env_steps,
        "train_plus_validation_vector_steps": _sum_optional(
            train_replay_vector_steps,
            validation_replay_vector_steps,
        ),
        "train_plus_validation_env_steps": _sum_optional(
            train_replay_env_steps,
            validation_replay_env_steps,
        ),
        "world_model_updates": world_model_updates,
        "policy_updates": policy_updates,
        "world_model_sampled_transitions": world_model_sampled_transitions,
        "world_model_replay_ratio": world_model_replay_ratio,
        "final_policy_eval_episodes": _int_param(
            params,
            "final_policy_eval_episodes",
        ),
    }


def _int_param(params: dict[str, Any], key: str) -> int | None:
    value = params.get(key)
    if value is None:
        return None
    return int(value)


def _sum_optional(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return left + right


def _phase_total(
    initial: int | None,
    iterations: int | None,
    online: int | None,
) -> int | None:
    if initial is None or iterations is None:
        return None
    resolved_online = initial if online is None else online
    return initial + iterations * resolved_online


def _product_optional(*values: int | None) -> int | None:
    if any(value is None for value in values):
        return None
    result = 1
    for value in values:
        assert value is not None
        result *= value
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/dmc_jepa_vector_hard_start"),
        help="Directory where launcher scripts, logs, and task runs are written.",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="dreamer_ac_online_adaptive_hard_start",
        help="Run preset. Use smoke first to verify a new pod.",
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["0"],
        help="CUDA device ids. Jobs are launched in batches, one per listed GPU.",
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--env-workers", type=int, default=None)
    parser.add_argument("--collect-steps", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--save-initial-replay", default=None)
    parser.add_argument("--load-initial-replay", default=None)
    parser.add_argument("--online-iterations", type=int, default=None)
    parser.add_argument("--online-collect-steps", type=int, default=None)
    parser.add_argument("--online-validation-steps", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--online-train-steps", type=int, default=None)
    parser.add_argument("--policy-train-steps", type=int, default=None)
    parser.add_argument("--online-policy-train-steps", type=int, default=None)
    parser.add_argument("--online-checkpoint-interval", type=int, default=None)
    parser.add_argument(
        "--isolated-rng-streams",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--deterministic-compute",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--policy-return-mode",
        choices=("reward-only", "lambda"),
        default=None,
    )
    parser.add_argument(
        "--policy-actor-baseline",
        choices=("none", "value"),
        default=None,
    )
    parser.add_argument(
        "--policy-return-normalization",
        choices=("none", "batch", "percentile", "ema-percentile"),
        default=None,
    )
    parser.add_argument("--policy-return-ema-decay", type=float, default=None)
    parser.add_argument("--value-clip", type=float, default=None)
    parser.add_argument(
        "--policy-gradient-mode",
        choices=("dynamics", "reinforce"),
        default=None,
    )
    parser.add_argument("--policy-actor-cvar-fraction", type=float, default=None)
    parser.add_argument("--policy-actor-cvar-coef", type=float, default=None)
    parser.add_argument(
        "--policy-eval-during-training",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--policy-selection-interval", type=int, default=None)
    parser.add_argument("--policy-model-selection-interval", type=int, default=None)
    parser.add_argument(
        "--policy-model-selection-metric",
        choices=(
            "policy/imagined_return",
            "policy/clipped_imagined_return",
            "policy/actor_score",
            "policy/actor_objective_score",
            "policy/actor_objective_cvar_score",
            "policy/heldout_model_score",
        ),
        default=None,
    )
    parser.add_argument(
        "--policy-model-selection-source",
        choices=("policy-starts", "validation-replay"),
        default=None,
    )
    parser.add_argument("--policy-model-selection-batch-size", type=int, default=None)
    parser.add_argument("--policy-model-selection-cvar-coef", type=float, default=None)
    parser.add_argument(
        "--policy-model-selection-uncertainty-penalty",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--policy-model-selection-action-saturation-penalty",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--policy-model-selection-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--policy-selection-episodes", type=int, default=None)
    parser.add_argument("--policy-eval-episodes", type=int, default=None)
    parser.add_argument("--policy-confirmation-episodes", type=int, default=None)
    parser.add_argument("--final-policy-eval-episodes", type=int, default=None)
    parser.add_argument("--final-policy-eval-seed", type=int, default=None)
    parser.add_argument("--policy-selection-std-penalty", type=float, default=None)
    parser.add_argument("--policy-selection-failure-penalty", type=float, default=None)
    parser.add_argument("--policy-failure-return-threshold", type=float, default=None)
    parser.add_argument("--policy-success-return-threshold", type=float, default=None)
    parser.add_argument(
        "--policy-soft-failure-return-threshold", type=float, default=None
    )
    parser.add_argument("--policy-soft-failure-penalty", type=float, default=None)
    parser.add_argument("--online-policy-std-penalty", type=float, default=None)
    parser.add_argument("--online-policy-failure-penalty", type=float, default=None)
    parser.add_argument(
        "--online-policy-champion",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--reward-prediction-mode",
        choices=("mse", "symlog-twohot"),
        default=None,
    )
    parser.add_argument(
        "--value-prediction-mode",
        choices=("mse", "symlog-twohot"),
        default=None,
    )
    parser.add_argument("--twohot-bins", type=int, default=None)
    parser.add_argument("--twohot-min", type=float, default=None)
    parser.add_argument("--twohot-max", type=float, default=None)
    parser.add_argument(
        "--clip-imagined-rewards",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--imagined-reward-min", type=float, default=None)
    parser.add_argument("--imagined-reward-max", type=float, default=None)
    parser.add_argument("--model-grad-clip-norm", type=float, default=None)
    parser.add_argument("--actor-grad-clip-norm", type=float, default=None)
    parser.add_argument("--critic-grad-clip-norm", type=float, default=None)
    parser.add_argument("--optimizer-warmup-steps", type=int, default=None)
    parser.add_argument("--adaptive-grad-clip", type=float, default=None)
    parser.add_argument("--optimizer-epsilon", type=float, default=None)
    parser.add_argument("--policy-uncertainty-coef", type=float, default=None)
    parser.add_argument("--policy-action-bound-coef", type=float, default=None)
    parser.add_argument("--policy-action-bound-limit", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--actor-learning-rate", type=float, default=None)
    parser.add_argument("--target-critic-ema-decay", type=float, default=None)
    parser.add_argument("--actor-hidden-dim", type=int, default=None)
    parser.add_argument("--critic-hidden-dim", type=int, default=None)
    parser.add_argument("--actor-num-layers", type=int, default=None)
    parser.add_argument("--critic-num-layers", type=int, default=None)
    parser.add_argument(
        "--actor-layer-norm",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--critic-layer-norm",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--stochastic-actor",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--stochastic-collection",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--actor-entropy-coef", type=float, default=None)
    parser.add_argument("--actor-log-std-min", type=float, default=None)
    parser.add_argument("--actor-log-std-max", type=float, default=None)
    parser.add_argument(
        "--input-symlog",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--activation", choices=("gelu", "silu"), default=None)
    parser.add_argument("--normalization", choices=("layer", "rms"), default=None)
    parser.add_argument("--actor-output-scale", type=float, default=None)
    parser.add_argument("--value-output-scale", type=float, default=None)
    parser.add_argument("--reward-output-scale", type=float, default=None)
    parser.add_argument("--policy-real-critic-interval", type=int, default=None)
    parser.add_argument("--policy-real-critic-updates", type=int, default=None)
    parser.add_argument("--policy-real-critic-batch-size", type=int, default=None)
    parser.add_argument("--policy-replay-critic-loss-coef", type=float, default=None)
    parser.add_argument("--policy-replay-critic-batch-size", type=int, default=None)
    parser.add_argument("--policy-replay-critic-horizon", type=int, default=None)
    parser.add_argument(
        "--policy-replay-critic-return-mode",
        choices=("reward-only", "lambda"),
        default=None,
    )
    parser.add_argument(
        "--policy-replay-critic-all-steps",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--policy-slow-value-regularization-coef",
        type=float,
        default=None,
    )
    parser.add_argument("--policy-hard-start-max-steps", type=int, default=None)
    parser.add_argument("--policy-hard-start-fraction", type=float, default=None)
    parser.add_argument("--policy-hard-critic-fraction", type=float, default=None)
    parser.add_argument(
        "--policy-hard-start-return-percentile",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--policy-hard-start-absolute-threshold",
        type=float,
        default=None,
    )
    parser.add_argument("--policy-hard-start-prefix-steps", type=int, default=None)
    parser.add_argument(
        "--policy-hard-start-recovery-windows",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--policy-hard-start-recovery-stride",
        type=int,
        default=None,
    )
    parser.add_argument("--policy-hard-start-mode-buckets", type=int, default=None)
    parser.add_argument(
        "--policy-hard-start-balance-modes",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--policy-hard-action-bound-coef", type=float, default=None)
    parser.add_argument(
        "--online-candidate-refit",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--online-freeze-encoder",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--online-reset-replay-env",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--online-candidate-eval-interval", type=int, default=None)
    parser.add_argument(
        "--online-candidate-min-recent-improvement",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--online-candidate-max-anchor-degradation",
        type=float,
        default=None,
    )
    parser.add_argument("--online-candidate-anchor-penalty", type=float, default=None)
    parser.add_argument("--online-anchor-batch-fraction", type=float, default=None)
    parser.add_argument("--online-control-value-weight", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--policy-batch-size", type=int, default=None)
    parser.add_argument("--imag-horizon", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--lambda-return", type=float, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
    )
    parser.add_argument(
        "--wandb-videos",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--wandb-video-every-phases", type=int, default=None)
    parser.add_argument("--wandb-video-frame-stride", type=int, default=None)
    parser.add_argument("--wandb-video-size", type=int, default=None)
    parser.add_argument("--wandb-video-fps", type=int, default=None)
    parser.add_argument("--wandb-video-camera", type=int, default=None)
    parser.add_argument(
        "--no-sync",
        dest="sync",
        action="store_false",
        default=True,
        help="Do not run uv sync at launcher start.",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start launcher immediately in the background.",
    )
    return parser.parse_args()


def apply_optional_overrides(args: argparse.Namespace, params: dict[str, Any]) -> None:
    for name in OVERRIDABLE_PARAMS:
        value = getattr(args, name)
        if value is not None:
            params[name] = value


def write_run_one(out_root: Path, params: dict[str, Any]) -> None:
    command_args = params_to_shell_args(params)
    body = dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        UV_BIN="${{UV_BIN:-}}"
        if [[ -z "$UV_BIN" ]]; then
          if command -v uv >/dev/null 2>&1; then
            UV_BIN="$(command -v uv)"
          elif [[ -x /root/.local/bin/uv ]]; then
            UV_BIN="/root/.local/bin/uv"
          else
            echo "uv not found; set UV_BIN to the uv executable path" >&2
            exit 127
          fi
        fi
        TASK="${{TASK:?TASK is required, for example reacher/easy}}"
        SEED="${{SEED:?SEED is required}}"
        SHORT="${{SHORT:-$(echo "$TASK" | tr '/-' '__')_seed${{SEED}}}}"
        OUTROOT="{out_root}"
        OUT="$OUTROOT/$SHORT"
        mkdir -p "$OUT"

        echo "==== starting dmc:$TASK seed=$SEED on CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}} ===="
        echo "out=$OUT"

        "$UV_BIN" run world-marl-validate-single-agent-world-model \\
          --env "dmc:$TASK" \\
          --seed "$SEED" \\
          {command_args} \\
          --out-dir "$OUT"

        echo "==== finished dmc:$TASK seed=$SEED ===="
        """
    )
    write_executable(out_root / "run_one.sh", body)


def params_to_shell_args(params: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in params.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                parts.append(flag)
            elif key in NEGATED_BOOL_PARAMS:
                parts.append("--no-" + key.replace("_", "-"))
            continue
        if isinstance(value, (list, tuple)):
            parts.append(flag)
            parts.extend(shlex.quote(str(item)) for item in value)
            continue
        parts.extend((flag, shlex.quote(str(value))))
    return " \\\n          ".join(parts)


def write_launcher(
    out_root: Path,
    jobs: list[dict[str, Any]],
    gpus: list[str],
    *,
    sync: bool,
    tracking: bool,
) -> None:
    jobs_block = "\n".join(
        f"  {shlex.quote(job['task'] + '|' + str(job['seed']) + '|' + job['short'])}"
        for job in jobs
    )
    gpus_block = " ".join(shlex.quote(gpu) for gpu in gpus)
    tracking_extra = " --extra tracking" if tracking else ""
    sync_block = (
        f'"$UV_BIN" sync --extra dmc --extra cuda12{tracking_extra}\n'
        if sync
        else 'echo "Skipping uv sync because launcher was generated with --no-sync"\n'
    )
    body = dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

        export UV_PROJECT_ENVIRONMENT="${{UV_PROJECT_ENVIRONMENT:-/tmp/wm-marl-venv}}"
        export UV_CACHE_DIR="${{UV_CACHE_DIR:-/tmp/uv-cache-wm-marl}}"
        export UV_LINK_MODE="${{UV_LINK_MODE:-copy}}"
        export XLA_PYTHON_CLIENT_PREALLOCATE="${{XLA_PYTHON_CLIENT_PREALLOCATE:-false}}"
        export JAX_PLATFORMS="${{JAX_PLATFORMS:-cuda}}"

        UV_BIN="${{UV_BIN:-}}"
        if [[ -z "$UV_BIN" ]]; then
          if command -v uv >/dev/null 2>&1; then
            UV_BIN="$(command -v uv)"
          elif [[ -x /root/.local/bin/uv ]]; then
            UV_BIN="/root/.local/bin/uv"
          else
            echo "uv not found; set UV_BIN to the uv executable path" >&2
            exit 127
          fi
        fi
        {sync_block}
        OUTROOT="{out_root}"
        GPUS=({gpus_block})
        JOBS=(
        {jobs_block}
        )

        index=0
        total="${{#JOBS[@]}}"
        while (( index < total )); do
          pids=()
          for gpu in "${{GPUS[@]}}"; do
            if (( index >= total )); then
              break
            fi
            IFS='|' read -r task seed short <<< "${{JOBS[$index]}}"
            log="$OUTROOT/$short.nohup.log"
            echo "launching dmc:$task seed=$seed on GPU $gpu -> $log"
            CUDA_VISIBLE_DEVICES="$gpu" TASK="$task" SEED="$seed" SHORT="$short" \\
              UV_BIN="$UV_BIN" bash "$OUTROOT/run_one.sh" > "$log" 2>&1 &
            pids+=("$!")
            index=$((index + 1))
          done
          for pid in "${{pids[@]}}"; do
            wait "$pid"
          done
        done

        echo "all DMC vector jobs finished"
        """
    )
    write_executable(out_root / "launcher.sh", body)


def write_tail(out_root: Path) -> None:
    body = dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
        OUTROOT="{out_root}"

        echo "== processes =="
        pgrep -af "world-marl-validate-single-agent-world-model|dmc_jepa_vector" || true
        echo
        echo "== gpu =="
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv 2>/dev/null || true
        echo
        echo "== summaries =="
        find "$OUTROOT" -path "*/summary.json" -print | sort || true
        echo
        echo "== latest logs =="
        find "$OUTROOT" -maxdepth 1 -name "*.nohup.log" -printf "%T@ %p\\n" 2>/dev/null \\
          | sort -n | tail -4 | cut -d' ' -f2- | while read -r log; do
              echo
              echo "==== $log ===="
              tail -n 12 "$log" | tr '\\r' '\\n' | tail -n 12
            done
        """
    )
    write_executable(out_root / "tail.sh", body)


def write_summarize(out_root: Path) -> None:
    body = dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
        python - <<'PY'
        import json
        import pathlib

        root = pathlib.Path({str(out_root)!r})
        paths = sorted(root.glob("*/*/summary.json"))
        if not paths:
            print("no summaries yet")
            raise SystemExit(0)

        print("job,passed,world,policy,initial,trained,trained_fail,trained_success,champion,final_eval,final_eval_std,final_fail,final_success,dreamer_train,dreamer_eps,dreamer_budget_reached,improve,online,model_accept,policy_accept,train_replay_steps,strict_total_steps,open_loop")
        for path in paths:
            job = path.parts[-3]
            summary = json.loads(path.read_text())
            values = [
                job,
                summary.get("passed"),
                summary.get("world_model_passed"),
                summary.get("policy_main_passed"),
                summary.get("aggregate_policy_initial_mean"),
                summary.get("aggregate_policy_trained_mean"),
                summary.get("aggregate_policy_trained_failure_rate"),
                summary.get("aggregate_policy_trained_success_rate"),
                summary.get("aggregate_policy_final_champion_return"),
                summary.get("aggregate_final_policy_eval_mean"),
                summary.get("aggregate_final_policy_eval_std"),
                summary.get("aggregate_final_policy_eval_failure_rate"),
                summary.get("aggregate_final_policy_eval_success_rate"),
                summary.get("aggregate_dreamer_style_train_return_mean"),
                summary.get("aggregate_dreamer_style_train_return_episodes"),
                summary.get("aggregate_dreamer_style_train_return_budget_reached"),
                summary.get("aggregate_policy_improvement"),
                summary.get("aggregate_policy_online_phase_improvement"),
                summary.get("aggregate_model_update_acceptance_rate"),
                summary.get("aggregate_policy_update_acceptance_rate"),
                summary.get("aggregate_real_train_replay_env_steps"),
                summary.get("aggregate_real_total_env_steps"),
                summary.get("aggregate_final_open_loop_loss"),
            ]
            print(",".join("" if value is None else str(value) for value in values))
        PY
        """
    )
    write_executable(out_root / "summarize.sh", body)


def write_executable(path: Path, text: str) -> None:
    path.write_text(dedent(text).lstrip(), encoding="utf-8")
    path.chmod(0o755)


def task_short_name(task: str) -> str:
    return task.replace("/", "_").replace("-", "_")


if __name__ == "__main__":
    main()
