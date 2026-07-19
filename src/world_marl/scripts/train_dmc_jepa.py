"""Train the maintained JEPA model-based RL algorithm on vector control tasks.

The runner intentionally exposes one publication-oriented training path:
collect reset-rich bootstrap replay, fit an action-conditioned JEPA world model,
train actor and critic heads in latent imagination, then interleave real data,
world-model updates, and policy updates. The latest policy is retained throughout;
real-environment interactions are never used to select checkpoints.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

from world_marl.determinism import configure_deterministic_environment

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from world_marl.checkpointing import load_params, save_checkpoint
from world_marl.envs.brax_adapter import BraxVectorAdapter, brax_env_name
from world_marl.envs.dmc_adapter import DMCVectorAdapter, dmc_env_name
from world_marl.jepa.config import canonical_jepa_config
from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.replay import ReplayBatch, SequenceReplayBuffer
from world_marl.jepa.reporting import (
    collection_report_summary as _collection_report_summary,
    dreamer_style_training_score as _dreamer_style_training_score,
    optional_value as _nested,
    real_step_accounting as _real_step_accounting,
    return_tail_metrics as _return_tail_metrics,
)
from world_marl.jepa.reproducibility import (
    JaxRngStreams,
    NumpyRngStreams,
    fingerprint_pytree,
)
from world_marl.jepa.schedule import (
    effective_recent_fraction as _effective_recent_fraction,
    recent_oversample_ratio as _recent_oversample_ratio,
    sample_policy_starts_with_reset_mix as _sample_policy_starts_with_reset_mix,
    sample_replay_batch as _sample_replay_batch,
    scheduled_online_actor_update_interval as _scheduled_online_actor_update_interval,
    scheduled_online_encoder_freeze as _scheduled_online_encoder_freeze,
    scheduled_policy_reset_start_fraction as _scheduled_policy_reset_start_fraction,
    scheduled_recent_world_model_fraction as _scheduled_recent_world_model_fraction,
    scheduled_value_clip as _scheduled_value_clip,
)
from world_marl.jepa.training import (
    continuous_policy_train_step,
    create_jepa_train_state,
    evaluate_open_loop,
    evaluate_world_model_loss,
    reset_policy_heads,
    select_continuous_actions,
    train_model_step,
)
from world_marl.jepa.training_snapshot import (
    load_training_snapshot,
    save_training_snapshot,
)
from world_marl.logging import (
    RunLogger,
    WandbConfig,
    dependency_versions,
    timestamp,
    to_jsonable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    environment = parser.add_argument_group("environment and data")
    environment.add_argument("--env", default="dmc:reacher/easy")
    environment.add_argument("--num-envs", type=int, default=16)
    environment.add_argument(
        "--env-workers",
        dest="env_workers",
        type=int,
        default=1,
    )
    environment.add_argument("--brax-backend", default=None)
    environment.add_argument("--max-cycles", type=int, default=1000)
    environment.add_argument("--collect-steps", type=int, default=320)
    environment.add_argument("--initial-reset-interval", type=int, default=80)
    environment.add_argument(
        "--initial-random-action-hold-steps",
        type=int,
        default=1,
        help=(
            "Number of bootstrap environment steps to hold each sampled random "
            "action; 1 preserves per-step i.i.d. random collection."
        ),
    )
    environment.add_argument("--validation-steps", type=int, default=80)
    environment.add_argument("--validation-seed", type=int, default=None)
    environment.add_argument("--replay-capacity", type=int, default=1_000_000)
    environment.add_argument("--batch-size", type=int, default=16)
    environment.add_argument("--chunk-length", type=int, default=64)

    world_model = parser.add_argument_group("JEPA world model")
    world_model.add_argument("--train-steps", type=int, default=1280)
    world_model.add_argument("--eval-interval", type=int, default=250)
    world_model.add_argument("--model-horizon", type=int, default=5)
    world_model.add_argument("--open-loop-horizon", type=int, default=5)
    world_model.add_argument("--context-window", type=int, default=8)
    world_model.add_argument("--latent-dim", type=int, default=128)
    world_model.add_argument("--model-dim", type=int, default=128)
    world_model.add_argument("--num-layers", type=int, default=2)
    world_model.add_argument("--num-heads", type=int, default=4)
    world_model.add_argument("--mlp-ratio", type=int, default=4)
    world_model.add_argument("--learning-rate", type=float, default=4e-5)
    world_model.add_argument("--model-grad-clip-norm", type=float, default=0.0)
    world_model.add_argument("--optimizer-warmup-steps", type=int, default=1000)
    world_model.add_argument("--adaptive-grad-clip", type=float, default=0.3)
    world_model.add_argument("--optimizer-epsilon", type=float, default=1e-8)
    world_model.add_argument("--reward-output-scale", type=float, default=0.0)
    world_model.add_argument(
        "--regularizer-weight",
        dest="regularizer_weight",
        type=float,
        default=0.05,
    )
    world_model.add_argument("--sigreg-knots", type=int, default=17)
    world_model.add_argument("--sigreg-num-proj", type=int, default=256)
    world_model.add_argument("--reward-weight", type=float, default=1.0)
    world_model.add_argument("--continue-weight", type=float, default=1.0)
    world_model.add_argument("--twohot-bins", type=int, default=255)
    world_model.add_argument("--twohot-min", type=float, default=-20.0)
    world_model.add_argument("--twohot-max", type=float, default=20.0)

    policy = parser.add_argument_group("actor and critic")
    policy.add_argument("--policy-train-steps", type=int, default=1280)
    policy.add_argument("--policy-batch-size", type=int, default=1024)
    policy.add_argument(
        "--policy-reset-start-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of online actor imagination starts sampled near episode "
            "starts in the growing main replay. This is reward-agnostic and "
            "does not add environment interactions."
        ),
    )
    policy.add_argument(
        "--policy-reset-start-fraction-start-env-steps",
        type=int,
        default=0,
        help=(
            "Training-transition threshold at which reset-aligned actor "
            "starts become active. Before this threshold their effective "
            "fraction is zero."
        ),
    )
    policy.add_argument(
        "--policy-reset-start-max-age",
        type=int,
        default=63,
        help="Largest within-episode observation age eligible for reset starts.",
    )
    policy.add_argument("--actor-learning-rate", type=float, default=4e-5)
    policy.add_argument("--actor-grad-clip-norm", type=float, default=10.0)
    policy.add_argument("--critic-grad-clip-norm", type=float, default=100.0)
    policy.add_argument("--actor-hidden-dim", type=int, default=64)
    policy.add_argument("--critic-hidden-dim", type=int, default=64)
    policy.add_argument("--actor-num-layers", type=int, default=3)
    policy.add_argument("--critic-num-layers", type=int, default=3)
    policy.add_argument(
        "--actor-layer-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    policy.add_argument(
        "--critic-layer-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    policy.add_argument("--actor-entropy-coef", type=float, default=3e-3)
    policy.add_argument(
        "--actor-log-std-min",
        type=float,
        default=-2.302585092994046,
    )
    policy.add_argument("--actor-log-std-max", type=float, default=0.0)
    policy.add_argument("--actor-output-scale", type=float, default=0.01)
    policy.add_argument("--critic-horizon", type=int, default=64)
    policy.add_argument("--imag-horizon", type=int, default=15)
    policy.add_argument("--policy-return-ema-decay", type=float, default=0.99)
    policy.add_argument(
        "--value-clip",
        type=float,
        default=100.0,
        help="Symmetric value-target clip; set to 0 to disable clipping.",
    )
    policy.add_argument(
        "--value-clip-final",
        type=float,
        default=None,
        help=(
            "Optional final value-target clip for a linear training-step schedule. "
            "Requires both value-clip schedule boundaries."
        ),
    )
    policy.add_argument(
        "--value-clip-schedule-start-env-steps",
        type=int,
        default=None,
        help="Training transition at which to begin changing the value clip.",
    )
    policy.add_argument(
        "--value-clip-schedule-end-env-steps",
        type=int,
        default=None,
        help="Training transition at which to reach the final value clip.",
    )
    policy.add_argument(
        "--policy-actor-kl-coef",
        type=float,
        default=0.0,
        help="Penalty coefficient for actor KL beyond the configured target.",
    )
    policy.add_argument(
        "--policy-actor-kl-target-per-dim",
        type=float,
        default=0.01,
        help="Allowed full-Gaussian actor KL per action dimension.",
    )
    policy.add_argument(
        "--policy-actor-kl-reference-interval",
        type=int,
        default=64,
        help="Actor updates between KL reference-policy refreshes.",
    )
    policy.add_argument("--value-output-scale", type=float, default=0.0)
    policy.add_argument("--target-critic-ema-decay", type=float, default=0.98)
    policy.add_argument("--policy-replay-critic-loss-coef", type=float, default=0.3)
    policy.add_argument("--policy-replay-critic-batch-size", type=int, default=16)
    policy.add_argument("--policy-replay-critic-horizon", type=int, default=64)
    policy.add_argument(
        "--policy-slow-value-regularization-coef",
        type=float,
        default=1.0,
    )
    policy.add_argument("--gamma", type=float, default=1.0 - 1.0 / 333.0)
    policy.add_argument("--lambda-return", type=float, default=0.95)

    online = parser.add_argument_group("online schedule")
    online.add_argument("--online-iterations", type=int, default=0)
    online.add_argument("--online-collect-steps", type=int, default=64)
    online.add_argument("--online-train-steps", type=int, default=1024)
    online.add_argument("--online-policy-train-steps", type=int, default=512)
    online.add_argument(
        "--online-policy-actor-update-interval",
        type=int,
        default=1,
        help=(
            "Online critic updates per actor update; one preserves the "
            "standard one-to-one cadence."
        ),
    )
    online.add_argument(
        "--online-policy-actor-update-interval-start-env-steps",
        type=int,
        default=0,
        help=(
            "Keep one actor update per critic update until this many training "
            "environment steps have been collected, then apply the configured "
            "online actor-update interval. Zero applies it immediately."
        ),
    )
    online.add_argument("--online-checkpoint-interval", type=int, default=16)
    online.add_argument(
        "--online-freeze-encoder-after-env-steps",
        type=int,
        default=None,
        help=(
            "Freeze the observation encoder once this many counted training "
            "environment steps have been collected. The predictor and heads "
            "continue training."
        ),
    )
    online.add_argument(
        "--online-recent-world-model-fraction",
        type=float,
        default=0.0,
        help="Fraction of online world-model batches sampled from recent replay.",
    )
    online.add_argument(
        "--online-recent-world-model-until-env-steps",
        type=int,
        default=None,
        help=(
            "Use the configured recent world-model fraction only while the "
            "phase starts below this many training environment steps, then "
            "switch world-model batches to uniform replay."
        ),
    )
    online.add_argument("--online-recent-replay-steps", type=int, default=320)
    online.add_argument(
        "--online-recent-replay-max-oversample",
        type=float,
        default=0.0,
        help=(
            "Cap the per-transition sampling probability ratio between recent "
            "and older replay entries; zero disables the cap."
        ),
    )

    reporting = parser.add_argument_group("reporting")
    reporting.add_argument("--failure-return-threshold", type=float, default=100.0)
    reporting.add_argument("--success-return-threshold", type=float, default=900.0)
    reporting.add_argument("--final-policy-eval-episodes", type=int, default=20)
    reporting.add_argument("--final-policy-eval-num-envs", type=int, default=None)
    reporting.add_argument("--final-policy-eval-seed", type=int, default=None)
    reporting.add_argument(
        "--dreamer-report-window-env-steps", type=int, default=10_000
    )
    reporting.add_argument("--dreamer-report-budget-env-steps", type=int, default=0)
    reporting.add_argument("--curve-eval-interval-env-steps", type=int, default=0)
    reporting.add_argument("--curve-eval-episodes", type=int, default=0)
    reporting.add_argument("--curve-eval-num-envs", type=int, default=None)
    reporting.add_argument("--curve-eval-seed", type=int, default=None)
    reproducibility = parser.add_argument_group("reproducibility and output")
    reproducibility.add_argument("--num-runs", type=int, default=1)
    reproducibility.add_argument("--seed", type=int, default=0)
    reproducibility.add_argument(
        "--isolated-rng-streams",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    reproducibility.add_argument(
        "--deterministic-compute",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    reproducibility.add_argument("--out-dir", default="runs/jepa")
    reproducibility.add_argument(
        "--training-snapshot-env-steps",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Training-step boundaries at which to save complete resumable "
            "snapshots. Unlike policy checkpoints, these include optimizer, "
            "replay, RNG, EMA, and DMC simulator state."
        ),
    )
    reproducibility.add_argument(
        "--resume-training-snapshot",
        default=None,
        help="Resume from a complete phase-boundary training snapshot.",
    )
    reproducibility.add_argument("--quiet", action="store_true")
    tracking = parser.add_argument_group("Weights & Biases")
    tracking.add_argument("--wandb-project", default=None)
    tracking.add_argument("--wandb-entity", default=None)
    tracking.add_argument("--wandb-name", default=None)
    tracking.add_argument("--wandb-group", default=None)
    tracking.add_argument("--wandb-tags", nargs="*", default=())
    tracking.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    tracking.add_argument("--wandb-videos", action="store_true")
    tracking.add_argument("--wandb-video-frame-stride", type=int, default=4)
    tracking.add_argument("--wandb-video-size", type=int, default=256)
    tracking.add_argument("--wandb-video-fps", type=int, default=20)
    tracking.add_argument("--wandb-video-camera", type=int, default=0)

    parser.set_defaults(**canonical_jepa_config())
    args = parser.parse_args()
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = (
        "num_envs",
        "env_workers",
        "max_cycles",
        "collect_steps",
        "initial_random_action_hold_steps",
        "validation_steps",
        "replay_capacity",
        "batch_size",
        "chunk_length",
        "train_steps",
        "eval_interval",
        "model_horizon",
        "open_loop_horizon",
        "context_window",
        "latent_dim",
        "model_dim",
        "num_layers",
        "num_heads",
        "mlp_ratio",
        "sigreg_knots",
        "sigreg_num_proj",
        "policy_batch_size",
        "actor_num_layers",
        "critic_num_layers",
        "critic_horizon",
        "imag_horizon",
        "policy_replay_critic_batch_size",
        "policy_replay_critic_horizon",
        "online_collect_steps",
        "online_recent_replay_steps",
        "online_checkpoint_interval",
        "online_policy_actor_update_interval",
        "num_runs",
        "wandb_video_frame_stride",
        "wandb_video_size",
        "wandb_video_fps",
    )
    for name in positive:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    nonnegative = (
        "seed",
        "policy_train_steps",
        "optimizer_warmup_steps",
        "online_iterations",
        "online_train_steps",
        "online_policy_train_steps",
        "online_policy_actor_update_interval_start_env_steps",
        "final_policy_eval_episodes",
        "dreamer_report_window_env_steps",
        "dreamer_report_budget_env_steps",
        "curve_eval_interval_env_steps",
        "curve_eval_episodes",
        "wandb_video_camera",
    )
    for name in nonnegative:
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    nonnegative_float = (
        "model_grad_clip_norm",
        "actor_grad_clip_norm",
        "critic_grad_clip_norm",
        "adaptive_grad_clip",
        "actor_output_scale",
        "value_output_scale",
        "reward_output_scale",
        "actor_entropy_coef",
        "policy_replay_critic_loss_coef",
        "policy_slow_value_regularization_coef",
    )
    for name in nonnegative_float:
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    min_sequence_steps = args.chunk_length + max(
        args.model_horizon,
        args.open_loop_horizon,
    )
    if args.collect_steps < min_sequence_steps:
        parser.error("--collect-steps is too short for the configured sequences")
    if args.validation_steps < min_sequence_steps:
        parser.error("--validation-steps is too short for the configured sequences")
    if args.initial_reset_interval is not None:
        if args.initial_reset_interval < min_sequence_steps:
            parser.error("--initial-reset-interval is too short for model sequences")
        if args.initial_reset_interval > args.collect_steps:
            parser.error("--initial-reset-interval must be <= --collect-steps")
    if (
        args.online_recent_world_model_until_env_steps is not None
        and args.online_recent_world_model_until_env_steps < 0
    ):
        parser.error("--online-recent-world-model-until-env-steps must be >= 0")
    if (
        args.online_freeze_encoder_after_env_steps is not None
        and args.online_freeze_encoder_after_env_steps < 0
    ):
        parser.error("--online-freeze-encoder-after-env-steps must be >= 0")
    if args.chunk_length < args.context_window:
        parser.error("--chunk-length must be >= --context-window")
    if not (args.env.startswith("dmc:") or args.env.startswith("brax:")):
        parser.error("--env must be dmc:<domain>/<task> or brax:<env>")
    if args.validation_seed is not None and args.validation_seed < 0:
        parser.error("--validation-seed must be >= 0")
    if args.final_policy_eval_seed is not None and args.final_policy_eval_seed < 0:
        parser.error("--final-policy-eval-seed must be >= 0")
    if args.curve_eval_seed is not None and args.curve_eval_seed < 0:
        parser.error("--curve-eval-seed must be >= 0")
    if (
        args.final_policy_eval_num_envs is not None
        and args.final_policy_eval_num_envs < 1
    ):
        parser.error("--final-policy-eval-num-envs must be >= 1")
    if args.curve_eval_num_envs is not None and args.curve_eval_num_envs < 1:
        parser.error("--curve-eval-num-envs must be >= 1")
    if (args.curve_eval_interval_env_steps == 0) != (args.curve_eval_episodes == 0):
        parser.error(
            "--curve-eval-interval-env-steps and --curve-eval-episodes must "
            "either both be zero or both be positive"
        )
    if not 0.0 <= args.online_recent_world_model_fraction <= 1.0:
        parser.error("--online-recent-world-model-fraction must be in [0, 1]")
    if not 0.0 <= args.policy_reset_start_fraction <= 1.0:
        parser.error("--policy-reset-start-fraction must be in [0, 1]")
    if args.policy_reset_start_fraction_start_env_steps < 0:
        parser.error("--policy-reset-start-fraction-start-env-steps must be >= 0")
    if args.policy_reset_start_max_age < 0:
        parser.error("--policy-reset-start-max-age must be >= 0")
    if (
        args.online_recent_replay_max_oversample != 0.0
        and args.online_recent_replay_max_oversample < 1.0
    ):
        parser.error("--online-recent-replay-max-oversample must be zero or >= 1")
    recent_min_steps = args.chunk_length + max(
        args.model_horizon,
        args.open_loop_horizon,
    )
    if args.online_recent_world_model_fraction > 0.0 and (
        args.online_recent_replay_steps < recent_min_steps
    ):
        parser.error(
            "--online-recent-replay-steps is too short for the configured "
            f"training sequences; need at least {recent_min_steps}"
        )
    if args.failure_return_threshold >= args.success_return_threshold:
        parser.error(
            "--failure-return-threshold must be below --success-return-threshold"
        )
    if args.actor_log_std_min >= args.actor_log_std_max:
        parser.error("--actor-log-std-min must be below --actor-log-std-max")
    if not 0.0 <= args.policy_return_ema_decay < 1.0:
        parser.error("--policy-return-ema-decay must be in [0, 1)")
    if not 0.0 <= args.target_critic_ema_decay < 1.0:
        parser.error("--target-critic-ema-decay must be in [0, 1)")
    if (
        args.policy_slow_value_regularization_coef > 0.0
        and args.target_critic_ema_decay == 0.0
    ):
        parser.error("slow-value regularization requires a target critic")
    if args.value_clip < 0.0:
        parser.error("--value-clip must be >= 0 (0 disables clipping)")
    value_clip_schedule = (
        args.value_clip_final,
        args.value_clip_schedule_start_env_steps,
        args.value_clip_schedule_end_env_steps,
    )
    if any(value is not None for value in value_clip_schedule):
        if not all(value is not None for value in value_clip_schedule):
            parser.error(
                "--value-clip-final and both value-clip schedule boundaries "
                "must be set together"
            )
        assert args.value_clip_final is not None
        assert args.value_clip_schedule_start_env_steps is not None
        assert args.value_clip_schedule_end_env_steps is not None
        if args.value_clip <= 0.0 or args.value_clip_final <= 0.0:
            parser.error("scheduled value clips must both be > 0")
        if args.value_clip_schedule_start_env_steps < 0:
            parser.error("--value-clip-schedule-start-env-steps must be >= 0")
        if (
            args.value_clip_schedule_end_env_steps
            <= args.value_clip_schedule_start_env_steps
        ):
            parser.error(
                "--value-clip-schedule-end-env-steps must be greater than "
                "--value-clip-schedule-start-env-steps"
            )
    if args.policy_actor_kl_coef < 0.0:
        parser.error("--policy-actor-kl-coef must be >= 0")
    if args.policy_actor_kl_target_per_dim < 0.0:
        parser.error("--policy-actor-kl-target-per-dim must be >= 0")
    if args.policy_actor_kl_reference_interval < 1:
        parser.error("--policy-actor-kl-reference-interval must be >= 1")
    if args.optimizer_epsilon <= 0.0:
        parser.error("--optimizer-epsilon must be > 0")
    if args.twohot_bins < 3 or args.twohot_min >= args.twohot_max:
        parser.error("invalid two-hot support")
    if args.online_iterations > 0 and args.policy_train_steps == 0:
        parser.error("--online-iterations requires policy training")
    if args.wandb_videos and not args.wandb_project:
        parser.error("--wandb-videos requires --wandb-project")
    if args.wandb_videos and not args.env.startswith("dmc:"):
        parser.error("--wandb-videos currently supports DMC only")
    if args.resume_training_snapshot is not None and args.num_runs != 1:
        parser.error("--resume-training-snapshot requires --num-runs 1")
    if args.resume_training_snapshot is not None and not args.env.startswith("dmc:"):
        parser.error("complete training snapshots currently support DMC only")
    if args.training_snapshot_env_steps and not args.env.startswith("dmc:"):
        parser.error("complete training snapshots currently support DMC only")
    phase_env_steps = args.online_collect_steps * args.num_envs
    initial_env_steps = args.collect_steps * args.num_envs
    final_env_steps = initial_env_steps + args.online_iterations * phase_env_steps
    for snapshot_env_steps in args.training_snapshot_env_steps:
        if snapshot_env_steps <= initial_env_steps:
            parser.error(
                "--training-snapshot-env-steps must be after bootstrap collection"
            )
        if (snapshot_env_steps - initial_env_steps) % phase_env_steps:
            parser.error(
                "--training-snapshot-env-steps must align with an online phase "
                f"boundary ({initial_env_steps} + k * {phase_env_steps})"
            )
        if snapshot_env_steps > final_env_steps:
            parser.error(
                "--training-snapshot-env-steps exceeds the configured training "
                f"budget of {final_env_steps}"
            )


def main() -> None:
    args = parse_args()
    _configure_deterministic_compute(args.deterministic_compute)
    experiment_dir = (
        Path(args.out_dir) / f"{_experiment_prefix(args.env)}_{timestamp()}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    outcomes = [
        run_one(
            args,
            run_dir=experiment_dir / f"run_{run_index:03d}",
            run_index=run_index,
        )
        for run_index in range(args.num_runs)
    ]
    summary = summarize(outcomes)
    RunLogger(experiment_dir).write_json("summary.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


def _configure_deterministic_compute(enabled: bool) -> None:
    if enabled:
        configure_deterministic_environment()
        jax.config.update("jax_default_matmul_precision", "highest")


def _env_backend(env: str) -> str:
    return "dmc" if env.startswith("dmc:") else "brax"


def _experiment_prefix(env: str) -> str:
    return f"{_env_backend(env)}_jepa"


def _make_vector_adapter(
    args: argparse.Namespace,
    *,
    seed: int,
    num_envs: int | None = None,
):
    adapter_num_envs = args.num_envs if num_envs is None else num_envs
    if args.env.startswith("dmc:"):
        return DMCVectorAdapter(
            dmc_env_name(args.env),
            num_envs=adapter_num_envs,
            max_cycles=args.max_cycles,
            seed=seed,
            num_workers=min(args.env_workers, adapter_num_envs),
        )
    return BraxVectorAdapter(
        brax_env_name(args.env),
        num_envs=adapter_num_envs,
        max_cycles=args.max_cycles,
        seed=seed,
        backend=args.brax_backend,
    )


def _wandb_run_config(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    seed: int,
    run_index: int,
) -> WandbConfig | None:
    if not args.wandb_project or args.wandb_mode == "disabled":
        return None
    run_name = args.wandb_name
    if run_name and args.num_runs > 1:
        run_name = f"{run_name}-seed{seed}"
    if not run_name:
        env_name = args.env.replace(":", "-").replace("/", "-")
        run_name = f"{env_name}-seed{seed}"
    return WandbConfig(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        group=args.wandb_group or run_dir.parents[1].name,
        tags=tuple(args.wandb_tags),
        mode=args.wandb_mode,
        config={"args": vars(args), "run_index": run_index, "seed": seed},
    )


def _jepa_config(args: argparse.Namespace, adapter) -> JepaConfig:
    return JepaConfig(
        observation_dim=int(np.prod(adapter.observation_shape)),
        action_dim=adapter.action_dim,
        action_mode="continuous",
        latent_dim=args.latent_dim,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        max_horizon=args.model_horizon,
        context_window=args.context_window,
        learning_rate=args.learning_rate,
        actor_learning_rate=args.actor_learning_rate,
        model_grad_clip_norm=args.model_grad_clip_norm,
        actor_grad_clip_norm=args.actor_grad_clip_norm,
        critic_grad_clip_norm=args.critic_grad_clip_norm,
        optimizer_warmup_steps=args.optimizer_warmup_steps,
        adaptive_grad_clip=args.adaptive_grad_clip,
        optimizer_epsilon=args.optimizer_epsilon,
        actor_hidden_dim=args.actor_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        actor_num_layers=args.actor_num_layers,
        critic_num_layers=args.critic_num_layers,
        actor_layer_norm=args.actor_layer_norm,
        critic_layer_norm=args.critic_layer_norm,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_max=args.actor_log_std_max,
        actor_output_scale=args.actor_output_scale,
        value_output_scale=args.value_output_scale,
        reward_output_scale=args.reward_output_scale,
        regularizer_weight=args.regularizer_weight,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        reward_weight=args.reward_weight,
        continue_weight=args.continue_weight,
        twohot_bins=args.twohot_bins,
        twohot_min=args.twohot_min,
        twohot_max=args.twohot_max,
        gamma=args.gamma,
        lambda_return=args.lambda_return,
    )


def _reproducibility_snapshot(
    state,
    *,
    phase: str,
    recent_replay: SequenceReplayBuffer | None = None,
    full_replay: SequenceReplayBuffer | None = None,
) -> dict[str, Any]:
    snapshot = {
        "phase": phase,
        "train_state_step": int(jax.device_get(state.step)),
        "params_sha256": fingerprint_pytree(state.params),
        "target_critic_sha256": fingerprint_pytree(state.target_critic_params),
    }
    if recent_replay is not None:
        snapshot["recent_replay_sha256"] = recent_replay.fingerprint()
        snapshot["recent_replay_size_per_env"] = recent_replay.size
    if full_replay is not None:
        snapshot["full_replay_sha256"] = full_replay.fingerprint()
        snapshot["full_replay_size_per_env"] = full_replay.size
    return snapshot


def _protocol_name(args: argparse.Namespace) -> str:
    del args
    return "reset_rich_interleaved_latest_policy"


@dataclasses.dataclass
class _TrainingPrefix:
    state: Any
    observations: np.ndarray
    replay: SequenceReplayBuffer
    online_recent_replay: SequenceReplayBuffer | None
    validation_replay: SequenceReplayBuffer
    validation_batch: ReplayBatch
    initial_train_env_steps: int
    train_env_steps: int
    validation_env_steps: int
    initial_metrics: dict[str, Any]
    initial_policy_metrics: dict[str, Any]
    online_history: list[dict[str, Any]]
    curve_evaluations: list[dict[str, Any]]
    next_curve_eval_step: int | None
    next_online_iteration: int


def _prepare_training_prefix(
    args: argparse.Namespace,
    logger: RunLogger,
    adapter,
    config: JepaConfig,
    *,
    seed: int,
    validation_seed: int,
    jax_rngs: JaxRngStreams,
    numpy_rngs: NumpyRngStreams,
    validation_jax_rngs: JaxRngStreams,
    validation_numpy_rngs: NumpyRngStreams,
    validation_sampling_rng: np.random.Generator,
) -> _TrainingPrefix:
    state = create_jepa_train_state(jax_rngs.take("initialization"), config)
    observations = adapter.reset()
    if args.resume_training_snapshot is not None:
        loaded = load_training_snapshot(
            args.resume_training_snapshot,
            target_train_state=state,
            target_policy_bundle_ema=None,
            adapter=adapter,
            jax_rng_streams={
                "training": jax_rngs,
                "validation": validation_jax_rngs,
            },
            numpy_rng_streams={
                "training": numpy_rngs,
                "validation": validation_numpy_rngs,
            },
        )
        metadata = loaded.metadata
        expected_config = dataclasses.asdict(config)
        if metadata["env"] != args.env:
            raise ValueError(
                "training snapshot environment does not match: "
                f"{metadata['env']} != {args.env}"
            )
        if int(metadata["seed"]) != seed:
            raise ValueError(
                f"training snapshot seed does not match: {metadata['seed']} != {seed}"
            )
        if metadata["jepa_config"] != expected_config:
            raise ValueError(
                "training snapshot JEPA architecture/config does not match"
            )
        canonical_replays = {"main", "online_recent", "validation"}
        legacy_replays = canonical_replays | {"bootstrap_start"}
        if set(loaded.replays) not in (canonical_replays, legacy_replays):
            raise ValueError(
                f"training snapshot replay set does not match: {sorted(loaded.replays)}"
            )
        if loaded.replays.get("bootstrap_start") is not None:
            raise ValueError(
                "snapshot contains the retired bootstrap-start replay and cannot "
                "be resumed by the canonical JEPA trainer"
            )
        replay = loaded.replays["main"]
        validation_replay = loaded.replays["validation"]
        if replay is None or validation_replay is None:
            raise ValueError("training snapshot is missing required replay")
        train_env_steps = int(metadata["train_env_steps"])
        next_online_iteration = int(metadata["next_online_iteration"])
        if next_online_iteration > args.online_iterations + 1:
            raise ValueError(
                "training snapshot is beyond the requested online iteration budget"
            )
        logger.set_train_env_steps(train_env_steps)
        logger.write_json(
            "resume_training_snapshot.json",
            {
                "source": str(Path(args.resume_training_snapshot).resolve()),
                "train_env_steps": train_env_steps,
                "next_online_iteration": next_online_iteration,
                "params_sha256": fingerprint_pytree(loaded.train_state.params),
                "main_replay_sha256": replay.fingerprint(),
                "environment_state_restored": True,
            },
        )
        legacy_validation_dones = loaded.arrays.get("validation_dones")
        validation_is_last = loaded.arrays.get(
            "validation_is_last",
            legacy_validation_dones,
        )
        if validation_is_last is None:
            raise ValueError("training snapshot is missing validation boundaries")
        validation_batch = ReplayBatch(
            observations=jnp.asarray(loaded.arrays["validation_observations"]),
            actions=jnp.asarray(loaded.arrays["validation_actions"]),
            rewards=jnp.asarray(loaded.arrays["validation_rewards"]),
            is_last=jnp.asarray(validation_is_last),
            is_terminal=jnp.asarray(
                loaded.arrays.get(
                    "validation_is_terminal",
                    validation_is_last,
                )
            ),
        )
        return _TrainingPrefix(
            state=loaded.train_state,
            observations=loaded.observations,
            replay=replay,
            online_recent_replay=loaded.replays["online_recent"],
            validation_replay=validation_replay,
            validation_batch=validation_batch,
            initial_train_env_steps=int(metadata["initial_train_env_steps"]),
            train_env_steps=train_env_steps,
            validation_env_steps=int(metadata["validation_env_steps"]),
            initial_metrics=dict(metadata["initial_metrics"]),
            initial_policy_metrics=dict(metadata["initial_policy_metrics"]),
            online_history=list(metadata["online_history"]),
            curve_evaluations=list(metadata["curve_evaluations"]),
            next_curve_eval_step=metadata["next_curve_eval_step"],
            next_online_iteration=next_online_iteration,
        )

    replay = _new_replay_buffer(
        capacity=max(2, math.ceil(args.replay_capacity / args.num_envs)),
        num_envs=args.num_envs,
        observation_dim=config.observation_dim,
        action_dim=config.action_dim,
    )
    online_recent_replay = (
        _new_replay_buffer(
            capacity=args.online_recent_replay_steps,
            num_envs=args.num_envs,
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
        )
        if args.online_recent_world_model_fraction > 0.0
        else None
    )

    # Bootstrap collection deliberately uses a separate adapter so the online
    # simulator remains at its first reset until learned collection starts.
    bootstrap_adapter = _make_vector_adapter(args, seed=seed)
    try:
        bootstrap_replays = [replay]
        if online_recent_replay is not None:
            bootstrap_replays.append(online_recent_replay)
        _, initial_train_env_steps = _collect_random_steps(
            bootstrap_adapter,
            bootstrap_adapter.reset(),
            numpy_rngs.get("initial_collection"),
            tuple(bootstrap_replays),
            steps=args.collect_steps,
            reset_interval=args.initial_reset_interval,
            action_hold_steps=args.initial_random_action_hold_steps,
            desc="collect reset-rich bootstrap replay",
            quiet=args.quiet,
        )
    finally:
        bootstrap_adapter.close()
    train_env_steps = initial_train_env_steps
    logger.set_train_env_steps(train_env_steps)
    logger.write_json(
        "train_replay.json",
        {
            "env_steps": initial_train_env_steps,
            "steps_per_env": replay.size,
            "size_per_env": replay.size,
            "collector_cut_count": replay.cut_count,
            "initial_reset_interval": args.initial_reset_interval,
            "initial_random_action_hold_steps": args.initial_random_action_hold_steps,
            "initial_segments_per_env": (
                math.ceil(args.collect_steps / args.initial_reset_interval)
                if args.initial_reset_interval is not None
                else 1
            ),
            "observation_dim": config.observation_dim,
            "action_dim": config.action_dim,
            "source": "isolated_reset_rich_collection",
        },
    )

    validation_replay = _collect_validation_replay(args, config, seed=validation_seed)
    validation_env_steps = args.validation_steps * args.num_envs
    logger.write_json(
        "validation_replay.json",
        {
            "env_steps": validation_env_steps,
            "steps_per_env": args.validation_steps,
            "size_per_env": validation_replay.size,
            "seed": validation_seed,
        },
    )
    initialized_snapshot = _reproducibility_snapshot(
        state,
        phase="initialized",
        full_replay=replay,
    )
    initialized_snapshot.update(
        {
            "initial_replay_sha256": replay.fingerprint(),
            "validation_replay_sha256": validation_replay.fingerprint(),
        }
    )
    logger.write_json("reproducibility_initialized.json", initialized_snapshot)

    validation_batch = validation_replay.sample(
        validation_sampling_rng,
        batch_size=args.batch_size,
        chunk_length=args.chunk_length,
        max_horizon=max(args.model_horizon, args.open_loop_horizon),
    )
    initial_metrics = _evaluate_model(
        state,
        validation_jax_rngs.take("evaluation"),
        validation_batch,
        config,
        chunk_length=args.chunk_length,
        open_loop_horizon=args.open_loop_horizon,
    )
    logger.write_json("model_metrics_initial.json", initial_metrics)

    model_rng = jax_rngs.current("world_model")
    state, model_rng, _ = _fit_world_model(
        args,
        logger,
        state,
        model_rng,
        replay,
        config,
        np_rng=numpy_rngs.get("world_model_replay"),
        steps=args.train_steps,
        phase="world_model",
        desc="fit initial world model",
        train_env_steps=train_env_steps,
    )
    jax_rngs.update("world_model", model_rng)
    initial_fit_metrics = _evaluate_model(
        state,
        validation_jax_rngs.take("evaluation"),
        validation_batch,
        config,
        chunk_length=args.chunk_length,
        open_loop_horizon=args.open_loop_horizon,
    )
    logger.write_json("model_metrics_initial_fit.json", initial_fit_metrics)
    logger.write_json(
        "reproducibility_initial_world_model.json",
        _reproducibility_snapshot(state, phase="initial_world_model"),
    )

    policy_rng = jax_rngs.current("policy")
    state, policy_rng, initial_policy_metrics = _train_policy(
        args,
        logger,
        state,
        config,
        replay,
        np_rng=numpy_rngs.get("policy_replay"),
        rng=policy_rng,
        action_low=adapter.action_low,
        action_high=adapter.action_high,
        phase="policy",
        train_steps=args.policy_train_steps,
        reset_actor=True,
        actor_update_interval=1,
        actor_entropy_coef=args.actor_entropy_coef,
        value_clip=_scheduled_value_clip(args, train_env_steps=train_env_steps),
    )
    jax_rngs.update("policy", policy_rng)
    logger.write_json("policy_initial_fit.json", initial_policy_metrics)
    logger.write_json(
        "reproducibility_initial_policy.json",
        _reproducibility_snapshot(state, phase="initial_policy"),
    )

    next_curve_eval_step = (
        args.curve_eval_interval_env_steps
        if args.curve_eval_interval_env_steps > 0
        else None
    )
    while next_curve_eval_step is not None and next_curve_eval_step <= train_env_steps:
        next_curve_eval_step += args.curve_eval_interval_env_steps
    return _TrainingPrefix(
        state=state,
        observations=observations,
        replay=replay,
        online_recent_replay=online_recent_replay,
        validation_replay=validation_replay,
        validation_batch=validation_batch,
        initial_train_env_steps=initial_train_env_steps,
        train_env_steps=train_env_steps,
        validation_env_steps=validation_env_steps,
        initial_metrics=initial_metrics,
        initial_policy_metrics=initial_policy_metrics,
        online_history=[],
        curve_evaluations=[],
        next_curve_eval_step=next_curve_eval_step,
        next_online_iteration=1,
    )


def run_one(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    run_index: int,
) -> dict[str, Any]:
    seed = args.seed + 10_000 * run_index
    validation_seed = (
        seed + 1_000_000 if args.validation_seed is None else args.validation_seed
    )
    logger = RunLogger(
        run_dir,
        wandb_config=_wandb_run_config(
            args,
            run_dir=run_dir,
            seed=seed,
            run_index=run_index,
        ),
    )
    adapter = None
    completed = False
    try:
        adapter = _make_vector_adapter(args, seed=seed)
        config = _jepa_config(args, adapter)
        resolved_config = {
            "args": vars(args),
            "run_index": run_index,
            "seed": seed,
            "observation_shape": adapter.observation_shape,
            "action_shape": adapter.action_shape,
            "action_low": adapter.action_low,
            "action_high": adapter.action_high,
            "env_backend": _env_backend(args.env),
            "jepa_config": dataclasses.asdict(config),
            "protocol": _protocol_name(args),
        }
        logger.write_json("config.json", resolved_config)
        logger.update_config(resolved_config)
        logger.write_json("versions.json", dependency_versions())

        jax_rngs = JaxRngStreams.create(seed, isolated=args.isolated_rng_streams)
        numpy_rngs = NumpyRngStreams.create(seed, isolated=args.isolated_rng_streams)
        validation_jax_rngs = (
            jax_rngs
            if args.validation_seed is None
            else JaxRngStreams.create(validation_seed, isolated=True)
        )
        validation_numpy_rngs = (
            numpy_rngs
            if args.validation_seed is None
            else NumpyRngStreams.create(validation_seed, isolated=True)
        )
        validation_sampling_rng = validation_numpy_rngs.get("validation_replay")
        logger.write_json(
            "rng_streams.json",
            {
                **jax_rngs.manifest(),
                **numpy_rngs.manifest(),
                "deterministic_compute": args.deterministic_compute,
                "jax_default_matmul_precision": str(
                    jax.config.jax_default_matmul_precision
                ),
                "xla_flags": os.environ.get("XLA_FLAGS"),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
                "validation_seed": validation_seed,
                "validation_seed_overridden": args.validation_seed is not None,
            },
        )

        prefix = _prepare_training_prefix(
            args,
            logger,
            adapter,
            config,
            seed=seed,
            validation_seed=validation_seed,
            jax_rngs=jax_rngs,
            numpy_rngs=numpy_rngs,
            validation_jax_rngs=validation_jax_rngs,
            validation_numpy_rngs=validation_numpy_rngs,
            validation_sampling_rng=validation_sampling_rng,
        )
        state = prefix.state
        observations = prefix.observations
        replay = prefix.replay
        online_recent_replay = prefix.online_recent_replay
        validation_replay = prefix.validation_replay
        validation_batch = prefix.validation_batch
        initial_train_env_steps = prefix.initial_train_env_steps
        train_env_steps = prefix.train_env_steps
        validation_env_steps = prefix.validation_env_steps
        initial_metrics = prefix.initial_metrics
        initial_policy_metrics = prefix.initial_policy_metrics
        online_history = prefix.online_history
        curve_evaluations = prefix.curve_evaluations
        next_curve_eval_step = prefix.next_curve_eval_step
        training_snapshot_targets = set(args.training_snapshot_env_steps)
        for online_index in range(
            prefix.next_online_iteration,
            args.online_iterations + 1,
        ):
            phase = f"online_{online_index:03d}"
            phase_start_train_env_steps = train_env_steps
            requested_recent_fraction = _scheduled_recent_world_model_fraction(
                args,
                train_env_steps=phase_start_train_env_steps,
            )
            collection_replays = [replay]
            if online_recent_replay is not None:
                collection_replays.append(online_recent_replay)
            observations, added_env_steps, collection = _collect_policy_steps(
                adapter,
                observations,
                state,
                config,
                tuple(collection_replays),
                steps=args.online_collect_steps,
                action_low=adapter.action_low,
                action_high=adapter.action_high,
                desc=f"{phase} collect policy replay",
                quiet=args.quiet,
                np_rng=numpy_rngs.get("online_collection"),
                stochastic_actions=True,
                train_env_step_offset=train_env_steps,
                failure_return_threshold=args.failure_return_threshold,
                success_return_threshold=args.success_return_threshold,
            )
            train_env_steps += added_env_steps
            logger.set_train_env_steps(train_env_steps)
            _log_collection_episode_reports(
                logger,
                collection,
                online_iteration=online_index,
            )
            collection = {
                **collection,
                "train_replay_total_env_steps": train_env_steps,
                "replay_size_per_env": replay.size,
                "online_recent_replay_size_per_env": (
                    online_recent_replay.size if online_recent_replay is not None else 0
                ),
            }
            recent_replay_size = (
                online_recent_replay.size if online_recent_replay is not None else 0
            )
            effective_recent_fraction = _effective_recent_fraction(
                requested_recent_fraction,
                full_replay_size=replay.size,
                recent_replay_size=recent_replay_size,
                max_oversample=args.online_recent_replay_max_oversample,
            )
            effective_recent_oversample = _recent_oversample_ratio(
                effective_recent_fraction,
                full_replay_size=replay.size,
                recent_replay_size=recent_replay_size,
            )
            collection.update(
                {
                    "online_recent_replay_fraction": requested_recent_fraction,
                    "online_recent_replay_requested_fraction": (
                        requested_recent_fraction
                    ),
                    "online_recent_replay_effective_fraction": (
                        effective_recent_fraction
                    ),
                    "online_recent_replay_max_oversample": (
                        args.online_recent_replay_max_oversample
                    ),
                    "online_recent_replay_effective_oversample": (
                        effective_recent_oversample
                    ),
                }
            )
            logger.append_metrics(
                {
                    "phase": "online_actor_replay",
                    "online_iteration": online_index,
                    "report": _collection_report_summary(collection),
                    **collection,
                }
            )

            model_rng = jax_rngs.current("world_model")
            freeze_online_encoder = _scheduled_online_encoder_freeze(
                args,
                train_env_steps=train_env_steps,
            )
            state, model_rng, _ = _fit_world_model(
                args,
                logger,
                state,
                model_rng,
                replay,
                config,
                np_rng=numpy_rngs.get("world_model_replay"),
                steps=args.online_train_steps,
                phase=f"{phase}_world_model",
                desc=f"{phase} fit world model",
                train_env_steps=train_env_steps,
                recent_replay=online_recent_replay,
                recent_fraction=effective_recent_fraction,
                freeze_encoder=freeze_online_encoder,
            )
            jax_rngs.update("world_model", model_rng)

            policy_rng = jax_rngs.current("policy")
            actor_update_interval = _scheduled_online_actor_update_interval(
                args,
                train_env_steps=phase_start_train_env_steps,
            )
            state, policy_rng, policy_metrics = _train_policy(
                args,
                logger,
                state,
                config,
                replay,
                np_rng=numpy_rngs.get("policy_replay"),
                rng=policy_rng,
                action_low=adapter.action_low,
                action_high=adapter.action_high,
                phase=f"{phase}_policy",
                train_steps=args.online_policy_train_steps,
                reset_actor=False,
                actor_update_interval=actor_update_interval,
                actor_entropy_coef=args.actor_entropy_coef,
                value_clip=_scheduled_value_clip(
                    args,
                    train_env_steps=train_env_steps,
                ),
                reset_start_fraction=_scheduled_policy_reset_start_fraction(
                    args,
                    train_env_steps=train_env_steps,
                ),
                reset_start_max_age=args.policy_reset_start_max_age,
            )
            jax_rngs.update("policy", policy_rng)

            curve_evaluation = None
            if (
                next_curve_eval_step is not None
                and train_env_steps >= next_curve_eval_step
            ):
                curve_evaluation = _curve_policy_evaluation(
                    args,
                    logger,
                    state,
                    config,
                    seed=seed,
                    action_low=adapter.action_low,
                    action_high=adapter.action_high,
                    phase=phase,
                    online_iteration=online_index,
                    train_env_steps=train_env_steps,
                    scheduled_env_steps=next_curve_eval_step,
                )
                curve_evaluations.append(curve_evaluation)
                while (
                    next_curve_eval_step is not None
                    and next_curve_eval_step <= train_env_steps
                ):
                    next_curve_eval_step += args.curve_eval_interval_env_steps

            checkpoint_phase = (
                online_index % args.online_checkpoint_interval == 0
                or online_index == args.online_iterations
            )
            reproducibility = None
            if checkpoint_phase:
                reproducibility = _reproducibility_snapshot(
                    state,
                    phase=phase,
                    recent_replay=online_recent_replay,
                    full_replay=replay,
                )
                logger.write_json(f"{phase}_reproducibility.json", reproducibility)
            online_history.append(
                {
                    "iteration": online_index,
                    "actor_replay": collection,
                    "policy": policy_metrics,
                    "policy_evaluation": curve_evaluation,
                    "reproducibility": reproducibility,
                    "world_model_train_steps": args.online_train_steps,
                    "policy_train_steps": args.online_policy_train_steps,
                    "policy_actor_update_interval": actor_update_interval,
                    "policy_actor_updates": policy_metrics["policy_actor_updates"],
                }
            )
            if train_env_steps in training_snapshot_targets:
                snapshot_dir = _save_branch_training_snapshot(
                    run_dir,
                    state,
                    args=args,
                    config=config,
                    seed=seed,
                    adapter=adapter,
                    observations=observations,
                    replay=replay,
                    online_recent_replay=online_recent_replay,
                    validation_replay=validation_replay,
                    validation_batch=validation_batch,
                    jax_rngs=jax_rngs,
                    numpy_rngs=numpy_rngs,
                    validation_jax_rngs=validation_jax_rngs,
                    validation_numpy_rngs=validation_numpy_rngs,
                    initial_train_env_steps=initial_train_env_steps,
                    train_env_steps=train_env_steps,
                    validation_env_steps=validation_env_steps,
                    initial_metrics=initial_metrics,
                    initial_policy_metrics=initial_policy_metrics,
                    online_history=online_history,
                    curve_evaluations=curve_evaluations,
                    next_curve_eval_step=next_curve_eval_step,
                    next_online_iteration=online_index + 1,
                )
                logger.write_json(
                    f"{phase}_training_snapshot.json",
                    {
                        "path": str(snapshot_dir),
                        "train_env_steps": train_env_steps,
                        "next_online_iteration": online_index + 1,
                    },
                )
            if checkpoint_phase:
                _save_recovery_checkpoint(
                    run_dir,
                    state,
                    args=args,
                    config=config,
                    seed=seed,
                    online_iteration=online_index,
                    train_env_steps=train_env_steps,
                )

        logger.write_json("online_history.json", online_history)
        training_score = _dreamer_style_training_score(
            online_history,
            window_env_steps=args.dreamer_report_window_env_steps,
            budget_env_steps=args.dreamer_report_budget_env_steps,
        )
        logger.write_json("dreamer_style_training_score.json", training_score)
        if training_score["enabled"]:
            logger.append_metrics(
                {"phase": "dreamer_style_training_score", **training_score}
            )

        final_metrics = _evaluate_model(
            state,
            validation_jax_rngs.take("evaluation"),
            validation_batch,
            config,
            chunk_length=args.chunk_length,
            open_loop_horizon=args.open_loop_horizon,
        )
        logger.write_json("model_metrics_final.json", final_metrics)
        logger.write_json(
            "reproducibility_final.json",
            _reproducibility_snapshot(
                state,
                phase="final",
                full_replay=replay,
            ),
        )

        checkpoint_dir = run_dir / "checkpoint"
        try:
            save_checkpoint(
                checkpoint_dir,
                state,
                metadata={
                    "algorithm": "single_agent_jepa_mbrl",
                    "checkpoint_kind": "final_latest_policy",
                    "env": args.env,
                    "env_backend": _env_backend(args.env),
                    "jepa_config": dataclasses.asdict(config),
                    "seed": seed,
                    "train_replay_env_steps": train_env_steps,
                },
            )
        except OSError as error:
            warnings.warn(
                f"Final checkpoint write failed: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            recovery_dir = run_dir / "checkpoint_latest"
            if (recovery_dir / "checkpoint.msgpack").is_file():
                checkpoint_dir = recovery_dir
        try:
            reload_diff = _reload_prediction_diff(
                state,
                config,
                checkpoint_dir=checkpoint_dir,
                batch=validation_batch,
                seed=seed + 99,
                chunk_length=args.chunk_length,
            )
        except OSError as error:
            warnings.warn(
                f"Checkpoint reload validation failed: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            reload_diff = float("inf")
        logger.write_json(
            "reload_evaluation.json",
            {"reload_max_abs_prediction_diff": reload_diff},
        )

        final_policy_eval = _final_policy_evaluation(
            args,
            logger,
            state,
            config,
            seed=seed,
            action_low=adapter.action_low,
            action_high=adapter.action_high,
        )
        outcome = {
            "run_index": run_index,
            "seed": seed,
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "protocol": _protocol_name(args),
            "target": (
                f"{_env_backend(args.env)}:"
                "p(z_next, reward, continue | z_history, action_history)"
            ),
            "initial_jepa_loss": initial_metrics["model/jepa_loss"],
            "final_jepa_loss": final_metrics["model/jepa_loss"],
            "initial_open_loop_loss": initial_metrics["model/open_loop_loss"],
            "final_open_loop_loss": final_metrics["model/open_loop_loss"],
            "final_reward_loss": final_metrics["model/reward_loss"],
            "final_continue_loss": final_metrics["model/continue_loss"],
            "reload_max_abs_prediction_diff": reload_diff,
            "final_model_metrics": final_metrics,
            "initial_policy_metrics": initial_policy_metrics,
            "latest_policy_metrics": (
                online_history[-1]["policy"]
                if online_history
                else initial_policy_metrics
            ),
            "online_iterations": args.online_iterations,
            "online_history": online_history,
            "policy_curve_evaluations": curve_evaluations,
            "dreamer_style_training_score": training_score,
            "dreamer_style_train_return_mean": training_score.get("mean_return"),
            "dreamer_style_train_return_std": training_score.get("std_return"),
            "dreamer_style_train_return_episodes": training_score.get("episodes"),
            "dreamer_style_train_return_budget_reached": training_score.get(
                "budget_reached"
            ),
            "final_policy_eval": final_policy_eval,
            "final_policy_eval_episodes": _nested(final_policy_eval, "episodes"),
            "final_policy_eval_mean": _nested(final_policy_eval, "mean_return"),
            "final_policy_eval_std": _nested(final_policy_eval, "std_return"),
            "final_policy_eval_failure_rate": _nested(
                final_policy_eval,
                "failure_rate",
            ),
            "final_policy_eval_success_rate": _nested(
                final_policy_eval,
                "success_rate",
            ),
            "final_policy_eval_return_p10": _nested(final_policy_eval, "return_p10"),
            "final_policy_eval_return_cvar10": _nested(
                final_policy_eval,
                "return_cvar10",
            ),
            "final_policy_eval_env_steps": _nested(final_policy_eval, "env_steps"),
            **_real_step_accounting(
                initial_train_env_steps=initial_train_env_steps,
                validation_env_steps=validation_env_steps,
                online_history=online_history,
                final_policy_eval=final_policy_eval,
            ),
        }
        logger.write_json("outcome.json", outcome)
        final_row = {
            "phase": "run_outcome",
            "budget/train_env_steps": outcome["real_train_replay_env_steps"],
            "budget/validation_env_steps": outcome["real_validation_replay_env_steps"],
            "budget/policy_eval_env_steps": outcome["real_policy_eval_env_steps"],
            "budget/total_real_env_steps": outcome["real_total_env_steps"],
            "model/final_jepa_loss": outcome["final_jepa_loss"],
            "model/final_open_loop_loss": outcome["final_open_loop_loss"],
            "model/final_reward_loss": outcome["final_reward_loss"],
            "model/final_continue_loss": outcome["final_continue_loss"],
            "run/checkpoint_reload_verified": reload_diff <= 1e-6,
        }
        if final_policy_eval is not None:
            final_row.update(
                {
                    "eval/return_mean": final_policy_eval["mean_return"],
                    "eval/return_std": final_policy_eval["std_return"],
                    "eval/return_p10": final_policy_eval["return_p10"],
                    "eval/return_cvar10": final_policy_eval["return_cvar10"],
                    "eval/failure_rate": final_policy_eval["failure_rate"],
                    "eval/success_rate": final_policy_eval["success_rate"],
                    "eval/episodes": final_policy_eval["episodes"],
                }
            )
        logger.append_metrics(final_row)
        logger.update_summary(final_row)
        completed = True
        return to_jsonable(outcome)
    finally:
        try:
            if adapter is not None:
                adapter.close()
        finally:
            logger.close(exit_code=0 if completed else 1)


def _save_branch_training_snapshot(
    run_dir: Path,
    state,
    *,
    args: argparse.Namespace,
    config: JepaConfig,
    seed: int,
    adapter,
    observations: np.ndarray,
    replay: SequenceReplayBuffer,
    online_recent_replay: SequenceReplayBuffer | None,
    validation_replay: SequenceReplayBuffer,
    validation_batch: ReplayBatch,
    jax_rngs: JaxRngStreams,
    numpy_rngs: NumpyRngStreams,
    validation_jax_rngs: JaxRngStreams,
    validation_numpy_rngs: NumpyRngStreams,
    initial_train_env_steps: int,
    train_env_steps: int,
    validation_env_steps: int,
    initial_metrics: dict[str, Any],
    initial_policy_metrics: dict[str, Any],
    online_history: list[dict[str, Any]],
    curve_evaluations: list[dict[str, Any]],
    next_curve_eval_step: int | None,
    next_online_iteration: int,
) -> Path:
    snapshot_dir = run_dir / "training_snapshots" / f"env_{train_env_steps:09d}"
    return save_training_snapshot(
        snapshot_dir,
        train_state=state,
        policy_bundle_ema=None,
        replays={
            "main": replay,
            "online_recent": online_recent_replay,
            "validation": validation_replay,
        },
        observations=observations,
        arrays={
            "validation_observations": jax.device_get(validation_batch.observations),
            "validation_actions": jax.device_get(validation_batch.actions),
            "validation_rewards": jax.device_get(validation_batch.rewards),
            "validation_is_last": jax.device_get(validation_batch.is_last),
            "validation_is_terminal": jax.device_get(validation_batch.is_terminal),
        },
        adapter=adapter,
        jax_rng_streams={
            "training": jax_rngs,
            "validation": validation_jax_rngs,
        },
        numpy_rng_streams={
            "training": numpy_rngs,
            "validation": validation_numpy_rngs,
        },
        metadata={
            "algorithm": "single_agent_jepa_mbrl",
            "checkpoint_kind": "complete_phase_boundary_training_snapshot",
            "env": args.env,
            "jepa_config": dataclasses.asdict(config),
            "seed": seed,
            "initial_train_env_steps": initial_train_env_steps,
            "train_env_steps": train_env_steps,
            "validation_env_steps": validation_env_steps,
            "initial_metrics": initial_metrics,
            "initial_policy_metrics": initial_policy_metrics,
            "online_history": online_history,
            "curve_evaluations": curve_evaluations,
            "next_curve_eval_step": next_curve_eval_step,
            "next_online_iteration": next_online_iteration,
            "params_sha256": fingerprint_pytree(state.params),
            "target_critic_sha256": fingerprint_pytree(state.target_critic_params),
            "main_replay_sha256": replay.fingerprint(),
            "online_recent_replay_sha256": (
                online_recent_replay.fingerprint()
                if online_recent_replay is not None
                else None
            ),
        },
    )


def _save_recovery_checkpoint(
    run_dir: Path,
    state,
    *,
    args: argparse.Namespace,
    config: JepaConfig,
    seed: int,
    online_iteration: int,
    train_env_steps: int,
) -> None:
    try:
        save_checkpoint(
            run_dir / "checkpoint_latest",
            state,
            metadata={
                "algorithm": "single_agent_jepa_mbrl",
                "checkpoint_kind": "online_recovery_latest_policy",
                "env": args.env,
                "env_backend": _env_backend(args.env),
                "jepa_config": dataclasses.asdict(config),
                "online_iteration": online_iteration,
                "seed": seed,
                "train_replay_env_steps": train_env_steps,
            },
        )
    except OSError as error:
        warnings.warn(
            f"Recovery checkpoint write failed; training continues: {error}",
            RuntimeWarning,
            stacklevel=2,
        )


def _new_replay_buffer(
    *,
    capacity: int,
    num_envs: int,
    observation_dim: int,
    action_dim: int,
) -> SequenceReplayBuffer:
    return SequenceReplayBuffer(
        capacity=max(2, capacity),
        num_envs=num_envs,
        observation_shape=(observation_dim,),
        action_shape=(action_dim,),
        action_dtype=np.float32,
    )


def _collect_random_steps(
    adapter,
    observations: np.ndarray,
    rng: np.random.Generator,
    replay: SequenceReplayBuffer | tuple[SequenceReplayBuffer, ...],
    *,
    steps: int,
    reset_interval: int | None = None,
    action_hold_steps: int = 1,
    desc: str,
    quiet: bool,
) -> tuple[np.ndarray, int]:
    held_actions = None
    held_steps = np.zeros((adapter.num_envs,), dtype=np.int64)
    for step_index in tqdm(range(steps), desc=desc, unit="step", disable=quiet):
        if held_actions is None:
            held_actions = adapter.sample_actions(rng)
            held_steps.fill(0)
        else:
            resample_mask = held_steps >= action_hold_steps
            if np.any(resample_mask):
                replacement_actions = adapter.sample_actions(rng)
                held_actions = np.array(held_actions, copy=True)
                held_actions[resample_mask] = replacement_actions[resample_mask]
                held_steps[resample_mask] = 0
        actions = held_actions
        step = adapter.step(actions)
        held_steps += 1
        forced_reset = (
            reset_interval is not None and (step_index + 1) % reset_interval == 0
        )
        _add_replay_step(
            replay,
            observations=observations[:, 0],
            actions=actions[:, 0],
            rewards=step.rewards[:, 0],
            is_last=step.dones[:, 0],
            is_terminal=step.dones[:, 0],
            cuts=(
                np.ones((adapter.num_envs,), dtype=np.float32) if forced_reset else None
            ),
        )
        if forced_reset:
            observations = adapter.reset()
            held_actions = None
            held_steps.fill(0)
        else:
            observations = step.observations
            done_mask = np.asarray(step.dones[:, 0]) > 0.0
            resample_done_mask = done_mask & (held_steps < action_hold_steps)
            if np.any(resample_done_mask):
                replacement_actions = adapter.sample_actions(rng)
                held_actions = np.array(held_actions, copy=True)
                held_actions[resample_done_mask] = replacement_actions[
                    resample_done_mask
                ]
                held_steps[resample_done_mask] = 0
    return observations, steps * adapter.num_envs


def _collect_policy_steps(
    adapter,
    observations: np.ndarray,
    state,
    config: JepaConfig,
    replay: SequenceReplayBuffer | tuple[SequenceReplayBuffer, ...],
    *,
    steps: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    desc: str,
    quiet: bool,
    np_rng: np.random.Generator,
    stochastic_actions: bool,
    train_env_step_offset: int,
    failure_return_threshold: float,
    success_return_threshold: float,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    action_low_jax = jnp.asarray(action_low, dtype=jnp.float32)
    action_high_jax = jnp.asarray(action_high, dtype=jnp.float32)
    action_key = jax.random.PRNGKey(int(np_rng.integers(0, 2**31 - 1)))
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    finish_collection_steps: list[int] = []
    finish_train_steps: list[int] = []
    progress = tqdm(range(steps), desc=desc, unit="step", disable=quiet)
    for step_index in progress:
        action_key, step_action_key = jax.random.split(action_key)
        actions = np.asarray(
            select_continuous_actions(
                state,
                jnp.asarray(observations[:, 0], dtype=jnp.float32),
                config,
                action_low_jax,
                action_high_jax,
                key=step_action_key,
                stochastic=stochastic_actions,
            )
        )
        step = adapter.step(actions[:, None, :])
        _add_replay_step(
            replay,
            observations=observations[:, 0],
            actions=actions,
            rewards=step.rewards[:, 0],
            is_last=step.dones[:, 0],
            is_terminal=step.dones[:, 0],
        )
        completed_count = len(step.completed_returns)
        completed_returns.extend(float(item[0]) for item in step.completed_returns)
        completed_lengths.extend(int(item) for item in step.completed_lengths)
        if completed_count:
            local_finish = (step_index + 1) * adapter.num_envs
            finish_collection_steps.extend([local_finish] * completed_count)
            finish_train_steps.extend(
                [train_env_step_offset + local_finish] * completed_count
            )
        if completed_returns:
            progress.set_postfix(
                episodes=len(completed_returns),
                mean_return=f"{np.mean(completed_returns):.3g}",
            )
        observations = step.observations

    metrics = {
        "env_steps": steps * adapter.num_envs,
        "steps_per_env": steps,
        "stochastic_actions": stochastic_actions,
        "completed_episodes": len(completed_returns),
        "mean_return": (
            float(np.mean(completed_returns)) if completed_returns else None
        ),
        "std_return": (float(np.std(completed_returns)) if completed_returns else None),
        "mean_length": (
            float(np.mean(completed_lengths)) if completed_lengths else None
        ),
        "returns": completed_returns,
        "lengths": completed_lengths,
        "episode_finish_collection_env_steps": finish_collection_steps,
        "episode_finish_train_env_steps": finish_train_steps,
        "train_env_step_offset": train_env_step_offset,
        **_return_tail_metrics(
            completed_returns,
            failure_threshold=failure_return_threshold,
            success_threshold=success_return_threshold,
        ),
    }
    return observations, steps * adapter.num_envs, metrics


def _add_replay_step(
    replay: SequenceReplayBuffer | tuple[SequenceReplayBuffer, ...],
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    is_last: np.ndarray,
    is_terminal: np.ndarray,
    cuts: np.ndarray | None = None,
) -> None:
    buffers = replay if isinstance(replay, tuple) else (replay,)
    for buffer in buffers:
        buffer.add_step(
            observations=observations,
            actions=actions,
            rewards=rewards,
            is_last=is_last,
            is_terminal=is_terminal,
            cuts=cuts,
        )


def _collect_validation_replay(
    args: argparse.Namespace,
    config: JepaConfig,
    *,
    seed: int,
) -> SequenceReplayBuffer:
    adapter = _make_vector_adapter(args, seed=seed)
    try:
        replay = _new_replay_buffer(
            capacity=args.validation_steps,
            num_envs=args.num_envs,
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
        )
        _collect_random_steps(
            adapter,
            adapter.reset(),
            np.random.default_rng(seed),
            replay,
            steps=args.validation_steps,
            desc="collect validation replay",
            quiet=args.quiet,
        )
        return replay
    finally:
        adapter.close()


def _log_collection_episode_reports(
    logger: RunLogger,
    metrics: dict[str, Any],
    *,
    online_iteration: int,
) -> None:
    returns = metrics.get("returns", ())
    lengths = metrics.get("lengths", ())
    finish_steps = metrics.get("episode_finish_train_env_steps", ())
    for index, episode_return in enumerate(returns):
        if index >= len(finish_steps):
            break
        row = {
            "phase": "online_episode",
            "online_iteration": online_iteration,
            "budget/train_env_steps": int(finish_steps[index]),
            "report/episode_return": float(episode_return),
        }
        if index < len(lengths):
            row["report/episode_length"] = int(lengths[index])
        logger.append_metrics(row)


def _fit_world_model(
    args: argparse.Namespace,
    logger: RunLogger,
    state,
    rng: jax.Array,
    replay: SequenceReplayBuffer,
    config: JepaConfig,
    *,
    np_rng: np.random.Generator,
    steps: int,
    phase: str,
    desc: str,
    train_env_steps: int,
    recent_replay: SequenceReplayBuffer | None = None,
    recent_fraction: float = 0.0,
    freeze_encoder: bool = False,
) -> tuple[Any, jax.Array, dict[str, Any]]:
    metrics: dict[str, Any] = {}
    progress = tqdm(
        range(1, steps + 1),
        desc=desc,
        unit="update",
        disable=args.quiet,
    )
    for step_index in progress:
        batch = _sample_replay_batch(
            replay,
            np_rng,
            recent_replay=recent_replay,
            recent_fraction=recent_fraction,
            batch_size=args.batch_size,
            chunk_length=args.chunk_length,
            max_horizon=max(args.model_horizon, args.open_loop_horizon),
        )
        rng, train_key = jax.random.split(rng)
        state, metrics = train_model_step(
            state,
            train_key,
            batch,
            config,
            chunk_length=args.chunk_length,
            freeze_encoder=freeze_encoder,
        )
        if (
            step_index == 1
            or step_index == steps
            or step_index % args.eval_interval == 0
        ):
            progress.set_postfix(
                loss=f"{float(metrics['model/total_loss']):.4g}",
                jepa=f"{float(metrics['model/jepa_loss']):.4g}",
            )
            logger.append_metrics(
                {
                    "phase": phase,
                    "update": step_index,
                    "budget/train_env_steps": train_env_steps,
                    "data/online_recent_replay_fraction": recent_fraction,
                    "model/online_encoder_frozen": float(freeze_encoder),
                    **metrics,
                }
            )
    return state, rng, to_jsonable(metrics)


def _train_policy(
    args: argparse.Namespace,
    logger: RunLogger,
    state,
    config: JepaConfig,
    replay: SequenceReplayBuffer,
    *,
    np_rng: np.random.Generator,
    rng: jax.Array,
    action_low: np.ndarray,
    action_high: np.ndarray,
    phase: str,
    train_steps: int,
    reset_actor: bool,
    actor_update_interval: int,
    actor_entropy_coef: float,
    value_clip: float,
    reset_start_fraction: float = 0.0,
    reset_start_max_age: int = 63,
) -> tuple[Any, jax.Array, dict[str, Any]]:
    if train_steps == 0:
        return (
            state,
            rng,
            {
                "policy_training_enabled": False,
                "policy_phase": phase,
                "policy_train_steps": 0,
            },
        )
    if reset_actor:
        rng, reset_key = jax.random.split(rng)
        state = reset_policy_heads(state, reset_key, config)

    action_low_jax = jnp.asarray(action_low, dtype=jnp.float32)
    action_high_jax = jnp.asarray(action_high, dtype=jnp.float32)
    metrics: dict[str, Any] = {}
    reset_start_indices = None
    if reset_start_fraction > 0.0:
        reset_start_indices = replay.episode_start_indices(
            max_age=reset_start_max_age,
            chunk_length=config.context_window,
            max_horizon=1,
        )
        if reset_start_indices[0].size == 0:
            raise ValueError(
                "main replay contains no valid reset-aligned policy starts"
            )
    actor_reference_params = jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        state.params,
    )
    actor_updates = 0
    progress = tqdm(
        range(1, train_steps + 1),
        desc=f"{phase} train actor-critic",
        unit="update",
        disable=args.quiet,
    )
    for step_index in progress:
        apply_actor_update = step_index % actor_update_interval == 0
        if (
            apply_actor_update
            and actor_updates % args.policy_actor_kl_reference_interval == 0
        ):
            actor_reference_params = jax.tree_util.tree_map(
                jax.lax.stop_gradient,
                state.params,
            )
        start_observations, start_actions = _sample_policy_starts_with_reset_mix(
            replay,
            np_rng,
            config=config,
            batch_size=args.policy_batch_size,
            reset_start_indices=reset_start_indices,
            reset_start_fraction=reset_start_fraction,
        )
        real_critic_batch = None
        if args.policy_replay_critic_loss_coef > 0.0:
            real_critic_batch = replay.sample(
                np_rng,
                batch_size=args.policy_replay_critic_batch_size,
                chunk_length=args.policy_replay_critic_horizon,
                max_horizon=1,
            )
        rng, policy_key = jax.random.split(rng)
        state, metrics = continuous_policy_train_step(
            state,
            policy_key,
            start_observations,
            config,
            action_low_jax,
            action_high_jax,
            imag_horizon=args.imag_horizon,
            return_normalization_ema_decay=args.policy_return_ema_decay,
            value_clip=value_clip,
            actor_reference_params=(
                actor_reference_params if args.policy_actor_kl_coef > 0.0 else None
            ),
            actor_kl_coef=args.policy_actor_kl_coef,
            actor_kl_target_per_dim=args.policy_actor_kl_target_per_dim,
            action_saturation_threshold=0.95,
            start_actions=start_actions,
            actor_entropy_coef=actor_entropy_coef,
            target_critic_params=(
                state.target_critic_params
                if args.target_critic_ema_decay > 0.0
                else None
            ),
            target_critic_ema_decay=args.target_critic_ema_decay,
            real_critic_batch=real_critic_batch,
            real_critic_loss_enabled=args.policy_replay_critic_loss_coef > 0.0,
            real_critic_loss_coef=args.policy_replay_critic_loss_coef,
            real_critic_horizon=args.policy_replay_critic_horizon,
            slow_value_regularization_coef=(args.policy_slow_value_regularization_coef),
            apply_actor_update=apply_actor_update,
        )
        if apply_actor_update:
            actor_updates += 1
        if (
            step_index == 1
            or step_index == train_steps
            or step_index % args.eval_interval == 0
        ):
            progress.set_postfix(
                loss=f"{float(metrics['policy/total_loss']):.4g}",
                score=f"{float(metrics['policy/imagined_return']):.4g}",
            )
            logger.append_metrics(
                {
                    "phase": f"{phase}_actor_critic",
                    "update": step_index,
                    **metrics,
                }
            )

    return (
        state,
        rng,
        {
            "policy_training_enabled": True,
            "policy_phase": phase,
            "policy_reset_actor": reset_actor,
            "policy_train_steps": train_steps,
            "policy_actor_update_interval": actor_update_interval,
            "policy_actor_updates": actor_updates,
            "policy_batch_size": args.policy_batch_size,
            "policy_imag_horizon": args.imag_horizon,
            "policy_return_mode": "lambda",
            "policy_actor_baseline": "value",
            "policy_return_normalization": "ema-percentile",
            "policy_gradient_mode": "reinforce",
            "policy_stochastic_actor": True,
            "policy_actor_entropy_coef": actor_entropy_coef,
            "policy_value_clip": value_clip,
            "policy_value_clip_initial": args.value_clip,
            "policy_value_clip_final": args.value_clip_final,
            "policy_reset_start_fraction": reset_start_fraction,
            "policy_reset_start_max_age": reset_start_max_age,
            "policy_reset_start_candidate_count": (
                int(reset_start_indices[0].size)
                if reset_start_indices is not None
                else 0
            ),
            "policy_target_critic_ema_decay": args.target_critic_ema_decay,
            "policy_actor_kl_coef": args.policy_actor_kl_coef,
            "policy_actor_kl_target_per_dim": args.policy_actor_kl_target_per_dim,
            "policy_actor_kl_reference_interval": (
                args.policy_actor_kl_reference_interval
            ),
            "policy_actor_kl_reference_mode": "phase",
            "policy_replay_critic_loss_coef": args.policy_replay_critic_loss_coef,
            "policy_final_metrics": to_jsonable(metrics),
        },
    )


def _final_policy_evaluation(
    args: argparse.Namespace,
    logger: RunLogger,
    state,
    config: JepaConfig,
    *,
    seed: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> dict[str, Any] | None:
    if args.final_policy_eval_episodes == 0:
        return None
    evaluation_seed = (
        seed + 9_000_000
        if args.final_policy_eval_seed is None
        else args.final_policy_eval_seed
    )
    num_envs = args.final_policy_eval_num_envs or min(
        args.num_envs,
        args.final_policy_eval_episodes,
    )
    evaluation = _evaluate_continuous_policy(
        args,
        state,
        config,
        seed=evaluation_seed,
        num_envs=num_envs,
        episodes=args.final_policy_eval_episodes,
        action_low=jnp.asarray(action_low, dtype=jnp.float32),
        action_high=jnp.asarray(action_high, dtype=jnp.float32),
        desc="evaluate final latest policy",
        video_logger=logger if args.wandb_videos else None,
        video_filename="videos/final_policy.mp4" if args.wandb_videos else None,
        video_key="videos/final/policy" if args.wandb_videos else None,
        video_caption="Final latest-policy evaluation",
    )
    evaluation = {**evaluation, "evaluation_seed": evaluation_seed}
    logger.write_json("final_policy_evaluation.json", evaluation)
    logger.append_metrics({"phase": "final_policy_evaluation", **evaluation})
    return evaluation


def _curve_policy_evaluation(
    args: argparse.Namespace,
    logger: RunLogger,
    state,
    config: JepaConfig,
    *,
    seed: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    phase: str,
    online_iteration: int,
    train_env_steps: int,
    scheduled_env_steps: int,
) -> dict[str, Any]:
    evaluation_seed = (
        args.curve_eval_seed
        if args.curve_eval_seed is not None
        else (
            args.final_policy_eval_seed
            if args.final_policy_eval_seed is not None
            else seed + 9_000_000
        )
    )
    num_envs = args.curve_eval_num_envs or min(
        args.num_envs,
        args.curve_eval_episodes,
    )
    evaluation = _evaluate_continuous_policy(
        args,
        state,
        config,
        seed=evaluation_seed,
        num_envs=num_envs,
        episodes=args.curve_eval_episodes,
        action_low=jnp.asarray(action_low, dtype=jnp.float32),
        action_high=jnp.asarray(action_high, dtype=jnp.float32),
        desc=f"evaluate latest policy at {train_env_steps} train steps",
    )
    evaluation = {
        **evaluation,
        "evaluation_seed": evaluation_seed,
        "online_iteration": online_iteration,
        "scheduled_train_env_steps": scheduled_env_steps,
        "train_env_steps": train_env_steps,
    }
    logger.write_json(f"{phase}_policy_evaluation.json", evaluation)
    logger.append_metrics(
        {
            "phase": "policy_curve_evaluation",
            "online_iteration": online_iteration,
            "budget/train_env_steps": train_env_steps,
            "eval/return_mean": evaluation["mean_return"],
            "eval/return_std": evaluation["std_return"],
            "eval/return_p10": evaluation["return_p10"],
            "eval/return_cvar10": evaluation["return_cvar10"],
            "eval/failure_rate": evaluation["failure_rate"],
            "eval/success_rate": evaluation["success_rate"],
            "eval/episodes": evaluation["episodes"],
        }
    )
    return evaluation


def _evaluate_continuous_policy(
    args: argparse.Namespace,
    state,
    config: JepaConfig,
    *,
    seed: int,
    num_envs: int,
    episodes: int,
    action_low: jax.Array,
    action_high: jax.Array,
    desc: str,
    video_logger: RunLogger | None = None,
    video_filename: str | None = None,
    video_key: str | None = None,
    video_caption: str = "",
) -> dict[str, Any]:
    adapter = _make_vector_adapter(args, seed=seed, num_envs=num_envs)
    try:
        observations = adapter.reset()
        video_frames: list[np.ndarray] = []
        capture_video = (
            video_logger is not None
            and video_filename is not None
            and video_key is not None
            and isinstance(adapter, DMCVectorAdapter)
        )
        if capture_video:
            try:
                video_frames.append(
                    adapter.render(
                        0,
                        height=args.wandb_video_size,
                        width=args.wandb_video_size,
                        camera_id=args.wandb_video_camera,
                    )
                )
            except Exception as error:
                warnings.warn(
                    f"Evaluation video capture failed: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                capture_video = False

        returns: list[float] = []
        lengths: list[int] = []
        step_calls = 0
        action_key = jax.random.PRNGKey(seed)
        with tqdm(
            total=episodes,
            desc=desc,
            unit="episode",
            disable=args.quiet,
        ) as progress:
            while len(returns) < episodes:
                before = len(returns)
                action_key, step_action_key = jax.random.split(action_key)
                actions = np.asarray(
                    select_continuous_actions(
                        state,
                        jnp.asarray(observations[:, 0], dtype=jnp.float32),
                        config,
                        action_low,
                        action_high,
                        key=step_action_key,
                        stochastic=False,
                    )
                )
                step = adapter.step(actions[:, None, :])
                step_calls += 1
                first_env_done = any(info.get("env_index") == 0 for info in step.infos)
                if (
                    capture_video
                    and not first_env_done
                    and step_calls % args.wandb_video_frame_stride == 0
                ):
                    try:
                        video_frames.append(
                            adapter.render(
                                0,
                                height=args.wandb_video_size,
                                width=args.wandb_video_size,
                                camera_id=args.wandb_video_camera,
                            )
                        )
                    except Exception as error:
                        warnings.warn(
                            f"Evaluation video capture failed: {error}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        capture_video = False
                if first_env_done:
                    capture_video = False
                returns.extend(float(item[0]) for item in step.completed_returns)
                lengths.extend(int(item) for item in step.completed_lengths)
                observations = step.observations
                progress.update(
                    max(0, min(len(returns), episodes) - min(before, episodes))
                )

        returns = returns[:episodes]
        lengths = lengths[:episodes]
        video_path = None
        if video_logger is not None and video_filename and video_key:
            video_path = video_logger.write_video(
                video_filename,
                video_frames,
                fps=args.wandb_video_fps,
                key=video_key,
                caption=video_caption or desc,
            )
        return {
            "episodes": len(returns),
            "num_envs": num_envs,
            "stochastic_actions": False,
            "env_steps": step_calls * num_envs,
            "completed_episode_steps": int(sum(lengths)),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "returns": returns,
            "lengths": lengths,
            "video_path": str(video_path) if video_path is not None else None,
            **_return_tail_metrics(
                returns,
                failure_threshold=args.failure_return_threshold,
                success_threshold=args.success_return_threshold,
            ),
        }
    finally:
        adapter.close()


def _evaluate_model(
    state,
    key: jax.Array,
    batch: ReplayBatch,
    config: JepaConfig,
    *,
    chunk_length: int,
    open_loop_horizon: int,
) -> dict[str, Any]:
    metrics = dict(
        evaluate_world_model_loss(
            state,
            key,
            batch,
            config,
            chunk_length=chunk_length,
        )
    )
    metrics.update(
        evaluate_open_loop(
            state,
            batch,
            config,
            horizon=open_loop_horizon,
        )
    )
    return to_jsonable(metrics)


def _reload_prediction_diff(
    state,
    config: JepaConfig,
    *,
    checkpoint_dir: Path,
    batch: ReplayBatch,
    seed: int,
    chunk_length: int,
) -> float:
    fresh = create_jepa_train_state(jax.random.PRNGKey(seed), config)
    fresh = fresh.replace(
        params=load_params(checkpoint_dir / "checkpoint.msgpack", fresh.params)
    )
    original = state.apply_fn(
        {"params": state.params},
        batch.observations,
        batch.actions,
        chunk_length=chunk_length,
        is_last=batch.is_last,
        method=JepaWorldModel.sequence_outputs,
    )["predicted_latents"]
    reloaded = fresh.apply_fn(
        {"params": fresh.params},
        batch.observations,
        batch.actions,
        chunk_length=chunk_length,
        is_last=batch.is_last,
        method=JepaWorldModel.sequence_outputs,
    )["predicted_latents"]
    return float(jnp.max(jnp.abs(original - reloaded)))


def summarize(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "algorithm": "single_agent_jepa_mbrl",
        "protocol": (
            outcomes[0].get("protocol", "reset_rich_interleaved_latest_policy")
            if outcomes
            else "reset_rich_interleaved_latest_policy"
        ),
        "runs_total": len(outcomes),
        "aggregate_initial_jepa_loss": _mean(outcomes, "initial_jepa_loss"),
        "aggregate_final_jepa_loss": _mean(outcomes, "final_jepa_loss"),
        "aggregate_initial_open_loop_loss": _mean(
            outcomes,
            "initial_open_loop_loss",
        ),
        "aggregate_final_open_loop_loss": _mean(
            outcomes,
            "final_open_loop_loss",
        ),
        "aggregate_final_policy_eval_mean": _mean(
            outcomes,
            "final_policy_eval_mean",
        ),
        "aggregate_final_policy_eval_std": _mean(
            outcomes,
            "final_policy_eval_std",
        ),
        "aggregate_final_policy_eval_return_p10": _mean(
            outcomes,
            "final_policy_eval_return_p10",
        ),
        "aggregate_final_policy_eval_return_cvar10": _mean(
            outcomes,
            "final_policy_eval_return_cvar10",
        ),
        "aggregate_final_policy_eval_failure_rate": _mean(
            outcomes,
            "final_policy_eval_failure_rate",
        ),
        "aggregate_final_policy_eval_success_rate": _mean(
            outcomes,
            "final_policy_eval_success_rate",
        ),
        "aggregate_final_policy_eval_episodes": _mean(
            outcomes,
            "final_policy_eval_episodes",
        ),
        "aggregate_final_policy_eval_env_steps": _mean(
            outcomes,
            "final_policy_eval_env_steps",
        ),
        "aggregate_dreamer_style_train_return_mean": _mean(
            outcomes,
            "dreamer_style_train_return_mean",
        ),
        "aggregate_dreamer_style_train_return_std": _mean(
            outcomes,
            "dreamer_style_train_return_std",
        ),
        "aggregate_dreamer_style_train_return_episodes": _mean(
            outcomes,
            "dreamer_style_train_return_episodes",
        ),
        "aggregate_real_train_replay_env_steps": _mean(
            outcomes,
            "real_train_replay_env_steps",
        ),
        "aggregate_real_validation_replay_env_steps": _mean(
            outcomes,
            "real_validation_replay_env_steps",
        ),
        "aggregate_real_train_plus_validation_env_steps": _mean(
            outcomes,
            "real_train_plus_validation_env_steps",
        ),
        "aggregate_real_policy_eval_env_steps": _mean(
            outcomes,
            "real_policy_eval_env_steps",
        ),
        "aggregate_real_total_env_steps": _mean(
            outcomes,
            "real_total_env_steps",
        ),
        "runs": outcomes,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


if __name__ == "__main__":
    main()
