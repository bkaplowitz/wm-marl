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
from functools import partial
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
from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.replay import ReplayBatch, SequenceReplayBuffer
from world_marl.jepa.reproducibility import (
    JaxRngStreams,
    NumpyRngStreams,
    fingerprint_pytree,
)
from world_marl.jepa.training import (
    continuous_critic_warmup_step,
    continuous_policy_train_step,
    create_jepa_train_state,
    evaluate_open_loop,
    evaluate_world_model_loss,
    reset_policy_heads,
    select_continuous_actions,
    train_model_step,
)
from world_marl.logging import (
    RunLogger,
    WandbConfig,
    dependency_versions,
    timestamp,
    to_jsonable,
)


CONTROL = "none"
MIN_TERMINAL_FRACTION_FOR_CONTINUE_BASELINE = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    environment = parser.add_argument_group("environment and data")
    environment.add_argument("--env", default="dmc:reacher/easy")
    environment.add_argument("--num-envs", type=int, default=16)
    environment.add_argument(
        "--env-workers",
        "--dmc-workers",
        dest="env_workers",
        type=int,
        default=1,
    )
    environment.add_argument("--brax-backend", default=None)
    environment.add_argument("--max-cycles", type=int, default=1000)
    environment.add_argument("--collect-steps", type=int, default=320)
    environment.add_argument("--initial-reset-interval", type=int, default=80)
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
    world_model.add_argument("--dynamics-ensemble-size", type=int, default=1)
    world_model.add_argument(
        "--target-gradient",
        choices=("stopgrad", "symmetric"),
        default="stopgrad",
    )
    world_model.add_argument(
        "--residual-dynamics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    world_model.add_argument("--learning-rate", type=float, default=4e-5)
    world_model.add_argument("--model-grad-clip-norm", type=float, default=0.0)
    world_model.add_argument("--optimizer-warmup-steps", type=int, default=1000)
    world_model.add_argument("--adaptive-grad-clip", type=float, default=0.3)
    world_model.add_argument("--optimizer-epsilon", type=float, default=1e-8)
    world_model.add_argument(
        "--input-symlog",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    world_model.add_argument("--activation", choices=("gelu", "silu"), default="silu")
    world_model.add_argument(
        "--normalization",
        choices=("layer", "rms"),
        default="rms",
    )
    world_model.add_argument("--reward-output-scale", type=float, default=0.0)
    world_model.add_argument(
        "--regularizer",
        choices=("sigreg", "none"),
        default="sigreg",
    )
    world_model.add_argument(
        "--regularizer-weight",
        "--sigreg-weight",
        dest="regularizer_weight",
        type=float,
        default=0.05,
    )
    world_model.add_argument("--sigreg-knots", type=int, default=17)
    world_model.add_argument("--sigreg-num-proj", type=int, default=256)
    world_model.add_argument("--reward-weight", type=float, default=1.0)
    world_model.add_argument("--continue-weight", type=float, default=1.0)
    world_model.add_argument(
        "--reward-prediction-mode",
        choices=("mse", "symlog-twohot"),
        default="symlog-twohot",
    )
    world_model.add_argument("--twohot-bins", type=int, default=255)
    world_model.add_argument("--twohot-min", type=float, default=-20.0)
    world_model.add_argument("--twohot-max", type=float, default=20.0)

    policy = parser.add_argument_group("actor and critic")
    policy.add_argument("--policy-train-steps", type=int, default=1280)
    policy.add_argument("--policy-batch-size", type=int, default=1024)
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
    policy.add_argument(
        "--stochastic-actor",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    policy.add_argument(
        "--stochastic-collection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    policy.add_argument("--actor-entropy-coef", type=float, default=3e-3)
    policy.add_argument("--actor-entropy-final-coef", type=float, default=None)
    policy.add_argument(
        "--actor-entropy-decay-start-env-steps",
        type=int,
        default=None,
    )
    policy.add_argument(
        "--actor-entropy-decay-end-env-steps",
        type=int,
        default=None,
    )
    policy.add_argument(
        "--actor-entropy-mode",
        choices=("gaussian", "tanh-normal"),
        default="tanh-normal",
    )
    policy.add_argument(
        "--actor-log-std-min",
        type=float,
        default=-2.302585092994046,
    )
    policy.add_argument("--actor-log-std-max", type=float, default=0.0)
    policy.add_argument("--actor-output-scale", type=float, default=0.01)
    policy.add_argument("--critic-warmup-steps", type=int, default=0)
    policy.add_argument("--critic-horizon", type=int, default=64)
    policy.add_argument("--imag-horizon", type=int, default=15)
    policy.add_argument(
        "--policy-return-mode",
        choices=("reward-only", "lambda"),
        default="lambda",
    )
    policy.add_argument(
        "--policy-actor-baseline",
        choices=("none", "value"),
        default="value",
    )
    policy.add_argument(
        "--policy-return-normalization",
        choices=("none", "batch", "percentile", "ema-percentile"),
        default="ema-percentile",
    )
    policy.add_argument(
        "--policy-gradient-mode",
        choices=("dynamics", "reinforce"),
        default="reinforce",
    )
    policy.add_argument("--policy-return-ema-decay", type=float, default=0.99)
    policy.add_argument(
        "--value-clip",
        type=float,
        default=100.0,
        help="Symmetric value-target clip; set to 0 to disable clipping.",
    )
    policy.add_argument(
        "--policy-normalized-advantage-clip",
        type=float,
        default=0.0,
        help=(
            "Symmetric clip applied after actor advantage normalization; "
            "set to 0 to disable. This does not clip critic targets."
        ),
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
    policy.add_argument(
        "--value-prediction-mode",
        choices=("mse", "symlog-twohot"),
        default="symlog-twohot",
    )
    policy.add_argument("--target-critic-ema-decay", type=float, default=0.98)
    policy.add_argument("--policy-replay-critic-loss-coef", type=float, default=0.3)
    policy.add_argument("--policy-replay-critic-batch-size", type=int, default=16)
    policy.add_argument("--policy-replay-critic-horizon", type=int, default=64)
    policy.add_argument(
        "--policy-replay-critic-return-mode",
        choices=("reward-only", "lambda"),
        default="lambda",
    )
    policy.add_argument(
        "--policy-replay-critic-all-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
        "--online-freeze-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Freeze the observation encoder during online world-model refits. "
            "This is intended as a latent-interface stability diagnostic."
        ),
    )
    online.add_argument(
        "--online-recent-replay-fraction",
        type=float,
        default=0.0,
        help=(
            "Default recent-replay fraction for each online learner. "
            "Component-specific fractions override this value."
        ),
    )
    online.add_argument(
        "--online-recent-world-model-fraction",
        type=float,
        default=None,
        help="Override recent replay for online world-model batches.",
    )
    online.add_argument(
        "--online-recent-policy-start-fraction",
        type=float,
        default=None,
        help="Override recent replay for actor imagination start states.",
    )
    online.add_argument(
        "--online-recent-critic-fraction",
        type=float,
        default=None,
        help="Override recent replay for real replay-critic targets.",
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
    reporting.add_argument(
        "--online-interface-diagnostic-batch-size",
        type=int,
        default=256,
        help=(
            "Replay observations used to separate world-model-induced latent "
            "interface drift from actor/critic update drift; set to 0 to disable."
        ),
    )

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
    reproducibility.add_argument("--quiet", action="store_true")
    reproducibility.add_argument("--allow-fail", action="store_true")

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

    args = parser.parse_args()
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = (
        "num_envs",
        "env_workers",
        "max_cycles",
        "collect_steps",
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
        "dynamics_ensemble_size",
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
        "critic_warmup_steps",
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
        "online_interface_diagnostic_batch_size",
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
    recent_fraction_names = (
        "online_recent_replay_fraction",
        "online_recent_world_model_fraction",
        "online_recent_policy_start_fraction",
        "online_recent_critic_fraction",
    )
    for name in recent_fraction_names:
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if (
        args.online_recent_replay_max_oversample != 0.0
        and args.online_recent_replay_max_oversample < 1.0
    ):
        parser.error("--online-recent-replay-max-oversample must be zero or >= 1")
    recent_min_steps = max(
        args.chunk_length + max(args.model_horizon, args.open_loop_horizon),
        args.context_window + 1,
        args.policy_replay_critic_horizon + 1,
    )
    requested_recent_fractions = _requested_recent_fractions(args)
    if max(requested_recent_fractions.values()) > 0.0 and (
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
    entropy_schedule = (
        args.actor_entropy_final_coef,
        args.actor_entropy_decay_start_env_steps,
        args.actor_entropy_decay_end_env_steps,
    )
    if any(value is not None for value in entropy_schedule):
        if any(value is None for value in entropy_schedule):
            parser.error(
                "entropy decay requires --actor-entropy-final-coef, "
                "--actor-entropy-decay-start-env-steps, and "
                "--actor-entropy-decay-end-env-steps"
            )
        assert args.actor_entropy_final_coef is not None
        assert args.actor_entropy_decay_start_env_steps is not None
        assert args.actor_entropy_decay_end_env_steps is not None
        if args.actor_entropy_final_coef < 0.0:
            parser.error("--actor-entropy-final-coef must be >= 0")
        if args.actor_entropy_final_coef > args.actor_entropy_coef:
            parser.error("--actor-entropy-final-coef must be <= --actor-entropy-coef")
        if args.actor_entropy_decay_start_env_steps < 0:
            parser.error("--actor-entropy-decay-start-env-steps must be >= 0")
        if (
            args.actor_entropy_decay_end_env_steps
            <= args.actor_entropy_decay_start_env_steps
        ):
            parser.error(
                "--actor-entropy-decay-end-env-steps must be greater than "
                "--actor-entropy-decay-start-env-steps"
            )
    if args.stochastic_collection and not args.stochastic_actor:
        parser.error("--stochastic-collection requires --stochastic-actor")
    if args.policy_gradient_mode == "reinforce" and not args.stochastic_actor:
        parser.error("--policy-gradient-mode reinforce requires --stochastic-actor")
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
    if args.policy_normalized_advantage_clip < 0.0:
        parser.error(
            "--policy-normalized-advantage-clip must be >= 0 (0 disables clipping)"
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
            run_dir=experiment_dir / CONTROL / f"run_{run_index:03d}",
            run_index=run_index,
        )
        for run_index in range(args.num_runs)
    ]
    summary = summarize(outcomes)
    RunLogger(experiment_dir).write_json("summary.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    if not args.allow_fail and not summary["passed"]:
        raise SystemExit(1)


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
        stochastic_actor=args.stochastic_actor,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_max=args.actor_log_std_max,
        input_symlog=args.input_symlog,
        activation=args.activation,
        normalization=args.normalization,
        actor_output_scale=args.actor_output_scale,
        value_output_scale=args.value_output_scale,
        reward_output_scale=args.reward_output_scale,
        regularizer=args.regularizer,
        regularizer_weight=args.regularizer_weight,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        reward_weight=args.reward_weight,
        continue_weight=args.continue_weight,
        reward_prediction_mode=args.reward_prediction_mode.replace("-", "_"),
        value_prediction_mode=args.value_prediction_mode.replace("-", "_"),
        twohot_bins=args.twohot_bins,
        twohot_min=args.twohot_min,
        twohot_max=args.twohot_max,
        dynamics_ensemble_size=args.dynamics_ensemble_size,
        gamma=args.gamma,
        lambda_return=args.lambda_return,
        residual_dynamics=args.residual_dynamics,
        target_gradient=args.target_gradient,
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


def _requested_recent_fractions(args: argparse.Namespace) -> dict[str, float]:
    default = float(args.online_recent_replay_fraction)
    return {
        "world_model": float(
            default
            if args.online_recent_world_model_fraction is None
            else args.online_recent_world_model_fraction
        ),
        "policy_start": float(
            default
            if args.online_recent_policy_start_fraction is None
            else args.online_recent_policy_start_fraction
        ),
        "critic": float(
            default
            if args.online_recent_critic_fraction is None
            else args.online_recent_critic_fraction
        ),
    }


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
        requested_recent_fractions = _requested_recent_fractions(args)
        resolved_config = {
            "args": vars(args),
            "run_index": run_index,
            "seed": seed,
            "control": CONTROL,
            "observation_shape": adapter.observation_shape,
            "action_shape": adapter.action_shape,
            "action_low": adapter.action_low,
            "action_high": adapter.action_high,
            "env_backend": _env_backend(args.env),
            "jepa_config": dataclasses.asdict(config),
            "online_recent_replay_requested_fractions": (requested_recent_fractions),
            "protocol": "reset_rich_interleaved_latest_policy",
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

        state = create_jepa_train_state(jax_rngs.take("initialization"), config)
        observations = adapter.reset()
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
            if max(requested_recent_fractions.values()) > 0.0
            else None
        )

        # Bootstrap collection uses its own adapter. This reproduces the
        # cache-loaded reference protocol without relying on an external NPZ or
        # advancing the online environments away from their first reset.
        bootstrap_adapter = _make_vector_adapter(args, seed=seed)
        try:
            _, initial_train_env_steps = _collect_random_steps(
                bootstrap_adapter,
                bootstrap_adapter.reset(),
                numpy_rngs.get("initial_collection"),
                (
                    (replay, online_recent_replay)
                    if online_recent_replay is not None
                    else replay
                ),
                steps=args.collect_steps,
                reset_interval=args.initial_reset_interval,
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

        validation_replay = _collect_validation_replay(
            args,
            config,
            seed=validation_seed,
        )
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
            action_low=adapter.action_low,
            action_high=adapter.action_high,
        )
        logger.write_json("model_metrics_initial.json", initial_metrics)

        model_rng = jax_rngs.current("world_model")
        state, model_rng, _, initial_model_losses = _fit_world_model(
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
        logger.plot_world_model_loss(
            initial_model_losses,
            filename="world_model_initial_loss.png",
        )
        initial_fit_metrics = _evaluate_model(
            state,
            validation_jax_rngs.take("evaluation"),
            validation_batch,
            config,
            chunk_length=args.chunk_length,
            open_loop_horizon=args.open_loop_horizon,
            action_low=adapter.action_low,
            action_high=adapter.action_high,
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
            actor_entropy_coef=_scheduled_actor_entropy_coef(
                args,
                train_env_steps=train_env_steps,
            ),
        )
        jax_rngs.update("policy", policy_rng)
        logger.write_json("policy_initial_fit.json", initial_policy_metrics)
        logger.write_json(
            "reproducibility_initial_policy.json",
            _reproducibility_snapshot(state, phase="initial_policy"),
        )

        online_history: list[dict[str, Any]] = []
        curve_evaluations: list[dict[str, Any]] = []
        next_curve_eval_step = (
            args.curve_eval_interval_env_steps
            if args.curve_eval_interval_env_steps > 0
            else None
        )
        while (
            next_curve_eval_step is not None and next_curve_eval_step <= train_env_steps
        ):
            next_curve_eval_step += args.curve_eval_interval_env_steps
        for online_index in range(1, args.online_iterations + 1):
            phase = f"online_{online_index:03d}"
            phase_start_train_env_steps = train_env_steps
            phase_replay = _new_replay_buffer(
                capacity=args.online_collect_steps,
                num_envs=args.num_envs,
                observation_dim=config.observation_dim,
                action_dim=config.action_dim,
            )
            collection_replays = [replay, phase_replay]
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
                stochastic_actions=args.stochastic_collection,
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
                "recent_replay_size_per_env": phase_replay.size,
                "online_recent_replay_size_per_env": (
                    online_recent_replay.size if online_recent_replay is not None else 0
                ),
            }
            recent_replay_size = (
                online_recent_replay.size if online_recent_replay is not None else 0
            )
            effective_recent_fractions = {
                component: _effective_recent_fraction(
                    requested_fraction,
                    full_replay_size=replay.size,
                    recent_replay_size=recent_replay_size,
                    max_oversample=args.online_recent_replay_max_oversample,
                )
                for component, requested_fraction in (
                    requested_recent_fractions.items()
                )
            }
            effective_recent_oversamples = {
                component: _recent_oversample_ratio(
                    effective_fraction,
                    full_replay_size=replay.size,
                    recent_replay_size=recent_replay_size,
                )
                for component, effective_fraction in (
                    effective_recent_fractions.items()
                )
            }
            collection.update(
                {
                    "online_recent_replay_fraction": (
                        args.online_recent_replay_fraction
                    ),
                    "online_recent_replay_requested_fraction": (
                        args.online_recent_replay_fraction
                    ),
                    "online_recent_replay_effective_fraction": (
                        effective_recent_fractions["world_model"]
                    ),
                    "online_recent_replay_max_oversample": (
                        args.online_recent_replay_max_oversample
                    ),
                    "online_recent_replay_effective_oversample": (
                        effective_recent_oversamples["world_model"]
                    ),
                    **{
                        f"online_recent_{component}_requested_fraction": (
                            requested_recent_fractions[component]
                        )
                        for component in requested_recent_fractions
                    },
                    **{
                        f"online_recent_{component}_effective_fraction": (
                            effective_recent_fractions[component]
                        )
                        for component in requested_recent_fractions
                    },
                    **{
                        f"online_recent_{component}_effective_oversample": (
                            effective_recent_oversamples[component]
                        )
                        for component in requested_recent_fractions
                    },
                }
            )
            logger.write_json(f"{phase}_actor_replay.json", collection)
            logger.append_metrics(
                {
                    "phase": "online_actor_replay",
                    "online_iteration": online_index,
                    "report": _collection_report_summary(collection),
                    **collection,
                }
            )

            interface_observations = None
            interface_before_world_model = None
            if args.online_interface_diagnostic_batch_size > 0:
                interface_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, 17_001, online_index])
                )
                interface_observations, _ = _sample_policy_starts(
                    replay,
                    interface_rng,
                    config=config,
                    batch_size=args.online_interface_diagnostic_batch_size,
                )
                interface_before_world_model = _policy_interface_snapshot(
                    state,
                    interface_observations,
                    config,
                )

            model_rng = jax_rngs.current("world_model")
            state, model_rng, _, online_model_losses = _fit_world_model(
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
                recent_fraction=effective_recent_fractions["world_model"],
                freeze_encoder=args.online_freeze_encoder,
            )
            jax_rngs.update("world_model", model_rng)
            logger.plot_world_model_loss(
                online_model_losses,
                filename=f"{phase}_world_model_loss.png",
            )
            model_metrics = _evaluate_model(
                state,
                validation_jax_rngs.take("evaluation"),
                validation_batch,
                config,
                chunk_length=args.chunk_length,
                open_loop_horizon=args.open_loop_horizon,
                action_low=adapter.action_low,
                action_high=adapter.action_high,
            )
            logger.write_json(f"{phase}_model_metrics.json", model_metrics)

            interface_after_world_model = None
            if interface_observations is not None:
                interface_after_world_model = _policy_interface_snapshot(
                    state,
                    interface_observations,
                    config,
                )

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
                actor_entropy_coef=_scheduled_actor_entropy_coef(
                    args,
                    train_env_steps=train_env_steps,
                ),
                recent_replay=online_recent_replay,
                start_recent_fraction=effective_recent_fractions["policy_start"],
                critic_recent_fraction=effective_recent_fractions["critic"],
            )
            policy_metrics.update(
                {
                    "policy_online_recent_start_requested_fraction": (
                        requested_recent_fractions["policy_start"]
                    ),
                    "policy_online_recent_start_effective_oversample": (
                        effective_recent_oversamples["policy_start"]
                    ),
                    "policy_online_recent_critic_requested_fraction": (
                        requested_recent_fractions["critic"]
                    ),
                    "policy_online_recent_critic_effective_oversample": (
                        effective_recent_oversamples["critic"]
                    ),
                    "policy_online_recent_replay_max_oversample": (
                        args.online_recent_replay_max_oversample
                    ),
                }
            )
            jax_rngs.update("policy", policy_rng)
            logger.write_json(f"{phase}_policy.json", policy_metrics)

            interface_metrics = None
            if (
                interface_observations is not None
                and interface_before_world_model is not None
                and interface_after_world_model is not None
            ):
                interface_after_policy = _policy_interface_snapshot(
                    state,
                    interface_observations,
                    config,
                )
                interface_metrics = {
                    "phase": "online_interface_drift",
                    "online_iteration": online_index,
                    "budget/train_env_steps": train_env_steps,
                    "interface/anchor_batch_size": (
                        args.online_interface_diagnostic_batch_size
                    ),
                    **_policy_interface_drift_metrics(
                        interface_before_world_model,
                        interface_after_world_model,
                        prefix="interface/world_model",
                    ),
                    **_policy_interface_drift_metrics(
                        interface_after_world_model,
                        interface_after_policy,
                        prefix="interface/policy_update",
                    ),
                    **_policy_interface_drift_metrics(
                        interface_before_world_model,
                        interface_after_policy,
                        prefix="interface/total_phase",
                    ),
                }
                logger.write_json(f"{phase}_interface_drift.json", interface_metrics)
                logger.append_metrics(interface_metrics)

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
            reproducibility = _reproducibility_snapshot(
                state,
                phase=phase,
                recent_replay=phase_replay,
                full_replay=replay if checkpoint_phase else None,
            )
            if online_recent_replay is not None:
                reproducibility.update(
                    {
                        "online_recent_replay_sha256": (
                            online_recent_replay.fingerprint()
                        ),
                        "online_recent_replay_size_per_env": (
                            online_recent_replay.size
                        ),
                    }
                )
            logger.write_json(f"{phase}_reproducibility.json", reproducibility)
            online_history.append(
                {
                    "iteration": online_index,
                    "actor_replay": collection,
                    "model_metrics": model_metrics,
                    "policy": policy_metrics,
                    "interface_drift": interface_metrics,
                    "policy_evaluation": curve_evaluation,
                    "reproducibility": reproducibility,
                    "world_model_train_steps": args.online_train_steps,
                    "policy_train_steps": args.online_policy_train_steps,
                    "policy_actor_update_interval": actor_update_interval,
                    "policy_actor_updates": policy_metrics["policy_actor_updates"],
                }
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
            action_low=adapter.action_low,
            action_high=adapter.action_high,
        )
        logger.write_json("model_metrics_final.json", final_metrics)
        logger.write_json(
            "reproducibility_final.json",
            _reproducibility_snapshot(state, phase="final", full_replay=replay),
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
        world_model_passed = _run_passed(
            initial_metrics,
            final_metrics,
            reload_diff,
        )
        outcome = {
            "run_index": run_index,
            "seed": seed,
            "control": CONTROL,
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "protocol": "reset_rich_interleaved_latest_policy",
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
            "world_model_passed": world_model_passed,
            "passed": world_model_passed,
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
            "run/world_model_passed": world_model_passed,
            "run/passed": world_model_passed,
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
    desc: str,
    quiet: bool,
) -> tuple[np.ndarray, int]:
    for step_index in tqdm(range(steps), desc=desc, unit="step", disable=quiet):
        actions = adapter.sample_actions(rng)
        step = adapter.step(actions)
        forced_reset = (
            reset_interval is not None and (step_index + 1) % reset_interval == 0
        )
        _add_replay_step(
            replay,
            observations=observations[:, 0],
            actions=actions[:, 0],
            rewards=step.rewards[:, 0],
            dones=step.dones[:, 0],
            cuts=(
                np.ones((adapter.num_envs,), dtype=np.float32) if forced_reset else None
            ),
        )
        observations = adapter.reset() if forced_reset else step.observations
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
            dones=step.dones[:, 0],
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
    dones: np.ndarray,
    cuts: np.ndarray | None = None,
) -> None:
    buffers = replay if isinstance(replay, tuple) else (replay,)
    for buffer in buffers:
        buffer.add_step(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
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


def _collection_report_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "return_mean": metrics.get("mean_return"),
        "return_std": metrics.get("std_return"),
        "return_p10": metrics.get("return_p10"),
        "return_cvar10": metrics.get("return_cvar10"),
        "failure_rate": metrics.get("failure_rate"),
        "success_rate": metrics.get("success_rate"),
        "completed_episodes": metrics.get("completed_episodes", 0),
    }


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


def _recent_batch_size(
    batch_size: int,
    *,
    recent_replay: SequenceReplayBuffer | None,
    recent_fraction: float,
) -> int:
    if recent_replay is None or recent_fraction <= 0.0:
        return 0
    return max(0, min(batch_size, int(round(batch_size * recent_fraction))))


def _effective_recent_fraction(
    requested_fraction: float,
    *,
    full_replay_size: int,
    recent_replay_size: int,
    max_oversample: float,
) -> float:
    """Bound recent replay pressure as the full replay grows.

    The recent buffer is a subset of the full replay. Under mixture sampling, a
    recent transition can therefore be drawn from either component while an
    older transition can only be drawn from the full component. This computes
    the largest mixture fraction whose per-transition probability ratio does
    not exceed ``max_oversample``.
    """

    if requested_fraction <= 0.0 or recent_replay_size <= 0:
        return 0.0
    if max_oversample <= 0.0:
        return float(requested_fraction)
    if max_oversample <= 1.0:
        return 0.0

    extra_weight = max_oversample - 1.0
    capped_fraction = (
        extra_weight
        * float(recent_replay_size)
        / (float(full_replay_size) + extra_weight * float(recent_replay_size))
    )
    return float(min(requested_fraction, capped_fraction))


def _recent_oversample_ratio(
    recent_fraction: float,
    *,
    full_replay_size: int,
    recent_replay_size: int,
) -> float | None:
    """Return recent-versus-old per-transition sampling probability."""

    if recent_replay_size <= 0 or full_replay_size <= recent_replay_size:
        return 1.0
    if recent_fraction <= 0.0:
        return 1.0
    if recent_fraction >= 1.0:
        return None
    return float(
        1.0
        + recent_fraction
        * float(full_replay_size)
        / ((1.0 - recent_fraction) * float(recent_replay_size))
    )


def _sample_replay_batch(
    replay: SequenceReplayBuffer,
    rng: np.random.Generator,
    *,
    recent_replay: SequenceReplayBuffer | None,
    recent_fraction: float,
    batch_size: int,
    chunk_length: int,
    max_horizon: int,
) -> ReplayBatch:
    recent_size = _recent_batch_size(
        batch_size,
        recent_replay=recent_replay,
        recent_fraction=recent_fraction,
    )
    if recent_size == 0:
        return replay.sample(
            rng,
            batch_size=batch_size,
            chunk_length=chunk_length,
            max_horizon=max_horizon,
        )
    assert recent_replay is not None
    full_size = batch_size - recent_size
    batches = []
    if full_size:
        batches.append(
            replay.sample(
                rng,
                batch_size=full_size,
                chunk_length=chunk_length,
                max_horizon=max_horizon,
            )
        )
    batches.append(
        recent_replay.sample(
            rng,
            batch_size=recent_size,
            chunk_length=chunk_length,
            max_horizon=max_horizon,
        )
    )
    if len(batches) == 1:
        return batches[0]
    return ReplayBatch(
        observations=jnp.concatenate([batch.observations for batch in batches], axis=0),
        actions=jnp.concatenate([batch.actions for batch in batches], axis=0),
        rewards=jnp.concatenate([batch.rewards for batch in batches], axis=0),
        dones=jnp.concatenate([batch.dones for batch in batches], axis=0),
    )


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
) -> tuple[Any, jax.Array, dict[str, Any], list[float]]:
    loss_history: list[jax.Array] = []
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
            control=CONTROL,
            freeze_encoder=freeze_encoder,
        )
        loss_history.append(metrics["model/total_loss"])
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
    return state, rng, to_jsonable(metrics), [float(loss) for loss in loss_history]


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
    recent_replay: SequenceReplayBuffer | None = None,
    start_recent_fraction: float = 0.0,
    critic_recent_fraction: float = 0.0,
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
    critic_metrics: dict[str, Any] = {}
    if args.critic_warmup_steps > 0:
        critic_progress = tqdm(
            range(1, args.critic_warmup_steps + 1),
            desc=f"{phase} warm replay critic",
            unit="update",
            disable=args.quiet,
        )
        for step_index in critic_progress:
            batch = _sample_replay_batch(
                replay,
                np_rng,
                recent_replay=recent_replay,
                recent_fraction=critic_recent_fraction,
                batch_size=args.policy_batch_size,
                chunk_length=args.critic_horizon,
                max_horizon=1,
            )
            state, critic_metrics = continuous_critic_warmup_step(
                state,
                batch,
                config,
                horizon=args.critic_horizon,
                value_clip=args.value_clip,
                target_critic_ema_decay=args.target_critic_ema_decay,
            )
            if (
                step_index == 1
                or step_index == args.critic_warmup_steps
                or step_index % args.eval_interval == 0
            ):
                critic_progress.set_postfix(
                    loss=f"{float(critic_metrics['critic/total_loss']):.4g}",
                )
                logger.append_metrics(
                    {
                        "phase": f"{phase}_critic_warmup",
                        "update": step_index,
                        **critic_metrics,
                    }
                )

    metrics: dict[str, Any] = {}
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
        start_observations, start_actions = _sample_mixed_policy_starts(
            replay,
            np_rng,
            config=config,
            batch_size=args.policy_batch_size,
            recent_replay=recent_replay,
            recent_fraction=start_recent_fraction,
        )
        real_critic_batch = None
        if args.policy_replay_critic_loss_coef > 0.0:
            real_critic_batch = _sample_replay_batch(
                replay,
                np_rng,
                recent_replay=recent_replay,
                recent_fraction=critic_recent_fraction,
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
            control=CONTROL,
            policy_return_mode=args.policy_return_mode,
            policy_actor_baseline=args.policy_actor_baseline,
            policy_return_normalization=args.policy_return_normalization,
            policy_gradient_mode=args.policy_gradient_mode,
            return_normalization_ema_decay=args.policy_return_ema_decay,
            value_clip=args.value_clip,
            normalized_advantage_clip=args.policy_normalized_advantage_clip,
            actor_reference_params=(
                actor_reference_params if args.policy_actor_kl_coef > 0.0 else None
            ),
            actor_kl_coef=args.policy_actor_kl_coef,
            actor_kl_target_per_dim=args.policy_actor_kl_target_per_dim,
            action_saturation_threshold=0.95,
            start_actions=start_actions,
            actor_entropy_coef=actor_entropy_coef,
            actor_entropy_mode=args.actor_entropy_mode,
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
            real_critic_return_mode=args.policy_replay_critic_return_mode,
            real_critic_all_steps=args.policy_replay_critic_all_steps,
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
            "policy_return_mode": args.policy_return_mode,
            "policy_actor_baseline": args.policy_actor_baseline,
            "policy_return_normalization": args.policy_return_normalization,
            "policy_gradient_mode": args.policy_gradient_mode,
            "policy_stochastic_actor": args.stochastic_actor,
            "policy_actor_entropy_coef": actor_entropy_coef,
            "policy_actor_entropy_initial_coef": args.actor_entropy_coef,
            "policy_actor_entropy_final_coef": args.actor_entropy_final_coef,
            "policy_online_recent_start_fraction": start_recent_fraction,
            "policy_online_recent_critic_fraction": critic_recent_fraction,
            "policy_target_critic_ema_decay": args.target_critic_ema_decay,
            "policy_actor_kl_coef": args.policy_actor_kl_coef,
            "policy_actor_kl_target_per_dim": args.policy_actor_kl_target_per_dim,
            "policy_actor_kl_reference_interval": (
                args.policy_actor_kl_reference_interval
            ),
            "policy_replay_critic_loss_coef": args.policy_replay_critic_loss_coef,
            "critic_warmup_steps": args.critic_warmup_steps,
            "critic_final_metrics": to_jsonable(critic_metrics),
            "policy_final_metrics": to_jsonable(metrics),
        },
    )


def _sample_mixed_policy_starts(
    replay: SequenceReplayBuffer,
    rng: np.random.Generator,
    *,
    config: JepaConfig,
    batch_size: int,
    recent_replay: SequenceReplayBuffer | None,
    recent_fraction: float,
) -> tuple[jax.Array, jax.Array]:
    recent_size = _recent_batch_size(
        batch_size,
        recent_replay=recent_replay,
        recent_fraction=recent_fraction,
    )
    if recent_size == 0:
        return _sample_policy_starts(
            replay,
            rng,
            config=config,
            batch_size=batch_size,
        )
    assert recent_replay is not None
    full_size = batch_size - recent_size
    chunks = []
    if full_size:
        chunks.append(
            _sample_policy_starts(
                replay,
                rng,
                config=config,
                batch_size=full_size,
            )
        )
    chunks.append(
        _sample_policy_starts(
            recent_replay,
            rng,
            config=config,
            batch_size=recent_size,
        )
    )
    return (
        jnp.concatenate([chunk[0] for chunk in chunks], axis=0),
        jnp.concatenate([chunk[1] for chunk in chunks], axis=0),
    )


def _sample_policy_starts(
    replay: SequenceReplayBuffer,
    rng: np.random.Generator,
    *,
    config: JepaConfig,
    batch_size: int,
) -> tuple[jax.Array, jax.Array]:
    observation_chunks = []
    action_chunks = []
    collected = 0
    attempts = 0
    sample_size = max(64, 2 * batch_size)
    while collected < batch_size and attempts < 64:
        attempts += 1
        batch = replay.sample(
            rng,
            batch_size=sample_size,
            chunk_length=config.context_window,
            max_horizon=1,
        )
        done_context = np.asarray(batch.dones[:, : config.context_window])
        valid_indices = np.flatnonzero(np.sum(done_context, axis=1) == 0.0)
        if valid_indices.size == 0:
            continue
        valid_indices = valid_indices[: batch_size - collected]
        observation_chunks.append(
            batch.observations[valid_indices, : config.context_window]
        )
        action_chunks.append(batch.actions[valid_indices, : config.context_window])
        collected += int(valid_indices.size)
    if collected < batch_size:
        raise ValueError(
            "could not sample enough policy starts without episode boundaries; "
            f"collected {collected}/{batch_size} after {attempts} attempts"
        )
    return (
        jnp.concatenate(observation_chunks, axis=0)[:batch_size],
        jnp.concatenate(action_chunks, axis=0)[:batch_size],
    )


def _policy_interface_snapshot(
    state,
    observations: jax.Array,
    config: JepaConfig,
) -> dict[str, np.ndarray]:
    """Evaluate the encoder and policy heads on fixed replay observations."""

    latents = state.apply_fn(
        {"params": state.params},
        observations,
        method=JepaWorldModel.encode,
    )
    policy_latents = latents[:, -1] if latents.ndim == 3 else latents
    means, log_stds, values = state.apply_fn(
        {"params": state.params},
        policy_latents,
        method=JepaWorldModel.actor_value_stats_from_latent,
    )
    return {
        "latents": np.asarray(jax.device_get(latents), dtype=np.float64),
        "means": np.asarray(jax.device_get(means), dtype=np.float64),
        "log_stds": np.asarray(jax.device_get(log_stds), dtype=np.float64),
        "values": np.asarray(jax.device_get(values), dtype=np.float64),
    }


def _policy_interface_drift_metrics(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    *,
    prefix: str,
) -> dict[str, float]:
    """Measure coordinate-sensitive and behavior-level phase-boundary drift."""

    before_latents = np.asarray(before["latents"], dtype=np.float64).reshape(
        (-1, before["latents"].shape[-1])
    )
    after_latents = np.asarray(after["latents"], dtype=np.float64).reshape(
        (-1, after["latents"].shape[-1])
    )
    latent_delta = after_latents - before_latents
    before_norm = np.linalg.norm(before_latents, axis=-1)
    after_norm = np.linalg.norm(after_latents, axis=-1)
    cosine = np.sum(before_latents * after_latents, axis=-1) / np.maximum(
        before_norm * after_norm,
        1e-12,
    )

    before_centered = before_latents - np.mean(before_latents, axis=0, keepdims=True)
    after_centered = after_latents - np.mean(after_latents, axis=0, keepdims=True)
    cross = before_centered.T @ after_centered
    before_gram = before_centered.T @ before_centered
    after_gram = after_centered.T @ after_centered
    cka_denominator = np.sqrt(
        np.sum(np.square(before_gram)) * np.sum(np.square(after_gram))
    )
    linear_cka = np.sum(np.square(cross)) / max(float(cka_denominator), 1e-12)

    before_means = np.asarray(before["means"], dtype=np.float64)
    after_means = np.asarray(after["means"], dtype=np.float64)
    before_log_stds = np.asarray(before["log_stds"], dtype=np.float64)
    after_log_stds = np.asarray(after["log_stds"], dtype=np.float64)
    variance_ratio = np.exp(2.0 * (before_log_stds - after_log_stds))
    squared_mean_delta = np.square(before_means - after_means) * np.exp(
        -2.0 * after_log_stds
    )
    kl_per_dimension = (
        after_log_stds
        - before_log_stds
        + 0.5 * (variance_ratio + squared_mean_delta - 1.0)
    )
    policy_kl_per_dim = np.mean(kl_per_dimension, axis=-1)

    before_actions = np.tanh(before_means)
    after_actions = np.tanh(after_means)
    action_delta = np.abs(after_actions - before_actions)
    before_values = np.asarray(before["values"], dtype=np.float64)
    after_values = np.asarray(after["values"], dtype=np.float64)
    value_delta = np.abs(after_values - before_values)

    return {
        f"{prefix}/latent_cosine_mean": float(np.mean(cosine)),
        f"{prefix}/latent_cosine_p10": float(np.percentile(cosine, 10.0)),
        f"{prefix}/latent_linear_cka": float(linear_cka),
        f"{prefix}/latent_relative_rms_delta": float(
            np.sqrt(np.mean(np.square(latent_delta)))
            / max(float(np.sqrt(np.mean(np.square(before_latents)))), 1e-12)
        ),
        f"{prefix}/latent_norm_ratio": float(
            np.mean(after_norm) / max(float(np.mean(before_norm)), 1e-12)
        ),
        f"{prefix}/policy_kl_per_action_dim_mean": float(np.mean(policy_kl_per_dim)),
        f"{prefix}/policy_kl_per_action_dim_p95": float(
            np.percentile(policy_kl_per_dim, 95.0)
        ),
        f"{prefix}/actor_mean_abs_delta": float(
            np.mean(np.abs(after_means - before_means))
        ),
        f"{prefix}/actor_log_std_abs_delta": float(
            np.mean(np.abs(after_log_stds - before_log_stds))
        ),
        f"{prefix}/normalized_action_mean_abs_delta": float(np.mean(action_delta)),
        f"{prefix}/normalized_action_mean_max_delta": float(np.max(action_delta)),
        f"{prefix}/action_mean_saturation_before": float(
            np.mean(np.abs(before_actions) >= 0.95)
        ),
        f"{prefix}/action_mean_saturation_after": float(
            np.mean(np.abs(after_actions) >= 0.95)
        ),
        f"{prefix}/value_abs_delta": float(np.mean(value_delta)),
        f"{prefix}/value_relative_abs_delta": float(
            np.mean(value_delta) / max(float(np.mean(np.abs(before_values))), 1.0)
        ),
    }


def _scheduled_actor_entropy_coef(
    args: argparse.Namespace,
    *,
    train_env_steps: int,
) -> float:
    final_coef = args.actor_entropy_final_coef
    start = args.actor_entropy_decay_start_env_steps
    end = args.actor_entropy_decay_end_env_steps
    if final_coef is None or start is None or end is None:
        return float(args.actor_entropy_coef)
    if train_env_steps <= start:
        return float(args.actor_entropy_coef)
    if train_env_steps >= end:
        return float(final_coef)
    progress = (train_env_steps - start) / (end - start)
    return float(
        args.actor_entropy_coef + progress * (final_coef - args.actor_entropy_coef)
    )


def _scheduled_online_actor_update_interval(
    args: argparse.Namespace,
    *,
    train_env_steps: int,
) -> int:
    if train_env_steps < args.online_policy_actor_update_interval_start_env_steps:
        return 1
    return int(args.online_policy_actor_update_interval)


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


def _return_tail_metrics(
    returns: list[float],
    *,
    failure_threshold: float,
    success_threshold: float,
) -> dict[str, Any]:
    if not returns:
        return {
            "failure_return_threshold": float(failure_threshold),
            "success_return_threshold": float(success_threshold),
            "failure_count": 0,
            "failure_rate": None,
            "success_count": 0,
            "success_rate": None,
            "return_min": None,
            "return_max": None,
            "return_p05": None,
            "return_p10": None,
            "return_p25": None,
            "return_cvar10": None,
            "nonfailure_mean_return": None,
        }
    values = np.asarray(returns, dtype=np.float32)
    failures = values < float(failure_threshold)
    successes = values >= float(success_threshold)
    tail_count = max(1, int(math.ceil(0.10 * values.size)))
    nonfailures = values[~failures]
    return {
        "failure_return_threshold": float(failure_threshold),
        "success_return_threshold": float(success_threshold),
        "failure_count": int(np.sum(failures)),
        "failure_rate": float(np.mean(failures)),
        "success_count": int(np.sum(successes)),
        "success_rate": float(np.mean(successes)),
        "return_min": float(np.min(values)),
        "return_max": float(np.max(values)),
        "return_p05": float(np.quantile(values, 0.05)),
        "return_p10": float(np.quantile(values, 0.10)),
        "return_p25": float(np.quantile(values, 0.25)),
        "return_cvar10": float(np.mean(np.sort(values)[:tail_count])),
        "nonfailure_mean_return": (
            float(np.mean(nonfailures)) if nonfailures.size else None
        ),
    }


def _dreamer_style_training_score(
    online_history: list[dict[str, Any]],
    *,
    window_env_steps: int,
    budget_env_steps: int,
) -> dict[str, Any]:
    enabled = window_env_steps > 0 and budget_env_steps > 0
    episodes: list[dict[str, Any]] = []
    for item in online_history:
        replay = item.get("actor_replay", {})
        returns = replay.get("returns") or []
        lengths = replay.get("lengths") or []
        finish_steps = replay.get("episode_finish_train_env_steps") or []
        if len(finish_steps) != len(returns):
            continue
        for index, (value, finish_step) in enumerate(zip(returns, finish_steps)):
            episodes.append(
                {
                    "online_iteration": item.get("iteration"),
                    "return": float(value),
                    "length": int(lengths[index]) if index < len(lengths) else None,
                    "finish_train_env_step": int(finish_step),
                }
            )

    final_step = max(
        (item["finish_train_env_step"] for item in episodes),
        default=None,
    )
    if not enabled or final_step is None:
        return {
            "enabled": enabled,
            "budget_env_steps": int(budget_env_steps),
            "window_env_steps": int(window_env_steps),
            "budget_reached": False,
            "final_train_env_step": final_step,
            "window_start_env_step": None,
            "window_end_env_step": None,
            "episodes": 0,
            "mean_return": None,
            "std_return": None,
            "returns": [],
            "episode_finish_train_env_steps": [],
        }
    budget_reached = final_step >= budget_env_steps
    window_end = budget_env_steps if budget_reached else final_step
    window_start = max(0, window_end - window_env_steps)
    selected = [
        item
        for item in episodes
        if window_start < item["finish_train_env_step"] <= window_end
    ]
    returns = [item["return"] for item in selected]
    return {
        "enabled": True,
        "budget_env_steps": int(budget_env_steps),
        "window_env_steps": int(window_env_steps),
        "budget_reached": bool(budget_reached),
        "final_train_env_step": int(final_step),
        "window_start_env_step": int(window_start),
        "window_end_env_step": int(window_end),
        "episodes": len(returns),
        "mean_return": float(np.mean(returns)) if returns else None,
        "std_return": float(np.std(returns)) if returns else None,
        "returns": returns,
        "episode_finish_train_env_steps": [
            item["finish_train_env_step"] for item in selected
        ],
        "episode_records": selected,
    }


def _real_step_accounting(
    *,
    initial_train_env_steps: int,
    validation_env_steps: int,
    online_history: list[dict[str, Any]],
    final_policy_eval: dict[str, Any] | None,
) -> dict[str, int]:
    online_env_steps = sum(
        int(item["actor_replay"]["env_steps"]) for item in online_history
    )
    curve_eval_env_steps = sum(
        int(_nested(item.get("policy_evaluation"), "env_steps") or 0)
        for item in online_history
    )
    curve_completed_eval_steps = sum(
        int(_nested(item.get("policy_evaluation"), "completed_episode_steps") or 0)
        for item in online_history
    )
    policy_eval_env_steps = curve_eval_env_steps + int(
        _nested(final_policy_eval, "env_steps") or 0
    )
    completed_eval_steps = (
        int(_nested(final_policy_eval, "completed_episode_steps") or 0)
        + curve_completed_eval_steps
    )
    train_env_steps = initial_train_env_steps + online_env_steps
    return {
        "real_initial_train_replay_env_steps": int(initial_train_env_steps),
        "real_online_actor_replay_env_steps": int(online_env_steps),
        "real_train_replay_env_steps": int(train_env_steps),
        "real_validation_replay_env_steps": int(validation_env_steps),
        "real_train_plus_validation_env_steps": int(
            train_env_steps + validation_env_steps
        ),
        "real_policy_eval_env_steps": policy_eval_env_steps,
        "real_policy_eval_completed_episode_steps": completed_eval_steps,
        "real_total_env_steps": int(
            train_env_steps + validation_env_steps + policy_eval_env_steps
        ),
    }


def _nested(payload: dict[str, Any] | None, key: str) -> Any:
    return None if payload is None else payload.get(key)


def _evaluate_model(
    state,
    key: jax.Array,
    batch: ReplayBatch,
    config: JepaConfig,
    *,
    chunk_length: int,
    open_loop_horizon: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> dict[str, Any]:
    metrics = dict(
        evaluate_world_model_loss(
            state,
            key,
            batch,
            config,
            chunk_length=chunk_length,
            control=CONTROL,
        )
    )
    metrics.update(
        evaluate_open_loop(
            state,
            batch,
            config,
            horizon=open_loop_horizon,
            control=CONTROL,
        )
    )
    metrics["model/continuous_action_low_high_sensitivity"] = (
        _continuous_action_sensitivity(
            state,
            batch,
            config,
            action_low=action_low,
            action_high=action_high,
        )
    )
    return to_jsonable(metrics)


@partial(jax.jit, static_argnames=("config",))
def _continuous_action_sensitivity(
    state,
    batch: ReplayBatch,
    config: JepaConfig,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> jax.Array:
    flat_obs = batch.observations[:, 0].reshape((-1, config.observation_dim))
    z = state.apply_fn({"params": state.params}, flat_obs, method=JepaWorldModel.encode)
    context = z[:, None, :]
    low = jnp.broadcast_to(
        jnp.asarray(action_low, dtype=jnp.float32),
        (z.shape[0], config.action_dim),
    )[:, None, :]
    high = jnp.broadcast_to(
        jnp.asarray(action_high, dtype=jnp.float32),
        (z.shape[0], config.action_dim),
    )[:, None, :]
    z_low, _, _ = state.apply_fn(
        {"params": state.params},
        context,
        low,
        method=JepaWorldModel.predict_next_from_history,
    )
    z_high, _, _ = state.apply_fn(
        {"params": state.params},
        context,
        high,
        method=JepaWorldModel.predict_next_from_history,
    )
    return jnp.mean(jnp.linalg.norm(z_high - z_low, axis=-1))


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
        dones=batch.dones,
        method=JepaWorldModel.sequence_outputs,
    )["predicted_latents"]
    reloaded = fresh.apply_fn(
        {"params": fresh.params},
        batch.observations,
        batch.actions,
        chunk_length=chunk_length,
        dones=batch.dones,
        method=JepaWorldModel.sequence_outputs,
    )["predicted_latents"]
    return float(jnp.max(jnp.abs(original - reloaded)))


def summarize(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    passed = bool(outcomes and all(item["passed"] for item in outcomes))
    return {
        "algorithm": "single_agent_jepa_mbrl",
        "protocol": "reset_rich_interleaved_latest_policy",
        "passed": passed,
        "world_model_passed": passed,
        "runs_passed": sum(bool(item["passed"]) for item in outcomes),
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


def _run_passed(
    initial_metrics: dict[str, Any],
    final_metrics: dict[str, Any],
    reload_diff: float,
) -> bool:
    return bool(
        _metrics_finite(final_metrics)
        and reload_diff <= 1e-6
        and final_metrics["model/open_loop_finite_fraction"] >= 1.0
        and final_metrics["model/jepa_loss"] <= initial_metrics["model/jepa_loss"]
        and final_metrics["model/open_loop_loss"]
        <= initial_metrics["model/open_loop_loss"]
        and final_metrics["model/reward_loss"]
        < final_metrics.get(
            "model/reward_constant_loss",
            final_metrics["model/reward_constant_mse"],
        )
        and _continue_criterion_passed(final_metrics)
    )


def _continue_criterion_passed(final_metrics: dict[str, Any]) -> bool:
    terminal_fraction = final_metrics.get("model/terminal_positive_fraction", 0.0)
    if terminal_fraction >= MIN_TERMINAL_FRACTION_FOR_CONTINUE_BASELINE:
        return (
            final_metrics["model/continue_loss"]
            < final_metrics["model/continue_constant_bce"]
        )
    return (
        math.isfinite(final_metrics["model/continue_loss"])
        and final_metrics.get("model/nonterminal_recall", 0.0) >= 0.95
    )


def _metrics_finite(metrics: dict[str, Any]) -> bool:
    return all(
        not isinstance(value, (int, float)) or math.isfinite(value)
        for value in metrics.values()
    )


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


if __name__ == "__main__":
    main()
