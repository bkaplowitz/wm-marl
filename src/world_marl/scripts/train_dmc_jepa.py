"""Validate a representation-space SIGReg-JEPA world model on single-agent rollouts.

By default this is the first single-agent rung: random replay collection plus
held-out latent prediction, reward prediction, and continue prediction. Passing
``--policy-train-steps`` enables the next rung: reset actor/value heads, freeze
the JEPA world model, train an actor inside the latent model, then evaluate that
actor in the real environment. Continuous backends (``dmc:``/``brax:``) train a
deterministic tanh actor with pathwise gradients; the discrete backend
(``gymnax:``) trains a categorical actor with REINFORCE on imagined returns.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from world_marl.checkpointing import load_params, save_checkpoint
from world_marl.envs.brax_adapter import BraxVectorAdapter, brax_env_name
from world_marl.envs.dmc_adapter import DMCVectorAdapter, dmc_env_name
from world_marl.envs.gymnax_adapter import GymnaxVectorAdapter, gymnax_env_name
from world_marl.jepa.decoder import (
    DecoderConfig,
    create_decoder_train_state,
    decode_open_loop_rollout,
    decoder_reconstruction_mse,
    encode_observations,
    select_display_trajectories,
    train_decoder_step,
)
from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.replay import ReplayBatch, SequenceReplayBuffer
from world_marl.jepa.training import (
    ControlMode,
    continuous_candidate_distill_step,
    continuous_policy_train_step,
    copy_policy_heads,
    create_jepa_train_state,
    critic_warmup_step,
    discrete_policy_train_step,
    evaluate_open_loop,
    evaluate_world_model_loss,
    reset_policy_heads,
    select_continuous_actions,
    select_discrete_actions,
    train_model_step,
)
from world_marl.jepa.validation import (
    action_contrast_metrics,
    best_passing_candidate_report,
    candidate_checkpoint_gate_summary,
    candidate_refit_gate_report,
    eval_completed_episode_steps,
    eval_env_steps,
    maybe_int,
    merge_online_policy_baseline,
    metrics_finite,
    online_history_metrics,
    real_step_accounting,
    run_passed,
    sample_online_candidate_batch,
    summarize,
)
from world_marl.logging import RunLogger, dependency_versions, timestamp, to_jsonable
from world_marl.scripts.plot_jepa_decoder import (
    FRAMES_FILENAME,
    TRACES_FILENAME,
    save_rollout_frames_plot,
    save_rollout_traces_plot,
)

FROZEN_RANDOM_WORLD_MODEL_CONTROL = "frozen-random-world-model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="dmc:cartpole/swingup")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="Number of vectorized environments.",
    )
    parser.add_argument(
        "--env-workers",
        "--dmc-workers",
        dest="env_workers",
        type=int,
        default=1,
        help="Worker threads for CPU/Python-backed adapters such as DMC.",
    )
    parser.add_argument(
        "--brax-backend",
        default=None,
        help="Optional Brax physics backend to pass through to brax.envs.create.",
    )
    parser.add_argument("--collect-steps", type=int, default=2048)
    parser.add_argument("--validation-steps", type=int, default=512)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--chunk-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--model-horizon", type=int, default=1)
    parser.add_argument("--open-loop-horizon", type=int, default=5)
    parser.add_argument(
        "--decoder-train-steps",
        type=int,
        default=0,
        help=(
            "Fit steps for a post-hoc observation decoder used only as a "
            "visual diagnostic (LeJEPA-style). 0 disables it; the world model "
            "itself never trains on reconstruction."
        ),
    )
    parser.add_argument("--decoder-hidden-dim", type=int, default=256)
    parser.add_argument("--decoder-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--decoder-rollout-horizon",
        type=int,
        default=0,
        help=(
            "Imagined rollout length for the decoder diagnostic frames "
            "(0 uses --open-loop-horizon)."
        ),
    )
    parser.add_argument(
        "--decoder-rollout-trajectories",
        type=int,
        default=4,
        help="Held-out trajectories shown in the decoder diagnostic figures.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=1,
        help=(
            "Latent/action history length for world-model training. Values >1 "
            "are currently supported for model-only validation, not policy "
            "imagination."
        ),
    )
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument(
        "--dynamics-ensemble-size",
        type=int,
        default=1,
        help=(
            "Number of independently initialized predictor/reward/continue "
            "heads sharing the encoder and transformer trunk. Values >1 enable "
            "ensemble disagreement diagnostics and conservative imagination."
        ),
    )
    parser.add_argument(
        "--target-gradient",
        choices=("stopgrad", "symmetric"),
        default="stopgrad",
        help=(
            "Whether JEPA target latents are stopped or receive symmetric "
            "gradients through the prediction loss."
        ),
    )
    parser.add_argument(
        "--no-residual-dynamics",
        dest="residual_dynamics",
        action="store_false",
        default=True,
        help="Disable residual latent transition z_next = norm(z + delta).",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--model-grad-clip-norm", type=float, default=100.0)
    parser.add_argument("--actor-grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--critic-grad-clip-norm", type=float, default=100.0)
    parser.add_argument(
        "--stochastic-actor",
        action="store_true",
        help=(
            "Use a tanh-normal actor during imagination training. Evaluation "
            "still uses the deterministic mean action."
        ),
    )
    parser.add_argument(
        "--stochastic-collection",
        action="store_true",
        help="Sample from the tanh-normal actor during online actor replay collection.",
    )
    parser.add_argument("--actor-entropy-coef", type=float, default=0.0)
    parser.add_argument("--actor-log-std-min", type=float, default=-5.0)
    parser.add_argument("--actor-log-std-max", type=float, default=2.0)
    parser.add_argument("--policy-train-steps", type=int, default=0)
    parser.add_argument("--policy-batch-size", type=int, default=None)
    parser.add_argument("--critic-warmup-steps", type=int, default=1000)
    parser.add_argument("--critic-horizon", type=int, default=32)
    parser.add_argument("--imag-horizon", type=int, default=5)
    parser.add_argument(
        "--policy-objective",
        choices=("candidate-distill", "direct"),
        default="direct",
        help=(
            "direct is the main algorithm and backpropagates reward-only or "
            "lambda returns through latent imagination. candidate-distill is a "
            "diagnostic planning-teacher baseline that scores sampled actions "
            "with the frozen latent model and trains the actor toward the best "
            "candidates."
        ),
    )
    parser.add_argument(
        "--policy-return-mode",
        choices=("reward-only", "lambda"),
        default="reward-only",
        help=(
            "Use finite-horizon predicted rewards by default. The lambda mode "
            "bootstraps from the learned value head and is experimental until "
            "the target critic is stronger."
        ),
    )
    parser.add_argument(
        "--policy-actor-baseline",
        choices=("none", "value"),
        default="none",
        help=(
            "Use no actor baseline or subtract the frozen value estimate from "
            "imagined returns before the actor objective."
        ),
    )
    parser.add_argument(
        "--policy-return-normalization",
        choices=("none", "batch"),
        default="none",
        help=(
            "Normalize imagined returns/advantages inside each actor update. "
            "Batch mode uses stop-gradient weighted batch statistics."
        ),
    )
    parser.add_argument("--value-clip", type=float, default=100.0)
    parser.add_argument("--action-saturation-threshold", type=float, default=0.95)
    parser.add_argument(
        "--uncertainty-penalty",
        type=float,
        default=0.0,
        help=(
            "Penalty subtracted from imagined rewards per unit of ensemble "
            "uncertainty. Requires --dynamics-ensemble-size > 1 to have effect."
        ),
    )
    parser.add_argument("--uncertainty-latent-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-reward-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-continue-weight", type=float, default=1.0)
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=float("inf"),
        help=(
            "Stop trusting an imagined trajectory after a transition exceeds "
            "this ensemble-uncertainty threshold."
        ),
    )
    parser.add_argument(
        "--uncertainty-budget",
        type=float,
        default=float("inf"),
        help=(
            "Stop trusting an imagined trajectory after cumulative ensemble "
            "uncertainty exceeds this budget."
        ),
    )
    parser.add_argument("--num-policy-candidates", type=int, default=64)
    parser.add_argument("--candidate-min-gap", type=float, default=1e-3)
    parser.add_argument("--policy-action-l2-coef", type=float, default=1e-3)
    parser.add_argument(
        "--policy-trust-coef",
        type=float,
        default=0.0,
        help=(
            "Direct-policy penalty for changing normalized actions from the "
            "actor at the start of the policy phase. This is mainly a "
            "conservative online-policy knob; leave at 0 for unconstrained "
            "initial actor learning."
        ),
    )
    parser.add_argument(
        "--online-policy-trust-coef",
        type=float,
        default=None,
        help=(
            "Override --policy-trust-coef for online policy phases. Use this "
            "to keep online actor updates close to the accepted actor while "
            "allowing the first offline policy phase to move freely."
        ),
    )
    parser.add_argument("--policy-eval-episodes", type=int, default=20)
    parser.add_argument("--policy-eval-num-envs", type=int, default=None)
    parser.add_argument("--policy-confirmation-episodes", type=int, default=0)
    parser.add_argument("--policy-confirmation-num-envs", type=int, default=None)
    parser.add_argument(
        "--final-policy-eval-episodes",
        type=int,
        default=0,
        help=(
            "Run an extra final evaluation of the accepted champion policy with "
            "this many episodes. This does not affect training or policy "
            "selection; it is intended for robust reporting."
        ),
    )
    parser.add_argument("--final-policy-eval-num-envs", type=int, default=None)
    parser.add_argument(
        "--policy-selection-interval",
        type=int,
        default=500,
        help=(
            "During frozen-model actor training, evaluate the actor every N "
            "updates on a fixed real-env selection set and keep the best actor. "
            "Use 0 to disable best-policy selection."
        ),
    )
    parser.add_argument("--policy-selection-episodes", type=int, default=20)
    parser.add_argument("--policy-selection-num-envs", type=int, default=None)
    parser.add_argument(
        "--online-iterations",
        type=int,
        default=0,
        help=(
            "After the initial offline world-model/policy fit, repeat: collect "
            "real replay with the selected actor, update the world model, then "
            "continue frozen-model policy training."
        ),
    )
    parser.add_argument("--online-collect-steps", type=int, default=None)
    parser.add_argument(
        "--online-validation-steps",
        type=int,
        default=None,
        help=(
            "Held-out current-policy stream length for online candidate-refit "
            "validation. Defaults to min(validation-steps, online-collect-steps)."
        ),
    )
    parser.add_argument("--online-train-steps", type=int, default=None)
    parser.add_argument("--online-policy-train-steps", type=int, default=None)
    parser.add_argument(
        "--online-policy-champion",
        dest="online_policy_champion",
        action="store_true",
        default=True,
        help=(
            "Keep the best real-env evaluated actor across online policy "
            "phases. Rejected policy proposals restore champion actor/value "
            "heads while preserving accepted world-model updates."
        ),
    )
    parser.add_argument(
        "--no-online-policy-champion",
        dest="online_policy_champion",
        action="store_false",
        help="Disable best-actor retention across online policy phases.",
    )
    parser.add_argument(
        "--online-policy-champion-tolerance",
        type=float,
        default=0.0,
        help=(
            "Allowed real-return regression when accepting an online policy "
            "proposal as the new champion."
        ),
    )
    parser.add_argument(
        "--online-reset-replay-env",
        dest="online_reset_replay_env",
        action="store_true",
        default=True,
        help=(
            "Reset vector environments before collecting each online actor-replay "
            "block so replay-return metrics are attributable to the current actor."
        ),
    )
    parser.add_argument(
        "--no-online-reset-replay-env",
        dest="online_reset_replay_env",
        action="store_false",
        help="Continue online actor replay from the current environment states.",
    )
    parser.add_argument(
        "--online-reset-actor",
        action="store_true",
        help="Reset actor/value heads at the start of each online policy phase.",
    )
    parser.add_argument(
        "--online-control-value-weight",
        type=float,
        default=0.0,
        help=(
            "Optional online refit loss weight for value-equivalent dynamics. "
            "The candidate's one-step predicted Q estimate is matched to a "
            "critic target computed from the real next latent."
        ),
    )
    parser.add_argument(
        "--online-anchor-batch-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of each online candidate-refit minibatch sampled from the "
            "initial random anchor replay. The remaining samples come from the "
            "latest actor replay. Full replay is still retained."
        ),
    )
    parser.add_argument(
        "--online-candidate-refit",
        action="store_true",
        help=(
            "Train online world-model updates as candidate states, then accept "
            "only if held-out recent-policy validation improves while anchor "
            "validation stays within tolerance."
        ),
    )
    parser.add_argument(
        "--online-candidate-gate-metric",
        choices=(
            "model/open_loop_loss",
            "model/jepa_loss",
        ),
        default="model/open_loop_loss",
    )
    parser.add_argument(
        "--online-candidate-min-recent-improvement",
        type=float,
        default=0.0,
        help="Required decrease in the gate metric on recent-policy validation.",
    )
    parser.add_argument(
        "--online-candidate-max-anchor-degradation",
        type=float,
        default=0.05,
        help="Maximum allowed increase in the gate metric on anchor validation.",
    )
    parser.add_argument(
        "--online-candidate-eval-interval",
        type=int,
        default=500,
        help=(
            "During online candidate refits, evaluate every N model updates and "
            "keep the best passing checkpoint. Use 0 to evaluate only the final "
            "candidate."
        ),
    )
    parser.add_argument(
        "--online-candidate-anchor-penalty",
        type=float,
        default=1.0,
        help=(
            "Penalty used to rank candidate checkpoints: score = recent "
            "improvement - penalty * max(anchor degradation, 0). Hard accept "
            "constraints are still enforced separately."
        ),
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda-return", type=float, default=0.95)
    parser.add_argument(
        "--regularizer-weight",
        "--sigreg-weight",
        dest="regularizer_weight",
        type=float,
        default=0.05,
        help="Weight for the selected representation regularizer.",
    )
    parser.add_argument(
        "--regularizer",
        choices=("sigreg", "none"),
        default="sigreg",
    )
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=256)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--continue-weight", type=float, default=1.0)
    parser.add_argument(
        "--reward-prediction-mode",
        choices=("mse", "symlog-twohot"),
        default="mse",
    )
    parser.add_argument(
        "--value-prediction-mode",
        choices=("mse", "symlog-twohot"),
        default="mse",
    )
    parser.add_argument("--twohot-bins", type=int, default=41)
    parser.add_argument("--twohot-min", type=float, default=-20.0)
    parser.add_argument("--twohot-max", type=float, default=20.0)
    parser.add_argument(
        "--clip-imagined-rewards",
        action="store_true",
        help=(
            "Clip model-predicted rewards inside imagined actor/planner "
            "objectives. This is useful for bounded-reward control suites."
        ),
    )
    parser.add_argument("--imagined-reward-min", type=float, default=0.0)
    parser.add_argument("--imagined-reward-max", type=float, default=1.0)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--out-dir", default="runs/dmc_jepa")
    parser.add_argument(
        "--controls",
        nargs="+",
        choices=(
            "none",
            "no-action-world-model",
            "shuffled-action-replay",
            FROZEN_RANDOM_WORLD_MODEL_CONTROL,
        ),
        default=(
            "none",
            "no-action-world-model",
            "shuffled-action-replay",
            FROZEN_RANDOM_WORLD_MODEL_CONTROL,
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--allow-fail", action="store_true")
    # Weights & Biases (disabled unless --wandb-project is set).
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    args = parser.parse_args()
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    min_sequence_steps = args.chunk_length + max(
        args.model_horizon, args.open_loop_horizon
    )
    for name in (
        "num_envs",
        "env_workers",
        "collect_steps",
        "validation_steps",
        "replay_capacity",
        "chunk_length",
        "batch_size",
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
        "num_runs",
        "max_cycles",
        "imag_horizon",
        "policy_eval_episodes",
        "policy_selection_episodes",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    for name in (
        "policy_train_steps",
        "critic_warmup_steps",
        "policy_selection_interval",
        "online_iterations",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.policy_confirmation_episodes < 0:
        parser.error("--policy-confirmation-episodes must be >= 0")
    if args.decoder_train_steps < 0:
        parser.error("--decoder-train-steps must be >= 0")
    if args.decoder_rollout_horizon < 0:
        parser.error("--decoder-rollout-horizon must be >= 0")
    if args.decoder_train_steps > 0:
        for name in ("decoder_hidden_dim", "decoder_rollout_trajectories"):
            if getattr(args, name) < 1:
                parser.error(f"--{name.replace('_', '-')} must be >= 1")
        if args.decoder_learning_rate <= 0.0:
            parser.error("--decoder-learning-rate must be > 0")
        decoder_horizon = args.decoder_rollout_horizon or args.open_loop_horizon
        if args.validation_steps < args.context_window + decoder_horizon:
            parser.error(
                "--validation-steps must cover context-window + decoder rollout "
                "horizon for the decoder diagnostic"
            )
    if args.final_policy_eval_episodes < 0:
        parser.error("--final-policy-eval-episodes must be >= 0")
    if args.online_collect_steps is not None and args.online_collect_steps < 1:
        parser.error("--online-collect-steps must be >= 1")
    if args.online_validation_steps is not None:
        if args.online_validation_steps < 1:
            parser.error("--online-validation-steps must be >= 1")
        if args.online_validation_steps < min_sequence_steps:
            parser.error(
                "--online-validation-steps must cover chunk-length + max model/open-loop horizon"
            )
    for name in ("online_train_steps", "online_policy_train_steps"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.online_policy_champion_tolerance < 0.0:
        parser.error("--online-policy-champion-tolerance must be >= 0")
    if args.online_control_value_weight < 0.0:
        parser.error("--online-control-value-weight must be >= 0")
    if not 0.0 <= args.online_anchor_batch_fraction <= 1.0:
        parser.error("--online-anchor-batch-fraction must be in [0, 1]")
    if args.online_candidate_min_recent_improvement < 0.0:
        parser.error("--online-candidate-min-recent-improvement must be >= 0")
    if args.online_candidate_max_anchor_degradation < 0.0:
        parser.error("--online-candidate-max-anchor-degradation must be >= 0")
    if args.online_candidate_eval_interval < 0:
        parser.error("--online-candidate-eval-interval must be >= 0")
    if args.online_candidate_anchor_penalty < 0.0:
        parser.error("--online-candidate-anchor-penalty must be >= 0")
    for name in ("critic_horizon",):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.num_policy_candidates < 2:
        parser.error("--num-policy-candidates must be >= 2")
    if args.candidate_min_gap < 0.0:
        parser.error("--candidate-min-gap must be >= 0")
    if args.policy_action_l2_coef < 0.0:
        parser.error("--policy-action-l2-coef must be >= 0")
    if args.policy_trust_coef < 0.0:
        parser.error("--policy-trust-coef must be >= 0")
    if (
        args.online_policy_trust_coef is not None
        and args.online_policy_trust_coef < 0.0
    ):
        parser.error("--online-policy-trust-coef must be >= 0")
    if args.value_clip <= 0.0:
        parser.error("--value-clip must be > 0")
    if args.actor_entropy_coef < 0.0:
        parser.error("--actor-entropy-coef must be >= 0")
    if args.actor_log_std_min >= args.actor_log_std_max:
        parser.error("--actor-log-std-min must be < --actor-log-std-max")
    if (
        args.stochastic_collection
        and not args.stochastic_actor
        and _action_mode(args.env) == "continuous"
    ):
        parser.error(
            "--stochastic-collection requires --stochastic-actor "
            "(discrete actors always define a sampling distribution)"
        )
    if args.twohot_bins < 3:
        parser.error("--twohot-bins must be >= 3")
    if args.twohot_min >= args.twohot_max:
        parser.error("--twohot-min must be < --twohot-max")
    if args.imagined_reward_min >= args.imagined_reward_max:
        parser.error("--imagined-reward-min must be < --imagined-reward-max")
    for name in (
        "model_grad_clip_norm",
        "actor_grad_clip_norm",
        "critic_grad_clip_norm",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if not 0.0 < args.action_saturation_threshold <= 1.0:
        parser.error("--action-saturation-threshold must be in (0, 1]")
    for name in (
        "uncertainty_penalty",
        "uncertainty_latent_weight",
        "uncertainty_reward_weight",
        "uncertainty_continue_weight",
        "uncertainty_threshold",
        "uncertainty_budget",
    ):
        value = getattr(args, name)
        if value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.policy_batch_size is not None and args.policy_batch_size < 1:
        parser.error("--policy-batch-size must be >= 1")
    if args.policy_eval_num_envs is not None and args.policy_eval_num_envs < 1:
        parser.error("--policy-eval-num-envs must be >= 1")
    if (
        args.policy_confirmation_num_envs is not None
        and args.policy_confirmation_num_envs < 1
    ):
        parser.error("--policy-confirmation-num-envs must be >= 1")
    if (
        args.policy_selection_num_envs is not None
        and args.policy_selection_num_envs < 1
    ):
        parser.error("--policy-selection-num-envs must be >= 1")
    if (
        args.final_policy_eval_num_envs is not None
        and args.final_policy_eval_num_envs < 1
    ):
        parser.error("--final-policy-eval-num-envs must be >= 1")
    if not (
        args.env.startswith("dmc:")
        or args.env.startswith("brax:")
        or args.env.startswith("gymnax:")
    ):
        parser.error(
            "--env must be formatted as dmc:<domain>/<task>, brax:<env>, "
            "or gymnax:<env_id>"
        )
    min_steps = min_sequence_steps
    if args.collect_steps < min_steps:
        parser.error(
            "--collect-steps must cover chunk-length + max model/open-loop horizon"
        )
    if args.validation_steps < min_steps:
        parser.error(
            "--validation-steps must cover chunk-length + max model/open-loop horizon"
        )
    if args.chunk_length < args.context_window:
        parser.error("--chunk-length must be >= --context-window")
    if (
        args.policy_train_steps > 0
        and args.critic_warmup_steps > 0
        and args.collect_steps < args.critic_horizon + 1
    ):
        parser.error("--collect-steps must cover critic-horizon + 1")
    if args.online_iterations > 0 and args.policy_train_steps == 0:
        parser.error("--online-iterations requires --policy-train-steps > 0")


def main() -> None:
    args = parse_args()
    experiment_dir = (
        Path(args.out_dir) / f"{_experiment_prefix(args.env)}_{timestamp()}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for control in args.controls:
        for run_index in range(args.num_runs):
            outcomes.append(
                run_one(
                    args,
                    run_dir=experiment_dir / control / f"run_{run_index:03d}",
                    run_index=run_index,
                    control=control,
                )
            )
    summary = summarize(outcomes)
    RunLogger(experiment_dir).write_json("summary.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    if not args.allow_fail and not summary["passed"]:
        raise SystemExit(1)


def _skip_world_model_fit(control: ControlMode) -> bool:
    return control == FROZEN_RANDOM_WORLD_MODEL_CONTROL


def _env_backend(env: str) -> str:
    if env.startswith("dmc:"):
        return "dmc"
    if env.startswith("brax:"):
        return "brax"
    if env.startswith("gymnax:"):
        return "gymnax"
    raise ValueError(f"unsupported env: {env!r}")


def _action_mode(env: str) -> str:
    return "discrete" if _env_backend(env) == "gymnax" else "continuous"


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
    if args.env.startswith("brax:"):
        return BraxVectorAdapter(
            brax_env_name(args.env),
            num_envs=adapter_num_envs,
            max_cycles=args.max_cycles,
            seed=seed,
            backend=args.brax_backend,
        )
    if args.env.startswith("gymnax:"):
        return GymnaxVectorAdapter(
            gymnax_env_name(args.env),
            num_envs=adapter_num_envs,
            max_cycles=args.max_cycles,
            seed=seed,
        )
    raise ValueError(f"unsupported env: {args.env!r}")


def _init_wandb(args: argparse.Namespace, *, run_index: int, control: ControlMode):
    """Create a W&B run when --wandb-project is set (returns None otherwise)."""
    if not args.wandb_project:
        return None
    import wandb

    env_slug = args.env.replace(":", "_").replace("/", "_")
    config = to_jsonable(vars(args))
    config.update({"run_index": run_index, "control": control})
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group or f"{env_slug}-jepa",
        name=f"{env_slug}-jepa-{control}-run{run_index:03d}",
        config=config,
        reinit=True,
    )


def run_one(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    run_index: int,
    control: ControlMode,
) -> dict[str, Any]:
    wandb_run = _init_wandb(args, run_index=run_index, control=control)
    logger = RunLogger(run_dir, wandb_run=wandb_run)
    seed = args.seed + 10_000 * run_index
    adapter = _make_vector_adapter(args, seed=seed)
    action_mode = _action_mode(args.env)
    try:
        config = JepaConfig(
            observation_dim=int(np.prod(adapter.observation_shape)),
            action_dim=adapter.action_dim,
            action_mode=action_mode,
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
            stochastic_actor=args.stochastic_actor,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
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
            clip_imagined_rewards=args.clip_imagined_rewards,
            imagined_reward_min=args.imagined_reward_min,
            imagined_reward_max=args.imagined_reward_max,
            dynamics_ensemble_size=args.dynamics_ensemble_size,
            gamma=args.gamma,
            lambda_return=args.lambda_return,
            residual_dynamics=args.residual_dynamics,
            target_gradient=args.target_gradient,
        )
        logger.write_json(
            "config.json",
            {
                "args": vars(args),
                "run_index": run_index,
                "seed": seed,
                "control": control,
                "observation_shape": adapter.observation_shape,
                "action_mode": action_mode,
                "action_shape": adapter.action_shape,
                "action_low": adapter.action_low,
                "action_high": adapter.action_high,
                "env_backend": _env_backend(args.env),
                "jepa_config": dataclasses.asdict(config),
                "protocol": (
                    "heldout_world_model_validation_with_optional_frozen_policy"
                    "_and_online_actor_replay"
                ),
            },
        )
        logger.write_json("versions.json", dependency_versions())

        rng = jax.random.PRNGKey(seed)
        rng, init_key = jax.random.split(rng)
        state = create_jepa_train_state(init_key, config)
        np_rng = np.random.default_rng(seed)
        replay_action_shape, replay_action_dtype = _replay_action_spec(
            adapter,
            action_mode,
        )
        replay = SequenceReplayBuffer(
            capacity=max(2, math.ceil(args.replay_capacity / args.num_envs)),
            num_envs=args.num_envs,
            observation_shape=(config.observation_dim,),
            action_shape=replay_action_shape,
            action_dtype=replay_action_dtype,
        )
        anchor_replay = SequenceReplayBuffer(
            capacity=max(2, args.collect_steps),
            num_envs=args.num_envs,
            observation_shape=(config.observation_dim,),
            action_shape=replay_action_shape,
            action_dtype=replay_action_dtype,
        )

        observations = adapter.reset()
        observations, env_steps = _collect_random_steps(
            adapter,
            observations,
            np_rng,
            (replay, anchor_replay),
            steps=args.collect_steps,
            desc=f"{control} collect train replay",
            quiet=args.quiet,
        )
        logger.write_json(
            "train_replay.json",
            {
                "env_steps": env_steps,
                "steps_per_env": args.collect_steps,
                "size_per_env": replay.size,
                "anchor_size_per_env": anchor_replay.size,
                "observation_dim": config.observation_dim,
                "action_dim": config.action_dim,
            },
        )
        initial_train_replay_env_steps = env_steps

        validation_replay = _collect_validation_replay(
            args,
            config,
            seed=seed + 1_000_000,
        )
        initial_validation_env_steps = args.validation_steps * args.num_envs
        logger.write_json(
            "validation_replay.json",
            {
                "env_steps": initial_validation_env_steps,
                "steps_per_env": args.validation_steps,
                "size_per_env": validation_replay.size,
                "seed": seed + 1_000_000,
            },
        )

        validation_batch = validation_replay.sample(
            np_rng,
            batch_size=args.batch_size,
            chunk_length=args.chunk_length,
            max_horizon=max(args.model_horizon, args.open_loop_horizon),
        )
        rng, eval_key = jax.random.split(rng)
        initial_metrics = _evaluate_model(
            state,
            eval_key,
            validation_batch,
            config,
            chunk_length=args.chunk_length,
            open_loop_horizon=args.open_loop_horizon,
            control=control,
            action_low=adapter.action_low,
            action_high=adapter.action_high,
        )
        logger.write_json("initial_model_metrics.json", initial_metrics)

        if _skip_world_model_fit(control):
            loss_history: list[float] = []
            logger.append_metrics(
                {
                    "phase": "world_model_skipped",
                    "update": 0,
                    "env_steps": env_steps,
                    "control": control,
                    "world_model_fit_skipped": True,
                }
            )
        else:
            state, rng, _, loss_history = _fit_world_model(
                args,
                logger,
                state,
                rng,
                replay,
                config,
                np_rng=np_rng,
                steps=args.train_steps,
                control=control,
                phase="world_model",
                desc=f"{control} fit world model",
                env_steps=env_steps,
            )
            logger.plot_world_model_loss(
                loss_history, filename="dmc_world_model_loss.png"
            )

        rng, eval_key = jax.random.split(rng)
        final_batch = validation_batch
        final_metrics = _evaluate_model(
            state,
            eval_key,
            final_batch,
            config,
            chunk_length=args.chunk_length,
            open_loop_horizon=args.open_loop_horizon,
            control=control,
            action_low=adapter.action_low,
            action_high=adapter.action_high,
        )
        logger.write_json("model_metrics_initial_fit.json", final_metrics)

        policy_outcome = _maybe_train_policy(
            args,
            logger,
            state,
            config,
            replay,
            control=control,
            seed=seed,
            np_rng=np_rng,
            rng=rng,
            action_low=adapter.action_low,
            action_high=adapter.action_high,
        )
        state = policy_outcome["state"]
        rng = policy_outcome["rng"]
        initial_policy_outcome = policy_outcome["outcome"]
        champion_state = state
        champion_policy_outcome = dict(initial_policy_outcome)
        champion_policy_return = champion_policy_outcome.get("policy_trained_mean")
        champion_policy_iteration = 0
        online_history: list[dict[str, Any]] = []
        for online_index in range(1, args.online_iterations + 1):
            phase = f"online_{online_index:03d}"
            online_collect_steps = args.online_collect_steps or args.collect_steps
            if args.online_reset_replay_env:
                observations = adapter.reset()
            recent_actor_replay = _new_replay_buffer(
                capacity=online_collect_steps,
                num_envs=args.num_envs,
                observation_dim=config.observation_dim,
                action_shape=replay_action_shape,
                action_dtype=replay_action_dtype,
            )
            observations, added_env_steps, collect_metrics = _collect_policy_steps(
                adapter,
                observations,
                state,
                config,
                (replay, recent_actor_replay),
                steps=online_collect_steps,
                action_low=adapter.action_low,
                action_high=adapter.action_high,
                desc=f"{control} {phase} collect actor replay",
                quiet=args.quiet,
                np_rng=np_rng,
                stochastic_actions=args.stochastic_collection,
            )
            env_steps += added_env_steps
            collect_payload = {
                **collect_metrics,
                "reset_env_before_collection": args.online_reset_replay_env,
                "total_env_steps": env_steps,
                "replay_size_per_env": replay.size,
                "anchor_replay_size_per_env": anchor_replay.size,
                "recent_actor_replay_size_per_env": recent_actor_replay.size,
            }
            logger.write_json(f"{phase}_actor_replay.json", collect_payload)
            logger.append_metrics(
                {
                    "phase": "online_actor_replay",
                    "online_iteration": online_index,
                    "control": control,
                    **collect_payload,
                }
            )

            online_train_steps = (
                args.online_train_steps
                if args.online_train_steps is not None
                else max(1, args.train_steps // 2)
            )
            if _skip_world_model_fit(control):
                online_train_steps = 0

            recent_validation_batch = None
            recent_validation_payload = None
            if args.online_candidate_refit and online_train_steps > 0:
                online_validation_steps = _online_validation_steps(
                    args,
                    online_collect_steps,
                )
                recent_validation_replay = _new_replay_buffer(
                    capacity=online_validation_steps,
                    num_envs=args.num_envs,
                    observation_dim=config.observation_dim,
                    action_shape=replay_action_shape,
                    action_dtype=replay_action_dtype,
                )
                observations, validation_env_steps, recent_validation_payload = (
                    _collect_policy_steps(
                        adapter,
                        observations,
                        state,
                        config,
                        recent_validation_replay,
                        steps=online_validation_steps,
                        action_low=adapter.action_low,
                        action_high=adapter.action_high,
                        desc=f"{control} {phase} collect recent validation",
                        quiet=args.quiet,
                        np_rng=np_rng,
                        stochastic_actions=args.stochastic_collection,
                    )
                )
                env_steps += validation_env_steps
                recent_validation_payload = {
                    **recent_validation_payload,
                    "total_env_steps": env_steps,
                    "recent_validation_size_per_env": recent_validation_replay.size,
                    "held_out_from_training_replay": True,
                }
                logger.write_json(
                    f"{phase}_recent_policy_validation_replay.json",
                    recent_validation_payload,
                )
                logger.append_metrics(
                    {
                        "phase": "online_recent_policy_validation_replay",
                        "online_iteration": online_index,
                        "control": control,
                        **recent_validation_payload,
                    }
                )
                recent_validation_batch = recent_validation_replay.sample(
                    np_rng,
                    batch_size=args.batch_size,
                    chunk_length=args.chunk_length,
                    max_horizon=max(args.model_horizon, args.open_loop_horizon),
                )

            pre_refit_state = state
            online_loss_history: list[float] = []
            candidate_report = None
            if online_train_steps > 0:
                if args.online_candidate_refit:
                    if recent_validation_batch is None:
                        raise RuntimeError("candidate refit requires recent validation")
                    state, rng, candidate_report, online_loss_history = (
                        _fit_candidate_world_model(
                            args,
                            logger,
                            pre_refit_state,
                            rng,
                            replay,
                            config,
                            np_rng=np_rng,
                            steps=online_train_steps,
                            control=control,
                            phase=f"{phase}_candidate_world_model",
                            desc=f"{control} {phase} fit candidate world model",
                            env_steps=env_steps,
                            anchor_replay=anchor_replay,
                            recent_replay=recent_actor_replay,
                            anchor_validation_batch=validation_batch,
                            recent_validation_batch=recent_validation_batch,
                            action_low=adapter.action_low,
                            action_high=adapter.action_high,
                            control_value_weight=args.online_control_value_weight,
                        )
                    )
                    logger.write_json(f"{phase}_candidate_refit.json", candidate_report)
                    logger.append_metrics(
                        {
                            "phase": "online_candidate_refit",
                            "online_iteration": online_index,
                            "control": control,
                            **candidate_report["gate"],
                            **candidate_report.get("checkpoint_selection", {}),
                        }
                    )
                else:
                    state, rng, _, online_loss_history = _fit_world_model(
                        args,
                        logger,
                        state,
                        rng,
                        replay,
                        config=config,
                        np_rng=np_rng,
                        steps=online_train_steps,
                        control=control,
                        phase=f"{phase}_world_model",
                        desc=f"{control} {phase} fit world model",
                        env_steps=env_steps,
                        freeze_encoder=True,
                        control_value_weight=args.online_control_value_weight,
                    )
            logger.plot_world_model_loss(
                online_loss_history,
                filename=f"{phase}_world_model_loss.png",
            )

            rng, eval_key = jax.random.split(rng)
            online_metrics = _evaluate_model(
                state,
                eval_key,
                validation_batch,
                config,
                chunk_length=args.chunk_length,
                open_loop_horizon=args.open_loop_horizon,
                control=control,
                action_low=adapter.action_low,
                action_high=adapter.action_high,
            )
            logger.write_json(f"{phase}_model_metrics.json", online_metrics)

            online_policy_train_steps = (
                args.online_policy_train_steps
                if args.online_policy_train_steps is not None
                else args.policy_train_steps
            )
            if online_policy_train_steps > 0:
                online_policy_outcome = _maybe_train_policy(
                    args,
                    logger,
                    state,
                    config,
                    replay,
                    control=control,
                    seed=seed,
                    np_rng=np_rng,
                    rng=rng,
                    action_low=adapter.action_low,
                    action_high=adapter.action_high,
                    phase=f"{phase}_policy",
                    train_steps=online_policy_train_steps,
                    reset_actor=args.online_reset_actor,
                )
                candidate_policy_state = online_policy_outcome["state"]
                rng = online_policy_outcome["rng"]
                candidate_policy_payload = online_policy_outcome["outcome"]
                candidate_policy_return = candidate_policy_payload.get(
                    "policy_trained_mean"
                )
                previous_champion_return = champion_policy_return
                policy_update_accepted = True
                if args.online_policy_champion:
                    policy_update_accepted = candidate_policy_return is not None and (
                        previous_champion_return is None
                        or candidate_policy_return
                        >= previous_champion_return
                        - args.online_policy_champion_tolerance
                    )
                if policy_update_accepted:
                    state = candidate_policy_state
                    champion_state = state
                    champion_policy_outcome = dict(candidate_policy_payload)
                    champion_policy_return = candidate_policy_return
                    champion_policy_iteration = online_index
                    online_policy_payload = dict(candidate_policy_payload)
                else:
                    state = copy_policy_heads(candidate_policy_state, champion_state)
                    online_policy_payload = dict(champion_policy_outcome)
                online_policy_payload.update(
                    {
                        "policy_champion_enabled": args.online_policy_champion,
                        "policy_update_accepted": policy_update_accepted,
                        "policy_candidate_trained_mean": candidate_policy_return,
                        "policy_previous_champion_mean": previous_champion_return,
                        "policy_champion_return": champion_policy_return,
                        "policy_champion_iteration": champion_policy_iteration,
                        "policy_champion_tolerance": (
                            args.online_policy_champion_tolerance
                        ),
                    }
                )
                candidate_policy_payload = {
                    **candidate_policy_payload,
                    "policy_champion_enabled": args.online_policy_champion,
                    "policy_update_accepted": policy_update_accepted,
                    "policy_candidate_trained_mean": candidate_policy_return,
                    "policy_previous_champion_mean": previous_champion_return,
                    "policy_champion_return": champion_policy_return,
                    "policy_champion_iteration": champion_policy_iteration,
                    "policy_champion_tolerance": args.online_policy_champion_tolerance,
                }
                policy_outcome = {
                    "state": state,
                    "rng": rng,
                    "outcome": online_policy_payload,
                }
                logger.append_metrics(
                    {
                        "phase": "online_policy_champion",
                        "online_iteration": online_index,
                        "control": control,
                        "policy_update_accepted": policy_update_accepted,
                        "policy_candidate_trained_mean": candidate_policy_return,
                        "policy_previous_champion_mean": previous_champion_return,
                        "policy_champion_return": champion_policy_return,
                        "policy_champion_iteration": champion_policy_iteration,
                        "policy_champion_tolerance": (
                            args.online_policy_champion_tolerance
                        ),
                    }
                )
            else:
                online_policy_payload = {"policy_training_enabled": False}
                candidate_policy_payload = {}
            online_history.append(
                {
                    "iteration": online_index,
                    "actor_replay": collect_payload,
                    "recent_policy_validation": recent_validation_payload,
                    "candidate_refit": candidate_report,
                    "model_metrics": online_metrics,
                    "policy": online_policy_payload,
                    "candidate_policy": candidate_policy_payload,
                    "world_model_train_steps": online_train_steps,
                    "policy_train_steps": online_policy_train_steps,
                }
            )
        if online_history:
            policy_outcome["outcome"] = merge_online_policy_baseline(
                policy_outcome["outcome"],
                initial_policy_outcome,
            )
            policy_outcome["outcome"].update(
                online_history_metrics(online_history, initial_policy_outcome)
            )
        if online_history:
            logger.write_json("online_history.json", online_history)

        rng, eval_key = jax.random.split(rng)
        final_metrics = _evaluate_model(
            state,
            eval_key,
            final_batch,
            config,
            chunk_length=args.chunk_length,
            open_loop_horizon=args.open_loop_horizon,
            control=control,
            action_low=adapter.action_low,
            action_high=adapter.action_high,
        )
        logger.write_json("model_metrics_final.json", final_metrics)

        if args.decoder_train_steps > 0:
            rng, decoder_key = jax.random.split(rng)
            decoder_summary = _fit_decoder_diagnostic(
                args,
                logger,
                state,
                config,
                replay,
                validation_replay,
                np_rng=np_rng,
                key=decoder_key,
                run_dir=run_dir,
                control=control,
            )
            logger.write_json("decoder_metrics.json", decoder_summary)

        checkpoint_dir = run_dir / "checkpoint"
        save_checkpoint(
            checkpoint_dir,
            state,
            metadata={
                "algorithm": "single_agent_sigreg_jepa_world_model",
                "env": args.env,
                "env_backend": _env_backend(args.env),
                "control": control,
                "policy_trained": args.policy_train_steps > 0,
                "jepa_config": dataclasses.asdict(config),
                "seed": seed,
            },
        )
        reload_diff = _reload_prediction_diff(
            state,
            config,
            checkpoint_dir=checkpoint_dir,
            batch=final_batch,
            seed=seed + 99,
            chunk_length=args.chunk_length,
        )
        reload = {"reload_max_abs_prediction_diff": reload_diff}
        logger.write_json("reload_evaluation.json", reload)

        final_policy_eval = None
        if args.final_policy_eval_episodes > 0:
            final_policy_eval_num_envs = args.final_policy_eval_num_envs or min(
                args.num_envs,
                args.final_policy_eval_episodes,
            )
            final_policy_eval = _evaluate_policy(
                args,
                state,
                config,
                seed=seed + 9_000_000,
                num_envs=final_policy_eval_num_envs,
                episodes=args.final_policy_eval_episodes,
                action_low=_as_action_bound(adapter.action_low),
                action_high=_as_action_bound(adapter.action_high),
                desc=f"{control} final eval champion policy",
            )
            logger.write_json(
                "final_champion_policy_evaluation.json",
                final_policy_eval,
            )
            logger.append_metrics(
                {
                    "phase": "final_champion_policy_evaluation",
                    "control": control,
                    "policy_champion_iteration": champion_policy_iteration,
                    "policy_champion_return": champion_policy_return,
                    **final_policy_eval,
                }
            )

        world_model_passed = run_passed(initial_metrics, final_metrics, reload_diff)
        outcome = {
            "run_index": run_index,
            "control": control,
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "target": (
                f"{_env_backend(args.env)}:"
                f"p(z_next, reward, continue | z, {action_mode}_action)"
            ),
            "initial_jepa_loss": initial_metrics["model/jepa_loss"],
            "final_jepa_loss": final_metrics["model/jepa_loss"],
            "initial_open_loop_loss": initial_metrics["model/open_loop_loss"],
            "final_open_loop_loss": final_metrics["model/open_loop_loss"],
            "final_reward_loss": final_metrics["model/reward_loss"],
            "final_reward_constant_mse": final_metrics["model/reward_constant_mse"],
            "final_continue_loss": final_metrics["model/continue_loss"],
            "final_continue_constant_bce": final_metrics["model/continue_constant_bce"],
            "jepa_loss_delta": initial_metrics["model/jepa_loss"]
            - final_metrics["model/jepa_loss"],
            "open_loop_loss_delta": initial_metrics["model/open_loop_loss"]
            - final_metrics["model/open_loop_loss"],
            "reload_max_abs_prediction_diff": reload_diff,
            "final_model_metrics": final_metrics,
            "final_policy_eval": final_policy_eval,
            "final_policy_eval_episodes": (
                final_policy_eval["episodes"] if final_policy_eval else None
            ),
            "final_policy_eval_mean": (
                final_policy_eval["mean_return"] if final_policy_eval else None
            ),
            "final_policy_eval_std": (
                final_policy_eval["std_return"] if final_policy_eval else None
            ),
            "final_policy_eval_env_steps": (
                final_policy_eval.get("env_steps") if final_policy_eval else None
            ),
            "final_policy_eval_completed_episode_steps": (
                final_policy_eval.get("completed_episode_steps")
                if final_policy_eval
                else None
            ),
            "online_iterations": args.online_iterations,
            "online_history": online_history,
            **policy_outcome["outcome"],
            "world_model_passed": world_model_passed,
            "passed": world_model_passed,
        }
        outcome.update(
            real_step_accounting(
                initial_train_replay_env_steps=initial_train_replay_env_steps,
                initial_validation_env_steps=initial_validation_env_steps,
                initial_policy_outcome=initial_policy_outcome,
                online_history=online_history,
                final_policy_eval=final_policy_eval,
            )
        )
        logger.write_json("outcome.json", outcome)
        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    key: value
                    for key, value in to_jsonable(outcome).items()
                    if isinstance(value, (int, float, bool, str))
                }
            )
        return to_jsonable(outcome)
    finally:
        adapter.close()
        if wandb_run is not None:
            wandb_run.finish()


def _collect_random_steps(
    adapter,
    observations: np.ndarray,
    rng: np.random.Generator,
    replay: SequenceReplayBuffer | tuple[SequenceReplayBuffer, ...],
    *,
    steps: int,
    desc: str,
    quiet: bool,
) -> tuple[np.ndarray, int]:
    for _ in tqdm(range(steps), desc=desc, unit="step", disable=quiet):
        actions = adapter.sample_actions(rng)
        step = adapter.step(actions)
        _add_replay_step(
            replay,
            observations=observations[:, 0],
            actions=actions[:, 0],
            rewards=step.rewards[:, 0],
            dones=step.dones[:, 0],
        )
        observations = step.observations
    return observations, steps * adapter.num_envs


def _collect_policy_steps(
    adapter,
    observations: np.ndarray,
    state,
    config: JepaConfig,
    replay: SequenceReplayBuffer | tuple[SequenceReplayBuffer, ...],
    *,
    steps: int,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
    desc: str,
    quiet: bool,
    np_rng: np.random.Generator,
    stochastic_actions: bool = False,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    discrete = config.action_mode == "discrete"
    action_low_jax = None if discrete else jnp.asarray(action_low, dtype=jnp.float32)
    action_high_jax = None if discrete else jnp.asarray(action_high, dtype=jnp.float32)
    action_key = jax.random.PRNGKey(int(np_rng.integers(0, 2**31 - 1)))
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    progress = tqdm(range(steps), desc=desc, unit="step", disable=quiet)
    for _ in progress:
        action_key, step_action_key = jax.random.split(action_key)
        if discrete:
            actions = np.asarray(
                select_discrete_actions(
                    state,
                    jnp.asarray(observations[:, 0], dtype=jnp.float32),
                    config,
                    key=step_action_key,
                    stochastic=stochastic_actions,
                )
            )
            step = adapter.step(actions[:, None])
        else:
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
        completed_returns.extend(float(item[0]) for item in step.completed_returns)
        completed_lengths.extend(int(item) for item in step.completed_lengths)
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
    }
    return observations, steps * adapter.num_envs, metrics


def _add_replay_step(
    replay: SequenceReplayBuffer | tuple[SequenceReplayBuffer, ...],
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
) -> None:
    buffers = replay if isinstance(replay, tuple) else (replay,)
    for buffer in buffers:
        buffer.add_step(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )


def _collect_validation_replay(
    args: argparse.Namespace,
    config: JepaConfig,
    *,
    seed: int,
) -> SequenceReplayBuffer:
    adapter = _make_vector_adapter(args, seed=seed)
    try:
        action_shape, action_dtype = _replay_action_spec(
            adapter,
            _action_mode(args.env),
        )
        replay = SequenceReplayBuffer(
            capacity=max(2, args.validation_steps),
            num_envs=args.num_envs,
            observation_shape=(config.observation_dim,),
            action_shape=action_shape,
            action_dtype=action_dtype,
        )
        observations = adapter.reset()
        _collect_random_steps(
            adapter,
            observations,
            np.random.default_rng(seed),
            replay,
            steps=args.validation_steps,
            desc="collect validation replay",
            quiet=args.quiet,
        )
        return replay
    finally:
        adapter.close()


def _online_validation_steps(
    args: argparse.Namespace, online_collect_steps: int
) -> int:
    min_steps = args.chunk_length + max(args.model_horizon, args.open_loop_horizon)
    if args.online_validation_steps is not None:
        return args.online_validation_steps
    return max(min_steps, min(args.validation_steps, online_collect_steps))


def _new_replay_buffer(
    *,
    capacity: int,
    num_envs: int,
    observation_dim: int,
    action_shape: tuple[int, ...],
    action_dtype: np.dtype | type,
) -> SequenceReplayBuffer:
    return SequenceReplayBuffer(
        capacity=max(2, capacity),
        num_envs=num_envs,
        observation_shape=(observation_dim,),
        action_shape=action_shape,
        action_dtype=action_dtype,
    )


def _replay_action_spec(
    adapter,
    action_mode: str,
) -> tuple[tuple[int, ...], type]:
    if action_mode == "discrete":
        return (), np.int32
    return (adapter.action_dim,), np.float32


def _as_action_bound(bound: np.ndarray | None) -> jax.Array | None:
    return None if bound is None else jnp.asarray(bound, dtype=jnp.float32)


def _evaluate_candidate_model_update(
    args: argparse.Namespace,
    baseline_state,
    candidate_state,
    *,
    anchor_key: jax.Array,
    recent_key: jax.Array,
    anchor_validation_batch: ReplayBatch,
    recent_validation_batch: ReplayBatch,
    config: JepaConfig,
    control: ControlMode,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
    baseline_anchor_metrics: dict[str, Any] | None = None,
    baseline_recent_policy_metrics: dict[str, Any] | None = None,
    update: int | None = None,
) -> dict[str, Any]:
    baseline_anchor = (
        baseline_anchor_metrics
        if baseline_anchor_metrics is not None
        else _evaluate_model(
            baseline_state,
            anchor_key,
            anchor_validation_batch,
            config,
            chunk_length=args.chunk_length,
            open_loop_horizon=args.open_loop_horizon,
            control=control,
            action_low=action_low,
            action_high=action_high,
        )
    )
    candidate_anchor = _evaluate_model(
        candidate_state,
        anchor_key,
        anchor_validation_batch,
        config,
        chunk_length=args.chunk_length,
        open_loop_horizon=args.open_loop_horizon,
        control=control,
        action_low=action_low,
        action_high=action_high,
    )
    baseline_recent = (
        baseline_recent_policy_metrics
        if baseline_recent_policy_metrics is not None
        else _evaluate_model(
            baseline_state,
            recent_key,
            recent_validation_batch,
            config,
            chunk_length=args.chunk_length,
            open_loop_horizon=args.open_loop_horizon,
            control=control,
            action_low=action_low,
            action_high=action_high,
        )
    )
    candidate_recent = _evaluate_model(
        candidate_state,
        recent_key,
        recent_validation_batch,
        config,
        chunk_length=args.chunk_length,
        open_loop_horizon=args.open_loop_horizon,
        control=control,
        action_low=action_low,
        action_high=action_high,
    )
    gate = candidate_refit_gate_report(
        baseline_anchor,
        candidate_anchor,
        baseline_recent,
        candidate_recent,
        metric=args.online_candidate_gate_metric,
        min_recent_improvement=args.online_candidate_min_recent_improvement,
        max_anchor_degradation=args.online_candidate_max_anchor_degradation,
        anchor_penalty=args.online_candidate_anchor_penalty,
    )
    return {
        "model_update_accepted": gate["model_update_accepted"],
        "candidate_update": update,
        "gate": gate,
        "baseline_anchor_metrics": baseline_anchor,
        "candidate_anchor_metrics": candidate_anchor,
        "baseline_recent_policy_metrics": baseline_recent,
        "candidate_recent_policy_metrics": candidate_recent,
    }


def _fit_candidate_world_model(
    args: argparse.Namespace,
    logger: RunLogger,
    state,
    rng: jax.Array,
    replay: SequenceReplayBuffer,
    config: JepaConfig,
    *,
    np_rng: np.random.Generator,
    steps: int,
    control: ControlMode,
    phase: str,
    desc: str,
    env_steps: int,
    anchor_replay: SequenceReplayBuffer,
    recent_replay: SequenceReplayBuffer,
    anchor_validation_batch: ReplayBatch,
    recent_validation_batch: ReplayBatch,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
    control_value_weight: float = 0.0,
) -> tuple[Any, jax.Array, dict[str, Any], list[float]]:
    rng, anchor_key, recent_key = jax.random.split(rng, 3)
    baseline_anchor = _evaluate_model(
        state,
        anchor_key,
        anchor_validation_batch,
        config,
        chunk_length=args.chunk_length,
        open_loop_horizon=args.open_loop_horizon,
        control=control,
        action_low=action_low,
        action_high=action_high,
    )
    baseline_recent = _evaluate_model(
        state,
        recent_key,
        recent_validation_batch,
        config,
        chunk_length=args.chunk_length,
        open_loop_horizon=args.open_loop_horizon,
        control=control,
        action_low=action_low,
        action_high=action_high,
    )

    candidate_state = state
    best_state = None
    best_report = None
    final_report = None
    reports: list[dict[str, Any]] = []
    loss_history: list[jax.Array] = []
    metrics: dict[str, Any] = {}
    eval_interval = args.online_candidate_eval_interval
    fit_steps = tqdm(
        range(1, steps + 1),
        desc=desc,
        unit="update",
        disable=args.quiet,
    )
    for step_index in fit_steps:
        batch = sample_online_candidate_batch(
            np_rng,
            replay=replay,
            anchor_replay=anchor_replay,
            recent_replay=recent_replay,
            batch_size=args.batch_size,
            chunk_length=args.chunk_length,
            max_horizon=max(args.model_horizon, args.open_loop_horizon),
            anchor_batch_fraction=args.online_anchor_batch_fraction,
        )
        rng, train_key = jax.random.split(rng)
        candidate_state, metrics = train_model_step(
            candidate_state,
            train_key,
            batch,
            config,
            chunk_length=args.chunk_length,
            control=control,
            freeze_encoder=True,
            control_value_weight=control_value_weight,
        )
        loss_history.append(metrics["model/total_loss"])
        if (
            step_index == 1
            or step_index == steps
            or step_index % args.eval_interval == 0
        ):
            fit_steps.set_postfix(
                loss=f"{float(metrics['model/total_loss']):.4g}",
                jepa=f"{float(metrics['model/jepa_loss']):.4g}",
            )
            logger.append_metrics(
                {
                    "phase": phase,
                    "update": step_index,
                    "env_steps": env_steps,
                    "control": control,
                    "online_encoder_frozen": True,
                    "online_control_value_weight": control_value_weight,
                    "online_anchor_batch_fraction": args.online_anchor_batch_fraction,
                    "anchor_replay_size_per_env": anchor_replay.size,
                    "recent_replay_size_per_env": recent_replay.size,
                    **metrics,
                }
            )

        should_evaluate = step_index == steps or (
            eval_interval > 0 and step_index % eval_interval == 0
        )
        if should_evaluate:
            report = _evaluate_candidate_model_update(
                args,
                state,
                candidate_state,
                anchor_key=anchor_key,
                recent_key=recent_key,
                anchor_validation_batch=anchor_validation_batch,
                recent_validation_batch=recent_validation_batch,
                config=config,
                control=control,
                action_low=action_low,
                action_high=action_high,
                baseline_anchor_metrics=baseline_anchor,
                baseline_recent_policy_metrics=baseline_recent,
                update=step_index,
            )
            reports.append(report)
            final_report = report
            if report["model_update_accepted"] and (
                best_report is None
                or report["gate"]["candidate_gate_score"]
                > best_report["gate"]["candidate_gate_score"]
            ):
                best_report = report
                best_state = candidate_state
            logger.append_metrics(
                {
                    "phase": "online_candidate_refit_checkpoint",
                    "update": step_index,
                    "env_steps": env_steps,
                    "control": control,
                    **report["gate"],
                }
            )

    if final_report is None:
        final_report = _evaluate_candidate_model_update(
            args,
            state,
            candidate_state,
            anchor_key=anchor_key,
            recent_key=recent_key,
            anchor_validation_batch=anchor_validation_batch,
            recent_validation_batch=recent_validation_batch,
            config=config,
            control=control,
            action_low=action_low,
            action_high=action_high,
            baseline_anchor_metrics=baseline_anchor,
            baseline_recent_policy_metrics=baseline_recent,
            update=steps,
        )
        reports.append(final_report)

    best_report = best_passing_candidate_report(reports)
    selected_report = best_report if best_report is not None else final_report
    selected_state = best_state if best_report is not None else state
    checkpoint_summaries = [
        candidate_checkpoint_gate_summary(report) for report in reports
    ]
    selected_report = {
        **selected_report,
        "checkpoint_selection": {
            "candidate_checkpointing_enabled": eval_interval > 0,
            "candidate_eval_interval": eval_interval,
            "candidate_checkpoint_count": len(reports),
            "candidate_checkpoint_updates": [
                item["candidate_update"] for item in checkpoint_summaries
            ],
            "candidate_final_update": steps,
            "candidate_final_update_accepted": final_report["model_update_accepted"],
            "candidate_final_gate_score": final_report["gate"].get(
                "candidate_gate_score"
            ),
            "candidate_best_passing_update": (
                best_report.get("candidate_update") if best_report is not None else None
            ),
            "candidate_selected_update": (
                selected_report.get("candidate_update")
                if selected_report.get("model_update_accepted")
                else None
            ),
        },
        "candidate_checkpoints": checkpoint_summaries,
        "final_candidate_gate": final_report["gate"],
    }
    return selected_state, rng, selected_report, [float(loss) for loss in loss_history]


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
    control: ControlMode,
    phase: str,
    desc: str,
    env_steps: int,
    freeze_encoder: bool = False,
    control_value_weight: float = 0.0,
) -> tuple[Any, jax.Array, dict[str, Any], list[float]]:
    loss_history: list[jax.Array] = []
    metrics: dict[str, Any] = {}
    fit_steps = tqdm(
        range(1, steps + 1),
        desc=desc,
        unit="update",
        disable=args.quiet,
    )
    for step_index in fit_steps:
        batch = replay.sample(
            np_rng,
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
            control=control,
            freeze_encoder=freeze_encoder,
            control_value_weight=control_value_weight,
        )
        # Keep device scalars unconverted so async dispatch overlaps host-side
        # replay sampling with device compute; sync only at logging intervals.
        loss_history.append(metrics["model/total_loss"])
        if (
            step_index == 1
            or step_index == steps
            or step_index % args.eval_interval == 0
        ):
            fit_steps.set_postfix(
                loss=f"{float(metrics['model/total_loss']):.4g}",
                jepa=f"{float(metrics['model/jepa_loss']):.4g}",
            )
            logger.append_metrics(
                {
                    "phase": phase,
                    "update": step_index,
                    "env_steps": env_steps,
                    "control": control,
                    "online_encoder_frozen": freeze_encoder,
                    "online_control_value_weight": control_value_weight,
                    **metrics,
                }
            )
    return state, rng, to_jsonable(metrics), [float(loss) for loss in loss_history]


def _fit_decoder_diagnostic(
    args: argparse.Namespace,
    logger: RunLogger,
    state,
    config: JepaConfig,
    replay: SequenceReplayBuffer,
    validation_replay: SequenceReplayBuffer,
    *,
    np_rng: np.random.Generator,
    key: jax.Array,
    run_dir: Path,
    control: ControlMode,
) -> dict[str, Any]:
    horizon = args.decoder_rollout_horizon or args.open_loop_horizon
    if not validation_replay.can_sample(
        chunk_length=config.context_window,
        max_horizon=horizon,
    ):
        raise ValueError(
            "validation replay is too small for the decoder rollout diagnostic "
            f"(need {config.context_window + horizon} steps, "
            f"have {validation_replay.size})"
        )
    decoder_config = DecoderConfig(
        latent_dim=config.latent_dim,
        observation_dim=config.observation_dim,
        hidden_dim=args.decoder_hidden_dim,
        learning_rate=args.decoder_learning_rate,
    )
    decoder_state = create_decoder_train_state(key, decoder_config)

    def flat_pairs(batch: ReplayBatch) -> tuple[jax.Array, jax.Array]:
        latents = encode_observations(state, batch.observations)
        return (
            latents.reshape((-1, config.latent_dim)),
            batch.observations.reshape((-1, config.observation_dim)),
        )

    loss_history: list[jax.Array] = []
    fit_steps = tqdm(
        range(1, args.decoder_train_steps + 1),
        desc=f"{control} fit diagnostic decoder",
        unit="update",
        disable=args.quiet,
    )
    for step_index in fit_steps:
        batch = replay.sample(
            np_rng,
            batch_size=args.batch_size,
            chunk_length=args.chunk_length,
            max_horizon=1,
        )
        decoder_state, loss = train_decoder_step(decoder_state, *flat_pairs(batch))
        loss_history.append(loss)
        if (
            step_index == 1
            or step_index == args.decoder_train_steps
            or step_index % args.eval_interval == 0
        ):
            fit_steps.set_postfix(recon=f"{float(loss):.4g}")
            logger.append_metrics(
                {
                    "phase": "observation_decoder",
                    "update": step_index,
                    "control": control,
                    "decoder/recon_loss": float(loss),
                }
            )

    validation_batch = validation_replay.sample(
        np_rng,
        batch_size=args.batch_size,
        chunk_length=args.chunk_length,
        max_horizon=1,
    )
    validation_mse = float(
        decoder_reconstruction_mse(decoder_state, *flat_pairs(validation_batch))
    )

    candidate_batch = validation_replay.sample(
        np_rng,
        batch_size=max(32, 8 * args.decoder_rollout_trajectories),
        chunk_length=config.context_window,
        max_horizon=horizon,
    )
    display_batch = select_display_trajectories(
        candidate_batch,
        context_window=config.context_window,
        horizon=horizon,
        count=args.decoder_rollout_trajectories,
    )
    rollout = decode_open_loop_rollout(
        state,
        decoder_state,
        display_batch,
        config,
        horizon=horizon,
    )
    rollout_np = {name: np.asarray(value) for name, value in rollout.items()}
    rollout_path = run_dir / "decoder_rollout.npz"
    np.savez(rollout_path, **rollout_np)
    title = (
        f"{args.env}: real vs imagined decoded rollout "
        f"(context={config.context_window}, horizon={horizon})"
    )
    frames_path = save_rollout_frames_plot(
        rollout_np,
        run_dir / FRAMES_FILENAME,
        title=title,
    )
    traces_path = save_rollout_traces_plot(
        rollout_np,
        run_dir / TRACES_FILENAME,
        title=title,
    )
    validity = rollout_np["validity"]
    cosine = rollout_np["open_loop_cosine"]
    valid_total = float(validity.sum())
    return {
        "decoder_config": dataclasses.asdict(decoder_config),
        "train_steps": args.decoder_train_steps,
        "final_train_recon_mse": float(loss_history[-1]),
        "validation_recon_mse": validation_mse,
        "rollout_horizon": horizon,
        "rollout_trajectories": int(cosine.shape[0]),
        "rollout_open_loop_cosine_mean": (
            float((cosine * validity).sum() / valid_total)
            if valid_total > 0.0
            else None
        ),
        "rollout_valid_fraction": float(validity.mean()),
        "rollout_npz": str(rollout_path),
        "frames_plot": str(frames_path),
        "traces_plot": str(traces_path),
    }


def _maybe_train_policy(
    args: argparse.Namespace,
    logger: RunLogger,
    state,
    config: JepaConfig,
    replay: SequenceReplayBuffer,
    *,
    control: ControlMode,
    seed: int,
    np_rng: np.random.Generator,
    rng: jax.Array,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
    phase: str = "policy",
    train_steps: int | None = None,
    reset_actor: bool = True,
    eval_seed_offset: int = 3_000_000,
    selection_seed_offset: int = 5_000_000,
    confirmation_seed_offset: int = 7_000_000,
) -> dict[str, Any]:
    policy_train_steps = args.policy_train_steps if train_steps is None else train_steps
    if policy_train_steps == 0:
        return {
            "state": state,
            "rng": rng,
            "outcome": {"policy_training_enabled": False},
        }
    if (
        args.policy_objective == "candidate-distill"
        and config.action_mode == "discrete"
    ):
        raise ValueError(
            "--policy-objective candidate-distill only supports continuous actions; "
            "use --policy-objective direct for discrete (gymnax) environments"
        )

    if reset_actor:
        rng, reset_key = jax.random.split(rng)
        state = reset_policy_heads(state, reset_key, config)
    eval_num_envs = args.policy_eval_num_envs or min(
        args.num_envs,
        args.policy_eval_episodes,
    )
    policy_batch_size = args.policy_batch_size or args.batch_size
    action_low_jax = _as_action_bound(action_low)
    action_high_jax = _as_action_bound(action_high)
    policy_eval_seed = seed + eval_seed_offset
    policy_selection_seed = seed + selection_seed_offset
    policy_confirmation_seed = seed + confirmation_seed_offset
    selection_enabled = args.policy_selection_interval > 0
    selection_num_envs = args.policy_selection_num_envs or min(
        args.num_envs,
        args.policy_selection_episodes,
    )
    confirmation_enabled = args.policy_confirmation_episodes > 0
    confirmation_num_envs = args.policy_confirmation_num_envs or min(
        args.num_envs,
        max(1, args.policy_confirmation_episodes),
    )
    artifact_prefix = "" if phase == "policy" else f"{phase}_"
    metric_phase_prefix = "" if phase == "policy" else f"{phase}_"

    random_eval = _evaluate_random_policy(
        args,
        seed=policy_eval_seed,
        num_envs=eval_num_envs,
        desc=f"{control} {phase} eval random policy",
    )
    initial_eval = _evaluate_policy(
        args,
        state,
        config,
        seed=policy_eval_seed,
        num_envs=eval_num_envs,
        action_low=action_low_jax,
        action_high=action_high_jax,
        desc=f"{control} {phase} eval initial policy",
    )
    logger.write_json(f"{artifact_prefix}random_policy_evaluation.json", random_eval)
    logger.write_json(f"{artifact_prefix}initial_policy_evaluation.json", initial_eval)

    confirmation_random_eval = None
    confirmation_initial_eval = None
    if confirmation_enabled:
        confirmation_random_eval = _evaluate_random_policy(
            args,
            seed=policy_confirmation_seed,
            num_envs=confirmation_num_envs,
            episodes=args.policy_confirmation_episodes,
            desc=f"{control} {phase} confirm random policy",
        )
        confirmation_initial_eval = _evaluate_policy(
            args,
            state,
            config,
            seed=policy_confirmation_seed,
            num_envs=confirmation_num_envs,
            episodes=args.policy_confirmation_episodes,
            action_low=action_low_jax,
            action_high=action_high_jax,
            desc=f"{control} {phase} confirm initial policy",
        )
        logger.write_json(
            f"{artifact_prefix}confirmation_random_policy_evaluation.json",
            confirmation_random_eval,
        )
        logger.write_json(
            f"{artifact_prefix}confirmation_initial_policy_evaluation.json",
            confirmation_initial_eval,
        )

    best_state = state
    best_policy_step = 0
    best_policy_metrics_json: dict[str, Any] = {
        "policy/selected_initial_actor": True,
    }
    selection_history: list[dict[str, Any]] = []
    best_selection_eval: dict[str, Any] | None = None
    best_selection_mean = -math.inf
    if selection_enabled:
        selection_eval = _evaluate_policy(
            args,
            state,
            config,
            seed=policy_selection_seed,
            num_envs=selection_num_envs,
            episodes=args.policy_selection_episodes,
            action_low=action_low_jax,
            action_high=action_high_jax,
            desc=f"{control} {phase} select initial policy",
        )
        best_selection_eval = selection_eval
        best_selection_mean = selection_eval["mean_return"]
        selection_record = _policy_selection_record(
            step=0,
            evaluation=selection_eval,
            selected=True,
        )
        selection_history.append(selection_record)
        logger.append_metrics(
            {
                "phase": "policy_selection",
                "update": 0,
                "control": control,
                "policy_phase": phase,
                **selection_record,
            }
        )
        logger.write_json(
            f"{artifact_prefix}policy_selection_initial.json", selection_eval
        )

    critic_metrics: dict[str, Any] = {}
    if args.critic_warmup_steps > 0:
        critic_steps = tqdm(
            range(1, args.critic_warmup_steps + 1),
            desc=f"{control} {phase} warm real-return critic",
            unit="update",
            disable=args.quiet,
        )
        for step_index in critic_steps:
            batch = replay.sample(
                np_rng,
                batch_size=policy_batch_size,
                chunk_length=args.critic_horizon,
                max_horizon=1,
            )
            state, critic_metrics = critic_warmup_step(
                state,
                batch,
                config,
                horizon=args.critic_horizon,
                value_clip=args.value_clip,
            )
            if (
                step_index == 1
                or step_index == args.critic_warmup_steps
                or step_index % args.eval_interval == 0
            ):
                critic_steps.set_postfix(
                    loss=f"{float(critic_metrics['critic/total_loss']):.4g}",
                    target=f"{float(critic_metrics['critic/target_mean']):.4g}",
                )
                logger.append_metrics(
                    {
                        "phase": f"{metric_phase_prefix}real_return_critic_warmup",
                        "update": step_index,
                        "control": control,
                        "policy_phase": phase,
                        **critic_metrics,
                    }
                )

    policy_trust_coef = args.policy_trust_coef
    if phase != "policy" and args.online_policy_trust_coef is not None:
        policy_trust_coef = args.online_policy_trust_coef
    reference_actor_params = jax.tree_util.tree_map(jax.lax.stop_gradient, state.params)
    policy_loss_history: list[jax.Array] = []
    metrics: dict[str, Any] = {}
    policy_steps = tqdm(
        range(1, policy_train_steps + 1),
        desc=f"{control} {phase} train frozen-policy",
        unit="update",
        disable=args.quiet,
    )
    for step_index in policy_steps:
        batch = replay.sample(
            np_rng,
            batch_size=policy_batch_size,
            chunk_length=config.context_window,
            max_horizon=1,
        )
        start_observations = batch.observations[:, : config.context_window]
        start_actions = batch.actions[:, : config.context_window]
        rng, policy_key = jax.random.split(rng)
        if args.policy_objective == "candidate-distill":
            state, metrics = continuous_candidate_distill_step(
                state,
                policy_key,
                start_observations,
                config,
                action_low_jax,
                action_high_jax,
                imag_horizon=args.imag_horizon,
                control=control,
                num_candidates=args.num_policy_candidates,
                candidate_min_gap=args.candidate_min_gap,
                action_l2_coef=args.policy_action_l2_coef,
                action_saturation_threshold=args.action_saturation_threshold,
                start_actions=start_actions,
            )
        elif config.action_mode == "discrete":
            state, metrics = discrete_policy_train_step(
                state,
                policy_key,
                start_observations,
                config,
                imag_horizon=args.imag_horizon,
                control=control,
                policy_return_mode=args.policy_return_mode,
                policy_actor_baseline=args.policy_actor_baseline,
                policy_return_normalization=args.policy_return_normalization,
                value_clip=args.value_clip,
                start_actions=start_actions,
                uncertainty_penalty=args.uncertainty_penalty,
                uncertainty_latent_weight=args.uncertainty_latent_weight,
                uncertainty_reward_weight=args.uncertainty_reward_weight,
                uncertainty_continue_weight=args.uncertainty_continue_weight,
                uncertainty_threshold=args.uncertainty_threshold,
                uncertainty_budget=args.uncertainty_budget,
                reference_actor_params=reference_actor_params,
                policy_trust_coef=policy_trust_coef,
                actor_entropy_coef=args.actor_entropy_coef,
            )
        else:
            state, metrics = continuous_policy_train_step(
                state,
                policy_key,
                start_observations,
                config,
                action_low_jax,
                action_high_jax,
                imag_horizon=args.imag_horizon,
                control=control,
                policy_return_mode=args.policy_return_mode,
                policy_actor_baseline=args.policy_actor_baseline,
                policy_return_normalization=args.policy_return_normalization,
                value_clip=args.value_clip,
                action_saturation_threshold=args.action_saturation_threshold,
                start_actions=start_actions,
                uncertainty_penalty=args.uncertainty_penalty,
                uncertainty_latent_weight=args.uncertainty_latent_weight,
                uncertainty_reward_weight=args.uncertainty_reward_weight,
                uncertainty_continue_weight=args.uncertainty_continue_weight,
                uncertainty_threshold=args.uncertainty_threshold,
                uncertainty_budget=args.uncertainty_budget,
                reference_actor_params=reference_actor_params,
                policy_trust_coef=policy_trust_coef,
                actor_entropy_coef=args.actor_entropy_coef,
            )
        policy_loss_history.append(metrics["policy/total_loss"])
        if (
            step_index == 1
            or step_index == policy_train_steps
            or step_index % args.eval_interval == 0
        ):
            policy_loss = float(metrics["policy/total_loss"])
            progress_score = float(
                metrics.get(
                    "policy/imagined_return",
                    metrics.get("policy/candidate_best_score", policy_loss),
                )
            )
            policy_steps.set_postfix(
                loss=f"{policy_loss:.4g}",
                score=f"{progress_score:.4g}",
            )
            logger.append_metrics(
                {
                    "phase": f"{metric_phase_prefix}frozen_model_policy",
                    "update": step_index,
                    "control": control,
                    "policy_phase": phase,
                    **metrics,
                }
            )
        if selection_enabled and (
            step_index == policy_train_steps
            or step_index % args.policy_selection_interval == 0
        ):
            selection_eval = _evaluate_policy(
                args,
                state,
                config,
                seed=policy_selection_seed,
                num_envs=selection_num_envs,
                episodes=args.policy_selection_episodes,
                action_low=action_low_jax,
                action_high=action_high_jax,
                desc=f"{control} {phase} select policy {step_index}",
            )
            selected = selection_eval["mean_return"] > best_selection_mean
            if selected:
                best_state = state
                best_policy_step = step_index
                best_policy_metrics_json = {
                    **to_jsonable(metrics),
                    "policy/selected_initial_actor": False,
                }
                best_selection_eval = selection_eval
                best_selection_mean = selection_eval["mean_return"]
            selection_record = _policy_selection_record(
                step=step_index,
                evaluation=selection_eval,
                selected=selected,
            )
            selection_history.append(selection_record)
            logger.append_metrics(
                {
                    "phase": "policy_selection",
                    "update": step_index,
                    "control": control,
                    "policy_phase": phase,
                    **selection_record,
                    "policy_selection_best_mean_return": best_selection_mean,
                    "policy_selection_best_step": best_policy_step,
                }
            )

    last_policy_metrics_json = to_jsonable(metrics)
    if selection_enabled:
        state = best_state
        logger.write_json(
            f"{artifact_prefix}policy_selection_history.json", selection_history
        )
        logger.write_json(
            f"{artifact_prefix}best_policy_selection_evaluation.json",
            {
                "best_policy_step": best_policy_step,
                "evaluation": best_selection_eval,
            },
        )
    trained_eval = _evaluate_policy(
        args,
        state,
        config,
        seed=policy_eval_seed,
        num_envs=eval_num_envs,
        action_low=action_low_jax,
        action_high=action_high_jax,
        desc=f"{control} {phase} eval trained policy",
    )
    logger.write_json(f"{artifact_prefix}trained_policy_evaluation.json", trained_eval)
    confirmation_trained_eval = None
    if confirmation_enabled:
        confirmation_trained_eval = _evaluate_policy(
            args,
            state,
            config,
            seed=policy_confirmation_seed,
            num_envs=confirmation_num_envs,
            episodes=args.policy_confirmation_episodes,
            action_low=action_low_jax,
            action_high=action_high_jax,
            desc=f"{control} {phase} confirm trained policy",
        )
        logger.write_json(
            f"{artifact_prefix}confirmation_trained_policy_evaluation.json",
            confirmation_trained_eval,
        )
    logger.plot_world_model_loss(
        [float(loss) for loss in policy_loss_history],
        filename=f"{artifact_prefix}frozen_model_policy_loss.png",
    )
    selection_env_steps = sum(
        maybe_int(item.get("policy_selection_env_steps")) for item in selection_history
    )
    selection_completed_episode_steps = sum(
        maybe_int(item.get("policy_selection_completed_episode_steps"))
        for item in selection_history
    )
    confirmation_eval_payloads = [
        confirmation_random_eval,
        confirmation_initial_eval,
        confirmation_trained_eval,
    ]
    confirmation_env_steps = sum(
        eval_env_steps(item) for item in confirmation_eval_payloads
    )
    confirmation_completed_episode_steps = sum(
        eval_completed_episode_steps(item) for item in confirmation_eval_payloads
    )
    policy_eval_payloads = [
        random_eval,
        initial_eval,
        trained_eval,
        *confirmation_eval_payloads,
    ]
    policy_nonselection_env_steps = sum(
        eval_env_steps(item) for item in policy_eval_payloads
    )
    policy_nonselection_completed_episode_steps = sum(
        eval_completed_episode_steps(item) for item in policy_eval_payloads
    )
    policy_total_eval_env_steps = policy_nonselection_env_steps + selection_env_steps
    policy_total_completed_episode_steps = (
        policy_nonselection_completed_episode_steps + selection_completed_episode_steps
    )

    initial_mean = initial_eval["mean_return"]
    trained_mean = trained_eval["mean_return"]
    random_mean = random_eval["mean_return"]
    confirmation_improvement = None
    confirmation_trained_minus_random = None
    if confirmation_enabled:
        confirmation_improvement = (
            confirmation_trained_eval["mean_return"]
            - confirmation_initial_eval["mean_return"]
        )
        confirmation_trained_minus_random = (
            confirmation_trained_eval["mean_return"]
            - confirmation_random_eval["mean_return"]
        )
    policy_metrics_json = (
        best_policy_metrics_json if selection_enabled else last_policy_metrics_json
    )
    critic_metrics_json = to_jsonable(critic_metrics)
    confirmation_passed = not confirmation_enabled or (
        confirmation_improvement is not None
        and confirmation_improvement > 0.0
        and confirmation_trained_minus_random is not None
        and confirmation_trained_minus_random > 0.0
    )
    outcome = {
        "policy_training_enabled": True,
        "policy_phase": phase,
        "policy_reset_actor": reset_actor,
        "policy_train_steps": policy_train_steps,
        "policy_objective": args.policy_objective,
        "policy_return_mode": args.policy_return_mode,
        "policy_actor_baseline": args.policy_actor_baseline,
        "policy_return_normalization": args.policy_return_normalization,
        "policy_stochastic_actor": args.stochastic_actor,
        "policy_stochastic_collection": args.stochastic_collection,
        "policy_actor_entropy_coef": args.actor_entropy_coef,
        "policy_actor_log_std_min": args.actor_log_std_min,
        "policy_actor_log_std_max": args.actor_log_std_max,
        "policy_imag_horizon": args.imag_horizon,
        "num_policy_candidates": args.num_policy_candidates,
        "candidate_min_gap": args.candidate_min_gap,
        "policy_action_l2_coef": args.policy_action_l2_coef,
        "policy_trust_coef": policy_trust_coef,
        "policy_base_trust_coef": args.policy_trust_coef,
        "online_policy_trust_coef": args.online_policy_trust_coef,
        "policy_eval_seed": policy_eval_seed,
        "policy_confirmation_enabled": confirmation_enabled,
        "policy_confirmation_seed": (
            policy_confirmation_seed if confirmation_enabled else None
        ),
        "policy_confirmation_episodes": args.policy_confirmation_episodes,
        "policy_confirmation_num_envs": (
            confirmation_num_envs if confirmation_enabled else None
        ),
        "policy_selection_enabled": selection_enabled,
        "policy_selection_seed": policy_selection_seed,
        "policy_selection_interval": args.policy_selection_interval,
        "policy_selection_episodes": args.policy_selection_episodes,
        "policy_selection_num_envs": selection_num_envs,
        "policy_selection_env_steps": selection_env_steps,
        "policy_selection_completed_episode_steps": selection_completed_episode_steps,
        "policy_nonselection_eval_env_steps": policy_nonselection_env_steps,
        "policy_nonselection_completed_episode_steps": (
            policy_nonselection_completed_episode_steps
        ),
        "policy_confirmation_env_steps": confirmation_env_steps,
        "policy_confirmation_completed_episode_steps": (
            confirmation_completed_episode_steps
        ),
        "policy_total_eval_env_steps": policy_total_eval_env_steps,
        "policy_total_completed_episode_steps": policy_total_completed_episode_steps,
        "best_policy_step": best_policy_step if selection_enabled else None,
        "best_policy_selection_mean": (
            best_selection_mean if selection_enabled else None
        ),
        "last_policy_metrics": last_policy_metrics_json,
        "critic_warmup_steps": args.critic_warmup_steps,
        "critic_horizon": args.critic_horizon,
        "critic_final_metrics": critic_metrics_json,
        "policy_random_mean": random_mean,
        "policy_initial_mean": initial_mean,
        "policy_trained_mean": trained_mean,
        "policy_improvement": trained_mean - initial_mean,
        "policy_primary_improvement": trained_mean - initial_mean,
        "policy_primary_improvement_key": "policy_improvement",
        "policy_trained_minus_random": trained_mean - random_mean,
        "policy_confirmation_random_mean": (
            confirmation_random_eval["mean_return"] if confirmation_enabled else None
        ),
        "policy_confirmation_initial_mean": (
            confirmation_initial_eval["mean_return"] if confirmation_enabled else None
        ),
        "policy_confirmation_trained_mean": (
            confirmation_trained_eval["mean_return"] if confirmation_enabled else None
        ),
        "policy_confirmation_improvement": confirmation_improvement,
        "policy_primary_confirmation_improvement": confirmation_improvement,
        "policy_confirmation_trained_minus_random": confirmation_trained_minus_random,
        "policy_confirmation_passed": confirmation_passed,
        "policy_final_metrics": policy_metrics_json,
        "policy_passed": bool(
            metrics_finite(policy_metrics_json)
            and metrics_finite(critic_metrics_json)
            and trained_mean > initial_mean
            and trained_mean > random_mean
            and confirmation_passed
            and policy_metrics_json.get("policy/action_saturation_fraction", 1.0) < 0.75
        ),
    }
    return {"state": state, "rng": rng, "outcome": outcome}


def _evaluate_random_policy(
    args: argparse.Namespace,
    *,
    seed: int,
    num_envs: int,
    desc: str,
    episodes: int | None = None,
) -> dict[str, Any]:
    target_episodes = args.policy_eval_episodes if episodes is None else episodes
    adapter = _make_vector_adapter(args, seed=seed, num_envs=num_envs)
    try:
        rng = np.random.default_rng(seed)
        observations = adapter.reset()
        del observations
        returns = []
        lengths = []
        step_calls = 0
        with tqdm(
            total=target_episodes,
            desc=desc,
            unit="episode",
            disable=args.quiet,
        ) as progress:
            while len(returns) < target_episodes:
                before = len(returns)
                step = adapter.step(adapter.sample_actions(rng))
                step_calls += 1
                returns.extend(float(item[0]) for item in step.completed_returns)
                lengths.extend(int(item) for item in step.completed_lengths)
                _update_episode_progress(
                    progress,
                    before,
                    len(returns),
                    target_episodes,
                )
        returns = returns[:target_episodes]
        lengths = lengths[:target_episodes]
        return {
            "episodes": len(returns),
            "num_envs": num_envs,
            "env_steps": step_calls * num_envs,
            "completed_episode_steps": int(sum(lengths)),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "returns": returns,
            "lengths": lengths,
        }
    finally:
        adapter.close()


def _evaluate_policy(
    args: argparse.Namespace,
    state,
    config: JepaConfig,
    *,
    seed: int,
    num_envs: int,
    action_low: jax.Array | None,
    action_high: jax.Array | None,
    desc: str,
    episodes: int | None = None,
) -> dict[str, Any]:
    discrete = config.action_mode == "discrete"
    target_episodes = args.policy_eval_episodes if episodes is None else episodes
    adapter = _make_vector_adapter(args, seed=seed, num_envs=num_envs)
    try:
        observations = adapter.reset()
        returns = []
        lengths = []
        step_calls = 0
        with tqdm(
            total=target_episodes,
            desc=desc,
            unit="episode",
            disable=args.quiet,
        ) as progress:
            while len(returns) < target_episodes:
                before = len(returns)
                if discrete:
                    actions = np.asarray(
                        select_discrete_actions(
                            state,
                            jnp.asarray(observations[:, 0], dtype=jnp.float32),
                            config,
                        )
                    )
                    step = adapter.step(actions[:, None])
                else:
                    actions = np.asarray(
                        select_continuous_actions(
                            state,
                            jnp.asarray(observations[:, 0], dtype=jnp.float32),
                            config,
                            action_low,
                            action_high,
                        )
                    )
                    step = adapter.step(actions[:, None, :])
                step_calls += 1
                returns.extend(float(item[0]) for item in step.completed_returns)
                lengths.extend(int(item) for item in step.completed_lengths)
                observations = step.observations
                _update_episode_progress(
                    progress,
                    before,
                    len(returns),
                    target_episodes,
                )
        returns = returns[:target_episodes]
        lengths = lengths[:target_episodes]
        return {
            "episodes": len(returns),
            "num_envs": num_envs,
            "env_steps": step_calls * num_envs,
            "completed_episode_steps": int(sum(lengths)),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "returns": returns,
            "lengths": lengths,
        }
    finally:
        adapter.close()


def _update_episode_progress(
    progress: tqdm,
    before: int,
    after: int,
    target_episodes: int,
) -> None:
    progress.update(max(0, min(after, target_episodes) - min(before, target_episodes)))


def _policy_selection_record(
    *,
    step: int,
    evaluation: dict[str, Any],
    selected: bool,
) -> dict[str, Any]:
    return {
        "policy_selection_step": step,
        "policy_selection_selected": selected,
        "policy_selection_mean_return": evaluation["mean_return"],
        "policy_selection_std_return": evaluation["std_return"],
        "policy_selection_mean_length": evaluation["mean_length"],
        "policy_selection_episodes": evaluation["episodes"],
        "policy_selection_env_steps": evaluation.get("env_steps"),
        "policy_selection_completed_episode_steps": evaluation.get(
            "completed_episode_steps"
        ),
    }


def _evaluate_model(
    state,
    key: jax.Array,
    batch: ReplayBatch,
    config: JepaConfig,
    *,
    chunk_length: int,
    open_loop_horizon: int,
    control: ControlMode,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
) -> dict[str, Any]:
    metrics = dict(
        evaluate_world_model_loss(
            state,
            key,
            batch,
            config,
            chunk_length=chunk_length,
            control=control,
        )
    )
    metrics.update(
        evaluate_open_loop(
            state,
            batch,
            config,
            horizon=open_loop_horizon,
            control=control,
        )
    )
    if config.action_mode == "continuous":
        metrics["model/continuous_action_low_high_sensitivity"] = (
            _continuous_action_sensitivity(
                state,
                batch,
                config,
                action_low=action_low,
                action_high=action_high,
                control=control,
            )
        )
    metrics.update(
        action_contrast_metrics(
            state,
            key,
            batch,
            config,
            chunk_length=chunk_length,
            control=control,
        )
    )
    return to_jsonable(metrics)


@partial(jax.jit, static_argnames=("config", "control"))
def _continuous_action_sensitivity(
    state,
    batch: ReplayBatch,
    config: JepaConfig,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    control: ControlMode,
) -> jax.Array:
    if control == "no-action-world-model":
        return jnp.asarray(0.0, dtype=jnp.float32)
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


if __name__ == "__main__":
    main()
