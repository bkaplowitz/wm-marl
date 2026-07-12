"""Validate a representation-space SIGReg-JEPA world model on single-agent rollouts.

By default this is the first single-agent rung: random continuous-control replay
plus held-out latent prediction, reward prediction, and continue prediction. Passing
``--policy-train-steps`` enables the next rung: reset actor/value heads, freeze
the JEPA world model, train a deterministic continuous actor inside the latent
model, then evaluate that actor in the real environment.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import warnings
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
from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.replay import ReplayBatch, SequenceReplayBuffer
from world_marl.jepa.training import (
    ControlMode,
    continuous_critic_warmup_step,
    continuous_policy_score,
    continuous_policy_train_step,
    copy_policy_heads,
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

MIN_TERMINAL_FRACTION_FOR_CONTINUE_BASELINE = 0.01
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
    parser.add_argument(
        "--save-initial-replay",
        type=Path,
        default=None,
        help=(
            "Save the initially collected train replay as an NPZ for exact "
            "diagnostic reuse."
        ),
    )
    parser.add_argument(
        "--load-initial-replay",
        type=Path,
        default=None,
        help=(
            "Load the initial train replay from an NPZ instead of collecting "
            "new random replay. Intended for controlled seed diagnostics."
        ),
    )
    parser.add_argument("--chunk-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--model-horizon", type=int, default=1)
    parser.add_argument("--open-loop-horizon", type=int, default=5)
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
    parser.add_argument("--optimizer-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--adaptive-grad-clip",
        type=float,
        default=0.0,
        help="Adaptive gradient clipping coefficient. Zero disables AGC.",
    )
    parser.add_argument("--optimizer-epsilon", type=float, default=1e-5)
    parser.add_argument(
        "--input-symlog",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply symlog to vector observations before the latent encoder.",
    )
    parser.add_argument("--activation", choices=("gelu", "silu"), default="gelu")
    parser.add_argument(
        "--normalization",
        choices=("layer", "rms"),
        default="layer",
    )
    parser.add_argument("--actor-output-scale", type=float, default=1.0)
    parser.add_argument("--value-output-scale", type=float, default=1.0)
    parser.add_argument("--reward-output-scale", type=float, default=1.0)
    parser.add_argument(
        "--target-critic-ema-decay",
        type=float,
        default=0.0,
        help=(
            "EMA decay for a target value head used only for lambda-return "
            "bootstrapping and value baselines. 0 disables the target critic."
        ),
    )
    parser.add_argument(
        "--actor-hidden-dim",
        type=int,
        default=0,
        help="Actor head hidden width. Use 0 to match --model-dim.",
    )
    parser.add_argument(
        "--critic-hidden-dim",
        type=int,
        default=0,
        help="Critic/value head hidden width. Use 0 to match --model-dim.",
    )
    parser.add_argument("--actor-num-layers", type=int, default=1)
    parser.add_argument("--critic-num-layers", type=int, default=1)
    parser.add_argument("--actor-layer-norm", action="store_true")
    parser.add_argument("--critic-layer-norm", action="store_true")
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
        choices=("none", "batch", "percentile", "ema-percentile"),
        default="none",
        help=(
            "Normalize imagined returns/advantages inside each actor update. "
            "Batch mode uses stop-gradient weighted batch statistics. "
            "Percentile mode divides by the stop-gradient p95-p5 range, "
            "while ema-percentile smooths that range across updates."
        ),
    )
    parser.add_argument(
        "--policy-gradient-mode",
        choices=("dynamics", "reinforce"),
        default="dynamics",
        help=(
            "Backpropagate actor gradients through imagined dynamics, or use a "
            "Dreamer-style score-function objective with stopped model paths."
        ),
    )
    parser.add_argument(
        "--policy-return-ema-decay",
        type=float,
        default=0.99,
        help="EMA decay for ema-percentile actor return normalization.",
    )
    parser.add_argument(
        "--policy-actor-cvar-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of lowest per-start imagined actor scores to include in "
            "the tail objective. Set below 1 with --policy-actor-cvar-coef "
            "to make actor updates focus on brittle starts."
        ),
    )
    parser.add_argument(
        "--policy-actor-cvar-coef",
        type=float,
        default=0.0,
        help=(
            "Blend factor for the lower-tail actor objective. Zero keeps the "
            "standard mean actor objective; one optimizes only the selected "
            "lower tail."
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
    parser.add_argument(
        "--policy-uncertainty-coef",
        type=float,
        default=0.0,
        help=(
            "Direct actor-loss penalty on imagined ensemble uncertainty. This "
            "is separate from --uncertainty-penalty, which changes imagined "
            "rewards before return computation."
        ),
    )
    parser.add_argument(
        "--policy-real-critic-interval",
        type=int,
        default=0,
        help=(
            "During actor training, run real-replay critic auxiliary updates "
            "every N actor updates. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--policy-real-critic-updates",
        type=int,
        default=1,
        help="Number of real-replay critic auxiliary updates per interval.",
    )
    parser.add_argument(
        "--policy-real-critic-batch-size",
        type=int,
        default=None,
        help="Batch size for real-replay critic auxiliary updates.",
    )
    parser.add_argument(
        "--policy-replay-critic-loss-coef",
        type=float,
        default=0.0,
        help=(
            "Soft replay critic loss coefficient mixed into the same critic "
            "update as imagined lambda-return training. This is gentler than "
            "--policy-real-critic-interval because it does not run a separate "
            "Adam step that can overwrite the critic scale."
        ),
    )
    parser.add_argument(
        "--policy-replay-critic-batch-size",
        type=int,
        default=None,
        help=(
            "Batch size for the soft replay critic loss. Defaults to "
            "--policy-batch-size."
        ),
    )
    parser.add_argument(
        "--policy-replay-critic-horizon",
        type=int,
        default=None,
        help=(
            "Real-replay return horizon for the soft critic loss. Defaults to "
            "--critic-horizon."
        ),
    )
    parser.add_argument(
        "--policy-replay-critic-return-mode",
        choices=("reward-only", "lambda"),
        default="reward-only",
        help="Target construction for the replay critic auxiliary loss.",
    )
    parser.add_argument(
        "--policy-replay-critic-all-steps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train replay values at every sequence state instead of only the first.",
    )
    parser.add_argument(
        "--policy-slow-value-regularization-coef",
        type=float,
        default=0.0,
        help="Cross-entropy regularization toward the EMA target value head.",
    )
    parser.add_argument(
        "--policy-hard-start-max-steps",
        type=int,
        default=0,
        help=(
            "Capacity, in stored transitions, for low-return episode prefixes "
            "used as generic hard starts during actor/critic training. Set 0 "
            "to disable hard-start replay."
        ),
    )
    parser.add_argument(
        "--policy-hard-start-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of each actor-start batch sampled from the hard-start "
            "buffer. The remaining starts come from the normal policy sampler."
        ),
    )
    parser.add_argument(
        "--policy-hard-critic-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of each soft replay-critic batch sampled from hard-start "
            "episode prefixes when --policy-replay-critic-loss-coef is active."
        ),
    )
    parser.add_argument(
        "--policy-hard-start-return-percentile",
        type=float,
        default=30.0,
        help=(
            "Within each real actor-collection block, add episodes at or below "
            "this return percentile to the hard-start buffer. Set 0 together "
            "with no absolute threshold to disable percentile admission."
        ),
    )
    parser.add_argument(
        "--policy-hard-start-absolute-threshold",
        type=float,
        default=None,
        help=(
            "Optional absolute return threshold for hard-start admission. "
            "When a percentile cutoff is also active, the stricter/lower "
            "cutoff is used so solved episodes are not admitted just because "
            "they are relatively low within a strong batch."
        ),
    )
    parser.add_argument(
        "--policy-hard-start-prefix-steps",
        type=int,
        default=64,
        help="Maximum early-episode transitions retained per hard episode.",
    )
    parser.add_argument(
        "--policy-hard-start-recovery-windows",
        type=int,
        default=1,
        help=(
            "Number of early windows retained from each low-return episode. "
            "Values above 1 train recovery from early failure states, not only "
            "from reset prefixes."
        ),
    )
    parser.add_argument(
        "--policy-hard-start-recovery-stride",
        type=int,
        default=8,
        help=(
            "Step offset between retained recovery windows when "
            "--policy-hard-start-recovery-windows is greater than 1."
        ),
    )
    parser.add_argument(
        "--policy-hard-start-mode-buckets",
        type=int,
        default=0,
        help=(
            "Number of task-agnostic observation-space buckets for hard-start "
            "failure modes. 0 disables bucketing."
        ),
    )
    parser.add_argument(
        "--policy-hard-start-balance-modes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Sample hard starts uniformly across non-empty failure-mode buckets "
            "instead of proportional to stored windows."
        ),
    )
    parser.add_argument(
        "--policy-hard-action-bound-coef",
        type=float,
        default=0.0,
        help=(
            "Additional actor action-bound penalty applied only to hard-start "
            "samples, targeting saturated recovery failures without stronger "
            "regularization on normal starts."
        ),
    )
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
    parser.add_argument(
        "--policy-action-bound-coef",
        type=float,
        default=0.0,
        help=(
            "Actor penalty for deterministic mean actions that approach the "
            "normalized action bounds. This targets brittle saturated policies."
        ),
    )
    parser.add_argument(
        "--policy-action-bound-limit",
        type=float,
        default=0.85,
        help=(
            "Start penalizing abs(tanh(action_mean)) above this normalized "
            "limit when --policy-action-bound-coef is positive."
        ),
    )
    parser.add_argument("--policy-eval-episodes", type=int, default=20)
    parser.add_argument("--policy-eval-num-envs", type=int, default=None)
    parser.add_argument(
        "--policy-eval-during-training",
        dest="policy_eval_during_training",
        action="store_true",
        default=True,
        help=(
            "Run random/current/trained real-environment policy evaluations "
            "inside each policy-training phase. These are useful diagnostics "
            "but add real environment interactions."
        ),
    )
    parser.add_argument(
        "--no-policy-eval-during-training",
        dest="policy_eval_during_training",
        action="store_false",
        help=(
            "Skip random/current/trained policy evaluations during training. "
            "Use this for fixed-schedule train-budget runs; final evaluation "
            "is still controlled by --final-policy-eval-episodes."
        ),
    )
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
        "--dreamer-report-window-env-steps",
        type=int,
        default=10_000,
        help=(
            "Compute a Dreamer-style training score from online actor-replay "
            "episodes that finish within the final N real train-replay env "
            "steps before --dreamer-report-budget-env-steps. Set to 0 to "
            "disable this reporting metric."
        ),
    )
    parser.add_argument(
        "--dreamer-report-budget-env-steps",
        type=int,
        default=500_000,
        help=(
            "Real train-replay env-step budget used for Dreamer-style reporting. "
            "Episodes finishing after this budget are excluded from the score."
        ),
    )
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
        "--policy-model-selection-interval",
        type=int,
        default=0,
        help=(
            "During frozen-model actor training, score actor checkpoints every N "
            "updates using a fixed imagined rollout batch and restore the best "
            "model-scored checkpoint. This does not use real environment eval."
        ),
    )
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
        default="policy/imagined_return",
        help="Metric maximized by --policy-model-selection-interval.",
    )
    parser.add_argument(
        "--policy-model-selection-source",
        choices=("policy-starts", "validation-replay"),
        default="policy-starts",
        help=(
            "Source contexts for model-side policy checkpoint selection. "
            "'policy-starts' reuses the actor training start sampler; "
            "'validation-replay' uses held-out real replay contexts."
        ),
    )
    parser.add_argument(
        "--policy-model-selection-batch-size",
        type=int,
        default=None,
        help=(
            "Batch size for model-side checkpoint selection contexts. Defaults "
            "to --policy-batch-size."
        ),
    )
    parser.add_argument(
        "--policy-model-selection-cvar-coef",
        type=float,
        default=0.5,
        help=(
            "Weight added to actor-objective CVaR when computing "
            "policy/heldout_model_score."
        ),
    )
    parser.add_argument(
        "--policy-model-selection-uncertainty-penalty",
        type=float,
        default=0.0,
        help=(
            "Penalty on ensemble uncertainty when computing policy/heldout_model_score."
        ),
    )
    parser.add_argument(
        "--policy-model-selection-action-saturation-penalty",
        type=float,
        default=0.0,
        help=(
            "Penalty on action saturation when computing policy/heldout_model_score."
        ),
    )
    parser.add_argument(
        "--policy-model-selection-diagnostics",
        action="store_true",
        help=(
            "When real-env policy selection is enabled, also score each candidate "
            "checkpoint with the model-side selector and log the paired real/model "
            "records. This is diagnostic only and does not affect selection."
        ),
    )
    parser.add_argument(
        "--policy-selection-std-penalty",
        type=float,
        default=0.0,
        help=(
            "Select frozen-policy checkpoints by mean_return - penalty * "
            "std_return. 0 preserves raw mean-return selection."
        ),
    )
    parser.add_argument(
        "--policy-selection-failure-penalty",
        type=float,
        default=0.0,
        help=(
            "Additional return-unit penalty for low-return failure episodes "
            "during checkpoint selection: score -= penalty * failure_rate."
        ),
    )
    parser.add_argument(
        "--policy-failure-return-threshold",
        type=float,
        default=100.0,
        help="Episode return below this threshold counts as a policy failure.",
    )
    parser.add_argument(
        "--policy-success-return-threshold",
        type=float,
        default=900.0,
        help="Episode return at or above this threshold counts as a success.",
    )
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
        "--online-checkpoint-interval",
        type=int,
        default=0,
        help=(
            "Atomically replace checkpoint_latest every N completed online "
            "phases. Zero disables recovery checkpoints."
        ),
    )
    parser.add_argument(
        "--online-freeze-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the observation encoder during online world-model updates.",
    )
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
        "--online-policy-std-penalty",
        type=float,
        default=0.0,
        help=(
            "Accept online champion updates by policy_trained_mean - penalty * "
            "policy_trained_std. 0 preserves raw mean-return championing."
        ),
    )
    parser.add_argument(
        "--online-policy-failure-penalty",
        type=float,
        default=0.0,
        help=(
            "Additional return-unit penalty for accepting online champions: "
            "score -= penalty * policy_trained_failure_rate."
        ),
    )
    parser.add_argument(
        "--policy-soft-failure-return-threshold",
        type=float,
        default=700.0,
        help=(
            "Return threshold for soft policy failures used by pooled champion "
            "acceptance diagnostics. Episodes at or below this return receive "
            "the soft-failure penalty."
        ),
    )
    parser.add_argument(
        "--policy-soft-failure-penalty",
        type=float,
        default=0.0,
        help=(
            "Return-unit penalty for pooled champion acceptance: score -= "
            "penalty * soft_failure_rate. Zero disables the soft-tail penalty."
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
        "--wandb-project",
        default=None,
        help="Enable optional W&B mirroring under this project.",
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=())
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument(
        "--wandb-videos",
        action="store_true",
        help=(
            "Record environment 0's first evaluation episode at selected "
            "policy phases and upload the encoded MP4 to W&B."
        ),
    )
    parser.add_argument("--wandb-video-every-phases", type=int, default=1)
    parser.add_argument("--wandb-video-frame-stride", type=int, default=4)
    parser.add_argument("--wandb-video-size", type=int, default=64)
    parser.add_argument("--wandb-video-fps", type=int, default=20)
    parser.add_argument("--wandb-video-camera", type=int, default=0)
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
        "optimizer_warmup_steps",
        "policy_selection_interval",
        "policy_model_selection_interval",
        "online_iterations",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.policy_selection_interval > 0 and args.policy_model_selection_interval > 0:
        parser.error(
            "--policy-selection-interval and --policy-model-selection-interval "
            "are mutually exclusive"
        )
    if (
        args.policy_model_selection_batch_size is not None
        and args.policy_model_selection_batch_size < 1
    ):
        parser.error("--policy-model-selection-batch-size must be >= 1")
    for name in (
        "policy_model_selection_cvar_coef",
        "policy_model_selection_uncertainty_penalty",
        "policy_model_selection_action_saturation_penalty",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.policy_model_selection_diagnostics and args.policy_selection_interval <= 0:
        parser.error(
            "--policy-model-selection-diagnostics requires "
            "--policy-selection-interval > 0"
        )
    if args.policy_confirmation_episodes < 0:
        parser.error("--policy-confirmation-episodes must be >= 0")
    if args.final_policy_eval_episodes < 0:
        parser.error("--final-policy-eval-episodes must be >= 0")
    if not args.policy_eval_during_training:
        if args.policy_selection_interval > 0:
            parser.error(
                "--no-policy-eval-during-training requires "
                "--policy-selection-interval 0"
            )
        if args.policy_confirmation_episodes > 0:
            parser.error(
                "--no-policy-eval-during-training requires "
                "--policy-confirmation-episodes 0"
            )
        if args.online_policy_champion:
            parser.error(
                "--no-policy-eval-during-training requires --no-online-policy-champion"
            )
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
    if args.policy_selection_std_penalty < 0.0:
        parser.error("--policy-selection-std-penalty must be >= 0")
    if args.policy_selection_failure_penalty < 0.0:
        parser.error("--policy-selection-failure-penalty must be >= 0")
    if args.policy_failure_return_threshold >= args.policy_success_return_threshold:
        parser.error(
            "--policy-failure-return-threshold must be < "
            "--policy-success-return-threshold"
        )
    if (
        args.policy_soft_failure_return_threshold
        <= args.policy_failure_return_threshold
    ):
        parser.error(
            "--policy-soft-failure-return-threshold must be greater than "
            "--policy-failure-return-threshold"
        )
    if args.online_policy_std_penalty < 0.0:
        parser.error("--online-policy-std-penalty must be >= 0")
    if args.online_policy_failure_penalty < 0.0:
        parser.error("--online-policy-failure-penalty must be >= 0")
    if args.policy_soft_failure_penalty < 0.0:
        parser.error("--policy-soft-failure-penalty must be >= 0")
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
    if args.policy_real_critic_interval < 0:
        parser.error("--policy-real-critic-interval must be >= 0")
    if args.policy_real_critic_updates < 1:
        parser.error("--policy-real-critic-updates must be >= 1")
    if (
        args.policy_real_critic_batch_size is not None
        and args.policy_real_critic_batch_size < 1
    ):
        parser.error("--policy-real-critic-batch-size must be >= 1")
    if args.policy_replay_critic_loss_coef < 0.0:
        parser.error("--policy-replay-critic-loss-coef must be >= 0")
    if args.policy_slow_value_regularization_coef < 0.0:
        parser.error("--policy-slow-value-regularization-coef must be >= 0")
    if (
        args.policy_slow_value_regularization_coef > 0.0
        and args.target_critic_ema_decay <= 0.0
    ):
        parser.error(
            "--policy-slow-value-regularization-coef requires "
            "--target-critic-ema-decay > 0"
        )
    if (
        args.policy_replay_critic_batch_size is not None
        and args.policy_replay_critic_batch_size < 1
    ):
        parser.error("--policy-replay-critic-batch-size must be >= 1")
    if (
        args.policy_replay_critic_horizon is not None
        and args.policy_replay_critic_horizon < 1
    ):
        parser.error("--policy-replay-critic-horizon must be >= 1")
    if args.policy_hard_start_max_steps < 0:
        parser.error("--policy-hard-start-max-steps must be >= 0")
    for name in ("policy_hard_start_fraction", "policy_hard_critic_fraction"):
        value = getattr(args, name)
        if not 0.0 <= value < 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1)")
    if not 0.0 <= args.policy_hard_start_return_percentile <= 100.0:
        parser.error("--policy-hard-start-return-percentile must be in [0, 100]")
    if args.policy_hard_start_prefix_steps < 1:
        parser.error("--policy-hard-start-prefix-steps must be >= 1")
    if args.policy_hard_start_recovery_windows < 1:
        parser.error("--policy-hard-start-recovery-windows must be >= 1")
    if args.policy_hard_start_recovery_stride < 1:
        parser.error("--policy-hard-start-recovery-stride must be >= 1")
    if args.policy_hard_start_mode_buckets < 0:
        parser.error("--policy-hard-start-mode-buckets must be >= 0")
    if args.policy_hard_start_max_steps == 0 and (
        args.policy_hard_start_fraction > 0.0 or args.policy_hard_critic_fraction > 0.0
    ):
        parser.error(
            "--policy-hard-start-max-steps must be > 0 when hard-start "
            "sampling fractions are enabled"
        )
    if args.policy_trust_coef < 0.0:
        parser.error("--policy-trust-coef must be >= 0")
    if (
        args.online_policy_trust_coef is not None
        and args.online_policy_trust_coef < 0.0
    ):
        parser.error("--online-policy-trust-coef must be >= 0")
    if args.policy_action_bound_coef < 0.0:
        parser.error("--policy-action-bound-coef must be >= 0")
    if args.policy_hard_action_bound_coef < 0.0:
        parser.error("--policy-hard-action-bound-coef must be >= 0")
    if not 0.0 < args.policy_action_bound_limit <= 1.0:
        parser.error("--policy-action-bound-limit must be in (0, 1]")
    if not 0.0 < args.policy_actor_cvar_fraction <= 1.0:
        parser.error("--policy-actor-cvar-fraction must be in (0, 1]")
    if args.policy_actor_cvar_coef < 0.0:
        parser.error("--policy-actor-cvar-coef must be >= 0")
    if args.value_clip <= 0.0:
        parser.error("--value-clip must be > 0")
    if not (0.0 <= args.target_critic_ema_decay < 1.0):
        parser.error("--target-critic-ema-decay must be in [0, 1)")
    if args.actor_entropy_coef < 0.0:
        parser.error("--actor-entropy-coef must be >= 0")
    if not 0.0 <= args.policy_return_ema_decay < 1.0:
        parser.error("--policy-return-ema-decay must be in [0, 1)")
    if args.policy_gradient_mode == "reinforce" and not args.stochastic_actor:
        parser.error("--policy-gradient-mode reinforce requires --stochastic-actor")
    if args.actor_log_std_min >= args.actor_log_std_max:
        parser.error("--actor-log-std-min must be < --actor-log-std-max")
    if args.stochastic_collection and not args.stochastic_actor:
        parser.error("--stochastic-collection requires --stochastic-actor")
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
        "adaptive_grad_clip",
        "actor_output_scale",
        "value_output_scale",
        "reward_output_scale",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.optimizer_epsilon <= 0.0:
        parser.error("--optimizer-epsilon must be > 0")
    for name in ("actor_hidden_dim", "critic_hidden_dim"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    for name in ("actor_num_layers", "critic_num_layers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if not 0.0 < args.action_saturation_threshold <= 1.0:
        parser.error("--action-saturation-threshold must be in (0, 1]")
    for name in (
        "uncertainty_penalty",
        "uncertainty_latent_weight",
        "uncertainty_reward_weight",
        "uncertainty_continue_weight",
        "uncertainty_threshold",
        "uncertainty_budget",
        "policy_uncertainty_coef",
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
    if not (args.env.startswith("dmc:") or args.env.startswith("brax:")):
        parser.error("--env must be formatted as dmc:<domain>/<task> or brax:<env>")
    min_steps = min_sequence_steps
    if args.collect_steps < min_steps:
        parser.error(
            "--collect-steps must cover chunk-length + max model/open-loop horizon"
        )
    if args.validation_steps < min_steps:
        parser.error(
            "--validation-steps must cover chunk-length + max model/open-loop horizon"
        )
    if args.load_initial_replay is not None and not args.load_initial_replay.exists():
        parser.error(
            f"--load-initial-replay does not exist: {args.load_initial_replay}"
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
    if args.online_checkpoint_interval < 0:
        parser.error("--online-checkpoint-interval must be >= 0")
    for name in (
        "wandb_video_every_phases",
        "wandb_video_frame_stride",
        "wandb_video_size",
        "wandb_video_fps",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.wandb_video_camera < 0:
        parser.error("--wandb-video-camera must be >= 0")
    if args.wandb_videos and not args.wandb_project:
        parser.error("--wandb-videos requires --wandb-project")
    if args.wandb_videos and args.wandb_mode == "disabled":
        parser.error("--wandb-videos cannot be used with --wandb-mode disabled")
    if args.wandb_videos and not args.env.startswith("dmc:"):
        parser.error("--wandb-videos currently supports only DMC environments")


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
    raise ValueError(f"unsupported env: {env!r}")


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
    raise ValueError(f"unsupported env: {args.env!r}")


def _wandb_run_config(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    seed: int,
    run_index: int,
    control: ControlMode,
) -> WandbConfig | None:
    if not args.wandb_project or args.wandb_mode == "disabled":
        return None
    suffix = f"{control}-seed{seed}"
    if args.wandb_name:
        run_name = args.wandb_name
        if args.num_runs > 1 or len(args.controls) > 1:
            run_name = f"{run_name}-{suffix}"
    else:
        env_name = args.env.replace(":", "-").replace("/", "-")
        run_name = f"{env_name}-{suffix}"
    return WandbConfig(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        group=args.wandb_group or run_dir.parents[1].name,
        tags=tuple(args.wandb_tags),
        mode=args.wandb_mode,
        config={
            "args": vars(args),
            "run_index": run_index,
            "seed": seed,
            "control": control,
        },
    )


def run_one(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    run_index: int,
    control: ControlMode,
) -> dict[str, Any]:
    seed = args.seed + 10_000 * run_index
    logger = RunLogger(
        run_dir,
        wandb_config=_wandb_run_config(
            args,
            run_dir=run_dir,
            seed=seed,
            run_index=run_index,
            control=control,
        ),
    )
    adapter = None
    completed = False
    try:
        adapter = _make_vector_adapter(args, seed=seed)
        config = JepaConfig(
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
            clip_imagined_rewards=args.clip_imagined_rewards,
            imagined_reward_min=args.imagined_reward_min,
            imagined_reward_max=args.imagined_reward_max,
            dynamics_ensemble_size=args.dynamics_ensemble_size,
            gamma=args.gamma,
            lambda_return=args.lambda_return,
            residual_dynamics=args.residual_dynamics,
            target_gradient=args.target_gradient,
        )
        resolved_config = {
            "args": vars(args),
            "run_index": run_index,
            "seed": seed,
            "control": control,
            "observation_shape": adapter.observation_shape,
            "action_shape": adapter.action_shape,
            "action_low": adapter.action_low,
            "action_high": adapter.action_high,
            "env_backend": _env_backend(args.env),
            "jepa_config": dataclasses.asdict(config),
            "protocol": (
                "heldout_world_model_validation_with_optional_frozen_policy"
                "_and_online_actor_replay"
            ),
        }
        logger.write_json("config.json", resolved_config)
        logger.update_config(resolved_config)
        logger.write_json("versions.json", dependency_versions())

        rng = jax.random.PRNGKey(seed)
        rng, init_key = jax.random.split(rng)
        state = create_jepa_train_state(init_key, config)
        np_rng = np.random.default_rng(seed)
        observations = adapter.reset()
        replay_source = "collected"
        if args.load_initial_replay is not None:
            replay = SequenceReplayBuffer.load_npz(
                args.load_initial_replay,
                capacity=max(
                    2,
                    math.ceil(args.replay_capacity / args.num_envs),
                    args.collect_steps,
                ),
            )
            anchor_replay = SequenceReplayBuffer.load_npz(
                args.load_initial_replay,
                capacity=max(2, args.collect_steps, replay.size),
            )
            if replay.num_envs != args.num_envs:
                raise ValueError(
                    "--load-initial-replay num_envs does not match --num-envs: "
                    f"{replay.num_envs} != {args.num_envs}"
                )
            if replay.observation_shape != (config.observation_dim,):
                raise ValueError(
                    "--load-initial-replay observation shape does not match env: "
                    f"{replay.observation_shape} != {(config.observation_dim,)}"
                )
            if replay.action_shape != (adapter.action_dim,):
                raise ValueError(
                    "--load-initial-replay action shape does not match env: "
                    f"{replay.action_shape} != {(adapter.action_dim,)}"
                )
            env_steps = replay.size * replay.num_envs
            replay_source = "loaded"
        else:
            replay = SequenceReplayBuffer(
                capacity=max(2, math.ceil(args.replay_capacity / args.num_envs)),
                num_envs=args.num_envs,
                observation_shape=(config.observation_dim,),
                action_shape=(adapter.action_dim,),
                action_dtype=np.float32,
            )
            anchor_replay = SequenceReplayBuffer(
                capacity=max(2, args.collect_steps),
                num_envs=args.num_envs,
                observation_shape=(config.observation_dim,),
                action_shape=(adapter.action_dim,),
                action_dtype=np.float32,
            )
            observations, env_steps = _collect_random_steps(
                adapter,
                observations,
                np_rng,
                (replay, anchor_replay),
                steps=args.collect_steps,
                desc=f"{control} collect train replay",
                quiet=args.quiet,
            )
            if args.save_initial_replay is not None:
                args.save_initial_replay.parent.mkdir(parents=True, exist_ok=True)
                anchor_replay.save_npz(args.save_initial_replay)
        logger.write_json(
            "train_replay.json",
            {
                "env_steps": env_steps,
                "steps_per_env": replay.size,
                "size_per_env": replay.size,
                "anchor_size_per_env": anchor_replay.size,
                "observation_dim": config.observation_dim,
                "action_dim": config.action_dim,
                "source": replay_source,
                "loaded_initial_replay": (
                    str(args.load_initial_replay)
                    if args.load_initial_replay is not None
                    else None
                ),
                "saved_initial_replay": (
                    str(args.save_initial_replay)
                    if args.save_initial_replay is not None
                    else None
                ),
            },
        )
        initial_train_replay_env_steps = env_steps
        logger.set_train_env_steps(initial_train_replay_env_steps)
        hard_start_replay = (
            HardStartReplayBuffer(
                max_steps=args.policy_hard_start_max_steps,
                observation_shape=(config.observation_dim,),
                action_shape=(adapter.action_dim,),
                mode_buckets=args.policy_hard_start_mode_buckets,
                balance_modes=args.policy_hard_start_balance_modes,
            )
            if args.policy_hard_start_max_steps > 0
            else None
        )
        logger.write_json(
            "hard_start_replay.json",
            (
                {
                    "hard_start_buffer_enabled": False,
                    "hard_start_max_steps": 0,
                }
                if hard_start_replay is None
                else hard_start_replay.summary()
            ),
        )

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
            policy_validation_replay=validation_replay,
        )
        state = policy_outcome["state"]
        rng = policy_outcome["rng"]
        initial_policy_outcome = policy_outcome["outcome"]
        champion_state = state
        champion_policy_outcome = dict(initial_policy_outcome)
        champion_policy_return = champion_policy_outcome.get("policy_trained_mean")
        champion_policy_score = _policy_outcome_score(
            champion_policy_outcome,
            std_penalty=args.online_policy_std_penalty,
            failure_penalty=args.online_policy_failure_penalty,
        )
        champion_policy_iteration = 0
        train_replay_env_steps = initial_train_replay_env_steps
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
                action_dim=adapter.action_dim,
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
                train_env_step_offset=train_replay_env_steps,
                failure_return_threshold=args.policy_failure_return_threshold,
                success_return_threshold=args.policy_success_return_threshold,
                hard_start_replay=hard_start_replay,
                hard_start_return_percentile=args.policy_hard_start_return_percentile,
                hard_start_absolute_threshold=(
                    args.policy_hard_start_absolute_threshold
                ),
                hard_start_prefix_steps=args.policy_hard_start_prefix_steps,
                hard_start_recovery_windows=args.policy_hard_start_recovery_windows,
                hard_start_recovery_stride=args.policy_hard_start_recovery_stride,
            )
            env_steps += added_env_steps
            train_replay_env_steps += added_env_steps
            logger.set_train_env_steps(train_replay_env_steps)
            collect_payload = {
                **collect_metrics,
                "reset_env_before_collection": args.online_reset_replay_env,
                "total_env_steps": env_steps,
                "train_replay_total_env_steps": train_replay_env_steps,
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
            recent_validation_replay = None
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
                    action_dim=adapter.action_dim,
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
                        failure_return_threshold=args.policy_failure_return_threshold,
                        success_return_threshold=args.policy_success_return_threshold,
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
                        freeze_encoder=args.online_freeze_encoder,
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
                    hard_start_replay=hard_start_replay,
                    policy_validation_replay=(
                        recent_validation_replay or validation_replay
                    ),
                )
                candidate_policy_state = online_policy_outcome["state"]
                rng = online_policy_outcome["rng"]
                candidate_policy_payload = online_policy_outcome["outcome"]
                candidate_policy_return = candidate_policy_payload.get(
                    "policy_trained_mean"
                )
                candidate_policy_score = _policy_outcome_score(
                    candidate_policy_payload,
                    std_penalty=args.online_policy_std_penalty,
                    failure_penalty=args.online_policy_failure_penalty,
                )
                previous_champion_return = champion_policy_return
                previous_champion_score = champion_policy_score
                policy_update_accepted = True
                if args.online_policy_champion:
                    policy_update_accepted = candidate_policy_score is not None and (
                        previous_champion_score is None
                        or candidate_policy_score
                        >= previous_champion_score
                        - args.online_policy_champion_tolerance
                    )
                if policy_update_accepted:
                    state = candidate_policy_state
                    champion_state = state
                    champion_policy_outcome = dict(candidate_policy_payload)
                    champion_policy_return = candidate_policy_return
                    champion_policy_score = candidate_policy_score
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
                        "policy_candidate_score": candidate_policy_score,
                        "policy_previous_champion_mean": previous_champion_return,
                        "policy_previous_champion_score": previous_champion_score,
                        "policy_champion_return": champion_policy_return,
                        "policy_champion_score": champion_policy_score,
                        "policy_champion_iteration": champion_policy_iteration,
                        "policy_champion_tolerance": (
                            args.online_policy_champion_tolerance
                        ),
                        "policy_champion_std_penalty": args.online_policy_std_penalty,
                        "policy_champion_failure_penalty": (
                            args.online_policy_failure_penalty
                        ),
                    }
                )
                candidate_policy_payload = {
                    **candidate_policy_payload,
                    "policy_champion_enabled": args.online_policy_champion,
                    "policy_update_accepted": policy_update_accepted,
                    "policy_candidate_trained_mean": candidate_policy_return,
                    "policy_candidate_score": candidate_policy_score,
                    "policy_previous_champion_mean": previous_champion_return,
                    "policy_previous_champion_score": previous_champion_score,
                    "policy_champion_return": champion_policy_return,
                    "policy_champion_score": champion_policy_score,
                    "policy_champion_iteration": champion_policy_iteration,
                    "policy_champion_tolerance": args.online_policy_champion_tolerance,
                    "policy_champion_std_penalty": args.online_policy_std_penalty,
                    "policy_champion_failure_penalty": (
                        args.online_policy_failure_penalty
                    ),
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
                        "policy_candidate_score": candidate_policy_score,
                        "policy_previous_champion_mean": previous_champion_return,
                        "policy_previous_champion_score": previous_champion_score,
                        "policy_champion_return": champion_policy_return,
                        "policy_champion_score": champion_policy_score,
                        "policy_champion_iteration": champion_policy_iteration,
                        "policy_champion_std_penalty": args.online_policy_std_penalty,
                        "policy_champion_failure_penalty": (
                            args.online_policy_failure_penalty
                        ),
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
            if args.online_checkpoint_interval > 0 and (
                online_index % args.online_checkpoint_interval == 0
                or online_index == args.online_iterations
            ):
                try:
                    save_checkpoint(
                        run_dir / "checkpoint_latest",
                        state,
                        metadata={
                            "algorithm": "single_agent_sigreg_jepa_world_model",
                            "checkpoint_kind": "online_recovery",
                            "env": args.env,
                            "env_backend": _env_backend(args.env),
                            "control": control,
                            "jepa_config": dataclasses.asdict(config),
                            "online_iteration": online_index,
                            "seed": seed,
                            "train_replay_env_steps": train_replay_env_steps,
                        },
                    )
                except OSError as error:
                    warnings.warn(
                        "Recovery checkpoint write failed; training will "
                        f"continue: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        if online_history:
            policy_outcome["outcome"] = _merge_online_policy_baseline(
                policy_outcome["outcome"],
                initial_policy_outcome,
            )
            policy_outcome["outcome"].update(
                _online_history_metrics(online_history, initial_policy_outcome)
            )
        if online_history:
            logger.write_json("online_history.json", online_history)

        dreamer_style_training_score = _dreamer_style_training_score(
            online_history,
            window_env_steps=args.dreamer_report_window_env_steps,
            budget_env_steps=args.dreamer_report_budget_env_steps,
        )
        logger.write_json(
            "dreamer_style_training_score.json",
            dreamer_style_training_score,
        )
        if dreamer_style_training_score["enabled"]:
            logger.append_metrics(
                {
                    "phase": "dreamer_style_training_score",
                    "control": control,
                    **dreamer_style_training_score,
                }
            )

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

        checkpoint_dir = run_dir / "checkpoint"
        checkpoint_metadata = {
            "algorithm": "single_agent_sigreg_jepa_world_model",
            "checkpoint_kind": "final",
            "env": args.env,
            "env_backend": _env_backend(args.env),
            "control": control,
            "policy_trained": args.policy_train_steps > 0,
            "jepa_config": dataclasses.asdict(config),
            "seed": seed,
            "train_replay_env_steps": train_replay_env_steps,
        }
        try:
            save_checkpoint(
                checkpoint_dir,
                state,
                metadata=checkpoint_metadata,
            )
        except OSError as error:
            warnings.warn(
                f"Final checkpoint write failed; final evaluation will continue: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            recovery_checkpoint_dir = run_dir / "checkpoint_latest"
            if (recovery_checkpoint_dir / "checkpoint.msgpack").is_file():
                checkpoint_dir = recovery_checkpoint_dir
        try:
            reload_diff = _reload_prediction_diff(
                state,
                config,
                checkpoint_dir=checkpoint_dir,
                batch=final_batch,
                seed=seed + 99,
                chunk_length=args.chunk_length,
            )
        except OSError as error:
            warnings.warn(
                f"Checkpoint reload validation failed; final evaluation will "
                f"continue: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            reload_diff = float("inf")
        reload = {"reload_max_abs_prediction_diff": reload_diff}
        logger.write_json("reload_evaluation.json", reload)

        final_policy_eval = None
        if args.final_policy_eval_episodes > 0:
            final_policy_eval_num_envs = args.final_policy_eval_num_envs or min(
                args.num_envs,
                args.final_policy_eval_episodes,
            )
            final_policy_eval = _evaluate_continuous_policy(
                args,
                state,
                config,
                seed=seed + 9_000_000,
                num_envs=final_policy_eval_num_envs,
                episodes=args.final_policy_eval_episodes,
                action_low=jnp.asarray(adapter.action_low, dtype=jnp.float32),
                action_high=jnp.asarray(adapter.action_high, dtype=jnp.float32),
                desc=f"{control} final eval champion policy",
                **(
                    {
                        "video_logger": logger,
                        "video_filename": "videos/final_champion.mp4",
                        "video_key": "videos/final/champion",
                        "video_caption": "Final champion policy evaluation",
                    }
                    if args.wandb_videos
                    else {}
                ),
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

        world_model_passed = _run_passed(initial_metrics, final_metrics, reload_diff)
        outcome = {
            "run_index": run_index,
            "control": control,
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "target": (
                f"{_env_backend(args.env)}:"
                "p(z_next, reward, continue | z, continuous_action)"
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
            "final_policy_eval_failure_rate": (
                final_policy_eval.get("failure_rate") if final_policy_eval else None
            ),
            "final_policy_eval_success_rate": (
                final_policy_eval.get("success_rate") if final_policy_eval else None
            ),
            "final_policy_eval_return_p10": (
                final_policy_eval.get("return_p10") if final_policy_eval else None
            ),
            "final_policy_eval_return_cvar10": (
                final_policy_eval.get("return_cvar10") if final_policy_eval else None
            ),
            "final_policy_eval_env_steps": (
                final_policy_eval.get("env_steps") if final_policy_eval else None
            ),
            "final_policy_eval_completed_episode_steps": (
                final_policy_eval.get("completed_episode_steps")
                if final_policy_eval
                else None
            ),
            "dreamer_style_training_score": dreamer_style_training_score,
            "dreamer_style_train_return_mean": dreamer_style_training_score.get(
                "mean_return"
            ),
            "dreamer_style_train_return_std": dreamer_style_training_score.get(
                "std_return"
            ),
            "dreamer_style_train_return_episodes": dreamer_style_training_score.get(
                "episodes"
            ),
            "dreamer_style_train_return_window_start_env_step": (
                dreamer_style_training_score.get("window_start_env_step")
            ),
            "dreamer_style_train_return_window_end_env_step": (
                dreamer_style_training_score.get("window_end_env_step")
            ),
            "dreamer_style_train_return_budget_reached": (
                dreamer_style_training_score.get("budget_reached")
            ),
            "online_iterations": args.online_iterations,
            "online_history": online_history,
            "hard_start_replay_final": (
                None if hard_start_replay is None else hard_start_replay.summary()
            ),
            **policy_outcome["outcome"],
            "world_model_passed": world_model_passed,
            "passed": world_model_passed,
        }
        outcome.update(
            _real_step_accounting(
                initial_train_replay_env_steps=initial_train_replay_env_steps,
                initial_validation_env_steps=initial_validation_env_steps,
                initial_policy_outcome=initial_policy_outcome,
                online_history=online_history,
                final_policy_eval=final_policy_eval,
            )
        )
        logger.write_json("outcome.json", outcome)
        final_row = {
            "phase": "run_outcome",
            "control": control,
            "budget/train_env_steps": outcome["real_train_replay_env_steps"],
            "budget/validation_env_steps": outcome["real_validation_replay_env_steps"],
            "budget/policy_eval_env_steps": outcome["real_policy_eval_env_steps"],
            "budget/total_real_env_steps": outcome["real_total_env_steps"],
            "model/final_jepa_loss": outcome["final_jepa_loss"],
            "model/final_open_loop_loss": outcome["final_open_loop_loss"],
            "model/final_reward_loss": outcome["final_reward_loss"],
            "model/final_continue_loss": outcome["final_continue_loss"],
            "run/world_model_passed": outcome["world_model_passed"],
            "run/passed": outcome["passed"],
        }
        if final_policy_eval is not None:
            final_row.update(
                {
                    "eval/return_mean": final_policy_eval["mean_return"],
                    "eval/return_std": final_policy_eval["std_return"],
                    "eval/return_p10": final_policy_eval.get("return_p10"),
                    "eval/return_cvar10": final_policy_eval.get("return_cvar10"),
                    "eval/failure_rate": final_policy_eval.get("failure_rate"),
                    "eval/success_rate": final_policy_eval.get("success_rate"),
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
    action_low: np.ndarray,
    action_high: np.ndarray,
    desc: str,
    quiet: bool,
    np_rng: np.random.Generator,
    stochastic_actions: bool = False,
    train_env_step_offset: int | None = None,
    failure_return_threshold: float = 100.0,
    success_return_threshold: float = 900.0,
    hard_start_replay: HardStartReplayBuffer | None = None,
    hard_start_return_percentile: float = 30.0,
    hard_start_absolute_threshold: float | None = None,
    hard_start_prefix_steps: int = 64,
    hard_start_recovery_windows: int = 1,
    hard_start_recovery_stride: int = 8,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    action_low_jax = jnp.asarray(action_low, dtype=jnp.float32)
    action_high_jax = jnp.asarray(action_high, dtype=jnp.float32)
    action_key = jax.random.PRNGKey(int(np_rng.integers(0, 2**31 - 1)))
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    hard_episode_records: list[dict[str, Any]] = []
    if hard_start_replay is None:
        episode_observations = episode_actions = episode_rewards = episode_dones = None
    else:
        episode_observations = [[] for _ in range(adapter.num_envs)]
        episode_actions = [[] for _ in range(adapter.num_envs)]
        episode_rewards = [[] for _ in range(adapter.num_envs)]
        episode_dones = [[] for _ in range(adapter.num_envs)]
    episode_finish_collection_env_steps: list[int] = []
    episode_finish_train_env_steps: list[int] = []
    progress = tqdm(range(steps), desc=desc, unit="step", disable=quiet)
    for step_index in progress:
        action_key, step_action_key = jax.random.split(action_key)
        obs_t = np.asarray(observations[:, 0], dtype=np.float32)
        actions = np.asarray(
            select_continuous_actions(
                state,
                jnp.asarray(obs_t, dtype=jnp.float32),
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
        if hard_start_replay is not None:
            rewards_t = np.asarray(step.rewards[:, 0], dtype=np.float32)
            dones_t = np.asarray(step.dones[:, 0], dtype=np.float32)
            for env_index in range(adapter.num_envs):
                episode_observations[env_index].append(obs_t[env_index].copy())
                episode_actions[env_index].append(actions[env_index].copy())
                episode_rewards[env_index].append(float(rewards_t[env_index]))
                episode_dones[env_index].append(float(dones_t[env_index]))
        completed_count = len(step.completed_returns)
        completed_returns.extend(float(item[0]) for item in step.completed_returns)
        completed_lengths.extend(int(item) for item in step.completed_lengths)
        if completed_count:
            completed_envs = _completed_env_indices(step, completed_count)
            local_finish_step = (step_index + 1) * adapter.num_envs
            episode_finish_collection_env_steps.extend(
                [local_finish_step] * completed_count
            )
            if train_env_step_offset is not None:
                episode_finish_train_env_steps.extend(
                    [train_env_step_offset + local_finish_step] * completed_count
                )
            if hard_start_replay is not None:
                for env_index in completed_envs:
                    rewards = np.asarray(episode_rewards[env_index], dtype=np.float32)
                    if rewards.size:
                        hard_episode_records.append(
                            {
                                "observations": np.asarray(
                                    episode_observations[env_index],
                                    dtype=np.float32,
                                ),
                                "actions": np.asarray(
                                    episode_actions[env_index],
                                    dtype=np.float32,
                                ),
                                "rewards": rewards,
                                "dones": np.asarray(
                                    episode_dones[env_index],
                                    dtype=np.float32,
                                ),
                                "return": float(np.sum(rewards)),
                            }
                        )
                    episode_observations[env_index] = []
                    episode_actions[env_index] = []
                    episode_rewards[env_index] = []
                    episode_dones[env_index] = []
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
        "episode_finish_collection_env_steps": episode_finish_collection_env_steps,
        **_return_tail_metrics(
            completed_returns,
            failure_threshold=failure_return_threshold,
            success_threshold=success_return_threshold,
        ),
    }
    if hard_start_replay is not None:
        metrics["hard_start_replay_update"] = hard_start_replay.add_completed_episodes(
            hard_episode_records,
            return_percentile=hard_start_return_percentile,
            absolute_threshold=hard_start_absolute_threshold,
            max_prefix_steps=hard_start_prefix_steps,
            recovery_windows=hard_start_recovery_windows,
            recovery_stride=hard_start_recovery_stride,
        )
    if train_env_step_offset is not None:
        metrics.update(
            {
                "train_env_step_offset": int(train_env_step_offset),
                "episode_finish_train_env_steps": episode_finish_train_env_steps,
            }
        )
    return observations, steps * adapter.num_envs, metrics


def _completed_env_indices(step, completed_count: int) -> list[int]:
    infos = getattr(step, "infos", ())
    info_envs = [
        int(info["env_index"])
        for info in infos
        if isinstance(info, dict) and "env_index" in info
    ]
    if len(info_envs) == completed_count:
        return info_envs
    dones = np.asarray(step.dones).reshape((-1,))
    done_envs = np.flatnonzero(dones > 0.5).astype(np.int64).tolist()
    return [int(item) for item in done_envs[:completed_count]]


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
        replay = SequenceReplayBuffer(
            capacity=max(2, args.validation_steps),
            num_envs=args.num_envs,
            observation_shape=(config.observation_dim,),
            action_shape=(config.action_dim,),
            action_dtype=np.float32,
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
    action_dim: int,
) -> SequenceReplayBuffer:
    return SequenceReplayBuffer(
        capacity=max(2, capacity),
        num_envs=num_envs,
        observation_shape=(observation_dim,),
        action_shape=(action_dim,),
        action_dtype=np.float32,
    )


class HardStartReplayBuffer:
    """Stores low-return episode prefixes for task-agnostic hard-start training."""

    def __init__(
        self,
        *,
        max_steps: int,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        mode_buckets: int = 0,
        balance_modes: bool = False,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.max_steps = int(max_steps)
        self.observation_shape = tuple(int(dim) for dim in observation_shape)
        self.action_shape = tuple(int(dim) for dim in action_shape)
        self.mode_buckets = int(mode_buckets)
        self.balance_modes = bool(balance_modes and self.mode_buckets > 1)
        if self.mode_buckets > 1:
            feature_dim = int(np.prod(self.observation_shape))
            rng = np.random.default_rng(17)
            projection = rng.normal(size=(feature_dim, self.mode_buckets))
            self._mode_projection = projection.astype(np.float32)
        else:
            self._mode_projection = None
        self._episodes: list[dict[str, Any]] = []
        self._steps = 0

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def episodes(self) -> int:
        return len(self._episodes)

    def add_completed_episodes(
        self,
        episodes: list[dict[str, Any]],
        *,
        return_percentile: float,
        absolute_threshold: float | None,
        max_prefix_steps: int,
        recovery_windows: int = 1,
        recovery_stride: int = 8,
    ) -> dict[str, Any]:
        returns = np.asarray([item["return"] for item in episodes], dtype=np.float32)
        percentile_cutoff = None
        if returns.size and return_percentile > 0.0:
            percentile_cutoff = float(np.quantile(returns, return_percentile / 100.0))
        if percentile_cutoff is not None and absolute_threshold is not None:
            effective_cutoff = min(percentile_cutoff, float(absolute_threshold))
        elif percentile_cutoff is not None:
            effective_cutoff = percentile_cutoff
        elif absolute_threshold is not None:
            effective_cutoff = float(absolute_threshold)
        else:
            effective_cutoff = None

        admitted = 0
        admitted_segments = 0
        admitted_returns: list[float] = []
        for episode in episodes:
            episode_return = float(episode["return"])
            keep = effective_cutoff is not None and episode_return <= effective_cutoff
            if keep:
                added = self.add_episode(
                    observations=episode["observations"],
                    actions=episode["actions"],
                    rewards=episode["rewards"],
                    dones=episode["dones"],
                    episode_return=episode_return,
                    max_prefix_steps=max_prefix_steps,
                    recovery_windows=recovery_windows,
                    recovery_stride=recovery_stride,
                )
                if added:
                    admitted += 1
                    admitted_segments += int(added)
                    admitted_returns.append(episode_return)

        summary = self.summary()
        return {
            **summary,
            "candidate_episodes": int(len(episodes)),
            "candidate_return_percentile": float(return_percentile),
            "candidate_percentile_cutoff": percentile_cutoff,
            "candidate_absolute_threshold": (
                None if absolute_threshold is None else float(absolute_threshold)
            ),
            "candidate_effective_cutoff": effective_cutoff,
            "admitted_episodes": int(admitted),
            "admitted_segments": int(admitted_segments),
            "admitted_fraction": (
                float(admitted / len(episodes)) if episodes else None
            ),
            "admitted_mean_return": (
                float(np.mean(admitted_returns)) if admitted_returns else None
            ),
            "admitted_return_p90": (
                float(np.quantile(np.asarray(admitted_returns), 0.90))
                if admitted_returns
                else None
            ),
            "hard_start_admitted_fraction": (
                float(admitted / len(episodes)) if episodes else None
            ),
            "hard_start_admitted_return_mean": (
                float(np.mean(admitted_returns)) if admitted_returns else None
            ),
            "hard_start_admitted_return_p90": (
                float(np.quantile(np.asarray(admitted_returns), 0.90))
                if admitted_returns
                else None
            ),
        }

    def add_episode(
        self,
        *,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        episode_return: float,
        max_prefix_steps: int,
        recovery_windows: int = 1,
        recovery_stride: int = 8,
    ) -> int:
        total_length = min(
            int(len(observations)),
            int(len(actions)),
            int(len(rewards)),
            int(len(dones)),
        )
        if total_length < 1:
            return 0
        starts = [
            start
            for start in (
                int(index) * int(recovery_stride)
                for index in range(max(1, int(recovery_windows)))
            )
            if start < total_length
        ]
        added = 0
        for source_start in starts:
            length = min(int(total_length - source_start), int(max_prefix_steps))
            if length < 1:
                continue
            stop = source_start + length
            segment_observations = np.asarray(
                observations[source_start:stop],
                dtype=np.float32,
            ).reshape((length, *self.observation_shape))
            segment_actions = np.asarray(
                actions[source_start:stop],
                dtype=np.float32,
            ).reshape((length, *self.action_shape))
            segment_rewards = np.asarray(
                rewards[source_start:stop],
                dtype=np.float32,
            ).reshape((length,))
            segment_dones = np.asarray(
                dones[source_start:stop],
                dtype=np.float32,
            ).reshape((length,))
            if not (
                np.all(np.isfinite(segment_observations))
                and np.all(np.isfinite(segment_actions))
                and np.all(np.isfinite(segment_rewards))
            ):
                continue
            self._episodes.append(
                {
                    "observations": segment_observations,
                    "actions": segment_actions,
                    "rewards": segment_rewards,
                    "dones": segment_dones,
                    "return": float(episode_return),
                    "length": int(length),
                    "source_start": int(source_start),
                    "mode_bucket": self._mode_bucket(segment_observations),
                }
            )
            self._steps += int(length)
            added += 1
        self._trim()
        return added

    def can_sample_starts(self, *, context_window: int) -> bool:
        return any(item["length"] >= context_window for item in self._episodes)

    def sample_starts(
        self,
        rng: np.random.Generator,
        *,
        batch_size: int,
        context_window: int,
    ) -> tuple[jax.Array, jax.Array]:
        indices = self._sample_episode_windows(
            rng,
            batch_size=batch_size,
            sequence_length=context_window,
        )
        observations = []
        actions = []
        for episode_index, start in indices:
            episode = self._episodes[episode_index]
            end = start + context_window
            observations.append(episode["observations"][start:end])
            actions.append(episode["actions"][start:end])
        return (
            jnp.asarray(np.stack(observations, axis=0), dtype=jnp.float32),
            jnp.asarray(np.stack(actions, axis=0), dtype=jnp.float32),
        )

    def can_sample_batch(self, *, chunk_length: int, max_horizon: int) -> bool:
        return any(
            item["length"] >= chunk_length + max_horizon for item in self._episodes
        )

    def sample_batch(
        self,
        rng: np.random.Generator,
        *,
        batch_size: int,
        chunk_length: int,
        max_horizon: int,
    ) -> ReplayBatch:
        sequence_length = chunk_length + max_horizon
        indices = self._sample_episode_windows(
            rng,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        observations = []
        actions = []
        rewards = []
        dones = []
        for episode_index, start in indices:
            episode = self._episodes[episode_index]
            obs_end = start + sequence_length
            trans_end = start + sequence_length - 1
            observations.append(episode["observations"][start:obs_end])
            actions.append(episode["actions"][start:trans_end])
            rewards.append(episode["rewards"][start:trans_end])
            dones.append(episode["dones"][start:trans_end])
        return ReplayBatch(
            observations=jnp.asarray(np.stack(observations, axis=0), dtype=jnp.float32),
            actions=jnp.asarray(np.stack(actions, axis=0), dtype=jnp.float32),
            rewards=jnp.asarray(np.stack(rewards, axis=0), dtype=jnp.float32),
            dones=jnp.asarray(np.stack(dones, axis=0), dtype=jnp.float32),
        )

    def summary(self) -> dict[str, Any]:
        returns = [float(item["return"]) for item in self._episodes]
        lengths = [int(item["length"]) for item in self._episodes]
        buckets: dict[int, int] = {}
        for item in self._episodes:
            bucket = int(item.get("mode_bucket", 0))
            buckets[bucket] = buckets.get(bucket, 0) + 1
        return {
            "hard_start_buffer_enabled": True,
            "hard_start_max_steps": int(self.max_steps),
            "hard_start_steps": int(self._steps),
            "hard_start_episodes": int(len(self._episodes)),
            "hard_start_mode_buckets": int(self.mode_buckets),
            "hard_start_balance_modes": bool(self.balance_modes),
            "hard_start_bucket_counts": {
                str(bucket): int(count) for bucket, count in sorted(buckets.items())
            },
            "hard_start_mean_return": (float(np.mean(returns)) if returns else None),
            "hard_start_min_return": float(np.min(returns)) if returns else None,
            "hard_start_max_return": float(np.max(returns)) if returns else None,
            "hard_start_return_p25": (
                float(np.quantile(np.asarray(returns), 0.25)) if returns else None
            ),
            "hard_start_buffer_return_p25": (
                float(np.quantile(np.asarray(returns), 0.25)) if returns else None
            ),
            "hard_start_buffer_return_p50": (
                float(np.quantile(np.asarray(returns), 0.50)) if returns else None
            ),
            "hard_start_buffer_return_p90": (
                float(np.quantile(np.asarray(returns), 0.90)) if returns else None
            ),
            "hard_start_mean_length": (float(np.mean(lengths)) if lengths else None),
        }

    def _sample_episode_windows(
        self,
        rng: np.random.Generator,
        *,
        batch_size: int,
        sequence_length: int,
    ) -> list[tuple[int, int]]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        candidates = [
            (index, int(item["length"] - sequence_length + 1))
            for index, item in enumerate(self._episodes)
            if item["length"] >= sequence_length
        ]
        if not candidates:
            raise ValueError("hard-start buffer has no eligible windows")
        if self.balance_modes:
            by_bucket: dict[int, list[tuple[int, int]]] = {}
            for episode_index, count in candidates:
                bucket = int(self._episodes[episode_index].get("mode_bucket", 0))
                by_bucket.setdefault(bucket, []).append((episode_index, count))
            non_empty_buckets = list(by_bucket)
            result: list[tuple[int, int]] = []
            bucket_picks = rng.choice(non_empty_buckets, size=(batch_size,))
            for bucket in bucket_picks:
                bucket_candidates = by_bucket[int(bucket)]
                counts = np.asarray(
                    [count for _, count in bucket_candidates],
                    dtype=np.float64,
                )
                probabilities = counts / np.sum(counts)
                candidate_index = int(
                    rng.choice(len(bucket_candidates), p=probabilities)
                )
                episode_index, count = bucket_candidates[candidate_index]
                result.append((episode_index, int(rng.integers(0, int(count)))))
            return result
        counts = np.asarray([count for _, count in candidates], dtype=np.int64)
        cumulative = np.cumsum(counts)
        total = int(cumulative[-1])
        picks = rng.integers(0, total, size=(batch_size,))
        result: list[tuple[int, int]] = []
        for pick in picks:
            candidate_index = int(np.searchsorted(cumulative, pick, side="right"))
            previous = (
                0 if candidate_index == 0 else int(cumulative[candidate_index - 1])
            )
            episode_index, _ = candidates[candidate_index]
            result.append((episode_index, int(pick - previous)))
        return result

    def _mode_bucket(self, observations: np.ndarray) -> int:
        if self._mode_projection is None:
            return 0
        feature = np.asarray(observations[0], dtype=np.float32).reshape((-1,))
        norm = float(np.linalg.norm(feature))
        if norm > 1e-6:
            feature = feature / norm
        scores = feature @ self._mode_projection
        return int(np.argmax(scores))

    def _trim(self) -> None:
        while self._steps > self.max_steps and self._episodes:
            removed = self._episodes.pop(0)
            self._steps -= int(removed["length"])


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
    action_low: np.ndarray,
    action_high: np.ndarray,
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
    gate = _candidate_refit_gate_report(
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


def _candidate_refit_gate_report(
    baseline_anchor: dict[str, Any],
    candidate_anchor: dict[str, Any],
    baseline_recent: dict[str, Any],
    candidate_recent: dict[str, Any],
    *,
    metric: str,
    min_recent_improvement: float,
    max_anchor_degradation: float,
    anchor_penalty: float = 1.0,
) -> dict[str, Any]:
    baseline_anchor_value = float(baseline_anchor[metric])
    candidate_anchor_value = float(candidate_anchor[metric])
    baseline_recent_value = float(baseline_recent[metric])
    candidate_recent_value = float(candidate_recent[metric])
    recent_improvement = baseline_recent_value - candidate_recent_value
    anchor_degradation = candidate_anchor_value - baseline_anchor_value
    gate_score = recent_improvement - anchor_penalty * max(anchor_degradation, 0.0)
    recent_improved = recent_improvement >= min_recent_improvement
    anchor_preserved = anchor_degradation <= max_anchor_degradation
    candidate_metrics_finite = _metrics_finite(candidate_anchor) and _metrics_finite(
        candidate_recent
    )
    return {
        "model_update_accepted": bool(
            candidate_metrics_finite and recent_improved and anchor_preserved
        ),
        "candidate_gate_metric": metric,
        "candidate_min_recent_improvement": min_recent_improvement,
        "candidate_max_anchor_degradation": max_anchor_degradation,
        "candidate_anchor_penalty": anchor_penalty,
        "candidate_gate_score": gate_score,
        "candidate_metrics_finite": candidate_metrics_finite,
        "recent_validation_baseline": baseline_recent_value,
        "recent_validation_candidate": candidate_recent_value,
        "recent_validation_improvement": recent_improvement,
        "recent_validation_improved": bool(recent_improved),
        "anchor_validation_baseline": baseline_anchor_value,
        "anchor_validation_candidate": candidate_anchor_value,
        "anchor_validation_degradation": anchor_degradation,
        "anchor_validation_preserved": bool(anchor_preserved),
    }


def _best_passing_candidate_report(
    reports: list[dict[str, Any]],
) -> dict[str, Any] | None:
    passing = [report for report in reports if report.get("model_update_accepted")]
    if not passing:
        return None
    return max(
        passing,
        key=lambda report: float(
            report.get("gate", {}).get("candidate_gate_score", 0.0)
        ),
    )


def _candidate_checkpoint_gate_summary(report: dict[str, Any]) -> dict[str, Any]:
    gate = report["gate"]
    return {
        "candidate_update": report.get("candidate_update"),
        "model_update_accepted": report.get("model_update_accepted"),
        "candidate_gate_score": gate.get("candidate_gate_score"),
        "recent_validation_improvement": gate.get("recent_validation_improvement"),
        "anchor_validation_degradation": gate.get("anchor_validation_degradation"),
        "recent_validation_improved": gate.get("recent_validation_improved"),
        "anchor_validation_preserved": gate.get("anchor_validation_preserved"),
        "candidate_metrics_finite": gate.get("candidate_metrics_finite"),
    }


def _concat_replay_batches(batches: list[ReplayBatch]) -> ReplayBatch:
    if len(batches) == 1:
        return batches[0]
    return ReplayBatch(
        observations=jnp.concatenate([batch.observations for batch in batches], axis=0),
        actions=jnp.concatenate([batch.actions for batch in batches], axis=0),
        rewards=jnp.concatenate([batch.rewards for batch in batches], axis=0),
        dones=jnp.concatenate([batch.dones for batch in batches], axis=0),
    )


def _sample_online_candidate_batch(
    np_rng: np.random.Generator,
    *,
    replay: SequenceReplayBuffer,
    anchor_replay: SequenceReplayBuffer,
    recent_replay: SequenceReplayBuffer,
    batch_size: int,
    chunk_length: int,
    max_horizon: int,
    anchor_batch_fraction: float,
) -> ReplayBatch:
    anchor_size = int(round(batch_size * anchor_batch_fraction))
    anchor_size = max(0, min(batch_size, anchor_size))
    recent_size = batch_size - anchor_size
    batches: list[ReplayBatch] = []
    if anchor_size > 0:
        batches.append(
            anchor_replay.sample(
                np_rng,
                batch_size=anchor_size,
                chunk_length=chunk_length,
                max_horizon=max_horizon,
            )
        )
    if recent_size > 0:
        batches.append(
            recent_replay.sample(
                np_rng,
                batch_size=recent_size,
                chunk_length=chunk_length,
                max_horizon=max_horizon,
            )
        )
    if batches:
        return _concat_replay_batches(batches)
    return replay.sample(
        np_rng,
        batch_size=batch_size,
        chunk_length=chunk_length,
        max_horizon=max_horizon,
    )


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
    action_low: np.ndarray,
    action_high: np.ndarray,
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
        batch = _sample_online_candidate_batch(
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
            freeze_encoder=args.online_freeze_encoder,
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
                    "online_encoder_frozen": args.online_freeze_encoder,
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

    best_report = _best_passing_candidate_report(reports)
    selected_report = best_report if best_report is not None else final_report
    selected_state = best_state if best_report is not None else state
    checkpoint_summaries = [
        _candidate_checkpoint_gate_summary(report) for report in reports
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
        # Keep device scalars on device so async dispatch can overlap host-side
        # replay sampling with accelerator compute. Convert only when logging.
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
    action_low: np.ndarray,
    action_high: np.ndarray,
    phase: str = "policy",
    train_steps: int | None = None,
    reset_actor: bool = True,
    eval_seed_offset: int = 3_000_000,
    selection_seed_offset: int = 5_000_000,
    confirmation_seed_offset: int = 7_000_000,
    hard_start_replay: HardStartReplayBuffer | None = None,
    policy_validation_replay: SequenceReplayBuffer | None = None,
) -> dict[str, Any]:
    policy_train_steps = args.policy_train_steps if train_steps is None else train_steps
    if policy_train_steps == 0:
        return {
            "state": state,
            "rng": rng,
            "outcome": {"policy_training_enabled": False},
        }

    if reset_actor:
        rng, reset_key = jax.random.split(rng)
        state = reset_policy_heads(state, reset_key, config)
    eval_num_envs = args.policy_eval_num_envs or min(
        args.num_envs,
        args.policy_eval_episodes,
    )
    policy_batch_size = args.policy_batch_size or args.batch_size
    action_low_jax = jnp.asarray(action_low, dtype=jnp.float32)
    action_high_jax = jnp.asarray(action_high, dtype=jnp.float32)
    policy_eval_seed = seed + eval_seed_offset
    policy_selection_seed = seed + selection_seed_offset
    policy_confirmation_seed = seed + confirmation_seed_offset
    policy_eval_enabled = args.policy_eval_during_training
    selection_enabled = args.policy_selection_interval > 0
    model_selection_enabled = args.policy_model_selection_interval > 0
    model_selection_diagnostics_enabled = args.policy_model_selection_diagnostics
    model_scoring_enabled = (
        model_selection_enabled or model_selection_diagnostics_enabled
    )
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

    random_eval = None
    initial_eval = None
    if policy_eval_enabled:
        random_eval = _evaluate_random_policy(
            args,
            seed=policy_eval_seed,
            num_envs=eval_num_envs,
            desc=f"{control} {phase} eval random policy",
        )
        initial_eval = _evaluate_continuous_policy(
            args,
            state,
            config,
            seed=policy_eval_seed,
            num_envs=eval_num_envs,
            action_low=action_low_jax,
            action_high=action_high_jax,
            desc=f"{control} {phase} eval initial policy",
            **_policy_video_options(
                args,
                logger,
                phase=phase,
                stage="initial",
            ),
        )
        logger.write_json(
            f"{artifact_prefix}random_policy_evaluation.json", random_eval
        )
        logger.write_json(
            f"{artifact_prefix}initial_policy_evaluation.json", initial_eval
        )

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
        confirmation_initial_eval = _evaluate_continuous_policy(
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
    model_selection_history: list[dict[str, Any]] = []
    model_selection_diagnostic_history: list[dict[str, Any]] = []
    best_selection_eval: dict[str, Any] | None = None
    best_selection_mean = -math.inf
    best_selection_score = -math.inf
    best_model_selection_score = -math.inf
    best_model_selection_diagnostic_score = -math.inf
    best_model_selection_diagnostic_step = 0
    if selection_enabled:
        selection_eval = _evaluate_continuous_policy(
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
        selection_score = _policy_evaluation_score(
            selection_eval,
            std_penalty=args.policy_selection_std_penalty,
            failure_penalty=args.policy_selection_failure_penalty,
        )
        best_selection_score = (
            selection_score if selection_score is not None else -math.inf
        )
        selection_record = _policy_selection_record(
            step=0,
            evaluation=selection_eval,
            selected=True,
            score=selection_score,
            std_penalty=args.policy_selection_std_penalty,
            failure_penalty=args.policy_selection_failure_penalty,
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
    policy_replay_critic_loss_coef = getattr(
        args,
        "policy_replay_critic_loss_coef",
        0.0,
    )
    policy_replay_critic_horizon = (
        getattr(args, "policy_replay_critic_horizon", None) or args.critic_horizon
    )
    policy_replay_critic_batch_size = (
        getattr(args, "policy_replay_critic_batch_size", None) or policy_batch_size
    )
    start_sampler, start_sampling_summary = _make_policy_start_sampler(
        args,
        config,
        replay,
        np_rng=np_rng,
        batch_size=policy_batch_size,
        hard_start_replay=hard_start_replay,
    )
    logger.write_json(
        f"{artifact_prefix}policy_start_sampling.json",
        start_sampling_summary,
    )
    model_selection_batch = None
    model_selection_key = None
    model_selection_sampling_summary: dict[str, Any] = {}

    def sample_validation_model_selection_batch():
        if policy_validation_replay is None:
            raise RuntimeError(
                "--policy-model-selection-source validation-replay requires "
                "a held-out validation replay"
            )
        selection_batch_size = (
            args.policy_model_selection_batch_size or policy_batch_size
        )
        validation_batch = policy_validation_replay.sample(
            np_rng,
            batch_size=selection_batch_size,
            chunk_length=config.context_window,
            max_horizon=1,
        )
        hard_mask = jnp.zeros((selection_batch_size,), dtype=jnp.float32)
        return (
            validation_batch.observations[:, : config.context_window],
            validation_batch.actions[:, : config.context_window],
            hard_mask,
        )

    def score_model_policy(candidate_state, step_index: int):
        if model_selection_batch is None or model_selection_key is None:
            raise RuntimeError("model policy selection is not initialized")
        start_observations, start_actions, hard_start_mask = model_selection_batch
        score_metrics = continuous_policy_score(
            candidate_state,
            model_selection_key,
            start_observations,
            config,
            action_low_jax,
            action_high_jax,
            imag_horizon=args.imag_horizon,
            control=control,
            policy_return_mode=args.policy_return_mode,
            policy_actor_baseline=args.policy_actor_baseline,
            policy_return_normalization=args.policy_return_normalization,
            policy_actor_cvar_fraction=args.policy_actor_cvar_fraction,
            policy_actor_cvar_coef=args.policy_actor_cvar_coef,
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
            policy_uncertainty_coef=args.policy_uncertainty_coef,
            policy_action_bound_coef=args.policy_action_bound_coef,
            policy_action_bound_limit=args.policy_action_bound_limit,
            policy_hard_action_bound_coef=args.policy_hard_action_bound_coef,
            hard_start_mask=hard_start_mask,
            actor_entropy_coef=args.actor_entropy_coef,
            target_critic_params=(
                candidate_state.target_critic_params
                if args.target_critic_ema_decay > 0.0
                else None
            ),
            policy_gradient_mode=args.policy_gradient_mode,
        )
        score_metrics_json = to_jsonable(
            {
                **score_metrics,
                "policy/heldout_model_score": _policy_heldout_model_score(
                    to_jsonable(score_metrics),
                    cvar_coef=args.policy_model_selection_cvar_coef,
                    uncertainty_penalty=(
                        args.policy_model_selection_uncertainty_penalty
                    ),
                    action_saturation_penalty=(
                        args.policy_model_selection_action_saturation_penalty
                    ),
                ),
                "policy/hard_start_fraction": start_sampling_summary.get(
                    "hard_start_actual_fraction",
                    0.0,
                ),
                "policy/model_selection_source": (args.policy_model_selection_source),
                "policy/model_selection_step": step_index,
            }
        )
        raw_score = score_metrics_json.get(args.policy_model_selection_metric)
        score = (
            float(raw_score)
            if isinstance(raw_score, (int, float)) and math.isfinite(raw_score)
            else -math.inf
        )
        return score, score_metrics_json

    if model_scoring_enabled:
        if args.policy_model_selection_source == "validation-replay":
            model_selection_batch = sample_validation_model_selection_batch()
            model_selection_sampling_summary = {
                "mode": "validation-replay",
                "batch_size": (
                    args.policy_model_selection_batch_size or policy_batch_size
                ),
                "context_window": config.context_window,
                "validation_replay_size_per_env": (
                    None
                    if policy_validation_replay is None
                    else policy_validation_replay.size
                ),
                "hard_start_actual_fraction": 0.0,
            }
        else:
            model_selection_batch = start_sampler()
            model_selection_sampling_summary = {
                **start_sampling_summary,
                "mode": "policy-starts",
            }
        rng, model_selection_key = jax.random.split(rng)
    if model_selection_enabled:
        best_model_selection_score, best_policy_metrics_json = score_model_policy(
            state,
            0,
        )
        best_policy_metrics_json = {
            **best_policy_metrics_json,
            "policy/model_selected_initial_actor": True,
        }
        model_selection_record = _policy_model_selection_record(
            step=0,
            metrics=best_policy_metrics_json,
            metric=args.policy_model_selection_metric,
            score=best_model_selection_score,
            selected=True,
        )
        model_selection_history.append(model_selection_record)
        logger.append_metrics(
            {
                "phase": "policy_model_selection",
                "update": 0,
                "control": control,
                "policy_phase": phase,
                **model_selection_record,
                "policy_model_selection_best_score": best_model_selection_score,
                "policy_model_selection_best_step": best_policy_step,
            }
        )
        logger.write_json(
            f"{artifact_prefix}policy_model_selection_initial.json",
            best_policy_metrics_json,
        )
    if model_selection_diagnostics_enabled:
        diagnostic_score, diagnostic_metrics = score_model_policy(state, 0)
        model_selected = diagnostic_score > best_model_selection_diagnostic_score
        if model_selected:
            best_model_selection_diagnostic_score = diagnostic_score
            best_model_selection_diagnostic_step = 0
        diagnostic_record = _policy_model_selection_record(
            step=0,
            metrics=diagnostic_metrics,
            metric=args.policy_model_selection_metric,
            score=diagnostic_score,
            selected=model_selected,
        )
        if best_selection_eval is not None:
            diagnostic_record.update(
                _policy_model_selection_diagnostic_real_fields(
                    selection_record=selection_history[-1],
                    evaluation=best_selection_eval,
                    selected_by_real=True,
                )
            )
        model_selection_diagnostic_history.append(diagnostic_record)
        logger.append_metrics(
            {
                "phase": "policy_model_selection_diagnostic",
                "update": 0,
                "control": control,
                "policy_phase": phase,
                **diagnostic_record,
                "policy_model_selection_diagnostic_best_score": (
                    best_model_selection_diagnostic_score
                ),
                "policy_model_selection_diagnostic_best_step": (
                    best_model_selection_diagnostic_step
                ),
            }
        )
        logger.write_json(
            f"{artifact_prefix}policy_model_selection_diagnostic_initial.json",
            diagnostic_record,
        )
    policy_steps = tqdm(
        range(1, policy_train_steps + 1),
        desc=f"{control} {phase} train frozen-policy",
        unit="update",
        disable=args.quiet,
    )
    for step_index in policy_steps:
        start_observations, start_actions, hard_start_mask = start_sampler()
        rng, policy_key = jax.random.split(rng)
        real_critic_batch = None
        real_critic_hard_fraction = 0.0
        if policy_replay_critic_loss_coef > 0.0:
            real_critic_batch, real_critic_hard_fraction = (
                _sample_mixed_replay_critic_batch(
                    replay,
                    hard_start_replay,
                    np_rng,
                    batch_size=policy_replay_critic_batch_size,
                    chunk_length=policy_replay_critic_horizon,
                    max_horizon=1,
                    hard_fraction=args.policy_hard_critic_fraction,
                )
            )
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
            policy_actor_cvar_fraction=args.policy_actor_cvar_fraction,
            policy_actor_cvar_coef=args.policy_actor_cvar_coef,
            policy_gradient_mode=args.policy_gradient_mode,
            return_normalization_ema_decay=args.policy_return_ema_decay,
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
            policy_uncertainty_coef=args.policy_uncertainty_coef,
            policy_action_bound_coef=args.policy_action_bound_coef,
            policy_action_bound_limit=args.policy_action_bound_limit,
            policy_hard_action_bound_coef=args.policy_hard_action_bound_coef,
            hard_start_mask=hard_start_mask,
            actor_entropy_coef=args.actor_entropy_coef,
            target_critic_params=(
                state.target_critic_params
                if args.target_critic_ema_decay > 0.0
                else None
            ),
            target_critic_ema_decay=args.target_critic_ema_decay,
            real_critic_batch=real_critic_batch,
            real_critic_loss_enabled=policy_replay_critic_loss_coef > 0.0,
            real_critic_loss_coef=policy_replay_critic_loss_coef,
            real_critic_horizon=policy_replay_critic_horizon,
            real_critic_return_mode=args.policy_replay_critic_return_mode,
            real_critic_all_steps=args.policy_replay_critic_all_steps,
            slow_value_regularization_coef=(args.policy_slow_value_regularization_coef),
        )
        metrics = {
            **metrics,
            "policy/hard_start_fraction": start_sampling_summary.get(
                "hard_start_actual_fraction", 0.0
            ),
            "policy/replay_critic_hard_fraction": real_critic_hard_fraction,
        }
        if (
            args.policy_real_critic_interval > 0
            and step_index % args.policy_real_critic_interval == 0
        ):
            real_critic_batch_size = (
                args.policy_real_critic_batch_size or policy_batch_size
            )
            real_critic_metrics: dict[str, Any] = {}
            for _ in range(args.policy_real_critic_updates):
                critic_batch = replay.sample(
                    np_rng,
                    batch_size=real_critic_batch_size,
                    chunk_length=args.critic_horizon,
                    max_horizon=1,
                )
                state, real_critic_metrics = continuous_critic_warmup_step(
                    state,
                    critic_batch,
                    config,
                    horizon=args.critic_horizon,
                    value_clip=args.value_clip,
                    target_critic_ema_decay=args.target_critic_ema_decay,
                )
            metrics = {
                **metrics,
                **{
                    f"policy/real_critic_aux_{key.split('/', 1)[-1]}": value
                    for key, value in real_critic_metrics.items()
                },
                "policy/real_critic_aux_interval": args.policy_real_critic_interval,
                "policy/real_critic_aux_updates": args.policy_real_critic_updates,
            }
            logger.append_metrics(
                {
                    "phase": f"{metric_phase_prefix}real_return_critic_aux",
                    "update": step_index,
                    "control": control,
                    "policy_phase": phase,
                    **{
                        f"policy/real_critic_aux_{key.split('/', 1)[-1]}": value
                        for key, value in real_critic_metrics.items()
                    },
                    "policy/real_critic_aux_interval": (
                        args.policy_real_critic_interval
                    ),
                    "policy/real_critic_aux_updates": (args.policy_real_critic_updates),
                }
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
                    policy_loss,
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
            selection_eval = _evaluate_continuous_policy(
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
            selection_score = _policy_evaluation_score(
                selection_eval,
                std_penalty=args.policy_selection_std_penalty,
                failure_penalty=args.policy_selection_failure_penalty,
            )
            selected = (
                selection_score is not None and selection_score > best_selection_score
            )
            if selected:
                best_state = state
                best_policy_step = step_index
                best_policy_metrics_json = {
                    **to_jsonable(metrics),
                    "policy/selected_initial_actor": False,
                }
                best_selection_eval = selection_eval
                best_selection_mean = selection_eval["mean_return"]
                best_selection_score = selection_score
            selection_record = _policy_selection_record(
                step=step_index,
                evaluation=selection_eval,
                selected=selected,
                score=selection_score,
                std_penalty=args.policy_selection_std_penalty,
                failure_penalty=args.policy_selection_failure_penalty,
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
                    "policy_selection_best_score": best_selection_score,
                    "policy_selection_best_step": best_policy_step,
                }
            )
            if model_selection_diagnostics_enabled:
                diagnostic_score, diagnostic_metrics = score_model_policy(
                    state,
                    step_index,
                )
                model_selected = (
                    diagnostic_score > best_model_selection_diagnostic_score
                )
                if model_selected:
                    best_model_selection_diagnostic_score = diagnostic_score
                    best_model_selection_diagnostic_step = step_index
                diagnostic_record = _policy_model_selection_record(
                    step=step_index,
                    metrics=diagnostic_metrics,
                    metric=args.policy_model_selection_metric,
                    score=diagnostic_score,
                    selected=model_selected,
                )
                diagnostic_record.update(
                    _policy_model_selection_diagnostic_real_fields(
                        selection_record=selection_record,
                        evaluation=selection_eval,
                        selected_by_real=selected,
                    )
                )
                model_selection_diagnostic_history.append(diagnostic_record)
                logger.append_metrics(
                    {
                        "phase": "policy_model_selection_diagnostic",
                        "update": step_index,
                        "control": control,
                        "policy_phase": phase,
                        **diagnostic_record,
                        "policy_model_selection_diagnostic_best_score": (
                            best_model_selection_diagnostic_score
                        ),
                        "policy_model_selection_diagnostic_best_step": (
                            best_model_selection_diagnostic_step
                        ),
                    }
                )
        if model_selection_enabled and (
            step_index == policy_train_steps
            or step_index % args.policy_model_selection_interval == 0
        ):
            model_selection_score, model_selection_metrics = score_model_policy(
                state,
                step_index,
            )
            selected = model_selection_score > best_model_selection_score
            if selected:
                best_state = state
                best_policy_step = step_index
                best_model_selection_score = model_selection_score
                best_policy_metrics_json = {
                    **model_selection_metrics,
                    "policy/model_selected_initial_actor": False,
                }
            model_selection_record = _policy_model_selection_record(
                step=step_index,
                metrics=model_selection_metrics,
                metric=args.policy_model_selection_metric,
                score=model_selection_score,
                selected=selected,
            )
            model_selection_history.append(model_selection_record)
            logger.append_metrics(
                {
                    "phase": "policy_model_selection",
                    "update": step_index,
                    "control": control,
                    "policy_phase": phase,
                    **model_selection_record,
                    "policy_model_selection_best_score": best_model_selection_score,
                    "policy_model_selection_best_step": best_policy_step,
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
                "best_policy_score": best_selection_score,
                "policy_selection_std_penalty": args.policy_selection_std_penalty,
                "evaluation": best_selection_eval,
            },
        )
    if model_selection_enabled:
        state = best_state
        logger.write_json(
            f"{artifact_prefix}policy_model_selection_history.json",
            model_selection_history,
        )
        logger.write_json(
            f"{artifact_prefix}best_policy_model_selection.json",
            {
                "best_policy_step": best_policy_step,
                "best_policy_model_selection_score": best_model_selection_score,
                "policy_model_selection_metric": args.policy_model_selection_metric,
                "metrics": best_policy_metrics_json,
            },
        )
    if model_selection_diagnostics_enabled:
        logger.write_json(
            f"{artifact_prefix}policy_model_selection_diagnostic_history.json",
            model_selection_diagnostic_history,
        )
        logger.write_json(
            f"{artifact_prefix}best_policy_model_selection_diagnostic.json",
            {
                "best_policy_model_selection_diagnostic_step": (
                    best_model_selection_diagnostic_step
                ),
                "best_policy_model_selection_diagnostic_score": (
                    best_model_selection_diagnostic_score
                ),
                "policy_model_selection_metric": args.policy_model_selection_metric,
            },
        )
    trained_eval = None
    if policy_eval_enabled:
        trained_eval = _evaluate_continuous_policy(
            args,
            state,
            config,
            seed=policy_eval_seed,
            num_envs=eval_num_envs,
            action_low=action_low_jax,
            action_high=action_high_jax,
            desc=f"{control} {phase} eval trained policy",
            **_policy_video_options(
                args,
                logger,
                phase=phase,
                stage="trained",
            ),
        )
        logger.write_json(
            f"{artifact_prefix}trained_policy_evaluation.json", trained_eval
        )
    confirmation_trained_eval = None
    if confirmation_enabled:
        confirmation_trained_eval = _evaluate_continuous_policy(
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
        _maybe_int(item.get("policy_selection_env_steps")) for item in selection_history
    )
    selection_completed_episode_steps = sum(
        _maybe_int(item.get("policy_selection_completed_episode_steps"))
        for item in selection_history
    )
    confirmation_eval_payloads = [
        confirmation_random_eval,
        confirmation_initial_eval,
        confirmation_trained_eval,
    ]
    confirmation_env_steps = sum(
        _eval_env_steps(item) for item in confirmation_eval_payloads
    )
    confirmation_completed_episode_steps = sum(
        _eval_completed_episode_steps(item) for item in confirmation_eval_payloads
    )
    policy_eval_payloads = [
        random_eval,
        initial_eval,
        trained_eval,
        *confirmation_eval_payloads,
    ]
    policy_nonselection_env_steps = sum(
        _eval_env_steps(item) for item in policy_eval_payloads
    )
    policy_nonselection_completed_episode_steps = sum(
        _eval_completed_episode_steps(item) for item in policy_eval_payloads
    )
    policy_total_eval_env_steps = policy_nonselection_env_steps + selection_env_steps
    policy_total_completed_episode_steps = (
        policy_nonselection_completed_episode_steps + selection_completed_episode_steps
    )

    initial_mean = _eval_metric(initial_eval, "mean_return")
    trained_mean = _eval_metric(trained_eval, "mean_return")
    random_mean = _eval_metric(random_eval, "mean_return")
    trained_score = _policy_evaluation_score(
        trained_eval,
        std_penalty=args.online_policy_std_penalty,
        failure_penalty=args.online_policy_failure_penalty,
    )
    policy_acceptance_stats = _pooled_policy_acceptance_stats(
        [
            best_selection_eval if selection_enabled else None,
            trained_eval,
            confirmation_trained_eval,
        ],
        failure_threshold=args.policy_failure_return_threshold,
        soft_failure_threshold=args.policy_soft_failure_return_threshold,
        failure_penalty=args.online_policy_failure_penalty,
        soft_failure_penalty=args.policy_soft_failure_penalty,
        std_penalty=args.online_policy_std_penalty,
    )
    confirmation_improvement = None
    confirmation_trained_minus_random = None
    if confirmation_enabled:
        confirmation_improvement = _optional_difference(
            _eval_metric(confirmation_trained_eval, "mean_return"),
            _eval_metric(confirmation_initial_eval, "mean_return"),
        )
        confirmation_trained_minus_random = _optional_difference(
            _eval_metric(confirmation_trained_eval, "mean_return"),
            _eval_metric(confirmation_random_eval, "mean_return"),
        )
    policy_metrics_json = (
        best_policy_metrics_json
        if (selection_enabled or model_selection_enabled)
        else last_policy_metrics_json
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
        "policy_return_mode": args.policy_return_mode,
        "policy_actor_baseline": args.policy_actor_baseline,
        "policy_return_normalization": args.policy_return_normalization,
        "policy_actor_cvar_fraction": args.policy_actor_cvar_fraction,
        "policy_actor_cvar_coef": args.policy_actor_cvar_coef,
        "policy_stochastic_actor": args.stochastic_actor,
        "policy_stochastic_collection": args.stochastic_collection,
        "policy_actor_entropy_coef": args.actor_entropy_coef,
        "policy_actor_log_std_min": args.actor_log_std_min,
        "policy_actor_log_std_max": args.actor_log_std_max,
        "policy_target_critic_ema_decay": args.target_critic_ema_decay,
        "policy_imag_horizon": args.imag_horizon,
        "policy_real_critic_interval": args.policy_real_critic_interval,
        "policy_real_critic_updates": args.policy_real_critic_updates,
        "policy_real_critic_batch_size": args.policy_real_critic_batch_size,
        "policy_replay_critic_loss_coef": policy_replay_critic_loss_coef,
        "policy_replay_critic_batch_size": policy_replay_critic_batch_size,
        "policy_replay_critic_horizon": policy_replay_critic_horizon,
        "policy_hard_start_fraction": args.policy_hard_start_fraction,
        "policy_hard_critic_fraction": args.policy_hard_critic_fraction,
        "policy_hard_start_max_steps": args.policy_hard_start_max_steps,
        "policy_hard_start_prefix_steps": args.policy_hard_start_prefix_steps,
        "policy_hard_start_recovery_windows": (args.policy_hard_start_recovery_windows),
        "policy_hard_start_recovery_stride": args.policy_hard_start_recovery_stride,
        "policy_hard_start_mode_buckets": args.policy_hard_start_mode_buckets,
        "policy_hard_start_balance_modes": args.policy_hard_start_balance_modes,
        "policy_hard_start_return_percentile": (
            args.policy_hard_start_return_percentile
        ),
        "policy_hard_start_absolute_threshold": (
            args.policy_hard_start_absolute_threshold
        ),
        "policy_start_sampling_summary": start_sampling_summary,
        "policy_trust_coef": policy_trust_coef,
        "policy_base_trust_coef": args.policy_trust_coef,
        "online_policy_trust_coef": args.online_policy_trust_coef,
        "policy_uncertainty_coef": args.policy_uncertainty_coef,
        "policy_action_bound_coef": args.policy_action_bound_coef,
        "policy_hard_action_bound_coef": args.policy_hard_action_bound_coef,
        "policy_action_bound_limit": args.policy_action_bound_limit,
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
        "policy_selection_std_penalty": args.policy_selection_std_penalty,
        "policy_selection_failure_penalty": args.policy_selection_failure_penalty,
        "policy_failure_return_threshold": args.policy_failure_return_threshold,
        "policy_success_return_threshold": args.policy_success_return_threshold,
        "policy_selection_env_steps": selection_env_steps,
        "policy_selection_completed_episode_steps": selection_completed_episode_steps,
        "policy_model_selection_enabled": model_selection_enabled,
        "policy_model_selection_source": args.policy_model_selection_source,
        "policy_model_selection_interval": args.policy_model_selection_interval,
        "policy_model_selection_metric": args.policy_model_selection_metric,
        "policy_model_selection_batch_size": (
            args.policy_model_selection_batch_size or policy_batch_size
        ),
        "policy_model_selection_cvar_coef": args.policy_model_selection_cvar_coef,
        "policy_model_selection_uncertainty_penalty": (
            args.policy_model_selection_uncertainty_penalty
        ),
        "policy_model_selection_action_saturation_penalty": (
            args.policy_model_selection_action_saturation_penalty
        ),
        "policy_model_selection_sampling_summary": (model_selection_sampling_summary),
        "policy_model_selection_score": (
            best_model_selection_score if model_selection_enabled else None
        ),
        "policy_model_selection_history": model_selection_history,
        "policy_model_selection_diagnostics_enabled": (
            model_selection_diagnostics_enabled
        ),
        "policy_model_selection_diagnostic_score": (
            best_model_selection_diagnostic_score
            if model_selection_diagnostics_enabled
            else None
        ),
        "policy_model_selection_diagnostic_best_step": (
            best_model_selection_diagnostic_step
            if model_selection_diagnostics_enabled
            else None
        ),
        "policy_model_selection_diagnostic_history": (
            model_selection_diagnostic_history
        ),
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
        "best_policy_step": (
            best_policy_step if (selection_enabled or model_selection_enabled) else None
        ),
        "best_policy_selection_mean": (
            best_selection_mean if selection_enabled else None
        ),
        "best_policy_selection_score": (
            best_selection_score if selection_enabled else None
        ),
        "last_policy_metrics": last_policy_metrics_json,
        "critic_warmup_steps": args.critic_warmup_steps,
        "critic_horizon": args.critic_horizon,
        "critic_final_metrics": critic_metrics_json,
        "policy_eval_during_training": policy_eval_enabled,
        "policy_random_mean": random_mean,
        "policy_random_std": _eval_metric(random_eval, "std_return"),
        "policy_random_failure_rate": _eval_metric(random_eval, "failure_rate"),
        "policy_random_success_rate": _eval_metric(random_eval, "success_rate"),
        "policy_initial_mean": initial_mean,
        "policy_initial_std": _eval_metric(initial_eval, "std_return"),
        "policy_initial_failure_rate": _eval_metric(initial_eval, "failure_rate"),
        "policy_initial_success_rate": _eval_metric(initial_eval, "success_rate"),
        "policy_trained_mean": trained_mean,
        "policy_trained_std": _eval_metric(trained_eval, "std_return"),
        "policy_trained_failure_rate": _eval_metric(trained_eval, "failure_rate"),
        "policy_trained_success_rate": _eval_metric(trained_eval, "success_rate"),
        "policy_trained_return_p10": _eval_metric(trained_eval, "return_p10"),
        "policy_trained_return_cvar10": _eval_metric(trained_eval, "return_cvar10"),
        "policy_trained_nonfailure_mean_return": _eval_metric(
            trained_eval,
            "nonfailure_mean_return",
        ),
        "policy_trained_score": trained_score,
        "online_policy_std_penalty": args.online_policy_std_penalty,
        "online_policy_failure_penalty": args.online_policy_failure_penalty,
        "policy_soft_failure_return_threshold": (
            args.policy_soft_failure_return_threshold
        ),
        "policy_soft_failure_penalty": args.policy_soft_failure_penalty,
        "policy_improvement": _optional_difference(trained_mean, initial_mean),
        "policy_primary_improvement": _optional_difference(trained_mean, initial_mean),
        "policy_primary_improvement_key": "policy_improvement",
        "policy_trained_minus_random": _optional_difference(
            trained_mean,
            random_mean,
        ),
        "policy_confirmation_random_mean": (
            _eval_metric(confirmation_random_eval, "mean_return")
            if confirmation_enabled
            else None
        ),
        "policy_confirmation_initial_mean": (
            _eval_metric(confirmation_initial_eval, "mean_return")
            if confirmation_enabled
            else None
        ),
        "policy_confirmation_trained_mean": (
            _eval_metric(confirmation_trained_eval, "mean_return")
            if confirmation_enabled
            else None
        ),
        "policy_confirmation_improvement": confirmation_improvement,
        "policy_primary_confirmation_improvement": confirmation_improvement,
        "policy_confirmation_trained_minus_random": confirmation_trained_minus_random,
        "policy_confirmation_passed": confirmation_passed,
        **policy_acceptance_stats,
        "policy_final_metrics": policy_metrics_json,
        "policy_passed": bool(
            _metrics_finite(policy_metrics_json)
            and _metrics_finite(critic_metrics_json)
            and trained_mean is not None
            and initial_mean is not None
            and random_mean is not None
            and trained_mean > initial_mean
            and trained_mean > random_mean
            and confirmation_passed
            and policy_metrics_json.get("policy/action_saturation_fraction", 1.0) < 0.75
        ),
    }
    return {"state": state, "rng": rng, "outcome": outcome}


def _make_policy_start_sampler(
    args: argparse.Namespace,
    config: JepaConfig,
    replay: SequenceReplayBuffer,
    *,
    np_rng: np.random.Generator,
    batch_size: int,
    hard_start_replay: HardStartReplayBuffer | None = None,
):
    hard_batch_size = _hard_sample_count(
        batch_size,
        fraction=args.policy_hard_start_fraction,
        enabled=(
            hard_start_replay is not None
            and hard_start_replay.can_sample_starts(
                context_window=config.context_window
            )
        ),
    )
    normal_batch_size = batch_size - hard_batch_size
    summary = {
        "mode": "replay",
        "batch_size": batch_size,
        "context_window": config.context_window,
        "replay_size_per_env": replay.size,
        "reject_done_crossing_contexts": True,
        "hard_start_requested_fraction": args.policy_hard_start_fraction,
        "hard_start_batch_size": hard_batch_size,
        "hard_start_actual_fraction": hard_batch_size / float(batch_size),
        "hard_start_buffer": (
            None if hard_start_replay is None else hard_start_replay.summary()
        ),
    }

    def sample_replay_batch(size: int):
        if size < 1:
            empty_obs = jnp.zeros(
                (0, config.context_window, config.observation_dim),
                dtype=jnp.float32,
            )
            if config.action_mode == "discrete":
                empty_actions = jnp.zeros((0, config.context_window), dtype=jnp.int32)
            else:
                empty_actions = jnp.zeros(
                    (0, config.context_window, config.action_dim),
                    dtype=jnp.float32,
                )
            return empty_obs, empty_actions

        observation_chunks = []
        action_chunks = []
        collected = 0
        attempts = 0
        sample_size = max(64, 2 * size)
        while collected < size and attempts < 64:
            attempts += 1
            batch = replay.sample(
                np_rng,
                batch_size=sample_size,
                chunk_length=config.context_window,
                max_horizon=1,
            )
            done_context = np.asarray(batch.dones[:, : config.context_window])
            valid_indices = np.flatnonzero(np.sum(done_context, axis=1) == 0.0)
            if valid_indices.size == 0:
                continue
            remaining = size - collected
            valid_indices = valid_indices[:remaining]
            observation_chunks.append(
                batch.observations[valid_indices, : config.context_window]
            )
            action_chunks.append(batch.actions[valid_indices, : config.context_window])
            collected += int(valid_indices.size)

        if collected < size:
            raise ValueError(
                "could not sample enough policy start contexts without done "
                f"boundaries after {attempts} attempts; collected {collected}/{size}"
            )
        return (
            jnp.concatenate(observation_chunks, axis=0)[:size],
            jnp.concatenate(action_chunks, axis=0)[:size],
        )

    def sample_replay():
        normal_observations, normal_actions = sample_replay_batch(normal_batch_size)
        if hard_batch_size == 0 or hard_start_replay is None:
            hard_mask = jnp.zeros((normal_batch_size,), dtype=jnp.float32)
            return normal_observations, normal_actions, hard_mask
        hard_observations, hard_actions = hard_start_replay.sample_starts(
            np_rng,
            batch_size=hard_batch_size,
            context_window=config.context_window,
        )
        observations = jnp.concatenate(
            [normal_observations, hard_observations],
            axis=0,
        )
        actions = jnp.concatenate([normal_actions, hard_actions], axis=0)
        hard_mask = jnp.concatenate(
            [
                jnp.zeros((normal_batch_size,), dtype=jnp.float32),
                jnp.ones((hard_batch_size,), dtype=jnp.float32),
            ],
            axis=0,
        )
        permutation = np_rng.permutation(batch_size)
        return observations[permutation], actions[permutation], hard_mask[permutation]

    return sample_replay, summary


def _hard_sample_count(batch_size: int, *, fraction: float, enabled: bool) -> int:
    if batch_size <= 1 or not enabled or fraction <= 0.0:
        return 0
    return max(1, min(batch_size - 1, int(round(batch_size * fraction))))


def _sample_mixed_replay_critic_batch(
    replay: SequenceReplayBuffer,
    hard_start_replay: HardStartReplayBuffer | None,
    np_rng: np.random.Generator,
    *,
    batch_size: int,
    chunk_length: int,
    max_horizon: int,
    hard_fraction: float,
) -> tuple[ReplayBatch, float]:
    hard_batch_size = _hard_sample_count(
        batch_size,
        fraction=hard_fraction,
        enabled=(
            hard_start_replay is not None
            and hard_start_replay.can_sample_batch(
                chunk_length=chunk_length,
                max_horizon=max_horizon,
            )
        ),
    )
    normal_batch_size = batch_size - hard_batch_size
    normal_batch = replay.sample(
        np_rng,
        batch_size=normal_batch_size,
        chunk_length=chunk_length,
        max_horizon=max_horizon,
    )
    if hard_batch_size == 0 or hard_start_replay is None:
        return normal_batch, 0.0
    hard_batch = hard_start_replay.sample_batch(
        np_rng,
        batch_size=hard_batch_size,
        chunk_length=chunk_length,
        max_horizon=max_horizon,
    )
    permutation = np_rng.permutation(batch_size)
    batch = ReplayBatch(
        observations=jnp.concatenate(
            [normal_batch.observations, hard_batch.observations],
            axis=0,
        )[permutation],
        actions=jnp.concatenate([normal_batch.actions, hard_batch.actions], axis=0)[
            permutation
        ],
        rewards=jnp.concatenate([normal_batch.rewards, hard_batch.rewards], axis=0)[
            permutation
        ],
        dones=jnp.concatenate([normal_batch.dones, hard_batch.dones], axis=0)[
            permutation
        ],
    )
    return batch, hard_batch_size / float(batch_size)


def _stable_phase_offset(phase: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(phase))


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
            **_return_tail_metrics(
                returns,
                failure_threshold=args.policy_failure_return_threshold,
                success_threshold=args.policy_success_return_threshold,
            ),
        }
    finally:
        adapter.close()


def _evaluate_continuous_policy(
    args: argparse.Namespace,
    state,
    config: JepaConfig,
    *,
    seed: int,
    num_envs: int,
    action_low: jax.Array,
    action_high: jax.Array,
    desc: str,
    episodes: int | None = None,
    stochastic_actions: bool = False,
    video_logger: RunLogger | None = None,
    video_filename: str | None = None,
    video_key: str | None = None,
    video_caption: str = "",
) -> dict[str, Any]:
    target_episodes = args.policy_eval_episodes if episodes is None else episodes
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
                    f"Evaluation video capture failed; evaluation will continue: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                capture_video = False
        returns = []
        lengths = []
        step_calls = 0
        action_key = jax.random.PRNGKey(seed)
        with tqdm(
            total=target_episodes,
            desc=desc,
            unit="episode",
            disable=args.quiet,
        ) as progress:
            while len(returns) < target_episodes:
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
                        stochastic=stochastic_actions,
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
                            "Evaluation video capture failed; evaluation will "
                            f"continue: {error}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        capture_video = False
                if first_env_done:
                    capture_video = False
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
            "stochastic_actions": bool(stochastic_actions),
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
                failure_threshold=args.policy_failure_return_threshold,
                success_threshold=args.policy_success_return_threshold,
            ),
        }
    finally:
        adapter.close()


def _policy_video_options(
    args: argparse.Namespace,
    logger: RunLogger,
    *,
    phase: str,
    stage: str,
) -> dict[str, Any]:
    if not args.wandb_videos or not args.env.startswith("dmc:"):
        return {}
    if stage == "initial" and phase != "policy":
        return {}
    if stage == "trained" and phase != "policy":
        try:
            online_index = int(phase.split("_", 2)[1])
        except (IndexError, ValueError):
            return {}
        if online_index % args.wandb_video_every_phases != 0:
            return {}
    return {
        "video_logger": logger,
        "video_filename": f"videos/{phase}_{stage}.mp4",
        "video_key": f"videos/{phase}/{stage}",
        "video_caption": f"{phase} {stage} policy evaluation",
    }


def _update_episode_progress(
    progress: tqdm,
    before: int,
    after: int,
    target_episodes: int,
) -> None:
    progress.update(max(0, min(after, target_episodes) - min(before, target_episodes)))


def _maybe_int(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


def _eval_env_steps(evaluation: dict[str, Any] | None) -> int:
    if evaluation is None:
        return 0
    return _maybe_int(evaluation.get("env_steps"))


def _eval_completed_episode_steps(evaluation: dict[str, Any] | None) -> int:
    if evaluation is None:
        return 0
    return _maybe_int(evaluation.get("completed_episode_steps"))


def _eval_metric(evaluation: dict[str, Any] | None, key: str) -> Any:
    if evaluation is None:
        return None
    return evaluation.get(key)


def _optional_difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


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
    sorted_values = np.sort(values)
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
        "return_cvar10": float(np.mean(sorted_values[:tail_count])),
        "nonfailure_mean_return": (
            float(np.mean(nonfailures)) if nonfailures.size else None
        ),
    }


def _policy_evaluation_score(
    evaluation: dict[str, Any] | None,
    *,
    std_penalty: float,
    failure_penalty: float = 0.0,
) -> float | None:
    if evaluation is None or evaluation.get("mean_return") is None:
        return None
    score = float(evaluation["mean_return"])
    if evaluation.get("std_return") is not None:
        score -= float(std_penalty) * float(evaluation["std_return"])
    if evaluation.get("failure_rate") is not None:
        score -= float(failure_penalty) * float(evaluation["failure_rate"])
    return score


def _policy_outcome_score(
    outcome: dict[str, Any] | None,
    *,
    std_penalty: float,
    failure_penalty: float = 0.0,
) -> float | None:
    if outcome is not None and outcome.get("policy_accept_score") is not None:
        return float(outcome["policy_accept_score"])
    if outcome is None or outcome.get("policy_trained_mean") is None:
        return None
    score = float(outcome["policy_trained_mean"])
    if outcome.get("policy_trained_std") is not None:
        score -= float(std_penalty) * float(outcome["policy_trained_std"])
    if outcome.get("policy_trained_failure_rate") is not None:
        score -= float(failure_penalty) * float(outcome["policy_trained_failure_rate"])
    return score


def _evaluation_returns(evaluation: dict[str, Any] | None) -> list[float]:
    if not isinstance(evaluation, dict):
        return []
    returns = evaluation.get("returns")
    if returns is None and isinstance(evaluation.get("evaluation"), dict):
        returns = evaluation["evaluation"].get("returns")
    if returns is None:
        return []
    return [float(item) for item in returns]


def _pooled_policy_acceptance_stats(
    evaluations: list[dict[str, Any] | None],
    *,
    failure_threshold: float,
    soft_failure_threshold: float,
    failure_penalty: float,
    soft_failure_penalty: float,
    std_penalty: float,
) -> dict[str, Any]:
    returns: list[float] = []
    for evaluation in evaluations:
        returns.extend(_evaluation_returns(evaluation))
    if not returns:
        return {
            "policy_accept_score": None,
            "policy_accept_mean": None,
            "policy_accept_std": None,
            "policy_accept_return_p05": None,
            "policy_accept_return_p10": None,
            "policy_accept_return_cvar10": None,
            "policy_accept_failure_rate": None,
            "policy_accept_soft_failure_rate": None,
            "policy_accept_success_mean": None,
            "policy_accept_episodes": 0,
        }

    values = np.asarray(returns, dtype=np.float32)
    failures = values <= float(failure_threshold)
    soft_failures = values <= float(soft_failure_threshold)
    successes = values > float(failure_threshold)
    sorted_values = np.sort(values)
    tail_count = max(1, int(math.ceil(0.10 * values.size)))
    mean = float(np.mean(values))
    std = float(np.std(values))
    failure_rate = float(np.mean(failures))
    soft_failure_rate = float(np.mean(soft_failures))
    score = (
        mean
        - float(failure_penalty) * failure_rate
        - float(soft_failure_penalty) * soft_failure_rate
        - float(std_penalty) * std
    )
    return {
        "policy_accept_score": float(score),
        "policy_accept_mean": mean,
        "policy_accept_std": std,
        "policy_accept_return_p05": float(np.quantile(values, 0.05)),
        "policy_accept_return_p10": float(np.quantile(values, 0.10)),
        "policy_accept_return_cvar10": float(np.mean(sorted_values[:tail_count])),
        "policy_accept_failure_rate": failure_rate,
        "policy_accept_soft_failure_rate": soft_failure_rate,
        "policy_accept_success_mean": (
            float(np.mean(values[successes])) if np.any(successes) else None
        ),
        "policy_accept_episodes": int(values.size),
    }


def _policy_selection_record(
    *,
    step: int,
    evaluation: dict[str, Any],
    selected: bool,
    score: float | None = None,
    std_penalty: float = 0.0,
    failure_penalty: float = 0.0,
) -> dict[str, Any]:
    if score is None:
        score = _policy_evaluation_score(
            evaluation,
            std_penalty=std_penalty,
            failure_penalty=failure_penalty,
        )
    return {
        "policy_selection_step": step,
        "policy_selection_selected": selected,
        "policy_selection_score": score,
        "policy_selection_std_penalty": std_penalty,
        "policy_selection_failure_penalty": failure_penalty,
        "policy_selection_mean_return": evaluation["mean_return"],
        "policy_selection_std_return": evaluation["std_return"],
        "policy_selection_failure_rate": evaluation.get("failure_rate"),
        "policy_selection_success_rate": evaluation.get("success_rate"),
        "policy_selection_return_p10": evaluation.get("return_p10"),
        "policy_selection_return_cvar10": evaluation.get("return_cvar10"),
        "policy_selection_nonfailure_mean_return": evaluation.get(
            "nonfailure_mean_return"
        ),
        "policy_selection_mean_length": evaluation["mean_length"],
        "policy_selection_episodes": evaluation["episodes"],
        "policy_selection_env_steps": evaluation.get("env_steps"),
        "policy_selection_completed_episode_steps": evaluation.get(
            "completed_episode_steps"
        ),
    }


def _policy_model_selection_record(
    *,
    step: int,
    metrics: dict[str, Any],
    metric: str,
    score: float,
    selected: bool,
) -> dict[str, Any]:
    return {
        "policy_model_selection_step": step,
        "policy_model_selection_selected": selected,
        "policy_model_selection_metric": metric,
        "policy_model_selection_score": score,
        "policy_model_selection_imagined_return": metrics.get("policy/imagined_return"),
        "policy_model_selection_clipped_imagined_return": metrics.get(
            "policy/clipped_imagined_return"
        ),
        "policy_model_selection_actor_score": metrics.get("policy/actor_score"),
        "policy_model_selection_actor_objective_score": metrics.get(
            "policy/actor_objective_score"
        ),
        "policy_model_selection_actor_objective_cvar_score": metrics.get(
            "policy/actor_objective_cvar_score"
        ),
        "policy_model_selection_heldout_model_score": metrics.get(
            "policy/heldout_model_score"
        ),
        "policy_model_selection_uncertainty": metrics.get("policy/uncertainty"),
        "policy_model_selection_action_saturation_fraction": metrics.get(
            "policy/action_saturation_fraction"
        ),
        "policy_model_selection_finite_fraction": metrics.get("policy/finite_fraction"),
    }


def _policy_heldout_model_score(
    metrics: dict[str, Any],
    *,
    cvar_coef: float,
    uncertainty_penalty: float,
    action_saturation_penalty: float,
) -> float:
    base = _finite_metric(metrics, "policy/actor_objective_score")
    cvar = _finite_metric(metrics, "policy/actor_objective_cvar_score")
    uncertainty = _finite_metric(metrics, "policy/uncertainty")
    action_saturation = _finite_metric(metrics, "policy/action_saturation_fraction")
    finite_fraction = _finite_metric(metrics, "policy/finite_fraction")
    if (
        base is None
        or cvar is None
        or uncertainty is None
        or action_saturation is None
        or finite_fraction is None
        or finite_fraction < 1.0
    ):
        return -math.inf
    return float(
        base
        + float(cvar_coef) * cvar
        - float(uncertainty_penalty) * uncertainty
        - float(action_saturation_penalty) * action_saturation
    )


def _finite_metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _policy_model_selection_diagnostic_real_fields(
    *,
    selection_record: dict[str, Any],
    evaluation: dict[str, Any],
    selected_by_real: bool,
) -> dict[str, Any]:
    return {
        "policy_model_selection_diagnostic_real_selected": selected_by_real,
        "policy_model_selection_diagnostic_real_score": selection_record.get(
            "policy_selection_score"
        ),
        "policy_model_selection_diagnostic_real_mean_return": evaluation.get(
            "mean_return"
        ),
        "policy_model_selection_diagnostic_real_std_return": evaluation.get(
            "std_return"
        ),
        "policy_model_selection_diagnostic_real_failure_rate": evaluation.get(
            "failure_rate"
        ),
        "policy_model_selection_diagnostic_real_success_rate": evaluation.get(
            "success_rate"
        ),
        "policy_model_selection_diagnostic_real_return_p10": evaluation.get(
            "return_p10"
        ),
        "policy_model_selection_diagnostic_real_return_cvar10": evaluation.get(
            "return_cvar10"
        ),
        "policy_model_selection_diagnostic_real_nonfailure_mean_return": (
            evaluation.get("nonfailure_mean_return")
        ),
    }


def _merge_online_policy_baseline(
    final_outcome: dict[str, Any],
    initial_outcome: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(final_outcome)
    phase_initial_mean = merged.get("policy_initial_mean")
    phase_random_mean = merged.get("policy_random_mean")
    phase_improvement = merged.get("policy_improvement")
    phase_confirmation_improvement = merged.get("policy_confirmation_improvement")
    merged["policy_online_phase_initial_mean"] = phase_initial_mean
    merged["policy_online_phase_random_mean"] = phase_random_mean
    merged["policy_online_phase_improvement"] = phase_improvement
    merged["policy_online_phase_confirmation_improvement"] = (
        phase_confirmation_improvement
    )
    pre_online_trained_mean = initial_outcome.get("policy_trained_mean")
    merged["policy_pre_online_trained_mean"] = pre_online_trained_mean
    merged["policy_online_total_improvement_vs_pre_online"] = (
        merged["policy_trained_mean"] - pre_online_trained_mean
        if pre_online_trained_mean is not None
        and merged.get("policy_trained_mean") is not None
        else None
    )
    merged["policy_initial_mean"] = initial_outcome.get("policy_initial_mean")
    merged["policy_random_mean"] = initial_outcome.get("policy_random_mean")
    merged["policy_improvement"] = _optional_difference(
        merged.get("policy_trained_mean"),
        merged.get("policy_initial_mean"),
    )
    primary_improvement = (
        phase_improvement
        if phase_improvement is not None
        else merged.get("policy_improvement")
    )
    merged["policy_primary_improvement"] = primary_improvement
    merged["policy_primary_improvement_key"] = "policy_online_phase_improvement"
    merged["policy_trained_minus_random"] = _optional_difference(
        merged.get("policy_trained_mean"),
        merged.get("policy_random_mean"),
    )
    confirmation_enabled = bool(merged.get("policy_confirmation_enabled", False))
    if (
        confirmation_enabled
        and initial_outcome.get("policy_confirmation_initial_mean") is not None
    ):
        merged["policy_confirmation_initial_mean"] = initial_outcome[
            "policy_confirmation_initial_mean"
        ]
        merged["policy_confirmation_random_mean"] = initial_outcome[
            "policy_confirmation_random_mean"
        ]
        merged["policy_confirmation_improvement"] = _optional_difference(
            merged.get("policy_confirmation_trained_mean"),
            merged.get("policy_confirmation_initial_mean"),
        )
        merged["policy_confirmation_trained_minus_random"] = _optional_difference(
            merged.get("policy_confirmation_trained_mean"),
            merged.get("policy_confirmation_random_mean"),
        )
    primary_confirmation_improvement = (
        phase_confirmation_improvement
        if phase_confirmation_improvement is not None
        else merged.get("policy_confirmation_improvement")
    )
    merged["policy_primary_confirmation_improvement"] = primary_confirmation_improvement
    confirmation_passed = not confirmation_enabled or (
        primary_confirmation_improvement is not None
        and primary_confirmation_improvement > 0.0
        and merged.get("policy_confirmation_trained_minus_random") is not None
        and merged["policy_confirmation_trained_minus_random"] > 0.0
    )
    merged["policy_confirmation_passed"] = confirmation_passed
    policy_metrics = merged.get("policy_final_metrics", {})
    critic_metrics = merged.get("critic_final_metrics", {})
    nonregressed_from_pre_online = (
        merged["policy_online_total_improvement_vs_pre_online"] is None
        or merged["policy_online_total_improvement_vs_pre_online"] >= 0.0
    )
    merged["policy_passed"] = bool(
        _metrics_finite(policy_metrics)
        and _metrics_finite(critic_metrics)
        and primary_improvement is not None
        and primary_improvement > 0.0
        and nonregressed_from_pre_online
        and merged.get("policy_trained_mean") is not None
        and merged.get("policy_random_mean") is not None
        and merged["policy_trained_mean"] > merged["policy_random_mean"]
        and confirmation_passed
        and policy_metrics.get("policy/action_saturation_fraction", 1.0) < 0.75
    )
    return merged


def _online_history_metrics(
    online_history: list[dict[str, Any]],
    initial_policy_outcome: dict[str, Any],
) -> dict[str, Any]:
    returns = [
        item["actor_replay"].get("mean_return")
        for item in online_history
        if item.get("actor_replay", {}).get("mean_return") is not None
    ]
    policy_improvements = [
        item["policy"].get("policy_improvement")
        for item in online_history
        if item.get("policy", {}).get("policy_improvement") is not None
    ]
    policy_passed = [
        bool(item["policy"].get("policy_passed", False))
        for item in online_history
        if item.get("policy", {}).get("policy_training_enabled", False)
    ]
    policy_candidate_returns = [
        item["candidate_policy"].get("policy_trained_mean")
        for item in online_history
        if item.get("candidate_policy", {}).get("policy_trained_mean") is not None
    ]
    policy_champion_returns = [
        item["policy"].get("policy_champion_return")
        for item in online_history
        if item.get("policy", {}).get("policy_champion_return") is not None
    ]
    policy_update_acceptances = [
        bool(item["policy"].get("policy_update_accepted", False))
        for item in online_history
        if item.get("policy", {}).get("policy_update_accepted") is not None
    ]
    model_jepa_losses = [
        item["model_metrics"].get("model/jepa_loss")
        for item in online_history
        if item.get("model_metrics", {}).get("model/jepa_loss") is not None
    ]
    model_open_loop_losses = [
        item["model_metrics"].get("model/open_loop_loss")
        for item in online_history
        if item.get("model_metrics", {}).get("model/open_loop_loss") is not None
    ]
    candidate_refits = [
        item["candidate_refit"]
        for item in online_history
        if item.get("candidate_refit") is not None
    ]
    candidate_acceptances = [
        bool(item.get("model_update_accepted", False)) for item in candidate_refits
    ]
    candidate_recent_improvements = [
        item["gate"].get("recent_validation_improvement")
        for item in candidate_refits
        if item.get("gate", {}).get("recent_validation_improvement") is not None
    ]
    candidate_anchor_degradations = [
        item["gate"].get("anchor_validation_degradation")
        for item in candidate_refits
        if item.get("gate", {}).get("anchor_validation_degradation") is not None
    ]
    candidate_selected_updates = [
        item.get("checkpoint_selection", {}).get("candidate_selected_update")
        for item in candidate_refits
        if item.get("checkpoint_selection", {}).get("candidate_selected_update")
        is not None
    ]
    candidate_final_acceptances = [
        item.get("checkpoint_selection", {}).get("candidate_final_update_accepted")
        for item in candidate_refits
        if item.get("checkpoint_selection", {}).get("candidate_final_update_accepted")
        is not None
    ]
    baseline = initial_policy_outcome.get("policy_trained_mean")
    if not returns:
        return {
            "online_actor_replay_iterations": 0,
            "online_actor_replay_returns": [],
            "online_actor_replay_first_mean": None,
            "online_actor_replay_final_mean": None,
            "online_actor_replay_delta": None,
            "online_actor_replay_vs_initial_policy": None,
            "online_actor_replay_trend_passed": False,
            "online_policy_phase_improvements": policy_improvements,
            "online_policy_phase_final_improvement": (
                policy_improvements[-1] if policy_improvements else None
            ),
            "online_policy_phase_passes": policy_passed,
            "online_policy_phase_passed": bool(policy_passed and all(policy_passed)),
            "online_policy_candidate_returns": policy_candidate_returns,
            "online_policy_champion_returns": policy_champion_returns,
            "online_policy_update_acceptances": policy_update_acceptances,
            "online_policy_update_acceptance_rate": (
                float(np.mean(policy_update_acceptances))
                if policy_update_acceptances
                else None
            ),
            "online_policy_final_champion_return": (
                policy_champion_returns[-1] if policy_champion_returns else None
            ),
            "online_model_jepa_losses": model_jepa_losses,
            "online_model_open_loop_losses": model_open_loop_losses,
            "online_candidate_refit_iterations": len(candidate_refits),
            "online_model_update_acceptances": candidate_acceptances,
            "online_model_update_acceptance_rate": (
                float(np.mean(candidate_acceptances)) if candidate_acceptances else None
            ),
            "online_candidate_recent_validation_improvements": (
                candidate_recent_improvements
            ),
            "online_candidate_anchor_validation_degradations": (
                candidate_anchor_degradations
            ),
            "online_candidate_recent_validation_improvement_final": (
                candidate_recent_improvements[-1]
                if candidate_recent_improvements
                else None
            ),
            "online_candidate_anchor_validation_degradation_final": (
                candidate_anchor_degradations[-1]
                if candidate_anchor_degradations
                else None
            ),
            "online_candidate_selected_updates": candidate_selected_updates,
            "online_candidate_final_update_acceptances": candidate_final_acceptances,
            "online_pipeline_completed": False,
        }
    delta = returns[-1] - returns[0] if len(returns) >= 2 else None
    vs_initial = returns[-1] - baseline if baseline is not None else None
    actor_replay_nonregression = vs_initial is None or vs_initial >= 0.0
    actor_replay_trend = len(returns) < 2 or returns[-1] > returns[0]
    return {
        "online_actor_replay_iterations": len(returns),
        "online_actor_replay_returns": returns,
        "online_actor_replay_first_mean": returns[0],
        "online_actor_replay_final_mean": returns[-1],
        "online_actor_replay_delta": delta,
        "online_actor_replay_vs_initial_policy": vs_initial,
        "online_actor_replay_trend_passed": (
            actor_replay_trend and actor_replay_nonregression
        ),
        "online_policy_phase_improvements": policy_improvements,
        "online_policy_phase_final_improvement": (
            policy_improvements[-1] if policy_improvements else None
        ),
        "online_policy_phase_passes": policy_passed,
        "online_policy_phase_passed": bool(policy_passed and all(policy_passed)),
        "online_policy_candidate_returns": policy_candidate_returns,
        "online_policy_champion_returns": policy_champion_returns,
        "online_policy_update_acceptances": policy_update_acceptances,
        "online_policy_update_acceptance_rate": (
            float(np.mean(policy_update_acceptances))
            if policy_update_acceptances
            else None
        ),
        "online_policy_final_champion_return": (
            policy_champion_returns[-1] if policy_champion_returns else None
        ),
        "online_model_jepa_losses": model_jepa_losses,
        "online_model_open_loop_losses": model_open_loop_losses,
        "online_candidate_refit_iterations": len(candidate_refits),
        "online_model_update_acceptances": candidate_acceptances,
        "online_model_update_acceptance_rate": (
            float(np.mean(candidate_acceptances)) if candidate_acceptances else None
        ),
        "online_candidate_recent_validation_improvements": (
            candidate_recent_improvements
        ),
        "online_candidate_anchor_validation_degradations": (
            candidate_anchor_degradations
        ),
        "online_candidate_recent_validation_improvement_final": (
            candidate_recent_improvements[-1] if candidate_recent_improvements else None
        ),
        "online_candidate_anchor_validation_degradation_final": (
            candidate_anchor_degradations[-1] if candidate_anchor_degradations else None
        ),
        "online_candidate_selected_updates": candidate_selected_updates,
        "online_candidate_final_update_acceptances": candidate_final_acceptances,
        "online_pipeline_completed": True,
    }


def _real_step_accounting(
    *,
    initial_train_replay_env_steps: int,
    initial_validation_env_steps: int,
    initial_policy_outcome: dict[str, Any],
    online_history: list[dict[str, Any]],
    final_policy_eval: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Count real environment interactions used by one run.

    The main sample-efficiency number is the train replay count. Validation and
    policy-selection/evaluation interactions are kept separate because they are
    real environment steps but are not added to the training replay.
    """

    online_actor_replay_env_steps = sum(
        _maybe_int(item.get("actor_replay", {}).get("env_steps"))
        for item in online_history
    )
    online_validation_env_steps = sum(
        _maybe_int((item.get("recent_policy_validation") or {}).get("env_steps"))
        for item in online_history
    )
    initial_policy_eval_env_steps = _maybe_int(
        initial_policy_outcome.get("policy_total_eval_env_steps")
    )
    online_policy_eval_env_steps = sum(
        _maybe_int(
            (item.get("candidate_policy") or {}).get("policy_total_eval_env_steps")
        )
        for item in online_history
    )
    initial_policy_completed_steps = _maybe_int(
        initial_policy_outcome.get("policy_total_completed_episode_steps")
    )
    online_policy_completed_steps = sum(
        _maybe_int(
            (item.get("candidate_policy") or {}).get(
                "policy_total_completed_episode_steps"
            )
        )
        for item in online_history
    )
    final_policy_eval_env_steps = _eval_env_steps(final_policy_eval)
    final_policy_completed_steps = _eval_completed_episode_steps(final_policy_eval)

    train_replay_env_steps = (
        initial_train_replay_env_steps + online_actor_replay_env_steps
    )
    validation_replay_env_steps = (
        initial_validation_env_steps + online_validation_env_steps
    )
    policy_eval_env_steps = (
        initial_policy_eval_env_steps
        + online_policy_eval_env_steps
        + final_policy_eval_env_steps
    )
    train_plus_validation_env_steps = (
        train_replay_env_steps + validation_replay_env_steps
    )
    total_real_env_steps = train_plus_validation_env_steps + policy_eval_env_steps

    return {
        "real_initial_train_replay_env_steps": int(initial_train_replay_env_steps),
        "real_online_actor_replay_env_steps": int(online_actor_replay_env_steps),
        "real_train_replay_env_steps": int(train_replay_env_steps),
        "real_initial_validation_env_steps": int(initial_validation_env_steps),
        "real_online_validation_env_steps": int(online_validation_env_steps),
        "real_validation_replay_env_steps": int(validation_replay_env_steps),
        "real_train_plus_validation_env_steps": int(train_plus_validation_env_steps),
        "real_initial_policy_eval_env_steps": int(initial_policy_eval_env_steps),
        "real_online_policy_eval_env_steps": int(online_policy_eval_env_steps),
        "real_final_policy_eval_env_steps": int(final_policy_eval_env_steps),
        "real_policy_eval_env_steps": int(policy_eval_env_steps),
        "real_policy_eval_completed_episode_steps": int(
            initial_policy_completed_steps
            + online_policy_completed_steps
            + final_policy_completed_steps
        ),
        "real_total_env_steps": int(total_real_env_steps),
    }


def _dreamer_style_training_score(
    online_history: list[dict[str, Any]],
    *,
    window_env_steps: int,
    budget_env_steps: int,
) -> dict[str, Any]:
    """Training-return score from actor replay episodes near a step budget.

    DreamerV3 reports training returns near the environment-step budget rather
    than selecting the best checkpoint. For our pipeline, the closest matching
    stream is online actor replay, because those episodes are real interactions
    with the current training policy and are added to the training replay.
    """

    enabled = window_env_steps > 0 and budget_env_steps > 0
    episodes: list[dict[str, Any]] = []
    for item in online_history:
        actor_replay = item.get("actor_replay", {})
        returns = actor_replay.get("returns") or []
        lengths = actor_replay.get("lengths") or []
        finish_steps = actor_replay.get("episode_finish_train_env_steps") or []
        if len(finish_steps) != len(returns):
            continue
        for index, (value, finish_step) in enumerate(zip(returns, finish_steps)):
            length = lengths[index] if index < len(lengths) else None
            episodes.append(
                {
                    "online_iteration": item.get("iteration"),
                    "return": float(value),
                    "length": int(length) if length is not None else None,
                    "finish_train_env_step": int(finish_step),
                }
            )

    final_train_env_step = (
        max((item["finish_train_env_step"] for item in episodes), default=None)
        if episodes
        else None
    )
    if not enabled or final_train_env_step is None:
        return {
            "enabled": enabled,
            "budget_env_steps": int(budget_env_steps),
            "window_env_steps": int(window_env_steps),
            "budget_reached": False,
            "final_train_env_step": final_train_env_step,
            "window_start_env_step": None,
            "window_end_env_step": None,
            "episodes": 0,
            "mean_return": None,
            "std_return": None,
            "returns": [],
            "episode_finish_train_env_steps": [],
        }

    budget_reached = final_train_env_step >= budget_env_steps
    window_end = budget_env_steps if budget_reached else final_train_env_step
    window_start = max(0, window_end - window_env_steps)
    window_episodes = [
        item
        for item in episodes
        if window_start < item["finish_train_env_step"] <= window_end
    ]
    returns = [item["return"] for item in window_episodes]
    finish_steps = [item["finish_train_env_step"] for item in window_episodes]
    return {
        "enabled": enabled,
        "budget_env_steps": int(budget_env_steps),
        "window_env_steps": int(window_env_steps),
        "budget_reached": bool(budget_reached),
        "final_train_env_step": int(final_train_env_step),
        "window_start_env_step": int(window_start),
        "window_end_env_step": int(window_end),
        "episodes": len(returns),
        "mean_return": float(np.mean(returns)) if returns else None,
        "std_return": float(np.std(returns)) if returns else None,
        "returns": returns,
        "episode_finish_train_env_steps": finish_steps,
        "episode_records": window_episodes,
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


def summarize(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    main = [outcome for outcome in outcomes if outcome["control"] == "none"]
    controls = [outcome for outcome in outcomes if outcome["control"] != "none"]
    policy_enabled = any(
        outcome.get("policy_training_enabled", False) for outcome in outcomes
    )
    confirmation_enabled = any(
        outcome.get("policy_confirmation_enabled", False) for outcome in outcomes
    )
    online_enabled = any(
        outcome.get("online_actor_replay_iterations", 0) > 0 for outcome in outcomes
    )
    main_passed = all(
        outcome.get("world_model_passed", outcome["passed"]) for outcome in main
    )
    controls_finite = all(
        _metrics_finite(outcome["final_model_metrics"]) for outcome in controls
    )
    main_open_loop = _mean(main, "final_open_loop_loss")
    main_jepa = _mean(main, "final_jepa_loss")
    control_open_loop = _mean(controls, "final_open_loop_loss")
    control_jepa = _mean(controls, "final_jepa_loss")
    policy_comparison_key = _policy_comparison_key(outcomes)
    paired = _paired_control_differences(
        outcomes,
        policy_key=policy_comparison_key,
    )
    main_beats_controls_open_loop = not controls or (
        main_open_loop is not None
        and control_open_loop is not None
        and main_open_loop < control_open_loop
    )
    main_beats_controls_jepa = not controls or (
        main_jepa is not None and control_jepa is not None and main_jepa < control_jepa
    )
    paired_open_loop_ok = not paired or all(
        item["mean_open_loop_advantage"] > 0.0
        and item["runs_main_better_open_loop"] >= item["required_majority_pairs"]
        for item in paired.values()
    )
    paired_jepa_ok = not paired or all(
        item["mean_jepa_advantage"] > 0.0
        and item["runs_main_better_jepa"] >= item["required_majority_pairs"]
        for item in paired.values()
    )
    policy_main_passed = True
    policy_main_successes = 0
    policy_required_successes = 0
    policy_aggregate_improved = True
    policy_aggregate_beats_random = True
    policy_main_beats_controls = True
    paired_policy_ok = True
    policy_confirmation_successes = 0
    online_trend_successes = 0
    online_trend_passed = True
    if policy_enabled:
        policy_main_successes = int(
            sum(outcome.get("policy_passed", False) for outcome in main)
        )
        policy_required_successes = max(1, math.ceil((2 * len(main)) / 3))
        policy_confirmation_successes = int(
            sum(outcome.get("policy_confirmation_passed", False) for outcome in main)
        )
        online_trend_successes = int(
            sum(
                outcome.get("online_actor_replay_trend_passed", False)
                for outcome in main
                if outcome.get("online_actor_replay_iterations", 0) > 0
            )
        )
        online_trend_passed = bool(
            not online_enabled or online_trend_successes >= policy_required_successes
        )
        main_policy_improvement = _mean(main, policy_comparison_key)
        main_policy_minus_random = _mean(main, "policy_trained_minus_random")
        policy_aggregate_improved = bool(
            main_policy_improvement is not None and main_policy_improvement > 0.0
        )
        policy_aggregate_beats_random = bool(
            main_policy_minus_random is not None and main_policy_minus_random > 0.0
        )
        policy_main_passed = bool(
            main
            and policy_main_successes >= policy_required_successes
            and policy_aggregate_improved
            and policy_aggregate_beats_random
            and online_trend_passed
        )
        control_policy_improvement = _mean(controls, policy_comparison_key)
        policy_main_beats_controls = not controls or (
            main_policy_improvement is not None
            and control_policy_improvement is not None
            and main_policy_improvement > control_policy_improvement
        )
        paired_policy_ok = not paired or all(
            item.get("mean_policy_primary_improvement_advantage") is not None
            and item["mean_policy_primary_improvement_advantage"] > 0.0
            and item["runs_main_better_policy_primary"]
            >= item["required_majority_pairs"]
            for item in paired.values()
        )
    return {
        "world_model_passed": bool(
            main
            and main_passed
            and controls_finite
            and main_beats_controls_open_loop
            and main_beats_controls_jepa
            and paired_open_loop_ok
            and paired_jepa_ok
        ),
        "passed": bool(
            main
            and main_passed
            and controls_finite
            and main_beats_controls_open_loop
            and main_beats_controls_jepa
            and paired_open_loop_ok
            and paired_jepa_ok
            and policy_main_passed
            and policy_main_beats_controls
            and paired_policy_ok
            and online_trend_passed
        ),
        "main_runs_passed": int(
            sum(
                outcome.get("world_model_passed", outcome["passed"]) for outcome in main
            )
        ),
        "main_runs": len(main),
        "controls_finite": controls_finite,
        "main_beats_controls_open_loop": main_beats_controls_open_loop,
        "main_beats_controls_jepa": main_beats_controls_jepa,
        "paired_open_loop_ok": paired_open_loop_ok,
        "paired_jepa_ok": paired_jepa_ok,
        "policy_training_enabled": policy_enabled,
        "milestone": (
            "single_agent_direct_latent_imagination_rl"
            if policy_enabled
            else "single_agent_jepa_world_model_validation"
        ),
        "policy_main_passed": policy_main_passed,
        "policy_main_successes": policy_main_successes,
        "policy_required_successes": policy_required_successes,
        "policy_confirmation_enabled": confirmation_enabled,
        "policy_confirmation_successes": policy_confirmation_successes,
        "policy_aggregate_improved": policy_aggregate_improved,
        "policy_aggregate_beats_random": policy_aggregate_beats_random,
        "policy_main_beats_controls": policy_main_beats_controls,
        "paired_policy_ok": paired_policy_ok,
        "online_training_enabled": online_enabled,
        "online_trend_successes": online_trend_successes,
        "online_trend_passed": online_trend_passed,
        "policy_comparison_key": policy_comparison_key,
        "paired_control_differences": paired,
        "aggregate_initial_jepa_loss": _mean(main, "initial_jepa_loss"),
        "aggregate_final_jepa_loss": main_jepa,
        "aggregate_control_final_jepa_loss": control_jepa,
        "aggregate_initial_open_loop_loss": _mean(main, "initial_open_loop_loss"),
        "aggregate_final_open_loop_loss": main_open_loop,
        "aggregate_control_final_open_loop_loss": control_open_loop,
        "aggregate_policy_random_mean": _mean(main, "policy_random_mean"),
        "aggregate_policy_initial_mean": _mean(main, "policy_initial_mean"),
        "aggregate_policy_trained_mean": _mean(main, "policy_trained_mean"),
        "aggregate_policy_trained_failure_rate": _mean(
            main,
            "policy_trained_failure_rate",
        ),
        "aggregate_policy_trained_success_rate": _mean(
            main,
            "policy_trained_success_rate",
        ),
        "aggregate_policy_trained_return_p10": _mean(
            main,
            "policy_trained_return_p10",
        ),
        "aggregate_policy_trained_return_cvar10": _mean(
            main,
            "policy_trained_return_cvar10",
        ),
        "aggregate_policy_improvement": _mean(main, "policy_improvement"),
        "aggregate_policy_online_phase_improvement": _mean(
            main,
            "policy_online_phase_improvement",
        ),
        "aggregate_policy_online_actor_replay_delta": _mean(
            main,
            "online_actor_replay_delta",
        ),
        "aggregate_policy_online_actor_replay_vs_initial": _mean(
            main,
            "online_actor_replay_vs_initial_policy",
        ),
        "aggregate_model_update_acceptance_rate": _mean(
            main,
            "online_model_update_acceptance_rate",
        ),
        "aggregate_candidate_recent_validation_improvement": _mean(
            main,
            "online_candidate_recent_validation_improvement_final",
        ),
        "aggregate_candidate_anchor_validation_degradation": _mean(
            main,
            "online_candidate_anchor_validation_degradation_final",
        ),
        "aggregate_candidate_selected_update": _flat_mean(
            main,
            "online_candidate_selected_updates",
        ),
        "aggregate_candidate_final_update_acceptance_rate": _flat_mean(
            main,
            "online_candidate_final_update_acceptances",
        ),
        "aggregate_policy_update_acceptance_rate": _flat_mean(
            main,
            "online_policy_update_acceptances",
        ),
        "aggregate_policy_final_champion_return": _mean(
            main,
            "online_policy_final_champion_return",
        ),
        "aggregate_final_policy_eval_mean": _mean(
            main,
            "final_policy_eval_mean",
        ),
        "aggregate_final_policy_eval_std": _mean(
            main,
            "final_policy_eval_std",
        ),
        "aggregate_final_policy_eval_failure_rate": _mean(
            main,
            "final_policy_eval_failure_rate",
        ),
        "aggregate_final_policy_eval_success_rate": _mean(
            main,
            "final_policy_eval_success_rate",
        ),
        "aggregate_final_policy_eval_return_p10": _mean(
            main,
            "final_policy_eval_return_p10",
        ),
        "aggregate_final_policy_eval_return_cvar10": _mean(
            main,
            "final_policy_eval_return_cvar10",
        ),
        "aggregate_final_policy_eval_episodes": _mean(
            main,
            "final_policy_eval_episodes",
        ),
        "aggregate_final_policy_eval_env_steps": _mean(
            main,
            "final_policy_eval_env_steps",
        ),
        "aggregate_dreamer_style_train_return_mean": _mean(
            main,
            "dreamer_style_train_return_mean",
        ),
        "aggregate_dreamer_style_train_return_std": _mean(
            main,
            "dreamer_style_train_return_std",
        ),
        "aggregate_dreamer_style_train_return_episodes": _mean(
            main,
            "dreamer_style_train_return_episodes",
        ),
        "aggregate_dreamer_style_train_return_budget_reached": _mean(
            main,
            "dreamer_style_train_return_budget_reached",
        ),
        "aggregate_real_train_replay_env_steps": _mean(
            main,
            "real_train_replay_env_steps",
        ),
        "aggregate_real_validation_replay_env_steps": _mean(
            main,
            "real_validation_replay_env_steps",
        ),
        "aggregate_real_train_plus_validation_env_steps": _mean(
            main,
            "real_train_plus_validation_env_steps",
        ),
        "aggregate_real_policy_eval_env_steps": _mean(
            main,
            "real_policy_eval_env_steps",
        ),
        "aggregate_real_total_env_steps": _mean(
            main,
            "real_total_env_steps",
        ),
        "aggregate_real_policy_eval_completed_episode_steps": _mean(
            main,
            "real_policy_eval_completed_episode_steps",
        ),
        "aggregate_policy_primary_improvement": _mean(
            main,
            policy_comparison_key,
        ),
        "aggregate_policy_primary_confirmation_improvement": _mean(
            main,
            "policy_primary_confirmation_improvement",
        ),
        "aggregate_policy_trained_minus_random": _mean(
            main,
            "policy_trained_minus_random",
        ),
        "aggregate_control_policy_improvement": _mean(
            controls,
            "policy_improvement",
        ),
        "aggregate_control_policy_online_phase_improvement": _mean(
            controls,
            "policy_online_phase_improvement",
        ),
        "aggregate_control_policy_primary_improvement": _mean(
            controls,
            policy_comparison_key,
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
    for value in metrics.values():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            return False
    return True


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := _metric_value(row, key)) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _flat_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = _metric_value(row, key)
        if value is None:
            continue
        if isinstance(value, list):
            values.extend(item for item in value if item is not None)
        else:
            values.append(value)
    if not values:
        return None
    return float(np.mean(values))


def _metric_value(row: dict[str, Any], key: str) -> Any | None:
    if key == "policy_primary_improvement":
        return row.get(
            "policy_primary_improvement",
            row.get(
                "policy_online_phase_improvement",
                row.get("policy_improvement"),
            ),
        )
    return row.get(key)


def _policy_comparison_key(outcomes: list[dict[str, Any]]) -> str:
    if any("policy_primary_improvement" in outcome for outcome in outcomes):
        return "policy_primary_improvement"
    if any("policy_online_phase_improvement" in outcome for outcome in outcomes):
        return "policy_online_phase_improvement"
    return "policy_improvement"


def _paired_control_differences(
    outcomes: list[dict[str, Any]],
    *,
    policy_key: str,
) -> dict[str, dict[str, Any]]:
    main_by_run = {
        outcome["run_index"]: outcome
        for outcome in outcomes
        if outcome["control"] == "none"
    }
    result: dict[str, dict[str, Any]] = {}
    for control in sorted({outcome["control"] for outcome in outcomes} - {"none"}):
        jepa_advantages = []
        open_loop_advantages = []
        policy_improvement_advantages = []
        policy_online_phase_advantages = []
        policy_primary_advantages = []
        for outcome in outcomes:
            if outcome["control"] != control:
                continue
            main = main_by_run.get(outcome["run_index"])
            if main is None:
                continue
            jepa_advantages.append(outcome["final_jepa_loss"] - main["final_jepa_loss"])
            open_loop_advantages.append(
                outcome["final_open_loop_loss"] - main["final_open_loop_loss"]
            )
            if "policy_improvement" in outcome and "policy_improvement" in main:
                policy_improvement_advantages.append(
                    main["policy_improvement"] - outcome["policy_improvement"]
                )
            main_online = _metric_value(main, "policy_online_phase_improvement")
            control_online = _metric_value(outcome, "policy_online_phase_improvement")
            if main_online is not None and control_online is not None:
                policy_online_phase_advantages.append(main_online - control_online)
            main_primary = _metric_value(main, policy_key)
            control_primary = _metric_value(outcome, policy_key)
            if main_primary is not None and control_primary is not None:
                policy_primary_advantages.append(main_primary - control_primary)
        result[control] = {
            "pairs": len(jepa_advantages),
            "required_majority_pairs": _required_majority(len(jepa_advantages)),
            "mean_jepa_advantage": (
                float(np.mean(jepa_advantages)) if jepa_advantages else None
            ),
            "mean_open_loop_advantage": (
                float(np.mean(open_loop_advantages)) if open_loop_advantages else None
            ),
            "runs_main_better_jepa": int(np.sum(np.asarray(jepa_advantages) > 0.0)),
            "runs_main_better_open_loop": int(
                np.sum(np.asarray(open_loop_advantages) > 0.0)
            ),
            "mean_policy_improvement_advantage": (
                float(np.mean(policy_improvement_advantages))
                if policy_improvement_advantages
                else None
            ),
            "mean_policy_online_phase_improvement_advantage": (
                float(np.mean(policy_online_phase_advantages))
                if policy_online_phase_advantages
                else None
            ),
            "mean_policy_primary_improvement_advantage": (
                float(np.mean(policy_primary_advantages))
                if policy_primary_advantages
                else None
            ),
            "runs_main_better_policy": int(
                np.sum(np.asarray(policy_improvement_advantages) > 0.0)
            ),
            "runs_main_better_policy_online_phase": int(
                np.sum(np.asarray(policy_online_phase_advantages) > 0.0)
            ),
            "runs_main_better_policy_primary": int(
                np.sum(np.asarray(policy_primary_advantages) > 0.0)
            ),
        }
    return result


def _required_majority(count: int) -> int:
    return max(1, math.ceil((2 * count) / 3)) if count > 0 else 0


if __name__ == "__main__":
    main()
