"""Validate a representation-space SIGReg-JEPA world model on DMC rollouts.

By default this is the first DMC rung: random continuous-control replay plus
held-out latent prediction, reward prediction, and continue prediction. Passing
``--policy-train-steps`` enables the next rung: reset actor/value heads, freeze
the JEPA world model, train a deterministic continuous actor inside the latent
model, then evaluate that actor in the real DMC environment.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from world_marl.checkpointing import load_params, save_checkpoint
from world_marl.envs.dmc_adapter import DMCVectorAdapter, dmc_env_name
from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.replay import ReplayBatch, SequenceReplayBuffer
from world_marl.jepa.training import (
    ControlMode,
    continuous_candidate_distill_step,
    continuous_critic_warmup_step,
    continuous_policy_train_step,
    create_jepa_train_state,
    evaluate_open_loop,
    reset_policy_heads,
    select_continuous_actions,
    train_model_step,
    world_model_loss,
)
from world_marl.logging import RunLogger, dependency_versions, timestamp, to_jsonable

MIN_TERMINAL_FRACTION_FOR_CONTINUE_BASELINE = 0.01
FROZEN_RANDOM_WORLD_MODEL_CONTROL = "frozen-random-world-model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="dmc:cartpole/swingup")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--dmc-workers", type=int, default=1)
    parser.add_argument("--collect-steps", type=int, default=2048)
    parser.add_argument("--validation-steps", type=int, default=512)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--chunk-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--model-horizon", type=int, default=1)
    parser.add_argument("--open-loop-horizon", type=int, default=5)
    parser.add_argument("--context-window", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--policy-train-steps", type=int, default=0)
    parser.add_argument("--policy-batch-size", type=int, default=None)
    parser.add_argument("--critic-warmup-steps", type=int, default=1000)
    parser.add_argument("--critic-horizon", type=int, default=32)
    parser.add_argument("--imag-horizon", type=int, default=5)
    parser.add_argument(
        "--policy-objective",
        choices=("candidate-distill", "direct"),
        default="candidate-distill",
        help=(
            "candidate-distill scores sampled actions with the frozen latent model "
            "and trains a direct actor toward the best candidates. direct "
            "backpropagates reward-only or lambda returns through the model."
        ),
    )
    parser.add_argument(
        "--policy-return-mode",
        choices=("reward-only", "lambda"),
        default="reward-only",
        help=(
            "Use finite-horizon predicted rewards by default. The old lambda mode "
            "bootstraps from the learned value head and is mainly a diagnostic."
        ),
    )
    parser.add_argument("--value-clip", type=float, default=100.0)
    parser.add_argument("--action-saturation-threshold", type=float, default=0.95)
    parser.add_argument("--num-policy-candidates", type=int, default=64)
    parser.add_argument("--candidate-min-gap", type=float, default=1e-3)
    parser.add_argument("--policy-action-l2-coef", type=float, default=1e-3)
    parser.add_argument("--policy-eval-episodes", type=int, default=20)
    parser.add_argument("--policy-eval-num-envs", type=int, default=None)
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
    parser.add_argument("--online-train-steps", type=int, default=None)
    parser.add_argument("--online-policy-train-steps", type=int, default=None)
    parser.add_argument(
        "--online-reset-actor",
        action="store_true",
        help="Reset actor/value heads at the start of each online policy phase.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda-return", type=float, default=0.95)
    parser.add_argument(
        "--isotropy-weight",
        "--sigreg-weight",
        dest="isotropy_weight",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--regularizer",
        choices=("sigreg", "isotropy", "none"),
        default="sigreg",
    )
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=256)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--continue-weight", type=float, default=1.0)
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
    args = parser.parse_args()
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in (
        "num_envs",
        "dmc_workers",
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
    if args.online_collect_steps is not None and args.online_collect_steps < 1:
        parser.error("--online-collect-steps must be >= 1")
    for name in ("online_train_steps", "online_policy_train_steps"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    for name in ("critic_horizon",):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.num_policy_candidates < 2:
        parser.error("--num-policy-candidates must be >= 2")
    if args.candidate_min_gap < 0.0:
        parser.error("--candidate-min-gap must be >= 0")
    if args.policy_action_l2_coef < 0.0:
        parser.error("--policy-action-l2-coef must be >= 0")
    if args.value_clip <= 0.0:
        parser.error("--value-clip must be > 0")
    if not 0.0 < args.action_saturation_threshold <= 1.0:
        parser.error("--action-saturation-threshold must be in (0, 1]")
    if args.policy_batch_size is not None and args.policy_batch_size < 1:
        parser.error("--policy-batch-size must be >= 1")
    if args.policy_eval_num_envs is not None and args.policy_eval_num_envs < 1:
        parser.error("--policy-eval-num-envs must be >= 1")
    if (
        args.policy_selection_num_envs is not None
        and args.policy_selection_num_envs < 1
    ):
        parser.error("--policy-selection-num-envs must be >= 1")
    if not args.env.startswith("dmc:"):
        parser.error("--env must be formatted as dmc:<domain>/<task>")
    if args.model_horizon != 1:
        parser.error("--model-horizon must be 1 for this DMC milestone")
    if args.context_window != 1:
        parser.error("--context-window must be 1 for this DMC milestone")
    min_steps = args.chunk_length + args.open_loop_horizon
    if args.collect_steps < min_steps:
        parser.error("--collect-steps must cover chunk-length + open-loop-horizon")
    if args.validation_steps < min_steps:
        parser.error("--validation-steps must cover chunk-length + open-loop-horizon")
    if args.policy_train_steps > 0 and args.collect_steps < args.critic_horizon + 1:
        parser.error("--collect-steps must cover critic-horizon + 1")
    if args.online_iterations > 0 and args.policy_train_steps == 0:
        parser.error("--online-iterations requires --policy-train-steps > 0")


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.out_dir) / f"dmc_jepa_{timestamp()}"
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


def run_one(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    run_index: int,
    control: ControlMode,
) -> dict[str, Any]:
    logger = RunLogger(run_dir)
    seed = args.seed + 10_000 * run_index
    adapter = DMCVectorAdapter(
        dmc_env_name(args.env),
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        seed=seed,
        num_workers=args.dmc_workers,
    )
    try:
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
            regularizer=args.regularizer,
            isotropy_weight=args.isotropy_weight,
            sigreg_knots=args.sigreg_knots,
            sigreg_num_proj=args.sigreg_num_proj,
            reward_weight=args.reward_weight,
            continue_weight=args.continue_weight,
            gamma=args.gamma,
            lambda_return=args.lambda_return,
        )
        logger.write_json(
            "config.json",
            {
                "args": vars(args),
                "run_index": run_index,
                "seed": seed,
                "control": control,
                "observation_shape": adapter.observation_shape,
                "action_shape": adapter.action_shape,
                "action_low": adapter.action_low,
                "action_high": adapter.action_high,
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
        replay = SequenceReplayBuffer(
            capacity=max(2, math.ceil(args.replay_capacity / args.num_envs)),
            num_envs=args.num_envs,
            observation_shape=(config.observation_dim,),
            action_shape=(adapter.action_dim,),
            action_dtype=np.float32,
        )

        observations = adapter.reset()
        observations, env_steps = _collect_random_steps(
            adapter,
            observations,
            np_rng,
            replay,
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
                "observation_dim": config.observation_dim,
                "action_dim": config.action_dim,
            },
        )

        validation_replay = _collect_validation_replay(
            args,
            config,
            seed=seed + 1_000_000,
        )
        logger.write_json(
            "validation_replay.json",
            {
                "env_steps": args.validation_steps * args.num_envs,
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
            logger.plot_world_model_loss(loss_history, filename="dmc_world_model_loss.png")

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
        online_history: list[dict[str, Any]] = []
        for online_index in range(1, args.online_iterations + 1):
            phase = f"online_{online_index:03d}"
            online_collect_steps = args.online_collect_steps or args.collect_steps
            observations, added_env_steps, collect_metrics = _collect_policy_steps(
                adapter,
                observations,
                state,
                config,
                replay,
                steps=online_collect_steps,
                action_low=adapter.action_low,
                action_high=adapter.action_high,
                desc=f"{control} {phase} collect actor replay",
                quiet=args.quiet,
            )
            env_steps += added_env_steps
            collect_payload = {
                **collect_metrics,
                "total_env_steps": env_steps,
                "replay_size_per_env": replay.size,
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
            online_loss_history: list[float] = []
            if online_train_steps > 0:
                state, rng, _, online_loss_history = _fit_world_model(
                    args,
                    logger,
                    state,
                    rng,
                    replay,
                    config,
                    np_rng=np_rng,
                    steps=online_train_steps,
                    control=control,
                    phase=f"{phase}_world_model",
                    desc=f"{control} {phase} fit world model",
                    env_steps=env_steps,
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
                state = online_policy_outcome["state"]
                rng = online_policy_outcome["rng"]
                policy_outcome = online_policy_outcome
                online_policy_payload = online_policy_outcome["outcome"]
            else:
                online_policy_payload = {"policy_training_enabled": False}
            online_history.append(
                {
                    "iteration": online_index,
                    "actor_replay": collect_payload,
                    "model_metrics": online_metrics,
                    "policy": online_policy_payload,
                    "world_model_train_steps": online_train_steps,
                    "policy_train_steps": online_policy_train_steps,
                }
            )
        if online_history:
            policy_outcome["outcome"] = _merge_online_policy_baseline(
                policy_outcome["outcome"],
                initial_policy_outcome,
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

        checkpoint_dir = run_dir / "checkpoint"
        save_checkpoint(
            checkpoint_dir,
            state,
            metadata={
                "algorithm": "dmc_sigreg_jepa_world_model",
                "env": args.env,
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

        world_model_passed = _run_passed(initial_metrics, final_metrics, reload_diff)
        outcome = {
            "run_index": run_index,
            "control": control,
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "target": "dmc:p(z_next, reward, continue | z, continuous_action)",
            "initial_jepa_loss": initial_metrics["model/jepa_loss"],
            "final_jepa_loss": final_metrics["model/jepa_loss"],
            "initial_open_loop_loss": initial_metrics["model/open_loop_loss"],
            "final_open_loop_loss": final_metrics["model/open_loop_loss"],
            "final_reward_loss": final_metrics["model/reward_loss"],
            "final_reward_constant_mse": final_metrics["model/reward_constant_mse"],
            "final_continue_loss": final_metrics["model/continue_loss"],
            "final_continue_constant_bce": final_metrics[
                "model/continue_constant_bce"
            ],
            "jepa_loss_delta": initial_metrics["model/jepa_loss"]
            - final_metrics["model/jepa_loss"],
            "open_loop_loss_delta": initial_metrics["model/open_loop_loss"]
            - final_metrics["model/open_loop_loss"],
            "reload_max_abs_prediction_diff": reload_diff,
            "final_model_metrics": final_metrics,
            "online_iterations": args.online_iterations,
            "online_history": online_history,
            **policy_outcome["outcome"],
            "world_model_passed": world_model_passed,
            "passed": world_model_passed,
        }
        logger.write_json("outcome.json", outcome)
        return to_jsonable(outcome)
    finally:
        adapter.close()


def _collect_random_steps(
    adapter: DMCVectorAdapter,
    observations: np.ndarray,
    rng: np.random.Generator,
    replay: SequenceReplayBuffer,
    *,
    steps: int,
    desc: str,
    quiet: bool,
) -> tuple[np.ndarray, int]:
    for _ in tqdm(range(steps), desc=desc, unit="step", disable=quiet):
        actions = adapter.sample_actions(rng)
        step = adapter.step(actions)
        replay.add_step(
            observations=observations[:, 0],
            actions=actions[:, 0],
            rewards=step.rewards[:, 0],
            dones=step.dones[:, 0],
        )
        observations = step.observations
    return observations, steps * adapter.num_envs


def _collect_policy_steps(
    adapter: DMCVectorAdapter,
    observations: np.ndarray,
    state,
    config: JepaConfig,
    replay: SequenceReplayBuffer,
    *,
    steps: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    desc: str,
    quiet: bool,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    action_low_jax = jnp.asarray(action_low, dtype=jnp.float32)
    action_high_jax = jnp.asarray(action_high, dtype=jnp.float32)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    progress = tqdm(range(steps), desc=desc, unit="step", disable=quiet)
    for _ in progress:
        actions = np.asarray(
            select_continuous_actions(
                state,
                jnp.asarray(observations[:, 0], dtype=jnp.float32),
                config,
                action_low_jax,
                action_high_jax,
            )
        )
        step = adapter.step(actions[:, None, :])
        replay.add_step(
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
        "completed_episodes": len(completed_returns),
        "mean_return": (
            float(np.mean(completed_returns)) if completed_returns else None
        ),
        "std_return": (
            float(np.std(completed_returns)) if completed_returns else None
        ),
        "mean_length": (
            float(np.mean(completed_lengths)) if completed_lengths else None
        ),
        "returns": completed_returns,
        "lengths": completed_lengths,
    }
    return observations, steps * adapter.num_envs, metrics


def _collect_validation_replay(
    args: argparse.Namespace,
    config: JepaConfig,
    *,
    seed: int,
) -> SequenceReplayBuffer:
    adapter = DMCVectorAdapter(
        dmc_env_name(args.env),
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        seed=seed,
        num_workers=args.dmc_workers,
    )
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
) -> tuple[Any, jax.Array, dict[str, Any], list[float]]:
    loss_history: list[float] = []
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
        )
        total_loss = float(metrics["model/total_loss"])
        jepa_loss = float(metrics["model/jepa_loss"])
        loss_history.append(total_loss)
        fit_steps.set_postfix(loss=f"{total_loss:.4g}", jepa=f"{jepa_loss:.4g}")
        if (
            step_index == 1
            or step_index == steps
            or step_index % args.eval_interval == 0
        ):
            logger.append_metrics(
                {
                    "phase": phase,
                    "update": step_index,
                    "env_steps": env_steps,
                    "control": control,
                    **metrics,
                }
            )
    return state, rng, to_jsonable(metrics), loss_history


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
    selection_enabled = args.policy_selection_interval > 0
    selection_num_envs = args.policy_selection_num_envs or min(
        args.num_envs,
        args.policy_selection_episodes,
    )
    artifact_prefix = "" if phase == "policy" else f"{phase}_"
    metric_phase_prefix = "" if phase == "policy" else f"{phase}_"

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
    )
    logger.write_json(f"{artifact_prefix}random_policy_evaluation.json", random_eval)
    logger.write_json(f"{artifact_prefix}initial_policy_evaluation.json", initial_eval)

    best_state = state
    best_policy_step = 0
    best_policy_metrics_json: dict[str, Any] = {
        "policy/selected_initial_actor": True,
    }
    selection_history: list[dict[str, Any]] = []
    best_selection_eval: dict[str, Any] | None = None
    best_selection_mean = -math.inf
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
        logger.write_json(f"{artifact_prefix}policy_selection_initial.json", selection_eval)

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
            )
            critic_loss = float(critic_metrics["critic/total_loss"])
            target_mean = float(critic_metrics["critic/target_mean"])
            critic_steps.set_postfix(
                loss=f"{critic_loss:.4g}",
                target=f"{target_mean:.4g}",
            )
            if (
                step_index == 1
                or step_index == args.critic_warmup_steps
                or step_index % args.eval_interval == 0
            ):
                logger.append_metrics(
                    {
                        "phase": f"{metric_phase_prefix}real_return_critic_warmup",
                        "update": step_index,
                        "control": control,
                        "policy_phase": phase,
                        **critic_metrics,
                    }
                )

    policy_loss_history: list[float] = []
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
            chunk_length=1,
            max_horizon=1,
        )
        rng, policy_key = jax.random.split(rng)
        if args.policy_objective == "candidate-distill":
            state, metrics = continuous_candidate_distill_step(
                state,
                policy_key,
                batch.observations[:, 0],
                config,
                action_low_jax,
                action_high_jax,
                imag_horizon=args.imag_horizon,
                control=control,
                num_candidates=args.num_policy_candidates,
                candidate_min_gap=args.candidate_min_gap,
                action_l2_coef=args.policy_action_l2_coef,
                action_saturation_threshold=args.action_saturation_threshold,
            )
        else:
            state, metrics = continuous_policy_train_step(
                state,
                policy_key,
                batch.observations[:, 0],
                config,
                action_low_jax,
                action_high_jax,
                imag_horizon=args.imag_horizon,
                control=control,
                policy_return_mode=args.policy_return_mode,
                value_clip=args.value_clip,
                action_saturation_threshold=args.action_saturation_threshold,
            )
        policy_loss = float(metrics["policy/total_loss"])
        progress_score = float(
            metrics.get(
                "policy/imagined_return",
                metrics.get("policy/candidate_best_score", policy_loss),
            )
        )
        policy_loss_history.append(policy_loss)
        policy_steps.set_postfix(
            loss=f"{policy_loss:.4g}",
            score=f"{progress_score:.4g}",
        )
        if (
            step_index == 1
            or step_index == policy_train_steps
            or step_index % args.eval_interval == 0
        ):
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
        logger.write_json(f"{artifact_prefix}policy_selection_history.json", selection_history)
        logger.write_json(
            f"{artifact_prefix}best_policy_selection_evaluation.json",
            {
                "best_policy_step": best_policy_step,
                "evaluation": best_selection_eval,
            },
        )
    trained_eval = _evaluate_continuous_policy(
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
    logger.plot_world_model_loss(
        policy_loss_history,
        filename=f"{artifact_prefix}frozen_model_policy_loss.png",
    )

    initial_mean = initial_eval["mean_return"]
    trained_mean = trained_eval["mean_return"]
    random_mean = random_eval["mean_return"]
    policy_metrics_json = (
        best_policy_metrics_json if selection_enabled else last_policy_metrics_json
    )
    critic_metrics_json = to_jsonable(critic_metrics)
    outcome = {
        "policy_training_enabled": True,
        "policy_phase": phase,
        "policy_reset_actor": reset_actor,
        "policy_train_steps": policy_train_steps,
        "policy_objective": args.policy_objective,
        "policy_return_mode": args.policy_return_mode,
        "policy_imag_horizon": args.imag_horizon,
        "num_policy_candidates": args.num_policy_candidates,
        "candidate_min_gap": args.candidate_min_gap,
        "policy_action_l2_coef": args.policy_action_l2_coef,
        "policy_eval_seed": policy_eval_seed,
        "policy_selection_enabled": selection_enabled,
        "policy_selection_seed": policy_selection_seed,
        "policy_selection_interval": args.policy_selection_interval,
        "policy_selection_episodes": args.policy_selection_episodes,
        "policy_selection_num_envs": selection_num_envs,
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
        "policy_final_metrics": policy_metrics_json,
        "policy_passed": bool(
            _metrics_finite(policy_metrics_json)
            and _metrics_finite(critic_metrics_json)
            and trained_mean > initial_mean
            and trained_mean > random_mean
            and policy_metrics_json.get("policy/action_saturation_fraction", 1.0)
            < 0.75
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
    adapter = DMCVectorAdapter(
        dmc_env_name(args.env),
        num_envs=num_envs,
        max_cycles=args.max_cycles,
        seed=seed,
        num_workers=min(args.dmc_workers, num_envs),
    )
    try:
        rng = np.random.default_rng(seed)
        observations = adapter.reset()
        del observations
        returns = []
        lengths = []
        with tqdm(
            total=target_episodes,
            desc=desc,
            unit="episode",
            disable=args.quiet,
        ) as progress:
            while len(returns) < target_episodes:
                before = len(returns)
                step = adapter.step(adapter.sample_actions(rng))
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
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "returns": returns,
            "lengths": lengths,
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
) -> dict[str, Any]:
    target_episodes = args.policy_eval_episodes if episodes is None else episodes
    adapter = DMCVectorAdapter(
        dmc_env_name(args.env),
        num_envs=num_envs,
        max_cycles=args.max_cycles,
        seed=seed,
        num_workers=min(args.dmc_workers, num_envs),
    )
    try:
        observations = adapter.reset()
        returns = []
        lengths = []
        with tqdm(
            total=target_episodes,
            desc=desc,
            unit="episode",
            disable=args.quiet,
        ) as progress:
            while len(returns) < target_episodes:
                before = len(returns)
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
    }


def _merge_online_policy_baseline(
    final_outcome: dict[str, Any],
    initial_outcome: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(final_outcome)
    phase_initial_mean = merged.get("policy_initial_mean")
    phase_random_mean = merged.get("policy_random_mean")
    phase_improvement = merged.get("policy_improvement")
    merged["policy_online_phase_initial_mean"] = phase_initial_mean
    merged["policy_online_phase_random_mean"] = phase_random_mean
    merged["policy_online_phase_improvement"] = phase_improvement
    merged["policy_initial_mean"] = initial_outcome["policy_initial_mean"]
    merged["policy_random_mean"] = initial_outcome["policy_random_mean"]
    merged["policy_improvement"] = (
        merged["policy_trained_mean"] - merged["policy_initial_mean"]
    )
    primary_improvement = (
        phase_improvement
        if phase_improvement is not None
        else merged["policy_improvement"]
    )
    merged["policy_primary_improvement"] = primary_improvement
    merged["policy_primary_improvement_key"] = "policy_online_phase_improvement"
    merged["policy_trained_minus_random"] = (
        merged["policy_trained_mean"] - merged["policy_random_mean"]
    )
    policy_metrics = merged.get("policy_final_metrics", {})
    critic_metrics = merged.get("critic_final_metrics", {})
    merged["policy_passed"] = bool(
        _metrics_finite(policy_metrics)
        and _metrics_finite(critic_metrics)
        and primary_improvement > 0.0
        and merged["policy_trained_mean"] > merged["policy_random_mean"]
        and policy_metrics.get("policy/action_saturation_fraction", 1.0) < 0.75
    )
    return merged


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
    _, metrics = world_model_loss(
        state.params,
        state.apply_fn,
        key,
        batch,
        config,
        chunk_length=chunk_length,
        control=control,
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
    main_beats_controls_open_loop = (
        not controls
        or (
            main_open_loop is not None
            and control_open_loop is not None
            and main_open_loop < control_open_loop
        )
    )
    main_beats_controls_jepa = (
        not controls
        or (
            main_jepa is not None
            and control_jepa is not None
            and main_jepa < control_jepa
        )
    )
    paired_open_loop_ok = not paired or all(
        item["mean_open_loop_advantage"] > 0.0 for item in paired.values()
    )
    paired_jepa_ok = not paired or all(
        item["mean_jepa_advantage"] > 0.0 for item in paired.values()
    )
    policy_main_passed = True
    policy_main_successes = 0
    policy_required_successes = 0
    policy_aggregate_improved = True
    policy_aggregate_beats_random = True
    policy_main_beats_controls = True
    paired_policy_ok = True
    if policy_enabled:
        policy_main_successes = int(
            sum(outcome.get("policy_passed", False) for outcome in main)
        )
        policy_required_successes = max(1, math.ceil((2 * len(main)) / 3))
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
        )
        control_policy_improvement = _mean(controls, policy_comparison_key)
        policy_main_beats_controls = (
            not controls
            or (
                main_policy_improvement is not None
                and control_policy_improvement is not None
                and main_policy_improvement > control_policy_improvement
            )
        )
        paired_policy_ok = not paired or all(
            item.get("mean_policy_primary_improvement_advantage") is not None
            and item["mean_policy_primary_improvement_advantage"] > 0.0
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
        ),
        "main_runs_passed": int(
            sum(outcome.get("world_model_passed", outcome["passed"]) for outcome in main)
        ),
        "main_runs": len(main),
        "controls_finite": controls_finite,
        "main_beats_controls_open_loop": main_beats_controls_open_loop,
        "main_beats_controls_jepa": main_beats_controls_jepa,
        "paired_open_loop_ok": paired_open_loop_ok,
        "paired_jepa_ok": paired_jepa_ok,
        "policy_training_enabled": policy_enabled,
        "policy_main_passed": policy_main_passed,
        "policy_main_successes": policy_main_successes,
        "policy_required_successes": policy_required_successes,
        "policy_aggregate_improved": policy_aggregate_improved,
        "policy_aggregate_beats_random": policy_aggregate_beats_random,
        "policy_main_beats_controls": policy_main_beats_controls,
        "paired_policy_ok": paired_policy_ok,
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
        "aggregate_policy_improvement": _mean(main, "policy_improvement"),
        "aggregate_policy_online_phase_improvement": _mean(
            main,
            "policy_online_phase_improvement",
        ),
        "aggregate_policy_primary_improvement": _mean(
            main,
            policy_comparison_key,
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
        < final_metrics["model/reward_constant_mse"]
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
    values = [
        value
        for row in rows
        if (value := _metric_value(row, key)) is not None
    ]
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
            jepa_advantages.append(
                outcome["final_jepa_loss"] - main["final_jepa_loss"]
            )
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
            "mean_jepa_advantage": (
                float(np.mean(jepa_advantages)) if jepa_advantages else None
            ),
            "mean_open_loop_advantage": (
                float(np.mean(open_loop_advantages))
                if open_loop_advantages
                else None
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


if __name__ == "__main__":
    main()
