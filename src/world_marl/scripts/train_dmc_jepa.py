"""Fit a decoder-free JEPA world model on DeepMind Control Suite rollouts.

This is the first DMC rung: random continuous-control replay plus latent
prediction, reward prediction, and continue prediction. It intentionally does
not train a continuous actor yet.
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

from world_marl.checkpointing import load_params, save_checkpoint
from world_marl.envs.dmc_adapter import DMCVectorAdapter, dmc_env_name
from world_marl.jepa.models import JepaConfig, JepaWorldModel
from world_marl.jepa.replay import ReplayBatch, SequenceReplayBuffer
from world_marl.jepa.training import (
    ControlMode,
    create_jepa_train_state,
    evaluate_open_loop,
    train_model_step,
    world_model_loss,
)
from world_marl.logging import RunLogger, dependency_versions, timestamp, to_jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="dmc:cartpole/swingup")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--collect-steps", type=int, default=2048)
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
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--reward-weight", type=float, default=1.0)
    parser.add_argument("--continue-weight", type=float, default=1.0)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--out-dir", default="runs/dmc_jepa")
    parser.add_argument(
        "--controls",
        nargs="+",
        choices=("none", "no-action-world-model", "shuffled-action-replay"),
        default=("none",),
    )
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in (
        "num_envs",
        "collect_steps",
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
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if not args.env.startswith("dmc:"):
        parser.error("--env must be formatted as dmc:<domain>/<task>")
    if args.model_horizon != 1:
        parser.error("--model-horizon must be 1 for this DMC milestone")
    if args.context_window != 1:
        parser.error("--context-window must be 1 for this DMC milestone")
    if args.collect_steps < args.chunk_length + args.open_loop_horizon:
        parser.error("--collect-steps must cover chunk-length + open-loop-horizon")


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


def run_one(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    run_index: int,
    control: ControlMode,
) -> dict[str, Any]:
    logger = RunLogger(run_dir)
    seed = args.seed + 10_000 * run_index + _control_seed_offset(control)
    adapter = DMCVectorAdapter(
        dmc_env_name(args.env),
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        seed=seed,
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
            regularizer=args.regularizer,
            isotropy_weight=args.isotropy_weight,
            sigreg_knots=args.sigreg_knots,
            sigreg_num_proj=args.sigreg_num_proj,
            reward_weight=args.reward_weight,
            continue_weight=args.continue_weight,
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
        )
        del observations
        logger.write_json(
            "replay.json",
            {
                "env_steps": env_steps,
                "steps_per_env": args.collect_steps,
                "size_per_env": replay.size,
                "observation_dim": config.observation_dim,
                "action_dim": config.action_dim,
            },
        )

        eval_batch = replay.sample(
            np_rng,
            batch_size=args.batch_size,
            chunk_length=args.chunk_length,
            max_horizon=max(args.model_horizon, args.open_loop_horizon),
        )
        rng, eval_key = jax.random.split(rng)
        initial_metrics = _evaluate_model(
            state,
            eval_key,
            eval_batch,
            config,
            chunk_length=args.chunk_length,
            open_loop_horizon=args.open_loop_horizon,
            control=control,
        )
        logger.write_json("initial_model_metrics.json", initial_metrics)

        loss_history: list[float] = []
        metrics = initial_metrics
        for step_index in range(1, args.train_steps + 1):
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
            loss_history.append(float(metrics["model/total_loss"]))
            if (
                step_index == 1
                or step_index == args.train_steps
                or step_index % args.eval_interval == 0
            ):
                logger.append_metrics(
                    {
                        "phase": "world_model",
                        "update": step_index,
                        "env_steps": env_steps,
                        "control": control,
                        **metrics,
                    }
                )

        rng, eval_key = jax.random.split(rng)
        final_batch = replay.sample(
            np_rng,
            batch_size=args.batch_size,
            chunk_length=args.chunk_length,
            max_horizon=max(args.model_horizon, args.open_loop_horizon),
        )
        final_metrics = _evaluate_model(
            state,
            eval_key,
            final_batch,
            config,
            chunk_length=args.chunk_length,
            open_loop_horizon=args.open_loop_horizon,
            control=control,
        )
        logger.write_json("model_metrics_final.json", final_metrics)
        logger.plot_world_model_loss(loss_history, filename="dmc_world_model_loss.png")

        checkpoint_dir = run_dir / "checkpoint"
        save_checkpoint(
            checkpoint_dir,
            state,
            metadata={
                "algorithm": "dmc_sigreg_jepa_world_model",
                "env": args.env,
                "control": control,
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

        outcome = {
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
            "reload_max_abs_prediction_diff": reload_diff,
            "final_model_metrics": final_metrics,
            "passed": _run_passed(initial_metrics, final_metrics, reload_diff),
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
) -> tuple[np.ndarray, int]:
    for _ in range(steps):
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


def _evaluate_model(
    state,
    key: jax.Array,
    batch: ReplayBatch,
    config: JepaConfig,
    *,
    chunk_length: int,
    open_loop_horizon: int,
    control: ControlMode,
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
    main_passed = all(outcome["passed"] for outcome in main)
    controls_finite = all(_metrics_finite(outcome["final_model_metrics"]) for outcome in controls)
    return {
        "passed": bool(main and main_passed and controls_finite),
        "main_runs_passed": int(sum(outcome["passed"] for outcome in main)),
        "main_runs": len(main),
        "controls_finite": controls_finite,
        "aggregate_initial_jepa_loss": _mean(main, "initial_jepa_loss"),
        "aggregate_final_jepa_loss": _mean(main, "final_jepa_loss"),
        "aggregate_initial_open_loop_loss": _mean(main, "initial_open_loop_loss"),
        "aggregate_final_open_loop_loss": _mean(main, "final_open_loop_loss"),
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
    )


def _metrics_finite(metrics: dict[str, Any]) -> bool:
    for value in metrics.values():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            return False
    return True


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(np.mean([row[key] for row in rows]))


def _control_seed_offset(control: str) -> int:
    return {
        "none": 0,
        "no-action-world-model": 100_000,
        "shuffled-action-replay": 200_000,
    }[control]


if __name__ == "__main__":
    main()
