"""Train a decoder-free isotropy-JEPA imagination actor-critic on Gymnax."""

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
from world_marl.envs.gymnax_adapter import GymnaxVectorAdapter, gymnax_env_name
from world_marl.evaluation import evaluate_policy, random_policy
from world_marl.jepa.models import JepaConfig
from world_marl.jepa.replay import SequenceReplayBuffer
from world_marl.jepa.training import (
    ControlMode,
    create_jepa_train_state,
    evaluate_open_loop,
    evaluate_world_model,
    policy_train_step,
    select_actions,
    train_model_step,
)
from world_marl.logging import RunLogger, dependency_versions, timestamp, to_jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="gymnax:CartPole-v1")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--total-env-steps", type=int, default=25_000)
    parser.add_argument("--env-steps-per-iter", type=int, default=16)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--chunk-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-updates-per-iter", type=int, default=2)
    parser.add_argument("--policy-updates-per-iter", type=int, default=1)
    parser.add_argument("--imag-horizon", type=int, default=5)
    parser.add_argument("--model-horizon", type=int, default=1)
    parser.add_argument("--context-window", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument(
        "--isotropy-weight",
        "--sigreg-weight",
        dest="isotropy_weight",
        type=float,
        default=0.05,
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda-return", type=float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=500)
    parser.add_argument("--out-dir", default="runs/jepa_cartpole")
    parser.add_argument(
        "--controls",
        nargs="+",
        choices=(
            "none",
            "no-action-world-model",
            "shuffled-action-replay",
            "no-sigreg",
            "weak-sigreg",
            "no-isotropy",
            "weak-isotropy",
        ),
        default=("none",),
    )
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in (
        "num_envs",
        "total_env_steps",
        "env_steps_per_iter",
        "replay_capacity",
        "chunk_length",
        "batch_size",
        "model_updates_per_iter",
        "policy_updates_per_iter",
        "imag_horizon",
        "model_horizon",
        "context_window",
        "latent_dim",
        "model_dim",
        "num_layers",
        "num_heads",
        "mlp_ratio",
        "eval_episodes",
        "eval_interval",
        "num_runs",
        "max_cycles",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if not args.env.startswith("gymnax:"):
        parser.error("--env must be formatted as gymnax:<env-id> for milestone 1")


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.out_dir) / f"jepa_{timestamp()}"
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
    env_name = gymnax_env_name(args.env)
    adapter = GymnaxVectorAdapter(
        env_name,
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        seed=seed,
    )
    try:
        config = JepaConfig(
            observation_dim=int(np.prod(adapter.observation_shape)),
            action_dim=adapter.action_dim,
            latent_dim=args.latent_dim,
            model_dim=args.model_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            max_horizon=args.model_horizon,
            context_window=args.context_window,
            learning_rate=args.learning_rate,
            actor_learning_rate=args.actor_learning_rate,
            isotropy_weight=args.isotropy_weight,
            gamma=args.gamma,
            lambda_return=args.lambda_return,
            entropy_coef=args.entropy_coef,
        )
        logger.write_json(
            "config.json",
            {
                "args": vars(args),
                "run_index": run_index,
                "seed": seed,
                "control": control,
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
        )

        random_eval = _evaluate_random(args, seed=seed + 1)
        initial_eval = _evaluate_state(args, state, config, seed=seed + 2)
        logger.write_json("random_baseline.json", random_eval)
        logger.write_json("initial_policy_evaluation.json", initial_eval)

        observations = adapter.reset()
        min_replay_steps = args.chunk_length + max(
            args.model_horizon,
            args.imag_horizon,
        )
        observations, rng, env_steps = _collect_steps(
            adapter,
            state,
            config,
            observations,
            rng,
            np_rng,
            replay,
            steps=min_replay_steps,
            random_actions=True,
        )

        updates = max(
            1,
            args.total_env_steps // (args.num_envs * args.env_steps_per_iter),
        )
        last_model_metrics: dict[str, Any] = {}
        eval_rows: list[dict[str, Any]] = []
        for update in range(1, updates + 1):
            observations, rng, collected = _collect_steps(
                adapter,
                state,
                config,
                observations,
                rng,
                np_rng,
                replay,
                steps=args.env_steps_per_iter,
                random_actions=False,
            )
            env_steps += collected

            model_metrics = {}
            for _ in range(args.model_updates_per_iter):
                batch = replay.sample(
                    np_rng,
                    batch_size=args.batch_size,
                    chunk_length=args.chunk_length,
                    max_horizon=max(args.model_horizon, args.imag_horizon),
                )
                rng, model_key = jax.random.split(rng)
                state, model_metrics = train_model_step(
                    state,
                    model_key,
                    batch,
                    config,
                    chunk_length=args.chunk_length,
                    control=control,
                )
            for _ in range(args.policy_updates_per_iter):
                batch = replay.sample(
                    np_rng,
                    batch_size=args.batch_size,
                    chunk_length=args.chunk_length,
                    max_horizon=max(args.model_horizon, args.imag_horizon),
                )
                rng, policy_key = jax.random.split(rng)
                state, policy_metrics = policy_train_step(
                    state,
                    policy_key,
                    batch.observations[:, 0],
                    config,
                    imag_horizon=args.imag_horizon,
                    control=control,
                )

            eval_batch = replay.sample(
                np_rng,
                batch_size=args.batch_size,
                chunk_length=args.chunk_length,
                max_horizon=max(args.model_horizon, args.imag_horizon),
            )
            rng, eval_key = jax.random.split(rng)
            eval_model_metrics = evaluate_world_model(
                state,
                eval_key,
                eval_batch,
                config,
                chunk_length=args.chunk_length,
                control=control,
            )
            open_loop_metrics = evaluate_open_loop(
                state,
                eval_batch,
                config,
                horizon=args.imag_horizon,
                control=control,
            )
            last_model_metrics = {
                **model_metrics,
                **policy_metrics,
                **eval_model_metrics,
                **open_loop_metrics,
            }
            row = {
                "update": update,
                "env_steps": env_steps,
                "control": control,
                **last_model_metrics,
            }
            logger.append_metrics(row)

            if update % args.eval_interval == 0 or update == updates:
                eval_result = _evaluate_state(args, state, config, seed=seed + 3 + update)
                eval_row = {
                    "update": update,
                    "env_steps": env_steps,
                    **eval_result,
                }
                eval_rows.append(eval_row)
                logger.append_metrics({f"eval/{key}": value for key, value in eval_row.items()})

        checkpoint_dir = run_dir / "checkpoint"
        save_checkpoint(
            checkpoint_dir,
            state,
            metadata={
                "algorithm": "isotropy_jepa",
                "env": args.env,
                "control": control,
                "jepa_config": dataclasses.asdict(config),
                "seed": seed,
            },
        )
        reload_eval, reload_diff = _reload_and_evaluate(
            args,
            state,
            config,
            checkpoint_dir=checkpoint_dir,
            seed=seed + 99,
        )
        logger.write_json("reload_evaluation.json", reload_eval)
        logger.write_json("model_metrics_final.json", last_model_metrics)
        logger.write_json("eval_returns.json", eval_rows)
        logger.plot_returns(
            [
                {
                    "update": row["update"],
                    "rollout_mean_reward": row["mean_return_per_agent"],
                    "episode_return_mean": row["mean_return_per_agent"],
                }
                for row in eval_rows
            ]
        )

        trained_eval = eval_rows[-1] if eval_rows else _evaluate_state(
            args,
            state,
            config,
            seed=seed + 4,
        )
        outcome = {
            "control": control,
            "run_dir": str(run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "random_mean": random_eval["mean_return_per_agent"],
            "initial_mean": initial_eval["mean_return_per_agent"],
            "trained_mean": trained_eval["mean_return_per_agent"],
            "improvement": (
                trained_eval["mean_return_per_agent"]
                - initial_eval["mean_return_per_agent"]
            ),
            "reload_mean": reload_eval["mean_return_per_agent"],
            "reload_abs_diff": reload_diff,
            "final_model_metrics": last_model_metrics,
        }
        logger.write_json("outcome.json", outcome)
        return to_jsonable(outcome)
    finally:
        adapter.close()


def _collect_steps(
    adapter: GymnaxVectorAdapter,
    state,
    config: JepaConfig,
    observations: np.ndarray,
    rng: jax.Array,
    np_rng: np.random.Generator,
    replay: SequenceReplayBuffer,
    *,
    steps: int,
    random_actions: bool,
) -> tuple[np.ndarray, jax.Array, int]:
    for _ in range(steps):
        if random_actions:
            actions = adapter.sample_actions(np_rng)
        else:
            rng, action_key = jax.random.split(rng)
            actions = np.asarray(
                select_actions(
                    state,
                    jnp.asarray(observations),
                    action_key,
                    config,
                    deterministic=False,
                ),
                dtype=np.int32,
            )
        step = adapter.step(actions)
        replay.add_step(
            observations=observations[:, 0],
            actions=actions[:, 0],
            rewards=step.rewards[:, 0],
            dones=step.dones[:, 0],
        )
        observations = step.observations
    return observations, rng, steps * adapter.num_envs


def _evaluate_random(args: argparse.Namespace, *, seed: int) -> dict[str, Any]:
    adapter = GymnaxVectorAdapter(
        gymnax_env_name(args.env),
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        seed=seed,
    )
    try:
        return evaluate_policy(
            adapter,
            random_policy(adapter, np.random.default_rng(seed)),
            episodes=args.eval_episodes,
        ).to_dict()
    finally:
        adapter.close()


def _evaluate_state(
    args: argparse.Namespace,
    state,
    config: JepaConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    adapter = GymnaxVectorAdapter(
        gymnax_env_name(args.env),
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        seed=seed,
    )
    key = jax.random.PRNGKey(seed)
    try:
        return evaluate_policy(
            adapter,
            _state_policy(state, config, key),
            episodes=args.eval_episodes,
        ).to_dict()
    finally:
        adapter.close()


def _state_policy(state, config: JepaConfig, key: jax.Array):
    def act(observations: np.ndarray) -> np.ndarray:
        nonlocal key
        key, action_key = jax.random.split(key)
        return np.asarray(
            select_actions(
                state,
                jnp.asarray(observations),
                action_key,
                config,
                deterministic=True,
            ),
            dtype=np.int32,
        )

    return act


def _reload_and_evaluate(
    args: argparse.Namespace,
    state,
    config: JepaConfig,
    *,
    checkpoint_dir: Path,
    seed: int,
) -> tuple[dict[str, Any], float]:
    fresh = create_jepa_train_state(jax.random.PRNGKey(seed), config)
    fresh = fresh.replace(
        params=load_params(checkpoint_dir / "checkpoint.msgpack", fresh.params)
    )
    original = _evaluate_state(args, state, config, seed=seed)
    reloaded = _evaluate_state(args, fresh, config, seed=seed)
    diff = abs(original["mean_return_per_agent"] - reloaded["mean_return_per_agent"])
    return reloaded, diff


def summarize(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    main = [outcome for outcome in outcomes if outcome["control"] == "none"]
    controls = [outcome for outcome in outcomes if outcome["control"] != "none"]
    required = max(1, math.ceil(len(main) * 2 / 3))
    improvements = np.asarray([outcome["improvement"] for outcome in main], dtype=float)
    trained = np.asarray([outcome["trained_mean"] for outcome in main], dtype=float)
    initial = np.asarray([outcome["initial_mean"] for outcome in main], dtype=float)
    reload_ok = all(outcome["reload_abs_diff"] <= 1e-5 for outcome in main)
    finite_metrics = all(_outcome_metrics_finite(outcome) for outcome in main)
    continue_ok = all(
        outcome["final_model_metrics"].get("model/continue_loss", np.inf)
        < outcome["final_model_metrics"].get("model/continue_constant_bce", -np.inf)
        for outcome in main
    )
    open_loop_ok = all(
        outcome["final_model_metrics"].get("model/open_loop_finite_fraction", 0.0)
        >= 1.0
        for outcome in main
    )
    successes = int(np.sum(improvements > 0.0))
    aggregate_improvement = float(trained.mean() - initial.mean()) if main else 0.0
    control_matches = False
    if controls and main:
        main_mean = aggregate_improvement
        control_matches = any(outcome["improvement"] >= main_mean for outcome in controls)
    passed = bool(
        main
        and successes >= required
        and aggregate_improvement > 0.0
        and reload_ok
        and finite_metrics
        and continue_ok
        and open_loop_ok
        and not control_matches
    )
    return {
        "passed": passed,
        "required_successes": required,
        "runs_improved": successes,
        "aggregate_initial_mean": float(initial.mean()) if main else None,
        "aggregate_trained_mean": float(trained.mean()) if main else None,
        "aggregate_improvement": aggregate_improvement,
        "reload_ok": reload_ok,
        "finite_metrics": finite_metrics,
        "continue_beats_constant": continue_ok,
        "open_loop_finite": open_loop_ok,
        "control_matches_main": control_matches,
        "runs": outcomes,
    }


def _outcome_metrics_finite(outcome: dict[str, Any]) -> bool:
    for value in outcome["final_model_metrics"].values():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            return False
    return True


def _control_seed_offset(control: str) -> int:
    return {
        "none": 0,
        "no-action-world-model": 100_000,
        "shuffled-action-replay": 200_000,
        "no-sigreg": 300_000,
        "weak-sigreg": 400_000,
        "no-isotropy": 300_000,
        "weak-isotropy": 400_000,
    }[control]


if __name__ == "__main__":
    main()
