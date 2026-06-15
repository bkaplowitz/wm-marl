"""End-to-end learning validation CLI."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.algs.ippo import (
  IPPOConfig,
  create_train_state as create_ippo_train_state,
  ppo_update,
)
from world_marl.algs.mappo import (
  MAPPOConfig,
  create_train_state as create_mappo_train_state,
  mappo_update,
)
from world_marl.checkpointing import load_metadata, load_params, save_checkpoint
from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter
from world_marl.evaluation import (
  evaluate_policy,
  mappo_train_state_policy,
  random_policy,
  train_state_policy,
)
from world_marl.logging import RunLogger, dependency_versions, timestamp, to_jsonable
from world_marl.training import (
  central_observation_shape,
  collect_mappo_rollout,
  collect_rollout,
  training_window_means,
)


@dataclass(frozen=True)
class RunOutcome:
  name: str
  run_dir: str
  control: str | None
  random_mean: float
  initial_mean: float
  trained_mean: float
  improvement: float
  random_improvement: float
  initial_improvement: float
  first_window_mean: float
  final_window_mean: float
  checkpoint_dir: str

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--algorithm", choices=("ippo", "mappo"), default="ippo")
  parser.add_argument("--substrate", default="coins")
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--rollout-steps", type=int, default=128)
  parser.add_argument("--total-env-steps", type=int, default=100_000)
  parser.add_argument("--eval-episodes", type=int, default=50)
  parser.add_argument("--num-runs", type=int, default=3)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--max-cycles", type=int, default=1000)
  parser.add_argument(
    "--observation-size",
    type=int,
    default=None,
    help="Optional square RGB downsample size, e.g. 22 or 44. Default keeps 88x88.",
  )
  parser.add_argument(
    "--append-agent-id",
    action="store_true",
    help="Append one-hot agent identity channels to each RGB observation.",
  )
  parser.add_argument(
    "--include-observation-scalars",
    action="store_true",
    help="Append scalar Melting Pot observation keys as constant image channels.",
  )
  parser.add_argument(
    "--stochastic-eval",
    action="store_true",
    help="Evaluate learned policies by sampling instead of taking argmax actions.",
  )
  parser.add_argument("--eval-max-steps", type=int, default=None)
  parser.add_argument("--out-dir", default="runs")
  parser.add_argument("--min-improvement", type=float, default=0.2)
  parser.add_argument(
    "--negative-control",
    choices=("none", "freeze-policy", "shuffle-rewards", "zero-advantages"),
    default="freeze-policy",
  )

  parser.add_argument("--learning-rate", type=float, default=5e-4)
  parser.add_argument("--gamma", type=float, default=0.99)
  parser.add_argument("--gae-lambda", type=float, default=0.95)
  parser.add_argument("--clip-eps", type=float, default=0.2)
  parser.add_argument("--ent-coef", type=float, default=0.01)
  parser.add_argument("--vf-coef", type=float, default=0.5)
  parser.add_argument("--max-grad-norm", type=float, default=0.5)
  parser.add_argument("--update-epochs", type=int, default=4)
  parser.add_argument("--num-minibatches", type=int, default=4)
  parser.add_argument("--activation", choices=("relu", "tanh"), default="relu")

  parser.add_argument(
    "--eval-checkpoint",
    default=None,
    help=argparse.SUPPRESS,
  )
  return parser.parse_args()


def algorithm_config_from_args(
  args: argparse.Namespace,
  control: str | None = None,
) -> IPPOConfig | MAPPOConfig:
  config_cls = MAPPOConfig if args.algorithm == "mappo" else IPPOConfig
  config = config_cls(
    learning_rate=args.learning_rate,
    gamma=args.gamma,
    gae_lambda=args.gae_lambda,
    clip_eps=args.clip_eps,
    ent_coef=args.ent_coef,
    vf_coef=args.vf_coef,
    max_grad_norm=args.max_grad_norm,
    update_epochs=args.update_epochs,
    num_minibatches=args.num_minibatches,
    activation=args.activation,
  )
  if control == "shuffle-rewards":
    return replace(config, shuffle_rewards=True)
  if control == "zero-advantages":
    return replace(config, zero_advantages=True)
  return config


def create_algorithm_train_state(
  algorithm: str,
  rng: jax.Array,
  adapter: MeltingPotVectorAdapter,
  config: IPPOConfig | MAPPOConfig,
):
  if algorithm == "mappo":
    return create_mappo_train_state(
      rng,
      adapter.observation_shape,
      central_observation_shape(adapter.observation_shape, adapter.num_agents),
      adapter.action_dim,
      config,
    )
  return create_ippo_train_state(
    rng,
    adapter.observation_shape,
    adapter.action_dim,
    config,
  )


def policy_from_train_state(
  algorithm: str,
  train_state,
  *,
  adapter: MeltingPotVectorAdapter,
  deterministic: bool,
  seed: int,
):
  policy_fn = mappo_train_state_policy if algorithm == "mappo" else train_state_policy
  return policy_fn(
    train_state,
    num_envs=adapter.num_envs,
    num_agents=adapter.num_agents,
    deterministic=deterministic,
    seed=seed,
  )


def evaluate_checkpoint_mode(args: argparse.Namespace) -> None:
  checkpoint_dir = Path(args.eval_checkpoint)
  metadata = load_metadata(checkpoint_dir)
  algorithm = metadata.get("algorithm", "ippo")
  substrate = args.substrate or metadata["substrate"]
  adapter = MeltingPotVectorAdapter(
    substrate=substrate,
    num_envs=args.num_envs,
    max_cycles=args.max_cycles,
    observation_size=(
      args.observation_size
      if args.observation_size is not None
      else metadata.get("observation_size")
    ),
    include_observation_scalars=(
      args.include_observation_scalars
      if args.include_observation_scalars
      else metadata.get("include_observation_scalars", False)
    ),
    append_agent_id=(
      args.append_agent_id
      if args.append_agent_id
      else metadata.get("append_agent_id", False)
    ),
  )
  try:
    config_payload = metadata.get("algorithm_config", metadata.get("ippo_config"))
    if config_payload is None:
      raise KeyError("checkpoint metadata missing algorithm_config")
    config = MAPPOConfig(**config_payload) if algorithm == "mappo" else IPPOConfig(**config_payload)
    train_state = create_algorithm_train_state(
      algorithm,
      jax.random.PRNGKey(0),
      adapter,
      config,
    )
    params = load_params(checkpoint_dir / "checkpoint.msgpack", train_state.params)
    train_state = train_state.replace(params=params)
    result = evaluate_policy(
      adapter,
      policy_from_train_state(
        algorithm,
        train_state,
        adapter=adapter,
        deterministic=not args.stochastic_eval,
        seed=args.seed,
      ),
      episodes=args.eval_episodes,
      max_steps=args.eval_max_steps,
    )
    print(json.dumps(to_jsonable(result.to_dict()), sort_keys=True))
  finally:
    adapter.close()


def evaluate_random_baseline(args: argparse.Namespace, seed: int) -> dict[str, Any]:
  adapter = MeltingPotVectorAdapter(
    substrate=args.substrate,
    num_envs=args.num_envs,
    max_cycles=args.max_cycles,
    observation_size=args.observation_size,
    include_observation_scalars=args.include_observation_scalars,
    append_agent_id=args.append_agent_id,
  )
  try:
    result = evaluate_policy(
      adapter,
      random_policy(adapter, np.random.default_rng(seed)),
      episodes=args.eval_episodes,
      max_steps=args.eval_max_steps,
    )
    return result.to_dict()
  finally:
    adapter.close()


def evaluate_checkpoint_subprocess(
  args: argparse.Namespace,
  checkpoint_dir: Path,
  *,
  seed: int,
) -> dict[str, Any]:
  command = [
    sys.executable,
    "-m",
    "world_marl.scripts.train_e2e",
    "--eval-checkpoint",
    str(checkpoint_dir),
    "--substrate",
    args.substrate,
    "--num-envs",
    str(args.num_envs),
    "--eval-episodes",
    str(args.eval_episodes),
    "--seed",
    str(seed),
    "--max-cycles",
    str(args.max_cycles),
  ]
  if args.observation_size is not None:
    command.extend(["--observation-size", str(args.observation_size)])
  if args.include_observation_scalars:
    command.append("--include-observation-scalars")
  if args.append_agent_id:
    command.append("--append-agent-id")
  if args.stochastic_eval:
    command.append("--stochastic-eval")
  if args.eval_max_steps is not None:
    command.extend(["--eval-max-steps", str(args.eval_max_steps)])
  result = subprocess.run(
    command,
    check=True,
    capture_output=True,
    text=True,
  )
  return json.loads(result.stdout.strip())


def run_training(
  args: argparse.Namespace,
  *,
  run_dir: Path,
  name: str,
  run_index: int,
  control: str | None,
) -> RunOutcome:
    logger = RunLogger(run_dir)
    seed = args.seed + run_index * 10_000 + (5_000 if control else 0)
    rng = jax.random.PRNGKey(seed)
    config = algorithm_config_from_args(args, control)
    freeze_policy = control == "freeze-policy"

    logger.write_json(
        "config.json",
        {
            "args": vars(args),
            "run_index": run_index,
            "control": control,
            "seed": seed,
            "algorithm": args.algorithm,
            "algorithm_config": dataclasses.asdict(config),
        },
    )
    logger.write_json("versions.json", dependency_versions())

    random_result = evaluate_random_baseline(args, seed=seed + 1)
    logger.write_json("random_baseline.json", random_result)

    adapter = MeltingPotVectorAdapter(
        substrate=args.substrate,
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        observation_size=args.observation_size,
        include_observation_scalars=args.include_observation_scalars,
        append_agent_id=args.append_agent_id,
    )
    rows: list[dict[str, Any]] = []
    try:
        observations = adapter.reset()
        rng, init_key = jax.random.split(rng)
        train_state = create_algorithm_train_state(
            args.algorithm,
            init_key,
            adapter,
            config,
        )
        initial_result = evaluate_policy(
            adapter,
            policy_from_train_state(
                args.algorithm,
                train_state,
                adapter=adapter,
                deterministic=not args.stochastic_eval,
                seed=seed + 2,
            ),
            episodes=args.eval_episodes,
            max_steps=args.eval_max_steps,
        ).to_dict()
        logger.write_json("initial_policy_evaluation.json", initial_result)
        observations = adapter.reset()

        if args.algorithm == "mappo":
            update_fn = jax.jit(
                lambda state, batch, last_values, update_rng: mappo_update(
                    state,
                    batch,
                    last_values,
                    update_rng,
                    config,
                )
            )
            collect_fn = collect_mappo_rollout
        else:
            update_fn = jax.jit(
                lambda state, batch, last_values, update_rng: ppo_update(
                    state,
                    batch,
                    last_values,
                    update_rng,
                    config,
                )
            )
            collect_fn = collect_rollout

        updates = max(1, args.total_env_steps // (args.num_envs * args.rollout_steps))
        env_steps = 0
        for update in range(1, updates + 1):
            rng, rollout_key, update_key = jax.random.split(rng, 3)
            rollout = collect_fn(
                adapter,
                train_state,
                observations,
                rollout_key,
                rollout_steps=args.rollout_steps,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )
            # Fit env model to rollout. Raw rollout or latent states?
            observations = rollout.next_observations
            update_metrics: dict[str, Any] = {}
            # Simulate from env model and use update_fn to fit to env rollouts.
            if not freeze_policy:
                train_state, update_metrics = update_fn(
                    train_state,
                    rollout.batch,
                    rollout.last_values,
                    update_key,
                )
                jax.block_until_ready(jnp.asarray(update_metrics["total_loss"]))

            env_steps += args.num_envs * args.rollout_steps
            row = {
                "update": update,
                "env_steps": env_steps,
                "control": control,
                **rollout.metrics,
                **{f"ppo/{key}": value for key, value in update_metrics.items()},
            }
            rows.append(to_jsonable(row))
            logger.append_metrics(row)

        first_window_mean, final_window_mean = training_window_means(rows)
        logger.plot_returns(rows)

        checkpoint_dir = run_dir / "checkpoint"
        save_checkpoint(
            checkpoint_dir,
            train_state,
            metadata={
                "substrate": args.substrate,
                "num_envs": args.num_envs,
                "num_agents": adapter.num_agents,
                "observation_shape": adapter.observation_shape,
                "raw_observation_shape": adapter.raw_observation_shape,
                "observation_size": adapter.observation_size,
                "include_observation_scalars": adapter.include_observation_scalars,
                "scalar_observation_keys": adapter.scalar_observation_keys,
                "append_agent_id": adapter.append_agent_id,
                "algorithm": args.algorithm,
                "central_observation_shape": (
                    central_observation_shape(
                        adapter.observation_shape, adapter.num_agents
                    )
                    if args.algorithm == "mappo"
                    else None
                ),
                "action_dim": adapter.action_dim,
                "algorithm_config": dataclasses.asdict(config),
                "ippo_config": (
                    dataclasses.asdict(config) if args.algorithm == "ippo" else None
                ),
                "seed": seed,
                "control": control,
            },
        )
    finally:
        adapter.close()

    reload_result = evaluate_checkpoint_subprocess(
        args,
        checkpoint_dir,
        seed=seed + 2,
    )
    logger.write_json("reload_evaluation.json", reload_result)

    random_mean = float(random_result["mean_return_per_agent"])
    initial_mean = float(initial_result["mean_return_per_agent"])
    trained_mean = float(reload_result["mean_return_per_agent"])
    random_improvement = trained_mean - random_mean
    initial_improvement = trained_mean - initial_mean
    outcome = RunOutcome(
        name=name,
        run_dir=str(run_dir),
        control=control,
        random_mean=random_mean,
        initial_mean=initial_mean,
        trained_mean=trained_mean,
        improvement=random_improvement,
        random_improvement=random_improvement,
        initial_improvement=initial_improvement,
        first_window_mean=first_window_mean,
        final_window_mean=final_window_mean,
        checkpoint_dir=str(checkpoint_dir),
    )
    logger.write_json("outcome.json", outcome.to_dict())
    return outcome


def summarize(
  outcomes: list[RunOutcome],
  control_outcome: RunOutcome | None,
  *,
  min_improvement: float,
) -> dict[str, Any]:
  improvements = np.asarray([outcome.improvement for outcome in outcomes], dtype=float)
  initial_improvements = np.asarray(
    [outcome.initial_improvement for outcome in outcomes],
    dtype=float,
  )
  trained = np.asarray([outcome.trained_mean for outcome in outcomes], dtype=float)
  random = np.asarray([outcome.random_mean for outcome in outcomes], dtype=float)
  initial = np.asarray([outcome.initial_mean for outcome in outcomes], dtype=float)
  first_windows = np.asarray(
    [outcome.first_window_mean for outcome in outcomes],
    dtype=float,
  )
  final_windows = np.asarray(
    [outcome.final_window_mean for outcome in outcomes],
    dtype=float,
  )

  required_successes = max(1, math.ceil(len(outcomes) * 2 / 3))
  runs_beating_random = int(np.sum(improvements > 0.0))
  runs_beating_initial = int(np.sum(initial_improvements > 0.0))
  aggregate_improvement = float(trained.mean() - random.mean())
  aggregate_initial_improvement = float(trained.mean() - initial.mean())
  curve_improved = bool(final_windows.mean() > first_windows.mean())

  control_would_pass = False
  if control_outcome is not None:
    control_would_pass = bool(
      control_outcome.initial_improvement >= min_improvement
    )

  passed = bool(
    runs_beating_random >= required_successes
    and runs_beating_initial >= required_successes
    and aggregate_improvement >= min_improvement
    and aggregate_initial_improvement >= min_improvement
    and curve_improved
    and not control_would_pass
  )

  return {
    "passed": passed,
    "required_successes": required_successes,
    "runs_beating_random": runs_beating_random,
    "runs_beating_initial": runs_beating_initial,
    "aggregate_random_mean": float(random.mean()),
    "aggregate_initial_mean": float(initial.mean()),
    "aggregate_trained_mean": float(trained.mean()),
    "aggregate_improvement": aggregate_improvement,
    "aggregate_random_improvement": aggregate_improvement,
    "aggregate_initial_improvement": aggregate_initial_improvement,
    "min_improvement": min_improvement,
    "curve_first_window_mean": float(first_windows.mean()),
    "curve_final_window_mean": float(final_windows.mean()),
    "curve_improved": curve_improved,
    "control_would_pass": control_would_pass,
    "runs": [outcome.to_dict() for outcome in outcomes],
    "control": control_outcome.to_dict() if control_outcome else None,
  }


def main() -> None:
  args = parse_args()
  if args.eval_checkpoint:
    evaluate_checkpoint_mode(args)
    return

  experiment_dir = Path(args.out_dir) / f"e2e_{timestamp()}"
  experiment_dir.mkdir(parents=True, exist_ok=True)
  outcomes = [
    run_training(
      args,
      run_dir=experiment_dir / f"run_{run_index:03d}",
      name=f"run_{run_index:03d}",
      run_index=run_index,
      control=None,
    )
    for run_index in range(args.num_runs)
  ]

  control_outcome = None
  if args.negative_control != "none":
    control_outcome = run_training(
      args,
      run_dir=experiment_dir / f"control_{args.negative_control}",
      name=f"control_{args.negative_control}",
      run_index=args.num_runs,
      control=args.negative_control,
    )

  summary = summarize(
    outcomes,
    control_outcome,
    min_improvement=args.min_improvement,
  )
  RunLogger(experiment_dir).write_json("summary.json", summary)
  print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
  if not summary["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()

# TODO:  Two timing e2e, one for fit-in-advance one for fit dyna style.
