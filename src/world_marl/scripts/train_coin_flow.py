"""Train a flow-matching joint-action sampler on Melting Pot coins rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax
import numpy as np

from world_marl.checkpointing import save_checkpoint
from world_marl.coin_flow import (
  collect_random_joint_actions,
  decode_joint_actions,
  fit_joint_action_gmm,
  flow_joint_action_policy,
  sample_flow_points,
  train_flow_for_gmm,
)
from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter
from world_marl.evaluation import evaluate_policy, random_policy
from world_marl.logging import RunLogger, dependency_versions, timestamp


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--substrate", default="coins")
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--max-cycles", type=int, default=500)
  parser.add_argument("--observation-size", type=int, default=44)
  parser.add_argument("--include-observation-scalars", action="store_true")
  parser.add_argument("--append-agent-id", action="store_true")
  parser.add_argument("--collect-steps", type=int, default=256)
  parser.add_argument("--gmm-std", type=float, default=0.10)
  parser.add_argument("--max-components", type=int, default=None)
  parser.add_argument("--train-steps", type=int, default=1000)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--hidden-dims", default="64,64,64,64")
  parser.add_argument("--flow-integration-steps", type=int, default=64)
  parser.add_argument("--generated-samples", type=int, default=256)
  parser.add_argument("--eval-episodes", type=int, default=10)
  parser.add_argument("--eval-max-steps", type=int, default=None)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out-dir", default="runs")
  return parser.parse_args()


def parse_hidden_dims(value: str) -> tuple[int, ...]:
  dims = tuple(int(item.strip()) for item in value.split(",") if item.strip())
  if not dims:
    raise ValueError("--hidden-dims must contain at least one integer")
  if any(dim < 1 for dim in dims):
    raise ValueError("--hidden-dims must be positive")
  return dims


def make_adapter(args: argparse.Namespace) -> MeltingPotVectorAdapter:
  return MeltingPotVectorAdapter(
    substrate=args.substrate,
    num_envs=args.num_envs,
    max_cycles=args.max_cycles,
    observation_size=args.observation_size,
    include_observation_scalars=args.include_observation_scalars,
    append_agent_id=args.append_agent_id,
  )


def main() -> None:
  args = parse_args()
  if args.substrate != "coins":
    raise SystemExit("world-marl-train-coin-flow currently targets --substrate coins")

  hidden_dims = parse_hidden_dims(args.hidden_dims)
  run_dir = Path(args.out_dir) / f"coin_flow_{timestamp()}"
  logger = RunLogger(run_dir)
  logger.write_json(
    "config.json",
    {
      "args": vars(args),
      "hidden_dims": hidden_dims,
      "purpose": (
        "Fit an empirical GMM over two-agent coins joint actions, train a "
        "JAX flow-matching sampler on that GMM, and evaluate generated joint "
        "actions in Melting Pot."
      ),
    },
  )
  logger.write_json("versions.json", dependency_versions())

  np_rng = np.random.default_rng(args.seed)
  adapter = make_adapter(args)
  try:
    dataset = collect_random_joint_actions(
      adapter,
      np_rng,
      rollout_steps=args.collect_steps,
    )
    env_metadata = {
      "substrate": adapter.substrate,
      "num_agents": adapter.num_agents,
      "action_dim": adapter.action_dim,
      "observation_shape": adapter.observation_shape,
      "raw_observation_shape": adapter.raw_observation_shape,
      "scalar_observation_keys": adapter.scalar_observation_keys,
    }
  finally:
    adapter.close()

  logger.write_json(
    "rollout_dataset.json",
    {
      **dataset.to_metadata(),
      "env": env_metadata,
    },
  )

  fitted = fit_joint_action_gmm(
    dataset.joint_actions,
    action_dim=dataset.action_dim,
    std=args.gmm_std,
    max_components=args.max_components,
  )
  logger.write_json("gmm.json", fitted.to_metadata())

  rng = jax.random.PRNGKey(args.seed)
  train_state, losses = train_flow_for_gmm(
    rng,
    fitted.gmm,
    train_steps=args.train_steps,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    hidden_dims=hidden_dims,
  )
  for step, loss in enumerate(losses, start=1):
    logger.append_metrics({"step": step, "flow/loss": loss})
  logger.write_json(
    "training_summary.json",
    {
      "initial_loss": losses[0],
      "final_loss": losses[-1],
      "min_loss": min(losses),
      "train_steps": args.train_steps,
    },
  )

  rng, sample_key = jax.random.split(rng)
  generated_points = np.asarray(
    sample_flow_points(
      train_state,
      sample_key,
      num_samples=args.generated_samples,
      integration_steps=args.flow_integration_steps,
    )
  )
  generated_actions = decode_joint_actions(generated_points, dataset.action_dim)
  unique_actions, generated_counts = np.unique(
    generated_actions,
    axis=0,
    return_counts=True,
  )
  logger.write_json(
    "generated_action_samples.json",
    {
      "points": generated_points.tolist(),
      "actions": generated_actions.astype(int).tolist(),
      "unique_action_pairs": unique_actions.astype(int).tolist(),
      "unique_action_counts": generated_counts.astype(int).tolist(),
    },
  )

  save_checkpoint(
    run_dir / "checkpoint",
    train_state,
    metadata={
      "kind": "coin_joint_action_flow",
      "substrate": args.substrate,
      "action_dim": dataset.action_dim,
      "num_agents": dataset.num_agents,
      "gmm": fitted.to_metadata(),
      "hidden_dims": hidden_dims,
      "flow_integration_steps": args.flow_integration_steps,
      "config": vars(args),
    },
  )

  eval_adapter = make_adapter(args)
  try:
    random_eval = evaluate_policy(
      eval_adapter,
      random_policy(eval_adapter, np.random.default_rng(args.seed + 1)),
      episodes=args.eval_episodes,
      max_steps=args.eval_max_steps,
    )
    flow_eval = evaluate_policy(
      eval_adapter,
      flow_joint_action_policy(
        train_state,
        num_envs=eval_adapter.num_envs,
        action_dim=eval_adapter.action_dim,
        seed=args.seed + 2,
        integration_steps=args.flow_integration_steps,
      ),
      episodes=args.eval_episodes,
      max_steps=args.eval_max_steps,
    )
  finally:
    eval_adapter.close()

  outcome: dict[str, Any] = {
    "random": random_eval.to_dict(),
    "flow": flow_eval.to_dict(),
    "flow_minus_random_mean_return_per_agent": (
      flow_eval.mean_return_per_agent - random_eval.mean_return_per_agent
    ),
  }
  logger.write_json("evaluation.json", outcome)
  print(logger.write_json("outcome.json", outcome).read_text(encoding="utf-8"))


if __name__ == "__main__":
  main()
