"""Train CoinGame flow-validation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from world_marl.coin_flow_experiments import (
  log_stage,
  run_conditional_action_validation,
)
from world_marl.logging import RunLogger, dependency_versions, timestamp


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--substrate", default="coins")
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--max-cycles", type=int, default=500)
  parser.add_argument("--observation-size", type=int, default=44)
  parser.add_argument("--include-observation-scalars", action="store_true")
  parser.add_argument("--append-agent-id", action="store_true")
  parser.add_argument(
    "--target-source",
    choices=("random", "checkpoint"),
    default="random",
    help="Source policy used to collect state-action imitation data.",
  )
  parser.add_argument(
    "--policy-checkpoint",
    default=None,
    help="IPPO/MAPPO checkpoint directory used when --target-source checkpoint.",
  )
  parser.add_argument(
    "--source-stochastic",
    action="store_true",
    help="Sample checkpoint policy actions while collecting/evaluating source actions.",
  )
  parser.add_argument("--collect-steps", type=int, default=256)
  parser.add_argument("--validation-fraction", type=float, default=0.25)
  parser.add_argument("--train-steps", type=int, default=1000)
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--hidden-dims", default="64,64,64,64")
  parser.add_argument(
    "--classifier-hidden-dims",
    default="128,128",
    help="Hidden dimensions for the categorical p(action | state) sanity baseline.",
  )
  parser.add_argument(
    "--classifier-learning-rate",
    type=float,
    default=None,
    help="Optional learning rate for the categorical baseline; defaults to --learning-rate.",
  )
  parser.add_argument("--flow-integration-steps", type=int, default=64)
  parser.add_argument("--generated-samples", type=int, default=256)
  parser.add_argument("--distribution-top-k", type=int, default=5)
  parser.add_argument("--eval-episodes", type=int, default=10)
  parser.add_argument("--eval-max-steps", type=int, default=None)
  parser.add_argument(
    "--skip-policy-eval",
    action="store_true",
    help="Only run heldout prediction validation; skip environment return evaluation.",
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--out-dir", default="runs")
  parser.add_argument(
    "--quiet",
    action="store_true",
    help="Disable terminal progress output.",
  )
  return parser.parse_args()


def parse_hidden_dims(value: str) -> tuple[int, ...]:
  dims = tuple(int(item.strip()) for item in value.split(",") if item.strip())
  if not dims:
    raise ValueError("--hidden-dims must contain at least one integer")
  if any(dim < 1 for dim in dims):
    raise ValueError("--hidden-dims must be positive")
  return dims


def main() -> None:
  args = parse_args()
  if args.substrate != "coins":
    raise SystemExit("world-marl-train-coin-flow currently targets --substrate coins")
  if args.target_source == "checkpoint" and args.policy_checkpoint is None:
    raise SystemExit("--policy-checkpoint is required with --target-source checkpoint")

  hidden_dims = parse_hidden_dims(args.hidden_dims)
  classifier_hidden_dims = parse_hidden_dims(args.classifier_hidden_dims)
  run_dir = Path(args.out_dir) / f"coin_flow_{timestamp()}"
  log_stage(args, f"writing artifacts to {run_dir}")
  logger = RunLogger(run_dir)
  logger.write_json(
    "config.json",
    {
      "args": vars(args),
      "hidden_dims": hidden_dims,
      "classifier_hidden_dims": classifier_hidden_dims,
      "target_source": args.target_source,
      "purpose": (
        "Validate whether flow matching can learn p(joint_action | state) "
        "from JaxMARL CoinGame rollouts."
      ),
    },
  )
  logger.write_json("versions.json", dependency_versions())

  run_conditional_action_validation(
    args,
    hidden_dims=hidden_dims,
    classifier_hidden_dims=classifier_hidden_dims,
    run_dir=run_dir,
    logger=logger,
    np_rng=np.random.default_rng(args.seed),
  )


if __name__ == "__main__":
  main()
