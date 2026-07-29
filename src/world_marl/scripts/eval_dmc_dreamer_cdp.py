"""Evaluate the latest checkpoint of an official Dreamer-CDP run."""

from __future__ import annotations

import argparse
from pathlib import Path

from world_marl.baselines.dreamer_cdp.launcher import verify_upstream
from world_marl.baselines.dreamerv3.evaluation import (
    DreamerV3EvaluationSpec,
    run_evaluation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--episode-length", type=int, default=1_000)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    spec = DreamerV3EvaluationSpec(
        experiment_dir=args.experiment_dir,
        episodes=args.episodes,
        envs=args.envs,
        episode_length=args.episode_length,
        eval_seed=args.eval_seed,
        python=args.python,
    )
    returncode, eval_dir = run_evaluation(
        spec, dry_run=args.dry_run, verify_fn=verify_upstream
    )
    print(f"Evaluation artifacts: {eval_dir}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
