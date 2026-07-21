"""Evaluate the latest saved checkpoint of an official DreamerV3 run."""

from __future__ import annotations

import argparse
from pathlib import Path

from world_marl.baselines.dreamerv3.evaluation import (
    DreamerV3EvaluationSpec,
    run_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--episode-length", type=int, default=1_000)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--success-threshold", type=float)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = DreamerV3EvaluationSpec(
        experiment_dir=args.experiment_dir,
        episodes=args.episodes,
        envs=args.envs,
        episode_length=args.episode_length,
        eval_seed=args.eval_seed,
        success_threshold=args.success_threshold,
        python=args.python,
    )
    returncode, eval_dir = run_evaluation(spec, dry_run=args.dry_run)
    print(f"Evaluation artifacts: {eval_dir.resolve()}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
