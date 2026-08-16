"""Evaluate the latest checkpoint of an official NE-Dreamer run."""

from __future__ import annotations

import argparse
from pathlib import Path

from dreamarl.baselines.nedreamer.evaluation import (
    NEDreamerEvaluationSpec,
    run_evaluation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    returncode, eval_dir = run_evaluation(
        NEDreamerEvaluationSpec(
            experiment_dir=args.experiment_dir,
            episodes=args.episodes,
            eval_seed=args.eval_seed,
            device=args.device,
            python=args.python,
        ),
        dry_run=args.dry_run,
    )
    print(f"Evaluation artifacts: {eval_dir}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
