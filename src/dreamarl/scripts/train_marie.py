"""Run an unmodified paper-era MARIE reference experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from dreamarl.baselines.dreamerv3.config import repository_root
from dreamarl.baselines.marie.config import (
    PAPER_BENCHMARKS,
    MARIERunSpec,
    default_marie_python,
    default_upstream_root,
)
from dreamarl.baselines.marie.launcher import run_training, timestamp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", choices=sorted(PAPER_BENCHMARKS), default="3m")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument(
        "--mode", choices=("disabled", "offline", "online"), default="online"
    )
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=repository_root() / "runs" / "marie"
    )
    parser.add_argument("--python", type=Path, default=default_marie_python())
    parser.add_argument("--upstream-root", type=Path, default=default_upstream_root())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    experiment_dir = args.experiment_dir or (
        args.output_root / args.map / f"seed_{args.seed}" / timestamp()
    )
    spec = MARIERunSpec(
        experiment_dir=experiment_dir,
        map_name=args.map,
        seed=args.seed,
        steps=args.steps,
        temperature=args.temperature,
        mode=args.mode,
        upstream_root=args.upstream_root,
        python=args.python,
    )
    print(f"Experiment: {spec.experiment_dir}")
    return run_training(spec, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
