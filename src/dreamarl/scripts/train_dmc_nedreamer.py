"""Launch the pinned official NE-Dreamer implementation on visual DMC."""

from __future__ import annotations

import argparse
from pathlib import Path

from dreamarl.baselines.nedreamer.config import (
    PHASE_1_TRAIN_TRANSITIONS,
    NEDreamerRunSpec,
    default_nedreamer_python,
    default_upstream_root,
    repository_root,
)
from dreamarl.baselines.nedreamer.launcher import run_training, timestamp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="dmc_walker_walk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--total-env-steps", type=int, default=PHASE_1_TRAIN_TRANSITIONS
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repository_root() / "runs" / "nedreamer" / "dmc_vision",
    )
    parser.add_argument("--python", type=Path, default=default_nedreamer_python())
    parser.add_argument("--upstream-root", type=Path, default=default_upstream_root())
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    experiment_dir = args.experiment_dir or (
        args.output_root / args.task / f"seed_{args.seed}" / timestamp()
    )
    spec = NEDreamerRunSpec(
        experiment_dir=experiment_dir,
        task=args.task,
        seed=args.seed,
        train_steps=args.total_env_steps,
        device=args.device,
        upstream_root=args.upstream_root,
        python=args.python,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        extra_overrides=tuple(args.override),
    )
    print(f"Experiment: {spec.experiment_dir}")
    return run_training(spec, resume=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
