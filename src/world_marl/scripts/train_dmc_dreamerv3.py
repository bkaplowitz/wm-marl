"""Launch the pinned official DreamerV3 implementation on DMC."""

from __future__ import annotations

import argparse
from pathlib import Path

from world_marl.baselines.dreamerv3.config import (
    COMPARISON_DMC_TRAIN_STEPS,
    OFFICIAL_DMC_CONFIG,
    OFFICIAL_DMC_TRAIN_STEPS,
    DreamerV3RunSpec,
    default_dreamerv3_python,
    default_upstream_root,
    repository_root,
)
from world_marl.baselines.dreamerv3.launcher import run_training, timestamp


def _override_args(values: list[str]) -> tuple[str, ...]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"override must be KEY=VALUE, got: {value}")
        key, raw = value.split("=", 1)
        if not key or not raw:
            raise ValueError(f"override must be KEY=VALUE, got: {value}")
        result.extend([f"--{key}", raw])
    return tuple(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="dmc_reacher_easy")
    parser.add_argument("--seed", type=int, default=0)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--total-env-steps",
        type=int,
        default=COMPARISON_DMC_TRAIN_STEPS,
    )
    budget.add_argument("--official-budget", action="store_true")
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repository_root() / "runs" / "dreamerv3",
    )
    parser.add_argument("--platform", choices=("cpu", "cuda", "tpu"), default="cuda")
    parser.add_argument("--python", type=Path, default=default_dreamerv3_python())
    parser.add_argument("--upstream-root", type=Path, default=default_upstream_root())
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Additional upstream config applied after dmc_proprio.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Explicit upstream flag override; recorded in launch.json.",
    )
    parser.add_argument("--save-every-seconds", type=int)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_steps = (
        OFFICIAL_DMC_TRAIN_STEPS if args.official_budget else args.total_env_steps
    )
    experiment_dir = args.experiment_dir or (
        args.output_root / args.task / f"seed_{args.seed}" / timestamp()
    )
    configs = [OFFICIAL_DMC_CONFIG, *args.config]
    if args.debug:
        configs.append("debug")
    spec = DreamerV3RunSpec(
        experiment_dir=experiment_dir,
        task=args.task,
        seed=args.seed,
        train_steps=train_steps,
        platform=args.platform,
        configs=tuple(configs),
        upstream_root=args.upstream_root,
        python=args.python,
        save_every_seconds=args.save_every_seconds,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        extra_args=_override_args(args.override),
    )
    print(f"Experiment: {spec.experiment_dir.resolve()}")
    return run_training(spec, resume=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
