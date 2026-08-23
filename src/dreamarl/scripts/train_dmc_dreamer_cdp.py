"""Train the pinned official Dreamer-CDP comparison on visual DMC."""

from __future__ import annotations

import argparse
from pathlib import Path

from dreamarl.baselines.dreamer_cdp.config import (
    M2_TRAIN_STEPS,
    DreamerCDPRunSpec,
    default_dreamer_cdp_python,
    default_upstream_root,
)
from dreamarl.baselines.dreamer_cdp.launcher import run_training, timestamp
from dreamarl.baselines.dreamerv3.config import repository_root


def _overrides(values: list[str]) -> tuple[str, ...]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"override must be KEY=VALUE, got: {value}")
        key, raw = value.split("=", 1)
        if not key or not raw:
            raise ValueError(f"override must be KEY=VALUE, got: {value}")
        result.extend([f"--{key}", raw])
    return tuple(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="dmc_reacher_easy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-env-steps", type=int, default=M2_TRAIN_STEPS)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=repository_root() / "runs" / "dreamer-cdp"
    )
    parser.add_argument("--platform", choices=("cpu", "cuda", "tpu"), default="cuda")
    parser.add_argument("--python", type=Path, default=default_dreamer_cdp_python())
    parser.add_argument("--upstream-root", type=Path, default=default_upstream_root())
    parser.add_argument("--save-every-seconds", type=int, default=1_800)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    overrides = list(args.override)
    if args.debug:
        # The upstream debug regex shrinks the CNN depth without updating its
        # hand-computed output width. Keep only that width production-shaped.
        overrides.append("agent.enc.simple.depth=64")
    experiment_dir = args.experiment_dir or (
        args.output_root / args.task / f"seed_{args.seed}" / timestamp()
    )
    spec = DreamerCDPRunSpec(
        experiment_dir=experiment_dir,
        task=args.task,
        seed=args.seed,
        train_steps=args.total_env_steps,
        platform=args.platform,
        configs=("dmc_vision", "debug") if args.debug else ("dmc_vision",),
        upstream_root=args.upstream_root,
        python=args.python,
        save_every_seconds=args.save_every_seconds,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        extra_args=_overrides(overrides),
    )
    print(f"Experiment: {spec.experiment_dir}")
    return run_training(spec, resume=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
