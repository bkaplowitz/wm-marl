"""Run the first-party, agent-axis-native DreaMARL implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

from world_marl.baselines.dreamer_cdp.config import default_dreamer_cdp_python
from world_marl.baselines.dreamerv3.config import repository_root
from world_marl.baselines.dreamer_cdp.config import default_upstream_root
from world_marl.dreamarl.config import DreaMARLRunSpec
from world_marl.dreamarl.launcher import run_training, timestamp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-agents",
        type=int,
        default=1,
        help="Extent of the explicit agent tensor axis; does not select a regime.",
    )
    parser.add_argument(
        "--interaction-context",
        choices=("none", "aligned", "shuffled"),
        default="none",
        help="World-model interaction arm; actor and critic remain local.",
    )
    parser.add_argument(
        "--local-memory",
        action="store_true",
        help="Enable the four-token local memory sidecar.",
    )
    parser.add_argument("--total-env-steps", type=int, default=250_000)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repository_root() / "runs" / "dreamarl",
    )
    parser.add_argument("--platform", choices=("cpu", "cuda", "tpu"), default="cuda")
    parser.add_argument("--python", type=Path, default=default_dreamer_cdp_python())
    parser.add_argument(
        "--infrastructure-root", type=Path, default=default_upstream_root()
    )
    parser.add_argument("--save-every-seconds", type=int, default=1_800)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    experiment_dir = args.experiment_dir or (
        args.output_root / args.task / f"seed_{args.seed}" / timestamp()
    )
    spec = DreaMARLRunSpec(
        experiment_dir=experiment_dir,
        task=args.task,
        seed=args.seed,
        train_steps=args.total_env_steps,
        num_agents=args.num_agents,
        interaction_context=args.interaction_context,
        local_memory=args.local_memory,
        platform=args.platform,
        infrastructure_root=args.infrastructure_root,
        python=args.python,
        save_every_seconds=args.save_every_seconds,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
    print(f"Experiment: {spec.experiment_dir}")
    return run_training(spec, resume=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
