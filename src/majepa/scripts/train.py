"""Train the locked MA-JEPA architecture."""

from __future__ import annotations

import argparse
from pathlib import Path

from majepa.config import MAJEPARunSpec
from majepa.launcher import run_training, timestamp
from majepa.runtime import infrastructure_root, repository_root, runtime_python


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-env-steps", type=int, default=50_000)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repository_root() / "runs" / "majepa",
    )
    parser.add_argument("--save-every-seconds", type=int, default=5_000)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-envs", type=int, default=4)
    parser.add_argument("--eval-seed-offset", type=int, default=50_000)
    parser.add_argument("--wm-critic-value-scale", type=float, default=0.0)
    parser.add_argument("--wm-joint-objective-scale", type=float, default=1.0)
    parser.add_argument("--wm-joint-prediction-gradient", action="store_true")
    parser.add_argument("--wm-joint-prediction-scale", type=float, default=1.0)
    parser.add_argument(
        "--local-prior", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--action-margin-loss-scale", type=float, default=0.1)
    parser.add_argument("--categorical-stoch", type=int, default=32)
    parser.add_argument("--categorical-classes", type=int, default=64)

    runtime = parser.add_argument_group("advanced runtime")
    runtime.add_argument("--platform", choices=("cpu", "cuda", "tpu"), default="cuda")
    runtime.add_argument("--python", type=Path, default=runtime_python())
    runtime.add_argument(
        "--infrastructure-root", type=Path, default=infrastructure_root()
    )
    runtime.add_argument("--resume", action="store_true")
    runtime.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.resume and args.experiment_dir is None:
        parser.error("--resume requires --experiment-dir")
    experiment_dir = args.experiment_dir or (
        args.output_root / args.task / f"seed_{args.seed}" / timestamp()
    )
    spec = MAJEPARunSpec(
        experiment_dir=experiment_dir,
        task=args.task,
        num_agents=args.num_agents,
        seed=args.seed,
        train_steps=args.total_env_steps,
        platform=args.platform,
        infrastructure_root=args.infrastructure_root,
        python=args.python,
        save_every_seconds=(
            args.save_every_seconds if args.save_every_seconds > 0 else None
        ),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        curve_eval_interval=args.eval_interval,
        curve_eval_episodes=args.eval_episodes,
        curve_eval_envs=args.eval_envs,
        curve_eval_seed_offset=args.eval_seed_offset,
        wm_critic_value_scale=args.wm_critic_value_scale,
        wm_joint_objective_scale=args.wm_joint_objective_scale,
        wm_joint_prediction_gradient=args.wm_joint_prediction_gradient,
        wm_joint_prediction_scale=args.wm_joint_prediction_scale,
        local_prior=args.local_prior,
        action_margin_loss_scale=args.action_margin_loss_scale,
        categorical_stoch=args.categorical_stoch,
        categorical_classes=args.categorical_classes,
    )
    print(f"Experiment: {spec.experiment_dir}")
    return run_training(spec, resume=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
