"""Train the locked first-party DreaMARL algorithm."""

from __future__ import annotations

import argparse
from pathlib import Path

from dreamarl.baselines.dreamerv3.config import (
    default_dreamerv3_python,
    default_upstream_root,
    repository_root,
)
from dreamarl.config import DreaMARLRunSpec
from dreamarl.launcher import run_training, timestamp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--marl-stage", choices=("b0", "b1", "b2", "jecc", "ctde"), default="b0"
    )
    parser.add_argument(
        "--replay-sampling",
        choices=("uniform", "recent"),
        default="uniform",
    )
    parser.add_argument(
        "--behavior-optimizer",
        choices=("joint", "separated", "grouped"),
        default=None,
    )
    parser.add_argument(
        "--behavior-objective", choices=("reinforce", "ppo"), default="reinforce"
    )
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument(
        "--repval-grad", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--anchor-batch", type=Path)
    parser.add_argument("--from-checkpoint", type=Path)
    parser.add_argument("--from-checkpoint-regex")
    parser.add_argument("--load-replay", action="store_true")
    parser.add_argument("--replay-source", type=Path)
    parser.add_argument("--report-actor-gradnorm", action="store_true")
    parser.add_argument("--agent-jepa-local-grad-scale", type=float, default=0.0)
    parser.add_argument("--agent-jepa-k0-scale", type=float, default=0.1)
    parser.add_argument("--agent-jepa-future-scale", type=float, default=1.0)
    parser.add_argument("--agent-jepa-future-set-scale", type=float, default=1.0)
    parser.add_argument("--ctde-rollout-steps", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ctde-multistep-anchors", type=int, default=128)
    parser.add_argument("--total-env-steps", type=int, default=50_000)
    parser.add_argument("--train-ratio", type=float, default=256.0)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repository_root() / "runs" / "dreamarl",
    )
    parser.add_argument("--platform", choices=("cpu", "cuda", "tpu"), default="cuda")
    parser.add_argument("--python", type=Path, default=default_dreamerv3_python())
    parser.add_argument(
        "--infrastructure-root", type=Path, default=default_upstream_root()
    )
    parser.add_argument("--save-every-seconds", type=int, default=1_800)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--curve-eval-interval", type=int, default=0)
    parser.add_argument("--curve-eval-episodes", type=int, default=20)
    parser.add_argument("--curve-eval-seed-offset", type=int, default=10_000)
    parser.add_argument(
        "--curve-eval-policy-mode",
        choices=("deterministic", "stochastic"),
        default="deterministic",
    )
    parser.add_argument(
        "--imagination-starts",
        type=int,
        default=0,
        help="number of final replay states used as imagination starts (0 means all)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    experiment_dir = args.experiment_dir or (
        args.output_root / args.task / f"seed_{args.seed}" / timestamp()
    )
    spec = DreaMARLRunSpec(
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
        curve_eval_interval=args.curve_eval_interval,
        curve_eval_episodes=args.curve_eval_episodes,
        curve_eval_seed_offset=args.curve_eval_seed_offset,
        curve_eval_policy_mode=args.curve_eval_policy_mode,
        imagination_starts=args.imagination_starts,
        train_ratio=args.train_ratio,
        replay_sampling=args.replay_sampling,
        behavior_optimizer=(
            args.behavior_optimizer
            or ("separated" if args.marl_stage in {"jecc", "ctde"} else "joint")
        ),
        behavior_objective=args.behavior_objective,
        ppo_epochs=args.ppo_epochs,
        ppo_clip=args.ppo_clip,
        repval_grad=args.repval_grad,
        anchor_batch=args.anchor_batch,
        from_checkpoint=args.from_checkpoint,
        from_checkpoint_regex=args.from_checkpoint_regex,
        load_replay=args.load_replay,
        replay_source=args.replay_source,
        report_actor_gradnorm=args.report_actor_gradnorm,
        marl_stage=args.marl_stage,
        agent_jepa_local_grad_scale=args.agent_jepa_local_grad_scale,
        agent_jepa_k0_scale=args.agent_jepa_k0_scale,
        agent_jepa_future_scale=args.agent_jepa_future_scale,
        agent_jepa_future_set_scale=args.agent_jepa_future_set_scale,
        ctde_rollout_steps=args.ctde_rollout_steps,
        ctde_multistep_anchors=args.ctde_multistep_anchors,
    )
    print(f"Experiment: {spec.experiment_dir}")
    return run_training(spec, resume=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
