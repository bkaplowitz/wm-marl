"""Run an explicit non-canonical DreaMARL ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

from dreamarl.baselines.dreamerv3.config import (
    default_dreamerv3_python,
    default_upstream_root,
    repository_root,
)
from dreamarl.ablations.config import AblationRunSpec
from dreamarl.ablations.contracts import verify_run_contract
from dreamarl.launcher import run_training, timestamp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-agents",
        type=int,
        required=True,
        help="Number of agents represented by the joint world state.",
    )
    parser.add_argument("--total-env-steps", type=int, default=50_000)
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
    parser.add_argument(
        "--final-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a final checkpoint in addition to the initial/periodic checkpoint.",
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--temporal-model",
        choices=("rssm", "parallel_transformer"),
        default="parallel_transformer",
        help="Temporal dynamics implementation; all other components are shared.",
    )
    parser.add_argument(
        "--world-model-objective",
        choices=("reconstruction", "embedding"),
        default="embedding",
        help=(
            "Use the reconstruction attribution control or the maintained "
            "decoder-free EMA-target JEPA objective."
        ),
    )
    parser.add_argument(
        "--embedding-target",
        choices=("ema", "online"),
        default="ema",
        help="Use a fixed EMA target or the fully differentiable online encoder.",
    )
    parser.add_argument(
        "--embedding-loss",
        choices=("cosine", "mse"),
        default="cosine",
    )
    parser.add_argument(
        "--posterior-jepa",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dynamics-jepa",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--spatial-jepa",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--spatial-mask-ratio", type=float, default=0.5)
    parser.add_argument(
        "--spatial-mask-topology",
        choices=("bernoulli", "fixed_count", "multiblock"),
        default="fixed_count",
    )
    parser.add_argument("--spatial-fill-value", type=int, default=128)
    parser.add_argument("--posterior-jepa-scale", type=float, default=2.0)
    parser.add_argument("--dynamics-jepa-scale", type=float, default=2.0)
    parser.add_argument("--spatial-jepa-scale", type=float, default=1.0)
    parser.add_argument(
        "--sigreg",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sigreg-scale", type=float, default=0.05)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=256)
    parser.add_argument(
        "--sigreg-aggregation",
        choices=("pooled", "per_timestep"),
        default="pooled",
        help="Pool batch and time or evaluate each timestep across the batch.",
    )
    parser.add_argument(
        "--posterior-context",
        choices=("observation", "history"),
        default="history",
        help=(
            "Condition the categorical posterior on only the current observation "
            "or on causal Transformer history and the current observation."
        ),
    )
    parser.add_argument(
        "--visual-encoder",
        choices=("simple", "vit"),
        default="simple",
        help="Use the DreamerV3 CNN or the compact 64px ViT control.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--curve-eval-interval",
        type=int,
        default=0,
        help="Run fixed-policy evaluation at this environment-step interval.",
    )
    parser.add_argument("--curve-eval-episodes", type=int, default=20)
    parser.add_argument("--curve-eval-seed-offset", type=int, default=10_000)
    parser.add_argument(
        "--curve-eval-policy-mode",
        choices=("deterministic", "stochastic"),
        default="deterministic",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    experiment_dir = args.experiment_dir or (
        args.output_root / args.task / f"seed_{args.seed}" / timestamp()
    )
    spec = AblationRunSpec(
        experiment_dir=experiment_dir,
        task=args.task,
        seed=args.seed,
        train_steps=args.total_env_steps,
        num_agents=args.num_agents,
        platform=args.platform,
        infrastructure_root=args.infrastructure_root,
        python=args.python,
        save_every_seconds=args.save_every_seconds,
        final_checkpoint=args.final_checkpoint,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        temporal_model=args.temporal_model,
        world_model_objective=args.world_model_objective,
        embedding_target=args.embedding_target,
        embedding_loss=args.embedding_loss,
        posterior_jepa=args.posterior_jepa,
        dynamics_jepa=args.dynamics_jepa,
        spatial_jepa=args.spatial_jepa,
        spatial_mask_ratio=args.spatial_mask_ratio,
        spatial_mask_topology=args.spatial_mask_topology,
        spatial_fill_value=args.spatial_fill_value,
        posterior_jepa_scale=args.posterior_jepa_scale,
        dynamics_jepa_scale=args.dynamics_jepa_scale,
        spatial_jepa_scale=args.spatial_jepa_scale,
        sigreg=args.sigreg,
        sigreg_scale=args.sigreg_scale,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        sigreg_aggregation=args.sigreg_aggregation,
        posterior_context=args.posterior_context,
        visual_encoder=args.visual_encoder,
        batch_size=args.batch_size,
        curve_eval_interval=args.curve_eval_interval,
        curve_eval_episodes=args.curve_eval_episodes,
        curve_eval_seed_offset=args.curve_eval_seed_offset,
        curve_eval_policy_mode=args.curve_eval_policy_mode,
    )
    print(f"Experiment: {spec.experiment_dir}")
    return run_training(
        spec,
        resume=args.resume,
        dry_run=args.dry_run,
        contract_verifier=verify_run_contract,
    )


if __name__ == "__main__":
    raise SystemExit(main())
