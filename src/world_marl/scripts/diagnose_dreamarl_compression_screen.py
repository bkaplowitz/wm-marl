"""Run the minimal B versus O_L versus O_J causal predictor screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from world_marl.dreamarl.compression_screen import (
    ScreenConfig,
    ScreenInput,
    evaluate_screen_predictor,
    parameter_count,
    prepare_screen_dataset,
    summarize_screen_errors,
    train_screen_predictor,
)
from world_marl.dreamarl.diagnostic_dataset import load_dataset, split_trajectories


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-width", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-trajectories", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--promotion-delta", type=float, default=0.02)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    return parser


def _mean_error(seed_results: dict[str, object], variant: ScreenInput) -> float:
    return float(seed_results[variant.value]["test"]["overall"]["mean"])


def run(args: argparse.Namespace) -> dict[str, object]:
    bundle = load_dataset(args.dataset)
    split = split_trajectories(bundle.arrays, seed=args.split_seed)
    prepared = prepare_screen_dataset(
        bundle.arrays,
        split,
        feature_width=args.feature_width,
        action_dim=int(bundle.manifest["action_dim"]),
        projection_seed=args.projection_seed,
    )
    variants = tuple(ScreenInput)
    results = {}
    expected_parameters = None
    for seed in args.seeds:
        config = ScreenConfig(
            feature_width=args.feature_width,
            hidden=args.hidden,
            heads=args.heads,
            temporal_layers=args.temporal_layers,
            learning_rate=args.learning_rate,
            steps=args.steps,
            batch_trajectories=args.batch_trajectories,
            seed=seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        seed_results = {}
        for variant in variants:
            params, history = train_screen_predictor(
                prepared, split.train, variant, config
            )
            count = parameter_count(params)
            if expected_parameters is None:
                expected_parameters = count
            elif count != expected_parameters:
                raise AssertionError("screen variants do not have equal capacity")
            validation = evaluate_screen_predictor(
                params, prepared, split.validation, variant, config
            )
            test = evaluate_screen_predictor(
                params, prepared, split.test, variant, config
            )
            seed_results[variant.value] = {
                "parameters": count,
                "training_history": history,
                "validation": summarize_screen_errors(validation, config),
                "test": summarize_screen_errors(test, config),
            }
        seed_results["effects"] = {
            "local_compression": _mean_error(seed_results, ScreenInput.BELIEF_LOCAL)
            - _mean_error(seed_results, ScreenInput.OBSERVATION_LOCAL),
            "other_agent_information": _mean_error(
                seed_results, ScreenInput.OBSERVATION_LOCAL
            )
            - _mean_error(seed_results, ScreenInput.OBSERVATION_JOINT),
        }
        results[str(seed)] = seed_results

    effects = {
        name: [float(result["effects"][name]) for result in results.values()]
        for name in ("local_compression", "other_agent_information")
    }
    promotion = {
        name: {
            "mean": float(np.mean(values)),
            "consistent_positive": bool(all(value > 0 for value in values)),
            "passes": bool(
                all(value > 0 for value in values)
                and np.mean(values) >= args.promotion_delta
            ),
        }
        for name, values in effects.items()
    }
    output = {
        "contract": "matched_causal_compression_screen_v1",
        "dataset": str(args.dataset.expanduser().resolve()),
        "dataset_manifest": bundle.manifest,
        "scope": {
            "environment_steps": 0,
            "actor_updates": 0,
            "critic_updates": 0,
            "trajectory_disjoint_split": True,
            "target": "frozen_next_belief",
            "inputs": {
                "B": "local_belief_history_and_local_action",
                "O_L": "local_observation_token_history_and_local_action",
                "O_J": "joint_observation_token_histories_and_local_action",
            },
        },
        "config": {
            "feature_width": args.feature_width,
            "hidden": args.hidden,
            "heads": args.heads,
            "temporal_layers": args.temporal_layers,
            "steps": args.steps,
            "batch_trajectories": args.batch_trajectories,
            "learning_rate": args.learning_rate,
            "split_seed": args.split_seed,
            "projection_seed": args.projection_seed,
            "predictor_seeds": args.seeds,
            "promotion_delta": args.promotion_delta,
        },
        "split": {
            "train": split.train.tolist(),
            "validation": split.validation.tolist(),
            "test": split.test.tolist(),
        },
        "seeds": results,
        "promotion": promotion,
        "replicate_on_second_world_model": bool(
            any(item["passes"] for item in promotion.values())
        ),
    }
    output_path = (
        args.output / "results.json"
        if args.output.exists() and args.output.is_dir()
        else args.output
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    _log_wandb(args, output)
    return output


def _log_wandb(args: argparse.Namespace, output: dict[str, object]) -> None:
    if not args.wandb_project:
        return
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or args.output.stem,
        config={**output["config"], **output["scope"]},
    )
    for seed_index, (seed, seed_results) in enumerate(output["seeds"].items()):
        metrics = {
            f"seed_{seed}/local_compression": seed_results["effects"][
                "local_compression"
            ],
            f"seed_{seed}/other_agent_information": seed_results["effects"][
                "other_agent_information"
            ],
        }
        for variant in ScreenInput:
            metrics[f"seed_{seed}/{variant.value}_test_error"] = seed_results[
                variant.value
            ]["test"]["overall"]["mean"]
        run.log(metrics, step=seed_index)
    run.finish()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run(args)
    for seed, result in output["seeds"].items():
        print(f"seed={seed}")
        for variant in ScreenInput:
            error = result[variant.value]["test"]["overall"]["mean"]
            print(f"  {variant.value:30s} error={error:.6f}")
        print(f"  effects={result['effects']}")
    print(f"promotion={output['promotion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
