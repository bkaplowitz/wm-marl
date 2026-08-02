"""Run the preregistered one-step DreaMARL information ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from world_marl.dreamarl.diagnostic_dataset import (
    load_dataset,
    split_trajectories,
)
from world_marl.dreamarl.information_ladder import (
    LadderConfig,
    available_rungs,
    evaluate_predictor,
    evaluate_residual_predictor,
    prepare_dataset,
    summarize_delta,
    summarize_errors,
    train_predictor,
    train_residual_predictor,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-width", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-trajectories", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    bundle = load_dataset(args.dataset)
    split = split_trajectories(bundle.arrays, seed=args.split_seed)
    action_dim = int(bundle.manifest["action_dim"])
    prepared = prepare_dataset(
        bundle.arrays,
        split,
        feature_width=args.feature_width,
        action_dim=action_dim,
    )
    rungs = available_rungs(bundle.arrays)
    seeds = {}
    for seed in args.seeds:
        config = LadderConfig(
            feature_width=args.feature_width,
            hidden=args.hidden,
            learning_rate=args.learning_rate,
            steps=args.steps,
            batch_trajectories=args.batch_trajectories,
            seed=seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        rung_results = {}
        previous = None
        x0_params = None
        x0_test = None
        parameter_count = None
        for rung_index, rung in enumerate(rungs):
            params, history = train_predictor(prepared, split.train, rung, config)
            current_parameter_count = sum(
                int(value.size) for value in jax.tree.leaves(params)
            )
            if parameter_count is None:
                parameter_count = current_parameter_count
            elif parameter_count != current_parameter_count:
                raise AssertionError("ladder rungs do not have equal capacity")
            validation = evaluate_predictor(params, prepared, split.validation, rung)
            test = evaluate_predictor(params, prepared, split.test, rung)
            result = {
                "parameters": current_parameter_count,
                "training_history": history,
                "validation": summarize_errors(
                    validation,
                    seed=seed + 100 * rung_index,
                    bootstrap_samples=args.bootstrap_samples,
                ),
                "test": summarize_errors(
                    test,
                    seed=seed + 1000 + 100 * rung_index,
                    bootstrap_samples=args.bootstrap_samples,
                ),
            }
            if rung_index == 0:
                x0_params = params
                x0_test = test
            else:
                residual_params, residual_history = train_residual_predictor(
                    x0_params, prepared, split.train, rung, config
                )
                residual_validation = evaluate_residual_predictor(
                    x0_params,
                    residual_params,
                    prepared,
                    split.validation,
                    rung,
                )
                residual_test = evaluate_residual_predictor(
                    x0_params, residual_params, prepared, split.test, rung
                )
                result["residual"] = {
                    "parameters": sum(
                        int(value.size)
                        for value in jax.tree.leaves(residual_params)
                    ),
                    "training_history": residual_history,
                    "validation": summarize_errors(
                        residual_validation,
                        seed=seed + 3000 + 100 * rung_index,
                        bootstrap_samples=args.bootstrap_samples,
                    ),
                    "test": summarize_errors(
                        residual_test,
                        seed=seed + 4000 + 100 * rung_index,
                        bootstrap_samples=args.bootstrap_samples,
                    ),
                    "test_gain_over_x0": summarize_delta(
                        x0_test,
                        residual_test,
                        seed=seed + 5000 + rung_index,
                        bootstrap_samples=args.bootstrap_samples,
                    ),
                }
            if previous is not None:
                result["test_delta_from_previous"] = summarize_delta(
                    previous,
                    test,
                    seed=seed + 2000 + rung_index,
                    bootstrap_samples=args.bootstrap_samples,
                )
            rung_results[rung.value] = result
            previous = test
        seeds[str(seed)] = rung_results

    output = {
        "contract": "equal_capacity_h1_information_ladder_v1",
        "scope": {
            "environment_steps": 0,
            "actor_updates": 0,
            "critic_updates": 0,
            "horizon": 1,
            "trajectory_disjoint_split": True,
            "held_out_policy_checkpoint_split": bool(
                np.unique(bundle.arrays["policy_checkpoint"]).size > 1
            ),
            "note": (
                "Checkpoint-stratified trajectory splits are used when multiple "
                "checkpoint-labelled datasets have been combined."
            ),
        },
        "dataset": str(args.dataset.expanduser().resolve()),
        "dataset_manifest": bundle.manifest,
        "split": {
            "train": split.train.tolist(),
            "validation": split.validation.tolist(),
            "test": split.test.tolist(),
        },
        "config": {
            "feature_width": args.feature_width,
            "hidden": args.hidden,
            "steps": args.steps,
            "batch_trajectories": args.batch_trajectories,
            "learning_rate": args.learning_rate,
            "split_seed": args.split_seed,
            "seeds": args.seeds,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "rungs": [rung.value for rung in rungs],
        "seeds": seeds,
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
    for seed, results in output["seeds"].items():
        for rung_index, (rung, result) in enumerate(results.items()):
            metrics = {
                f"seed_{seed}/test_error": result["test"]["overall"]["mean"],
                f"seed_{seed}/test_error_ci_low": result["test"]["overall"]["low"],
                f"seed_{seed}/test_error_ci_high": result["test"]["overall"]["high"],
            }
            if "test_delta_from_previous" in result:
                metrics[f"seed_{seed}/delta"] = result[
                    "test_delta_from_previous"
                ]["mean"]
                metrics[f"seed_{seed}/delta_ci_low"] = result[
                    "test_delta_from_previous"
                ]["low"]
                metrics[f"seed_{seed}/delta_ci_high"] = result[
                    "test_delta_from_previous"
                ]["high"]
            run.log({**metrics, "rung": rung}, step=rung_index)
    run.finish()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run(args)
    for seed, results in output["seeds"].items():
        print(f"seed={seed}")
        for rung, result in results.items():
            error = result["test"]["overall"]
            delta = result.get("test_delta_from_previous")
            suffix = "" if delta is None else f" delta={delta['mean']:+.6f}"
            print(f"  {rung:42s} error={error['mean']:.6f}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
