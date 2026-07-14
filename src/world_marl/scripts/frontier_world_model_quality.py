"""Evaluate DreamerV3 and Genie2 on JAX-native Brax and MJX DMC control."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any

import jax
import numpy as np

from world_marl.scripts.compare_visual_wm import ARM_COMMANDS, build_arm_command
from world_marl.world_model_foundation.collect import make_single_agent_adapter
from world_marl.world_model_foundation.metrics import scanned_episode_metrics


MODELS = tuple(ARM_COMMANDS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--brax-env", default="brax:reacher")
    parser.add_argument("--brax-backend", default="mjx")
    parser.add_argument("--brax-collect-steps", type=int, default=256)
    parser.add_argument("--brax-num-envs", type=int, default=16)
    parser.add_argument("--brax-train-steps", type=int, default=1000)
    parser.add_argument("--brax-policy-train-steps", type=int, default=1000)
    parser.add_argument("--brax-eval-episodes", type=int, default=16)
    parser.add_argument("--dmc-task", default="cartpole/swingup")
    parser.add_argument("--dmc-collect-steps", type=int, default=81_920)
    parser.add_argument("--dmc-num-envs", type=int, default=2)
    parser.add_argument("--dmc-train-steps", type=int, default=3_000)
    parser.add_argument("--dmc-policy-train-steps", type=int, default=1_500)
    parser.add_argument("--dmc-eval-episodes", type=int, default=8)
    parser.add_argument(
        "--genie-policy-objective",
        choices=("reinforce", "candidate-distill"),
        default="candidate-distill",
    )
    parser.add_argument("--genie-num-policy-candidates", type=int, default=64)
    parser.add_argument("--genie-candidate-min-gap", type=float, default=0.0)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--reference-random-return", type=float, required=True)
    parser.add_argument("--reference-trained-return", type=float, required=True)
    parser.add_argument("--reference-real-env-transitions", type=int, default=163_840)
    parser.add_argument("--reference-env", default="dmc:cartpole/swingup")
    parser.add_argument(
        "--reference-label", default="singlerl_jepa_dmc_cartpole_swingup"
    )
    parser.add_argument("--reference-source", default="unspecified")
    parser.add_argument("--min-reference-fraction", type=float, default=0.5)
    parser.add_argument("--max-loss-ratio", type=float, default=0.9)
    parser.add_argument("--min-dmc-random-improvement", type=float, default=50.0)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.genie_num_policy_candidates < 2:
        parser.error("--genie-num-policy-candidates must be at least 2")
    if args.genie_candidate_min_gap < 0.0:
        parser.error("--genie-candidate-min-gap must be non-negative")
    return args


def assess_quality(
    *,
    brax_runs: list[dict[str, Any]],
    dmc_rows: list[dict[str, Any]],
    expected_seed_count: int,
    expected_real_env_transitions: int | None = None,
    reference_random_return: float,
    reference_trained_return: float,
    min_reference_fraction: float,
    max_loss_ratio: float,
    min_dmc_random_improvement: float,
) -> dict[str, Any]:
    reference_gap = reference_trained_return - reference_random_return
    if not math.isfinite(reference_gap) or reference_gap <= 0:
        raise ValueError("reference trained return must exceed reference random return")

    dmc_by_model = {str(row["model"]): row for row in dmc_rows}
    random_row = dmc_by_model.get("random_action")
    random_dmc_return = (
        float(random_row["mean_return"]) if random_row is not None else math.nan
    )
    random_dmc_seed_count = (
        int(random_row.get("successful_seed_count", expected_seed_count))
        if random_row is not None
        else 0
    )
    model_reports: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        runs = [row for row in brax_runs if row.get("model") == model]
        loss_ratios = []
        for row in runs:
            initial_loss = row.get("initial_loss")
            final_loss = row.get("final_loss")
            if initial_loss is None or final_loss is None:
                continue
            initial_loss = float(initial_loss)
            final_loss = float(final_loss)
            if (
                initial_loss > 0
                and math.isfinite(initial_loss)
                and math.isfinite(final_loss)
            ):
                loss_ratios.append(final_loss / initial_loss)
        learning_gate_results = [
            bool(row["learning_gate_passed"])
            for row in runs
            if "learning_gate_passed" in row
        ]
        returns = [
            float(row["evaluation_return"])
            for row in runs
            if math.isfinite(float(row["evaluation_return"]))
        ]
        seed_count_passed = len(runs) == expected_seed_count
        scalar_loss_gate_passed = not loss_ratios or all(
            ratio <= max_loss_ratio for ratio in loss_ratios
        )
        learning_passed = (
            seed_count_passed
            and len(learning_gate_results) == expected_seed_count
            and all(learning_gate_results)
            and scalar_loss_gate_passed
        )
        brax_mean = statistics.mean(returns) if returns else math.nan
        brax_execution_passed = len(returns) == expected_seed_count

        dmc_row = dmc_by_model.get(model)
        dmc_mean = float(dmc_row["mean_return"]) if dmc_row is not None else math.nan
        dmc_seed_count = (
            int(dmc_row.get("successful_seed_count", expected_seed_count))
            if dmc_row is not None
            else 0
        )
        transition_counts = (
            [int(value) for value in dmc_row.get("real_env_transition_counts", [])]
            if dmc_row is not None
            else []
        )
        transition_budget_passed = expected_real_env_transitions is None or (
            len(transition_counts) == expected_seed_count
            and all(
                count == expected_real_env_transitions for count in transition_counts
            )
        )
        dmc_improvement = dmc_mean - random_dmc_return
        dmc_reference_fraction = (
            (dmc_mean - reference_random_return) / reference_gap
            if math.isfinite(dmc_mean)
            else math.nan
        )
        dmc_reference_passed = (
            math.isfinite(dmc_reference_fraction)
            and dmc_reference_fraction >= min_reference_fraction
        )
        dmc_measured_random_passed = (
            math.isfinite(dmc_improvement)
            and dmc_improvement >= min_dmc_random_improvement
        )
        dmc_quality_passed = (
            random_dmc_seed_count == expected_seed_count
            and dmc_seed_count == expected_seed_count
            and transition_budget_passed
            and dmc_measured_random_passed
            and dmc_reference_passed
        )
        passed = learning_passed and brax_execution_passed and dmc_quality_passed
        model_reports[model] = {
            "passed": passed,
            "seed_count": len(runs),
            "learning_passed": learning_passed,
            "learning_gate_results": learning_gate_results,
            "loss_ratios": loss_ratios,
            "scalar_loss_gate_passed": scalar_loss_gate_passed,
            "max_observed_loss_ratio": max(loss_ratios, default=None),
            "brax_mean_return": brax_mean,
            "brax_execution_passed": brax_execution_passed,
            "dmc_mean_return": dmc_mean,
            "real_env_transition_counts": transition_counts,
            "transition_budget_passed": transition_budget_passed,
            "dmc_random_return": random_dmc_return,
            "dmc_random_improvement": dmc_improvement,
            "dmc_measured_random_passed": dmc_measured_random_passed,
            "dmc_reference_fraction": dmc_reference_fraction,
            "dmc_reference_passed": dmc_reference_passed,
            "dmc_quality_passed": dmc_quality_passed,
        }

    return {
        "passed": all(row["passed"] for row in model_reports.values()),
        "expected_seed_count": expected_seed_count,
        "dmc_random_seed_count": random_dmc_seed_count,
        "reference": {
            "random_return": reference_random_return,
            "trained_return": reference_trained_return,
            "real_env_transitions": expected_real_env_transitions,
        },
        "thresholds": {
            "max_loss_ratio": max_loss_ratio,
            "min_reference_fraction": min_reference_fraction,
            "min_dmc_random_improvement": min_dmc_random_improvement,
        },
        "models": model_reports,
    }


def _run_brax(args: argparse.Namespace, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    records = []
    for model in MODELS:
        for seed in seeds:
            out_dir = args.out_dir / "brax" / model / f"seed_{seed}"
            policy_train_steps = (
                args.brax_train_steps
                if model == "dreamer_v3_baseline"
                else args.brax_policy_train_steps
            )
            command = build_arm_command(
                model,
                env=args.brax_env,
                out_dir=out_dir,
                collect_steps=args.brax_collect_steps,
                num_envs=args.brax_num_envs,
                max_cycles=args.max_cycles,
                train_steps=args.brax_train_steps,
                policy_train_steps=policy_train_steps,
                eval_episodes=args.brax_eval_episodes,
                allow_fail=True,
                seed=seed,
                brax_backend=args.brax_backend,
            )
            if model == "genie2_continuous_jax":
                command.extend(
                    [
                        "--policy-objective",
                        args.genie_policy_objective,
                        "--num-policy-candidates",
                        str(args.genie_num_policy_candidates),
                        "--candidate-min-gap",
                        str(args.genie_candidate_min_gap),
                    ]
                )
            record = {
                "environment_family": "brax",
                "model": model,
                "seed": seed,
                "command": command,
                "policy_objective": (
                    args.genie_policy_objective
                    if model == "genie2_continuous_jax"
                    else "dreamer_v3_imagination"
                ),
            }
            if not args.dry_run:
                result = subprocess.run(command, check=False)
                record["return_code"] = int(result.returncode)
            records.append(record)
    return records


def _run_dmc(
    args: argparse.Namespace,
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    records = []
    env = f"dmc:{args.dmc_task}"
    for model in MODELS:
        for seed in seeds:
            out_dir = args.out_dir / "dmc_vector" / model / f"seed_{seed}"
            policy_train_steps = (
                args.dmc_train_steps
                if model == "dreamer_v3_baseline"
                else args.dmc_policy_train_steps
            )
            command = build_arm_command(
                model,
                env=env,
                out_dir=out_dir,
                collect_steps=args.dmc_collect_steps,
                num_envs=args.dmc_num_envs,
                max_cycles=args.max_cycles,
                train_steps=args.dmc_train_steps,
                policy_train_steps=policy_train_steps,
                eval_episodes=args.dmc_eval_episodes,
                allow_fail=True,
                seed=seed,
            )
            if model == "genie2_continuous_jax":
                command.extend(
                    [
                        "--policy-objective",
                        args.genie_policy_objective,
                        "--num-policy-candidates",
                        str(args.genie_num_policy_candidates),
                        "--candidate-min-gap",
                        str(args.genie_candidate_min_gap),
                    ]
                )
            record = {
                "environment_family": "dmc",
                "model": model,
                "seed": seed,
                "command": command,
                "policy_objective": (
                    args.genie_policy_objective
                    if model == "genie2_continuous_jax"
                    else "dreamer_v3_imagination"
                ),
            }
            if not args.dry_run:
                result = subprocess.run(command, check=False)
                record["return_code"] = int(result.returncode)
            records.append(record)
    return records


def _evaluate_random_dmc(args: argparse.Namespace, seeds: tuple[int, ...]) -> None:
    target_episodes = max(args.dmc_eval_episodes, 1)
    evaluation_num_envs = math.gcd(args.dmc_num_envs, target_episodes)
    evaluation_steps = math.ceil(target_episodes / evaluation_num_envs) * (
        args.max_cycles + 1
    )
    for seed in seeds:
        adapter = make_single_agent_adapter(
            f"dmc:{args.dmc_task}",
            num_envs=evaluation_num_envs,
            max_cycles=args.max_cycles,
            seed=seed + 40_000,
        )
        try:
            observations = np.asarray(adapter.reset(), dtype=np.float32)
            ys = adapter.scan_random_sequence(
                evaluation_steps,
                key=jax.random.PRNGKey(seed + 50_000),
                observations=observations,
            )
            _, _, rewards, _, lasts = ys
            episode_metrics = scanned_episode_metrics(
                rewards,
                lasts,
                target_episodes=target_episodes,
                policy_source="random_action",
                arrival_aligned=True,
            )
            summary = {
                "model": "random_action",
                "seed": seed,
                "env": f"dmc:{args.dmc_task}",
                "environment_backend": "mujoco_playground",
                "physics_backend": "mjx",
                "observation_mode": "vector",
                "evaluation_execution": "jax_scan",
                "real_env_return": statistics.mean(
                    float(row["return"]) for row in episode_metrics
                ),
            }
            _write_json(
                args.out_dir
                / "dmc_vector"
                / "random_action"
                / f"seed_{seed}"
                / "summary.json",
                summary,
            )
        finally:
            close = getattr(adapter, "close", None)
            if close is not None:
                close()


def _load_brax_runs(args: argparse.Namespace, seeds: tuple[int, ...]) -> list[dict]:
    rows = []
    for model in MODELS:
        for seed in seeds:
            run_dir = args.out_dir / "brax" / model / f"seed_{seed}"
            summary = json.loads((run_dir / "summary.json").read_text())
            outcome = json.loads((run_dir / "outcome.json").read_text())
            row = {
                "model": model,
                "seed": seed,
                "learning_gate_passed": bool(summary["learning_gate_passed"]),
                "evaluation_return": summary.get(
                    "real_env_bridged_return", summary.get("real_env_return")
                ),
            }
            if "initial_loss" in outcome and "final_loss" in outcome:
                row.update(
                    initial_loss=outcome["initial_loss"],
                    final_loss=outcome["final_loss"],
                )
            rows.append(row)
    return rows


def _load_dmc_rows(
    args: argparse.Namespace,
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = []
    for model in (*MODELS, "random_action"):
        returns = []
        real_env_transition_counts = []
        for seed in seeds:
            summary_path = (
                args.out_dir / "dmc_vector" / model / f"seed_{seed}" / "summary.json"
            )
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            value = summary.get(
                "real_env_bridged_return",
                summary.get("real_env_return"),
            )
            if value is not None and math.isfinite(float(value)):
                returns.append(float(value))
                if model != "random_action" and "real_env_transitions" in summary:
                    real_env_transition_counts.append(
                        int(summary["real_env_transitions"])
                    )
        rows.append(
            {
                "model": model,
                "mean_return": statistics.mean(returns) if returns else math.nan,
                "successful_seed_count": len(returns),
                "real_env_transition_counts": real_env_transition_counts,
            }
        )
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dmc_env = f"dmc:{args.dmc_task}"
    if dmc_env != args.reference_env:
        raise ValueError(
            "reference environment must match the evaluated DMC environment: "
            f"{args.reference_env!r} != {dmc_env!r}"
        )
    seeds = tuple(args.seed or (0, 1, 2))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    brax_commands = _run_brax(args, seeds)
    dmc_commands = _run_dmc(args, seeds)
    _write_json(args.out_dir / "commands.json", brax_commands + dmc_commands)
    _write_json(
        args.out_dir / "suite.json",
        {
            "seeds": seeds,
            "brax_env": args.brax_env,
            "dmc_task": args.dmc_task,
            "reference_env": args.reference_env,
            "reference_label": args.reference_label,
            "reference_source": args.reference_source,
            "reference_real_env_transitions": args.reference_real_env_transitions,
            "brax_runs": brax_commands,
            "dmc_runs": dmc_commands,
            "dmc_observation_mode": "vector",
            "dmc_physics_backend": "mjx",
        },
    )
    if args.dry_run:
        return 0

    _evaluate_random_dmc(args, seeds)
    brax_runs = _load_brax_runs(args, seeds)
    dmc_rows = _load_dmc_rows(args, seeds)
    _write_json(args.out_dir / "dmc_vector" / "aggregate.json", dmc_rows)
    report = assess_quality(
        brax_runs=brax_runs,
        dmc_rows=dmc_rows,
        expected_seed_count=len(seeds),
        expected_real_env_transitions=args.reference_real_env_transitions,
        reference_random_return=args.reference_random_return,
        reference_trained_return=args.reference_trained_return,
        min_reference_fraction=args.min_reference_fraction,
        max_loss_ratio=args.max_loss_ratio,
        min_dmc_random_improvement=args.min_dmc_random_improvement,
    )
    report["reference"].update(
        {
            "environment": args.reference_env,
            "label": args.reference_label,
            "source": args.reference_source,
        }
    )
    _write_json(args.out_dir / "quality_report.json", report)
    return 0 if report["passed"] or args.allow_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
