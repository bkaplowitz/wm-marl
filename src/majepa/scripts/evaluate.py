"""Evaluate the latest canonical MA-JEPA checkpoint without selection."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from majepa.config import (
    algorithm_config_profiles,
    environment_config_profile,
    evaluation_defaults,
)
from majepa.launcher import runtime_environment
from majepa.runtime import absolute_path, repository_root


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_checkpoint(experiment: Path) -> Path:
    root = experiment / "run" / "ckpt"
    latest = root / "latest"
    checkpoint = root / latest.read_text(encoding="utf-8").strip()
    if not (checkpoint / "done").is_file():
        raise FileNotFoundError(f"incomplete checkpoint: {checkpoint}")
    return checkpoint


def _manifest_algorithm(manifest: dict[str, object]) -> str:
    """Reject checkpoints from historical experiment branches."""

    algorithm = manifest.get("algorithm")
    if algorithm == "ma-jepa":
        return str(algorithm)
    raise ValueError("checkpoint was not produced by MA-JEPA")


def _evaluation_protocol(manifest, *, episodes, envs, eval_seed):
    defaults = evaluation_defaults()
    resolved_episodes = int(defaults["episodes"] if episodes is None else episodes)
    resolved_envs = int(defaults["envs"] if envs is None else envs)
    resolved_seed = (
        int(manifest.get("seed", 0)) if eval_seed is None else int(eval_seed)
    )
    if resolved_episodes < 1 or resolved_envs < 1:
        raise ValueError("evaluation requires positive episode and environment counts")
    return resolved_episodes, min(resolved_envs, resolved_episodes), resolved_seed


def main(argv: list[str] | None = None) -> int:
    defaults = evaluation_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--eval-seed", type=int)
    parser.add_argument(
        "--policy-mode",
        choices=("deterministic", "stochastic"),
        default=defaults["policy_mode"],
    )
    parser.add_argument("--python", type=Path)
    parser.add_argument("--platform", choices=("cpu", "cuda", "tpu"))
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    experiment = args.experiment_dir.expanduser().resolve()
    manifest = json.loads((experiment / "launch.json").read_text(encoding="utf-8"))
    if manifest.get("implementation") != "MA-JEPA":
        raise ValueError("checkpoint was not produced by MA-JEPA")

    algorithm = _manifest_algorithm(manifest)
    episodes, envs, eval_seed = _evaluation_protocol(
        manifest,
        episodes=args.episodes,
        envs=args.envs,
        eval_seed=args.eval_seed,
    )
    environment_config_profile(str(manifest["task"]), int(manifest["num_agents"]))
    profiles = [*algorithm_config_profiles(algorithm), "eval_only"]
    evaluation = experiment / "evaluation" / f"seed_{eval_seed}_{_timestamp()}"
    python = absolute_path(args.python or Path(str(manifest["python"])))
    gradients = manifest.get("world_model_gradients", {})
    model_controls = manifest.get("model_controls", {})
    command = [
        str(python),
        "-m",
        "majepa.main",
        "--logdir",
        str(evaluation),
        "--configs",
        *profiles,
        "--task",
        str(manifest["task"]),
        "--seed",
        str(eval_seed),
        "--agent.num_agents",
        str(manifest["num_agents"]),
        "--agent.world_model_gradients.critic_value_scale",
        str(gradients.get("critic_value_scale", 0.0)),
        "--agent.world_model_gradients.joint_objective_scale",
        str(gradients.get("joint_objective_scale", 1.0)),
        "--agent.world_model_gradients.joint_prediction",
        str(gradients.get("joint_prediction", False)),
        "--agent.world_model_gradients.joint_prediction_scale",
        str(gradients.get("joint_prediction_scale", 1.0)),
        "--agent.dyn.parallel_transformer.local_prior",
        str(model_controls.get("local_prior", True)),
        "--agent.loss_scales.ctde_multistep_jepa_action",
        str(model_controls.get("action_margin_loss_scale", 0.1)),
        "--agent.dyn.parallel_transformer.stoch",
        str(model_controls.get("categorical_stoch", 32)),
        "--agent.dyn.parallel_transformer.classes",
        str(model_controls.get("categorical_classes", 64)),
        "--run.from_checkpoint",
        str(_latest_checkpoint(experiment)),
        "--run.eval_eps",
        str(episodes),
        "--run.eval_policy_mode",
        "eval" if args.policy_mode == "deterministic" else "eval_sample",
        "--run.envs",
        str(envs),
        "--run.steps",
        str(manifest.get("train_env_steps_budget", 50_000)),
        "--jax.platform",
        args.platform or str(manifest["platform"]),
        "--logger.outputs",
        "jsonl",
        "wandb",
    ]
    evaluation.parent.mkdir(parents=True, exist_ok=True)
    evaluation.with_suffix(".launch.json").write_text(
        json.dumps(
            {
                "algorithm": algorithm,
                "command": command,
                "configs": profiles,
                "episodes": episodes,
                "envs": envs,
                "eval_seed": eval_seed,
                "worker_offset": defaults["worker_offset"],
                "policy_mode": args.policy_mode,
                "source_experiment": str(experiment),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(" ".join(command))
    if args.dry_run:
        return 0

    env = runtime_environment(
        task=str(manifest["task"]),
        infrastructure_root=Path(str(manifest["infrastructure_root"])),
        artifact_dir=evaluation,
        wandb_job_type="final100",
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=f"{experiment.name}_fixed_eval",
    )
    return subprocess.run(
        command, cwd=repository_root(), env=env, check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
