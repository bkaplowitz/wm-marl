"""Evaluate a historical DreaMARL ablation checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import platform as host_platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dreamarl.baselines.dreamerv3.config import absolute_path
from dreamarl.runtime import repository_root


def _timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_checkpoint(experiment: Path) -> Path:
    root = experiment / "run" / "ckpt"
    latest = root / "latest"
    if not latest.is_file():
        raise FileNotFoundError(f"checkpoint pointer is missing: {latest}")
    checkpoint = root / latest.read_text(encoding="utf-8").strip()
    if not checkpoint.is_dir() or not (checkpoint / "done").is_file():
        raise FileNotFoundError(f"latest checkpoint is incomplete: {checkpoint}")
    return checkpoint


def _model_arguments(manifest: dict[str, object]) -> list[str]:
    """Reconstruct the complete world-model shape recorded at launch."""

    arguments = [
        "--agent.dyn.typ",
        str(manifest["world_model"]),
        "--agent.enc.typ",
        str(manifest.get("visual_encoder", "simple")),
        "--agent.objective",
        str(manifest.get("world_model_objective", "reconstruction")),
        "--agent.embedding_target",
        str(manifest.get("embedding_target", "ema")),
        "--agent.embedding_loss",
        str(manifest.get("embedding_loss", "cosine")),
        "--agent.posterior_jepa",
        str(bool(manifest.get("posterior_jepa", False))),
        "--agent.dynamics_jepa",
        str(bool(manifest.get("dynamics_jepa", False))),
        "--agent.spatial_jepa.enabled",
        str(bool(manifest.get("spatial_jepa", False))),
        "--agent.spatial_jepa.mask_ratio",
        str(manifest.get("spatial_mask_ratio", 0.5)),
        "--agent.spatial_jepa.topology",
        str(manifest.get("spatial_mask_topology", "bernoulli")),
        "--agent.spatial_jepa.fill_value",
        str(manifest.get("spatial_fill_value", 128)),
        "--agent.loss_scales.posterior_jepa",
        str(manifest.get("posterior_jepa_scale", 1.0)),
        "--agent.loss_scales.dynamics_jepa",
        str(manifest.get("dynamics_jepa_scale", 1.0)),
        "--agent.loss_scales.spatial_jepa",
        str(manifest.get("spatial_jepa_scale", 1.0)),
        "--agent.sigreg.enabled",
        str(bool(manifest.get("sigreg", False))),
        "--agent.sigreg.knots",
        str(manifest.get("sigreg_knots", 17)),
        "--agent.sigreg.num_proj",
        str(manifest.get("sigreg_num_proj", 256)),
        "--agent.sigreg.aggregation",
        str(manifest.get("sigreg_aggregation", "pooled")),
        "--agent.loss_scales.sigreg",
        str(manifest.get("sigreg_scale", 0.05)),
    ]
    if manifest["world_model"] == "parallel_transformer":
        arguments.extend(
            [
                "--agent.dyn.parallel_transformer.posterior_context",
                str(manifest.get("posterior_context", "history")),
            ]
        )
    return arguments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument(
        "--policy-mode",
        choices=("deterministic", "stochastic"),
        default="deterministic",
        help=(
            "Use deterministic distribution predictions or sampled actions. "
            "The latter matches the pinned DreamerV3 policy protocol."
        ),
    )
    parser.add_argument("--python", type=Path)
    parser.add_argument("--platform", choices=("cpu", "cuda", "tpu"))
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    experiment = args.experiment_dir.expanduser().resolve()
    manifest_path = experiment / "launch.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = experiment / "evaluation" / f"seed_{args.eval_seed}_{_timestamp()}"
    python = absolute_path(args.python or Path(manifest["python"]))
    checkpoint = _latest_checkpoint(experiment)
    platform = args.platform or manifest["platform"]
    outputs = ["jsonl", "scope"]
    if args.wandb_project:
        outputs.append("wandb")
    configs = manifest["configs"]
    command = [
        str(python),
        "-m",
        "dreamarl.ablations.main",
        "--logdir",
        str(evaluation),
        "--configs",
        *configs,
        "--task",
        manifest["task"],
        "--seed",
        str(args.eval_seed),
        "--agent.num_agents",
        str(manifest["num_agents"]),
        *_model_arguments(manifest),
        "--script",
        "eval_only",
        "--run.from_checkpoint",
        str(checkpoint),
        "--run.eval_eps",
        str(args.episodes),
        "--run.eval_policy_mode",
        "eval" if args.policy_mode == "deterministic" else "eval_sample",
        "--run.envs",
        str(min(args.envs, args.episodes)),
        "--jax.platform",
        platform,
        "--jax.precompile",
        "False",
        "--logger.outputs",
        *outputs,
    ]
    if manifest.get("visual_encoder") == "vjepa":
        environment = (
            "dmc" if str(manifest["task"]).startswith("dmc_") else "meltingpot"
        )
        command.extend([f"--env.{environment}.size", "224", "224"])
    evaluation.parent.mkdir(parents=True, exist_ok=True)
    evaluation.with_suffix(".launch.json").write_text(
        json.dumps(
            {
                "command": command,
                "episodes": args.episodes,
                "eval_seed": args.eval_seed,
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

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "glfw" if host_platform.system() == "Darwin" else "egl")
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(manifest["infrastructure_root"])),
            str(repository_root() / "src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["PYTHONUNBUFFERED"] = "1"
    if args.wandb_project:
        env["WANDB_PROJECT"] = args.wandb_project
        env["WANDB_NAME"] = f"{experiment.name}_fixed_eval"
    if args.wandb_entity:
        env["WANDB_ENTITY"] = args.wandb_entity
    return subprocess.run(
        command, cwd=repository_root(), env=env, check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
