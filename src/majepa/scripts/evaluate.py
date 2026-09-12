"""Evaluate the final checkpoint with its saved training configuration."""

import argparse
from pathlib import Path

import elements

from majepa.main import run


def evaluation_config(
    run_dir, *, episodes=100, envs=4, output_dir=None, seed_offset=100000
):
    """Reuse exact model sizes; never reconstruct a checkpoint from today's defaults."""
    run_dir = Path(run_dir).expanduser().resolve()
    if episodes < 1 or envs < 1 or seed_offset < 0:
        raise ValueError(
            "Episodes/envs must be positive; seed offset must be nonnegative"
        )
    checkpoint_root = run_dir / "ckpt"
    checkpoint = checkpoint_root / (checkpoint_root / "latest").read_text().strip()
    if not (checkpoint / "done").is_file():
        raise FileNotFoundError(f"Incomplete checkpoint: {checkpoint}")
    output_dir = Path(output_dir or run_dir / f"final{episodes}").expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output already exists: {output_dir}")
    config = elements.Config.load(run_dir / "config.yaml")
    return config.update(
        {
            "logdir": str(output_dir),
            "script": "eval_only",
            "run.from_checkpoint": str(checkpoint),
            "run.eval_eps": int(episodes),
            "run.envs": min(int(envs), int(episodes)),
            "run.eval_worker_offset": int(seed_offset),
            "run.eval_policy_mode": "eval",
            "run.curve_eval_interval": 0,
            "jax.precompile": False,
        }
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed-offset", type=int, default=100000)
    args = parser.parse_args(argv)
    run(evaluation_config(**vars(args)))


if __name__ == "__main__":
    main()
