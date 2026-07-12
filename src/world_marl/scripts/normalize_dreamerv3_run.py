"""Regenerate normalized artifacts for an existing DreamerV3 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_marl.baselines.dreamerv3.artifacts import normalize_training_artifacts


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("experiment_dir", type=Path)
  parser.add_argument("--bin-size", type=int, default=10_000)
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  launch = json.loads(
    (args.experiment_dir / "launch.json").read_text(encoding="utf-8")
  )
  summary = normalize_training_artifacts(
    args.experiment_dir,
    upstream_root=launch["upstream_root"],
    task=launch["task"],
    seed=int(launch["seed"]),
    train_steps_budget=int(launch["train_env_steps_budget"]),
    bin_size=args.bin_size,
  )
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
