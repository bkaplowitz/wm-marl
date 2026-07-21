"""Create the isolated environment for the official DreamerV3 baseline."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

from world_marl.baselines.dreamerv3.config import (
    default_upstream_root,
    repository_root,
)
from world_marl.baselines.dreamerv3.environment import prepare_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=repository_root() / ".venv-dreamerv3",
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=default_upstream_root(),
    )
    parser.add_argument(
        "--accelerator",
        choices=("cpu", "cuda12"),
        default="cpu" if platform.system() == "Darwin" else "cuda12",
    )
    parser.add_argument("--recreate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    python = prepare_environment(
        venv_dir=args.venv_dir,
        upstream_root=args.upstream_root,
        accelerator=args.accelerator,
        recreate=args.recreate,
    )
    print(f"DreamerV3 environment ready: {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
