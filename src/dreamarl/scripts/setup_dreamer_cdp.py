"""Create the isolated environment for official Dreamer-CDP."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

from dreamarl.baselines.dreamer_cdp.config import default_upstream_root
from dreamarl.baselines.dreamer_cdp.environment import prepare_environment
from dreamarl.baselines.dreamerv3.config import repository_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv-dir", type=Path, default=repository_root() / ".venv-dreamer-cdp"
    )
    parser.add_argument("--upstream-root", type=Path, default=default_upstream_root())
    parser.add_argument(
        "--accelerator",
        choices=("cpu", "cuda12"),
        default="cpu" if platform.system() == "Darwin" else "cuda12",
    )
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args(argv)
    python = prepare_environment(
        venv_dir=args.venv_dir,
        upstream_root=args.upstream_root,
        accelerator=args.accelerator,
        recreate=args.recreate,
    )
    print(f"Dreamer-CDP environment ready: {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
