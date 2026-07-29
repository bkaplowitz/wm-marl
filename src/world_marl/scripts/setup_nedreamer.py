"""Create the isolated environment for the official NE-Dreamer baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from world_marl.baselines.nedreamer.config import default_upstream_root, repository_root
from world_marl.baselines.nedreamer.environment import prepare_environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv-dir", type=Path, default=repository_root() / ".venv-nedreamer"
    )
    parser.add_argument("--upstream-root", type=Path, default=default_upstream_root())
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args(argv)
    python = prepare_environment(
        venv_dir=args.venv_dir,
        upstream_root=args.upstream_root,
        recreate=args.recreate,
    )
    print(f"NE-Dreamer environment ready: {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
