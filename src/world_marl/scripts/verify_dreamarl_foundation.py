"""Run the DreaMARL environment and replay contract gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_marl.dreamarl.foundation import verify_foundation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--time-steps", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-cycles", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = verify_foundation(
        output=args.output,
        time_steps=args.time_steps,
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
