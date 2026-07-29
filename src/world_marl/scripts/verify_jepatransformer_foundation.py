"""Verify pinned sources and the fixed visual-DMC research protocol."""

from __future__ import annotations

import json

from world_marl.jepa_transformer.foundation import verify_foundation


def main() -> int:
    print(json.dumps(verify_foundation(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
