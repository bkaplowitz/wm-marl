from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "model",
    "status",
    "final_loss",
    "learning_gate_passed",
    "summary_path",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        type=Path,
        required=True,
        help="Path to a summary.json artifact. Repeat once per arm.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/visual_wm_compare"))
    return parser.parse_args(argv)


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return {
        "model": payload.get("model", path.parent.name),
        "status": payload.get("status", "unknown"),
        "final_loss": payload.get("final_loss"),
        "learning_gate_passed": payload.get("learning_gate_passed"),
        "summary_path": str(path),
    }


def write_comparison(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (out_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = [load_summary(path) for path in args.summary]
    write_comparison(args.out_dir, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
