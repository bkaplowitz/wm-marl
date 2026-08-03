"""Extract fixed complete raw replay windows for checkpoint comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from world_marl.scripts.diagnose_dreamarl_rollouts import RAW_KEYS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=20)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> Path:
    if args.trajectories < 1 or args.length < 16:
        raise ValueError("trajectories must be positive and length must be at least 16")
    replay_dir = args.replay_dir.expanduser().resolve()
    files = sorted(replay_dir.glob("*.npz"))
    generator = np.random.default_rng(args.seed)
    generator.shuffle(files)
    windows = []
    provenance = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            missing = set(RAW_KEYS) - set(archive.files)
            if missing:
                raise ValueError(f"{path} is missing keys: {sorted(missing)}")
            first = np.asarray(archive["is_first"], bool)
            last = np.asarray(archive["is_last"], bool)
            starts = np.flatnonzero(first)
            generator.shuffle(starts)
            for start in starts:
                start = int(start)
                stop = start + args.length
                if stop > len(first) or last[start : stop - 1].any():
                    continue
                windows.append(
                    {key: np.asarray(archive[key][start:stop]) for key in RAW_KEYS}
                )
                provenance.append(
                    {
                        "source_chunk": path.name,
                        "source_chunk_sha256": _sha256(path),
                        "source_start": start,
                    }
                )
                break
        if len(windows) == args.trajectories:
            break
    if len(windows) != args.trajectories:
        raise ValueError(
            f"requested {args.trajectories} windows but found {len(windows)}"
        )

    arrays = {key: np.stack([window[key] for window in windows]) for key in RAW_KEYS}
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    manifest = {
        "contract": "fixed_complete_replay_windows_v1",
        "replay_dir": str(replay_dir),
        "trajectories": args.trajectories,
        "length": args.length,
        "seed": args.seed,
        "dataset_sha256": _sha256(output),
        "provenance": provenance,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    return 0 if run(_parser().parse_args(argv)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
