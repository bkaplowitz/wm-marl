"""Compare a frozen DreaMARL checkpoint on fixed complete replay windows."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import elements
import numpy as np
import ruamel.yaml as yaml

from world_marl.dreamarl.main import make_agent


RAW_KEYS = (
    "image",
    "reward",
    "is_first",
    "is_last",
    "is_terminal",
    "action",
    "stepid",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", default="cuda")
    return parser


def _load_config(run_dir: Path, platform: str, report_length: int) -> elements.Config:
    data = yaml.YAML(typ="safe").load((run_dir / "config.yaml").read_text())
    # The discarded unified-memory screen temporarily added this resolved key.
    # Its only maintained value, ``residual``, is the current sidecar behavior.
    data["agent"]["dyn"]["jepa_transformer"].pop("memory_mode", None)
    data["agent"]["report_video"] = False
    config = elements.Config(data)
    return config.update(
        logdir=str(run_dir),
        batch_size=1,
        replay_context=0,
        report_length=report_length,
        jax={
            **dict(config.jax),
            "platform": platform,
            "prealloc": False,
            "profiler": False,
            "expect_devices": 0,
        },
    )


def _checkpoint_path(run_dir: Path) -> Path:
    root = run_dir / "ckpt"
    latest = (root / "latest").read_text().strip()
    checkpoint = root / latest
    if not (checkpoint / "done").exists():
        raise FileNotFoundError(f"incomplete checkpoint: {checkpoint}")
    return checkpoint


def _load_dataset(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(RAW_KEYS) - set(archive.files)
        if missing:
            raise ValueError(f"dataset is missing keys: {sorted(missing)}")
        arrays = {key: archive[key] for key in RAW_KEYS}
    leading = {key: value.shape[:2] for key, value in arrays.items()}
    if len(set(leading.values())) != 1:
        raise ValueError(f"unaligned trajectory axes: {leading}")
    manifest_path = path.with_suffix(path.suffix + ".json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return arrays, manifest


def _scalar_metrics(metrics) -> dict[str, float]:
    result = {}
    for key, value in metrics.items():
        array = np.asarray(value)
        if key.startswith("world_model/") and array.ndim == 0:
            result[key] = float(array)
    return result


def _aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted(set.intersection(*(set(row) for row in rows)))
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in keys
    }


def _complete_batch(agent, batch):
    length = next(iter(batch.values())).shape[1]
    for key, space in agent.spaces.items():
        if key not in batch:
            batch[key] = np.zeros((1, length, *space.shape), space.dtype)
    return batch


def run(args: argparse.Namespace) -> dict[str, object]:
    run_dir = args.run_dir.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    arrays, manifest = _load_dataset(dataset_path)
    report_length = next(iter(arrays.values())).shape[1]
    config = _load_config(run_dir, args.platform, report_length)
    agent = make_agent(config)
    checkpoint = _checkpoint_path(run_dir)
    elements.checkpoint.load(checkpoint, {"agent": agent.load})
    with agent.n_batches.lock:
        agent.n_batches.value = 0

    rows = []
    batches = [
        _complete_batch(
            agent,
            {key: value[index : index + 1] for key, value in arrays.items()},
        )
        for index in range(next(iter(arrays.values())).shape[0])
    ]
    stream = iter(agent.stream(itertools.cycle(batches)))
    for _ in range(len(batches)):
        batch = next(stream)
        carry = agent.init_report(1)
        _, metrics = agent.report(carry, batch)
        rows.append(_scalar_metrics(metrics))

    output = {
        "contract": "fixed_complete_replay_windows_v1",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_path),
        "dataset_manifest": manifest,
        "trajectories": len(rows),
        "metrics": _aggregate(rows),
        "per_trajectory": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run(args)
    print(json.dumps(output["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
