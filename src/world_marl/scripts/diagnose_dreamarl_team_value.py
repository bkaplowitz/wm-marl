"""Compare local and pooled team reward/value prediction on frozen replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_marl.dreamarl.team_value_probe import (
    ProbeConfig,
    load_replay_dataset,
    paired_episode_bootstrap,
    train_probe,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--predictor-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--states-per-episode", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=332 / 333)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)

    dataset = load_replay_dataset(
        args.replay_dir,
        states_per_episode=args.states_per_episode,
        gamma=args.gamma,
        seed=args.dataset_seed,
    )
    config = ProbeConfig(
        hidden=args.hidden,
        steps=args.steps,
        batch_size=args.batch_size,
    )
    comparisons = []
    for seed in args.predictor_seeds:
        local = train_probe(dataset, "local", config, seed=seed)
        joint = train_probe(dataset, "joint", config, seed=seed)
        comparisons.append(
            {
                "seed": seed,
                "local": {key: value for key, value in local.items() if not key.startswith("test_")},
                "joint": {key: value for key, value in joint.items() if not key.startswith("test_")},
                "paired_test": paired_episode_bootstrap(
                    dataset, local, joint, seed=seed + 20_000
                ),
            }
        )
    result = {
        "dataset": dataset.manifest,
        "config": {
            "hidden": config.hidden,
            "learning_rate": config.learning_rate,
            "steps": config.steps,
            "batch_size": config.batch_size,
            "grad_clip": config.grad_clip,
            "predictor_seeds": args.predictor_seeds,
        },
        "comparison": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
