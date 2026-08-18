"""Read-only architectural diagnostics for trained DreaMARL checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import elements


def utility_probe(make_agent, make_replay, make_stream, args):
    """Evaluate frozen B1 utility readouts on replay without training."""

    if not args.from_checkpoint:
        raise ValueError("utility_probe requires run.from_checkpoint")
    if args.probe_batches < 1:
        raise ValueError("utility_probe requires at least one probe batch")

    agent = make_agent()
    replay = make_replay()
    # Standalone diagnostics do not restore the replay through the training
    # checkpoint object, so populate the in-memory index from its chunk files.
    replay.load()
    if not len(replay):
        raise RuntimeError(f"no replay data found at {args.probe_source!r}")

    checkpoint_path = Path(str(args.from_checkpoint))
    if checkpoint_path.is_file():
        checkpoint_path = checkpoint_path.parent / checkpoint_path.read_text(
            encoding="utf-8"
        ).strip()
    checkpoint = elements.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(str(checkpoint_path), keys=["agent"])

    stream = iter(agent.stream(make_stream(replay, "report")))
    carry = agent.init_report(args.batch_size)
    aggregate = elements.Agg()
    for _ in range(int(args.probe_batches)):
        carry, metrics = agent.report(carry, next(stream))
        aggregate.add(
            {
                key: value
                for key, value in metrics.items()
                if "agent_jepa/probe/" in key
            }
        )
    result = {
        key: float(value)
        for key, value in aggregate.result().items()
    }
    result.update(
        checkpoint=str(checkpoint_path),
        replay=str(args.probe_source),
        batches=int(args.probe_batches),
    )
    logdir = elements.Path(args.logdir)
    logdir.mkdir()
    (logdir / "utility_probe.json").write(
        json.dumps(result, indent=2, sort_keys=True)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
