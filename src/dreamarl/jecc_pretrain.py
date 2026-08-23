"""Replay-only fitting for the JECC acceptance gate."""

from __future__ import annotations

import json
import re
import time
from functools import partial as bind

import elements
import jax
import numpy as np


def pretrain(make_agent, make_replay, make_stream, args):
    """Fit only JECC modules for an exact number of frozen replay updates."""

    if not args.from_checkpoint:
        raise ValueError("JECC pretraining requires a frozen B0 checkpoint")
    if re.match(str(args.from_checkpoint_regex), "opt/state"):
        raise ValueError(
            "JECC pretraining must exclude the source optimizer; use "
            "run.from_checkpoint_regex='^(?!opt/).*'"
        )
    updates = int(args.jecc_pretrain_updates)
    if updates < 1:
        raise ValueError("JECC pretraining updates must be positive")

    agent = make_agent()
    replay = make_replay()
    replay.load()
    if not len(replay):
        raise RuntimeError("JECC pretraining replay is empty")

    elements.checkpoint.load(
        args.from_checkpoint,
        {"agent": bind(agent.load, regex=args.from_checkpoint_regex)},
    )
    stream = iter(agent.stream(make_stream(replay, "train")))
    carry = agent.init_train(args.batch_size)
    started = time.monotonic()

    def prepare_batch():
        batch = next(stream)
        if "_environment_step" in agent.spaces:
            reference = batch["is_first"]
            batch["_environment_step"] = jax.device_put(
                np.zeros(reference.shape, np.int32), reference.sharding
            )
        return batch

    # The compiled agent's first call initializes parameters and does not run
    # an optimizer update. Prime that call before counting the fitting budget.
    carry, _, _ = agent.train(carry, prepare_batch())
    latest_metrics = {}
    for update in range(1, updates + 1):
        carry, _, latest_metrics = agent.train(carry, prepare_batch())
        if update == 1 or update % 100 == 0 or update == updates:
            print(f"JECC replay update {update}/{updates}")

    actual_updates = int(np.asarray(latest_metrics["jecc_opt/updates"]))
    if actual_updates != updates:
        raise RuntimeError(
            f"JECC optimizer completed {actual_updates} updates, expected {updates}"
        )

    logdir = elements.Path(args.logdir)
    checkpoint = elements.Checkpoint(logdir / "ckpt")
    checkpoint.agent = agent
    checkpoint.save()
    elapsed = time.monotonic() - started
    summary = {
        "source_checkpoint": str(args.from_checkpoint),
        "replay_items": int(len(replay)),
        "jecc_updates": updates,
        "elapsed_seconds": elapsed,
        "updates_per_second": updates / max(elapsed, 1e-8),
        "latest_metrics": {
            key: float(np.asarray(value))
            for key, value in latest_metrics.items()
            if np.asarray(value).ndim == 0
        },
    }
    (logdir / "jecc_pretrain_summary.json").write(
        json.dumps(summary, indent=2, sort_keys=True)
    )


__all__ = ["pretrain"]
