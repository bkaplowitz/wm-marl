"""First-party stream adapter with a separate diagnostic seed sequence."""

import elements
import embodied
import hashlib
import json
from embodied.jax import internal
import numpy as np


def _prepare_batch(agent, data, counter, offset=0):
    for key, value in data.items():
        if np.issubdtype(value.dtype, np.floating) and np.isnan(value).any():
            raise ValueError(f"NaN in replay field {key}")
    data = internal.device_put(data, agent.train_sharded)
    with counter.lock:
        index = counter.value
        counter.value += 1
    seed = agent._seeds(offset + index, agent.train_mirrored)
    return {**data, "seed": seed}


class ReplaySnapshotStream:
    """One batch ahead, with every replay read on the collector's thread.

    prime() captures the first pair as soon as a full replay window exists,
    just as eager Prefetch can. take() refills BEFORE the current learner call
    and its replay write-back. No collector/write-back callback runs between
    the two views. This fixes the read ordering without delaying startup reads
    until training begins, or removing the existing one-batch lag.

    This requires local replay exclusively mutated by the serial train loop.
    Replay.sample owns its returned arrays; device transfer can remain async.
    """

    def __init__(self, agent, source, on_snapshot=None, *, behavior_source=None,
                 startup_behavior_min_starts=1):
        self.agent = agent
        self.source = iter(source)
        self.behavior_source = iter(behavior_source) if behavior_source is not None else None
        self.startup_behavior_min_starts = int(startup_behavior_min_starts)
        if self.startup_behavior_min_starts < 1:
            raise ValueError("Startup behavior eligibility must be positive")
        if self.startup_behavior_min_starts != 1 and self.behavior_source is None:
            raise ValueError("Staggered startup requires separate replay views")
        self.startup_world = None
        self.startup_complete = False
        self.pending = None
        self.sampled_at = None
        self.on_snapshot = on_snapshot

    def prime(self, environment_step, eligible_starts=None):
        if self.pending is not None:
            return False
        if self.behavior_source is not None and not self.startup_complete:
            if eligible_starts is None:
                raise ValueError("First staggered read requires replay eligibility")
            if self.startup_world is None:
                self.startup_world = next(self.source)
            if eligible_starts < self.startup_behavior_min_starts:
                return False
            data, self.startup_world = self.startup_world, None
        else:
            data = next(self.source)
        if self.behavior_source is not None:
            other = next(self.behavior_source)
            if data['is_first'].shape != other['is_first'].shape:
                raise ValueError("Replay view shapes differ")
            data = {**data, **{f'_behavior_replay/{key}': value
                              for key, value in other.items()}}
        self.startup_complete = True
        index = int(self.agent.n_batches)
        if self.on_snapshot is not None:
            self.on_snapshot(index, int(environment_step), data)
        self.pending = _prepare_batch(self.agent, data, self.agent.n_batches)
        self.sampled_at = int(environment_step)
        return True

    def take(self, environment_step):
        if self.pending is None:
            raise RuntimeError("Prime replay after its first eligible window")
        batch, sampled_at = self.pending, self.sampled_at
        self.pending = None
        self.prime(environment_step)
        return batch, sampled_at


def synchronous_report_stream(agent, source):
    """Read report batches on demand, with no training RNG consumption."""
    counter = elements.Counter()
    for data in source:
        yield _prepare_batch(agent, data, counter, 1 << 32)


def replay_batch_fingerprints(data):
    """Hash actual batch contents, excluding nondeterministic storage UUIDs."""
    result = {}
    for view, prefix in (("world", ""), ("behavior", "_behavior_replay/")):
        raw, history = hashlib.sha256(), hashlib.sha256()
        count = 0
        for name in sorted(data):
            if name.startswith("_behavior_replay/") != bool(prefix):
                continue
            key = name[len(prefix):]
            if key in {"stepid", "seed"}:
                continue
            value = np.ascontiguousarray(data[name])
            hasher = history if key.startswith("dyn/") else raw
            hasher.update(json.dumps([key, str(value.dtype), value.shape]).encode())
            hasher.update(memoryview(value).cast("B"))
            count += 1
        if count:
            result[view] = {"data_sha256": raw.hexdigest(),
                            "history_sha256": history.hexdigest()}
    return result


class ReplaySnapshotTrace:
    """Optional bounded host-side audit of initial sampled batch contents."""

    def __init__(self, path, limit):
        self.path = path
        self.limit = int(limit)

    def __call__(self, index, environment_step, data):
        if index >= self.limit:
            return
        record = {"batch_index": index, "sampled_at": environment_step,
                  "views": replay_batch_fingerprints(data)}
        # Hash before JAX transport: instrumentation does not consume RNG or
        # read device buffers while the learner owns/donates them.
        with self.path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def isolated_report_stream(agent, source):
    """Match the pinned JAX stream transport without consuming train seeds.

    The report counter is intentionally not part of the learner checkpoint:
    diagnostics cannot affect training, including after a checkpoint reload.
    The offset gives diagnostics a disjoint practical counter range.
    """
    counter = elements.Counter()

    def prepare(data):
        for key, value in data.items():
            if np.issubdtype(value.dtype, np.floating) and np.isnan(value).any():
                raise ValueError(f"NaN in report field {key}")
        data = internal.device_put(data, agent.train_sharded)
        with counter.lock:
            index = counter.value
            counter.value += 1
        seed = agent._seeds((1 << 32) + index, agent.train_mirrored)
        return {**data, "seed": seed}

    return embodied.streams.Prefetch(source, prepare)
