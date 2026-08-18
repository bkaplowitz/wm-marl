"""Replay selectors for non-stationary multi-agent world-model training."""

import threading

import elements
import embodied
from embodied.core import limiters, selectors
import numpy as np


class ExponentialRecency:
    """Sample sequence starts with probability proportional to decay**age."""

    def __init__(self, decay, seed=0):
        self.decay = float(decay)
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("recency decay must be in (0, 1]")
        self.rng = np.random.default_rng(seed)
        self.step = 0
        self.steps = {}
        self.items = {}
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.items)

    def __call__(self):
        with self.lock:
            count = len(self.items)
            if not count:
                raise IndexError("cannot sample an empty recency selector")
            if self.decay == 1.0:
                age = int(self.rng.integers(0, count))
            else:
                uniform = self.rng.random()
                truncated = 1.0 - self.decay**count
                age = int(
                    np.floor(
                        np.log1p(-uniform * truncated) / np.log(self.decay)
                    )
                )
                age = min(age, count - 1)
            return self.items[self.step - 1 - age]

    def __setitem__(self, key, stepids):
        del stepids
        with self.lock:
            self.steps[key] = self.step
            self.items[self.step] = key
            self.step += 1

    def __delitem__(self, key):
        with self.lock:
            step = self.steps.pop(key)
            del self.items[step]


class RecentWorldUniformBehaviorReplay(embodied.replay.Replay):
    """Share storage while exposing distinct world and behavior samplers."""

    def __init__(self, *, recency_decay=0.9998, seed=0, **kwargs):
        kwargs.pop("online", None)
        decay = float(recency_decay)
        if not 0.0 < decay <= 1.0:
            raise ValueError("recency_decay must be in (0, 1]")
        super().__init__(
            selector=ExponentialRecency(decay, seed=seed),
            online=False,
            seed=seed,
            **kwargs,
        )
        self.behavior_sampler = selectors.Uniform(seed=seed + 1)
        self._sample_age = {"world": [], "behavior": []}
        self._sample_age_lock = threading.Lock()

    def _insert(self, chunkid, index):
        itemid = self.itemid
        super()._insert(chunkid, index)
        stepids = self._getseq(chunkid, index, ["stepid"])["stepid"]
        self.behavior_sampler[itemid] = stepids

    def _remove(self):
        itemid = self.fifo[0]
        del self.behavior_sampler[itemid]
        super()._remove()

    @elements.timer.section("replay_sample")
    def sample(self, batch, mode="train"):
        if mode == "train_world":
            selector, kind, train = self.sampler, "world", True
        elif mode == "train_behavior":
            selector, kind, train = self.behavior_sampler, "behavior", True
        elif mode in {"report", "eval"}:
            selector, kind, train = self.behavior_sampler, "behavior", False
        else:
            raise ValueError(f"unsupported dual replay mode: {mode!r}")
        message = f"Replay buffer {self.name} is empty"
        limiters.wait(lambda: len(selector), message)
        samples = [self._sample_from(selector, kind, train) for _ in range(batch)]
        sequences, is_online = zip(*samples)
        data = self._assemble_batch(sequences, 0, self.length)
        return self._annotate_batch(data, is_online, True)

    def _sample_from(self, selector, kind, train):
        if train:
            self.metrics["samples"] += 1
        while True:
            try:
                with elements.timer.section("sample"):
                    itemid = selector()
                chunkid, index = self.items[itemid]
                sequence = self._getseq(chunkid, index, concat=False)
                age = max(self.itemid - 1 - itemid, 0)
                with self._sample_age_lock:
                    self._sample_age[kind].append(age)
                return sequence, False
            except KeyError:
                continue

    def stats(self):
        result = super().stats()
        with self._sample_age_lock:
            ages = self._sample_age
            self._sample_age = {"world": [], "behavior": []}
        for kind, values in ages.items():
            if values:
                array = np.asarray(values, np.float32)
                result[f"{kind}_sample_age_mean"] = float(array.mean())
                result[f"{kind}_sample_age_p50"] = float(np.percentile(array, 50))
                result[f"{kind}_sample_age_p95"] = float(np.percentile(array, 95))
        return result


__all__ = ["RecentWorldUniformBehaviorReplay"]
