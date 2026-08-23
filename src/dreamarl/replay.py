"""Replay selectors for non-stationary multi-agent world-model training."""

import threading

import embodied
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
        self.sampled_ages = []
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.items)

    def __bool__(self):
        # Upstream Replay selects `selector or Uniform(...)` while the replay
        # is still empty, so the selector object itself must remain truthy.
        return True

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
                age = int(np.floor(np.log1p(-uniform * truncated) / np.log(self.decay)))
                age = min(age, count - 1)
            self.sampled_ages.append(age)
            return self.items[self.step - 1 - age]

    def pop_sampled_ages(self):
        with self.lock:
            ages = self.sampled_ages
            self.sampled_ages = []
            return ages

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


class RecentReplay(embodied.replay.Replay):
    """Single-stream replay with exponentially decayed sampling by item age."""

    def __init__(self, *, recency_decay=0.9998, seed=0, **kwargs):
        decay = float(recency_decay)
        if not 0.0 < decay <= 1.0:
            raise ValueError("recency_decay must be in (0, 1]")
        kwargs.pop("online", None)
        super().__init__(
            selector=ExponentialRecency(decay, seed=seed),
            online=False,
            seed=seed,
            **kwargs,
        )
        self.sampled_ages = []
        self.sampled_ages_lock = threading.Lock()

    def _sample(self, mode):
        sequence, is_online = super()._sample(mode)
        if mode == "train":
            # ExponentialRecency records the sampled age without changing the
            # replay API or the learner batch shape.
            with self.sampled_ages_lock:
                self.sampled_ages.extend(self.sampler.pop_sampled_ages())
        return sequence, is_online

    def stats(self):
        result = super().stats()
        with self.sampled_ages_lock:
            values = self.sampled_ages
            self.sampled_ages = []
        if values:
            array = np.asarray(values, np.float32)
            result["sample_age_mean"] = float(array.mean())
            result["sample_age_p50"] = float(np.percentile(array, 50))
            result["sample_age_p95"] = float(np.percentile(array, 95))
        return result


__all__ = ["ExponentialRecency", "RecentReplay"]
