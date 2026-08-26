"""Replay selectors for non-stationary multi-agent world-model training."""

import threading

import elements
import embodied
import numpy as np


class ExponentialRecency:
    """Sample sequence starts with probability proportional to decay**age."""

    def __init__(self, decay, seed=0, track_ages=True):
        self.decay = float(decay)
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("recency decay must be in (0, 1]")
        self.rng = np.random.default_rng(seed)
        self.step = 0
        self.steps = {}
        self.items = {}
        self.sampled_ages = []
        self.track_ages = bool(track_ages)
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
            if self.track_ages:
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


class DualViewReplay(embodied.replay.Replay):
    """One replay store exposed through separate world and behavior views.

    The world-model view samples recent sequence starts using exponential
    recency weighting. The behavior view samples the same physical items
    uniformly with an independent selector. Keeping the selectors here, next
    to the single item table, makes it impossible for the two views to drift
    onto different replay contents.
    """

    dual_view = True
    _WORLD_MODES = frozenset({"train", "train_world"})
    _BEHAVIOR_MODES = frozenset({"train_behavior", "report", "eval"})

    def __init__(
        self, *, optimized_length=None, recency_decay=0.9998, seed=0, **kwargs
    ):
        decay = float(recency_decay)
        if not 0.0 < decay <= 1.0:
            raise ValueError("recency_decay must be in (0, 1]")
        if kwargs.pop("online", False):
            raise ValueError("dual-view replay requires replay.online=False")
        replay_length = int(kwargs["length"])
        self.optimized_length = int(optimized_length or replay_length)
        if not 0 < self.optimized_length <= replay_length:
            raise ValueError(
                "optimized_length must be positive and no larger than replay length"
            )
        super().__init__(
            selector=ExponentialRecency(decay, seed=seed, track_ages=False),
            online=False,
            seed=seed,
            **kwargs,
        )
        # A separate RNG is important: a world sample must never determine the
        # corresponding behavior root, even when both streams advance in lock
        # step in the learner.
        self.behavior_sampler = embodied.selectors.Uniform(seed=int(seed) + 1)
        self._view_stats = {
            "world_samples": 0,
            "behavior_samples": 0,
            "world_ages": [],
            "behavior_ages": [],
        }
        self._view_stats_lock = threading.Lock()

    def _insert(self, chunkid, index):
        """Insert one item into both selectors without duplicating storage."""

        itemid = self.itemid
        super()._insert(chunkid, index)
        # The base replay owns the item table and recency selector. This second
        # selector stores only integer item identifiers.
        self.behavior_sampler[itemid] = ()

    def _remove(self):
        """Evict the same FIFO item from both selector views."""

        itemid = self.fifo[0]
        del self.behavior_sampler[itemid]
        super()._remove()

    def _sample(self, mode):
        if mode not in self._WORLD_MODES | self._BEHAVIOR_MODES:
            raise AssertionError(mode)

        is_world = mode in self._WORLD_MODES
        selector = self.sampler if is_world else self.behavior_sampler
        is_training = mode in {"train", "train_world", "train_behavior"}
        if is_training:
            # Preserve upstream replay-ratio accounting while exposing the two
            # view-specific counts separately below.
            self.metrics["samples"] += 1

        while True:
            try:
                with elements.timer.section("sample"):
                    itemid = selector()
                chunkid, index = self.items[itemid]
                sequence = self._getseq(chunkid, index, concat=False)
                if is_training:
                    age = self.itemid - 1 - int(itemid)
                    view = "world" if is_world else "behavior"
                    with self._view_stats_lock:
                        self._view_stats[f"{view}_samples"] += 1
                        self._view_stats[f"{view}_ages"].append(age)
                return sequence, False
            except KeyError:
                # Match upstream replay's tolerance for a selector racing an
                # eviction. Both selectors are corrected by the eviction path.
                continue

    def stats(self):
        result = super().stats()
        with self._view_stats_lock:
            values = self._view_stats
            self._view_stats = {
                "world_samples": 0,
                "behavior_samples": 0,
                "world_ages": [],
                "behavior_ages": [],
            }

        result["world_samples"] = values["world_samples"]
        result["behavior_samples"] = values["behavior_samples"]
        inserts = result["inserts"]
        for view in ("world", "behavior"):
            samples = values[f"{view}_samples"]
            result[f"{view}_replay_ratio"] = (
                self.optimized_length * samples / inserts if inserts else np.nan
            )
            result[f"{view}_read_ratio"] = (
                self.length * samples / inserts if inserts else np.nan
            )
        result["optimized_replay_ratio"] = (
            result["world_replay_ratio"] + result["behavior_replay_ratio"]
        )
        for view in ("world", "behavior"):
            ages = values[f"{view}_ages"]
            if ages:
                array = np.asarray(ages, np.float32)
                result[f"{view}_sample_age_mean"] = float(array.mean())
                result[f"{view}_sample_age_p50"] = float(np.percentile(array, 50))
                result[f"{view}_sample_age_p95"] = float(np.percentile(array, 95))
        return result


__all__ = ["DualViewReplay", "ExponentialRecency", "RecentReplay"]
