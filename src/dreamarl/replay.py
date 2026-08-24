"""Replay strategies for non-stationary multi-agent world-model training."""

from collections import defaultdict, deque
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


class EliteRecentReplay:
    """Mix recent experience with complete high-return episodes.

    The recent stream remains authoritative and receives every environment step.
    A second bounded stream receives complete episodes whose team return clears a
    rolling, monotonically non-decreasing threshold. Training batches contain an
    exact fixed fraction from that elite stream once it has enough data to sample.
    """

    def __init__(
        self,
        *,
        length,
        capacity,
        directory=None,
        chunksize=1024,
        online=False,
        recency_decay=0.9998,
        elite_fraction=0.1875,
        elite_capacity=12_500,
        elite_quantile=0.75,
        elite_min_episodes=32,
        elite_return_window=256,
        seed=0,
        **kwargs,
    ):
        del online
        if kwargs:
            raise TypeError(f"unsupported replay arguments: {sorted(kwargs)}")
        if not 0.0 < elite_fraction < 1.0:
            raise ValueError("elite_fraction must be in (0, 1)")
        if elite_capacity < length:
            raise ValueError("elite_capacity must be at least one replay sequence")
        if not 0.0 < elite_quantile < 1.0:
            raise ValueError("elite_quantile must be in (0, 1)")
        if elite_min_episodes < 1 or elite_return_window < elite_min_episodes:
            raise ValueError("elite return window must cover the minimum episodes")

        recent_directory = directory / "recent" if directory else None
        elite_directory = directory / "elite" if directory else None
        common = dict(length=length, chunksize=chunksize)
        self.recent = RecentReplay(
            **common,
            capacity=capacity,
            directory=recent_directory,
            recency_decay=recency_decay,
            seed=seed,
        )
        self.elite = embodied.replay.Replay(
            **common,
            capacity=int(elite_capacity),
            directory=elite_directory,
            online=False,
            seed=seed + 1,
            name="elite",
        )
        self.elite_fraction = float(elite_fraction)
        self.elite_quantile = float(elite_quantile)
        self.elite_min_episodes = int(elite_min_episodes)
        self.return_history = deque(maxlen=int(elite_return_window))
        self.episode_buffers = defaultdict(list)
        self.threshold = -np.inf
        self.completed_episodes = 0
        self.elite_episodes = 0
        self.recent_samples = 0
        self.elite_samples = 0
        self.rng = np.random.default_rng(seed + 2)
        self.lock = threading.RLock()

    def __len__(self):
        return len(self.recent)

    @staticmethod
    def _flag(step, key):
        return bool(np.asarray(step.get(key, False)).any())

    @staticmethod
    def _copy_step(step):
        return {
            key: np.asarray(value).copy()
            for key, value in step.items()
            if not key.startswith("log/")
        }

    @staticmethod
    def _episode_return(episode):
        rewards = [np.asarray(step["reward"], np.float64).mean() for step in episode]
        return float(np.sum(rewards, dtype=np.float64))

    def add(self, step, worker=0):
        self.recent.add(step, worker)
        copied = self._copy_step(step)
        with self.lock:
            buffer = self.episode_buffers[worker]
            if self._flag(copied, "is_first") and buffer:
                buffer.clear()
            buffer.append(copied)
            if not self._flag(copied, "is_last"):
                return

            score = self._episode_return(buffer)
            qualifies = (
                self.completed_episodes < self.elite_min_episodes
                or score >= self.threshold
            )
            self.completed_episodes += 1
            self.return_history.append(score)
            if len(self.return_history) >= self.elite_min_episodes:
                candidate = float(
                    np.quantile(np.asarray(self.return_history), self.elite_quantile)
                )
                self.threshold = max(self.threshold, candidate)
            if qualifies:
                for episode_step in buffer:
                    self.elite.add(episode_step, worker)
                self.elite_episodes += 1
            buffer.clear()

    def sample(self, batch, mode="train"):
        if mode != "train" or not len(self.elite):
            data = self.recent.sample(batch, mode)
            if mode == "train":
                with self.lock:
                    self.recent_samples += batch
            return data

        elite_batch = min(batch - 1, max(1, round(batch * self.elite_fraction)))
        recent_batch = batch - elite_batch
        recent = self.recent.sample(recent_batch, mode)
        elite = self.elite.sample(elite_batch, mode)
        if recent.keys() != elite.keys():
            raise ValueError("recent and elite replay schemas differ")
        data = {
            key: np.concatenate([recent[key], elite[key]], axis=0) for key in recent
        }
        with self.lock:
            order = self.rng.permutation(batch)
            self.recent_samples += recent_batch
            self.elite_samples += elite_batch
        return {key: value[order] for key, value in data.items()}

    def update(self, data):
        self.recent.update(dict(data))
        self.elite.update(dict(data))

    def save(self):
        self.recent.save()
        self.elite.save()
        with self.lock:
            return {
                "return_history": list(self.return_history),
                "episode_buffers": dict(self.episode_buffers),
                "threshold": self.threshold,
                "completed_episodes": self.completed_episodes,
                "elite_episodes": self.elite_episodes,
                "recent_samples": self.recent_samples,
                "elite_samples": self.elite_samples,
                "rng_state": self.rng.bit_generator.state,
            }

    def load(self, data=None):
        self.recent.load()
        self.elite.load()
        if not data:
            return
        with self.lock:
            self.return_history.clear()
            self.return_history.extend(data.get("return_history", ()))
            self.episode_buffers.clear()
            self.episode_buffers.update(data.get("episode_buffers", {}))
            self.threshold = float(data.get("threshold", -np.inf))
            self.completed_episodes = int(data.get("completed_episodes", 0))
            self.elite_episodes = int(data.get("elite_episodes", 0))
            self.recent_samples = int(data.get("recent_samples", 0))
            self.elite_samples = int(data.get("elite_samples", 0))
            if "rng_state" in data:
                self.rng.bit_generator.state = data["rng_state"]

    def stats(self):
        result = self.recent.stats()
        elite = self.elite.stats()
        with self.lock:
            total_samples = self.recent_samples + self.elite_samples
            result.update(
                {
                    "elite_items": elite["items"],
                    "elite_ram_gb": elite["ram_gb"],
                    "elite_sample_fraction": (
                        self.elite_samples / total_samples if total_samples else 0.0
                    ),
                    "elite_return_threshold": (
                        self.threshold if np.isfinite(self.threshold) else 0.0
                    ),
                    "elite_episode_fraction": (
                        self.elite_episodes / self.completed_episodes
                        if self.completed_episodes
                        else 0.0
                    ),
                }
            )
            self.recent_samples = 0
            self.elite_samples = 0
        return result


__all__ = ["EliteRecentReplay", "ExponentialRecency", "RecentReplay"]
