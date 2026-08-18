from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from dreamarl import evaluation


class _Counter:
    def __init__(self, value: int):
        self.value = value
        self.lock = threading.Lock()


class _Agent:
    def __init__(self):
        self.n_actions = _Counter(37)
        self.policy_lock = threading.Lock()
        self.pending_sync = {"params": object()}

    def init_policy(self, batch_size):
        return [None] * batch_size

    def policy(self, *args, mode):
        del args
        assert mode in {"eval", "eval_sample"}
        self.last_mode = mode
        with self.n_actions.lock:
            self.n_actions.value += 1
        return None


class _Driver:
    def __init__(self, functions, parallel):
        del parallel
        self.workers = len(functions)
        self.callbacks = []

    def on_step(self, callback):
        self.callbacks.append(callback)

    def reset(self, init_policy):
        init_policy(self.workers)

    def __call__(self, policy, steps):
        del steps
        policy(None, None)
        for worker in range(self.workers):
            transition = {
                "is_first": True,
                "is_last": True,
                "reward": np.array([worker + 1.0, worker + 3.0], np.float32),
            }
            for callback in self.callbacks:
                callback(transition, worker)

    def close(self):
        pass


def test_inline_evaluation_preserves_training_policy_rng(monkeypatch) -> None:
    monkeypatch.setattr(evaluation.embodied, "Driver", _Driver)
    agent = _Agent()

    summary = evaluation.evaluate_current_policy(
        agent,
        lambda index: SimpleNamespace(index=index),
        episodes=4,
        envs=2,
        debug=True,
    )

    assert summary["episodes"] == 4
    assert summary["return_mean"] == 2.5
    assert summary["per_agent_return_mean"] == 2.5
    assert summary["team_return_mean"] == 5.0
    assert summary["team_returns"] == [4.0, 6.0, 4.0, 6.0]
    assert agent.last_mode == "eval"
    assert agent.n_actions.value == 37
    assert agent.pending_sync is not None


def test_inline_evaluation_supports_dreamer_sampled_policy(monkeypatch) -> None:
    monkeypatch.setattr(evaluation.embodied, "Driver", _Driver)
    agent = _Agent()

    evaluation.evaluate_current_policy(
        agent,
        lambda index: SimpleNamespace(index=index),
        episodes=1,
        envs=1,
        debug=True,
        policy_mode="eval_sample",
    )

    assert agent.last_mode == "eval_sample"
    assert agent.n_actions.value == 37
    assert agent.pending_sync is not None
