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
                "log/battle_won": np.float32(worker == 0),
                "log/timeout": np.float32(worker == 1),
                "log/legacy_reward": np.float32(worker + 1.0),
                "log/corrected_reward": np.float32(worker + 2.0),
                "log/enemy_damage": np.float32(7.0),
                "log/enemy_health_damage": np.float32(5.0),
                "log/enemy_shield_damage": np.float32(2.0),
                "log/enemy_shield_regen": np.float32(1.0),
                "log/enemy_deaths_step": np.float32(1.0),
                "log/ally_deaths_step": np.float32(0.0),
                "log/dead_allies": np.float32(0.0),
                "log/dead_enemies": np.float32(1.0),
                "log/ally_survivors": np.float32(2.0),
                "log/enemy_survivors": np.float32(1.0),
                "log/action_noop_count": np.float32(0.0),
                "log/action_stop_count": np.float32(1.0),
                "log/action_move_count": np.float32(2.0),
                "log/action_attack_count": np.float32(1.0),
                "log/action_target_switch_count": np.float32(worker),
                "log/attack_target_0_count": np.float32(worker == 0),
                "log/attack_target_1_count": np.float32(worker == 1),
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
    assert summary["win_rate"] == 0.5
    assert summary["wins"] == 2
    assert summary["timeout_rate"] == 0.5
    assert summary["legacy_return_mean"] == 1.5
    assert summary["corrected_return_mean"] == 2.5
    assert summary["legacy_corrected_gap_mean"] == -1.0
    assert summary["enemy_damage_mean"] == 7.0
    assert summary["enemy_shield_regen_mean"] == 1.0
    assert summary["enemy_survivors_mean"] == 1.0
    assert summary["action_attack_fraction"] == 0.25
    assert summary["action_move_fraction"] == 0.5
    assert summary["attack_target_0_fraction"] == 0.5
    assert summary["attack_target_1_fraction"] == 0.5
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
