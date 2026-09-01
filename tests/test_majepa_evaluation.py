from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from majepa import evaluation


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


@pytest.mark.parametrize(("episodes", "envs"), ((0, 1), (1, 0)))
def test_inline_evaluation_rejects_invalid_protocol(episodes, envs) -> None:
    with pytest.raises(ValueError, match="positive episode and environment counts"):
        evaluation.evaluate_current_policy(
            _Agent(),
            lambda index: SimpleNamespace(index=index),
            episodes=episodes,
            envs=envs,
            debug=True,
        )


class _Checkpoint:
    loaded = []

    def load(self, path, keys):
        self.loaded.append((path, keys))


class _RecordingLogger:
    def __init__(self, logdir):
        self.logdir = logdir
        self.events = []

    def add(self, metrics, prefix=None):
        self.events.append(("add", prefix, dict(metrics)))

    def write(self):
        self.events.append(("write",))

    def close(self):
        self.events.append(
            (
                "close",
                (self.logdir / "evaluation_episodes.jsonl").is_file(),
                (self.logdir / "evaluation_summary.json").is_file(),
            )
        )


def test_eval_only_uses_heldout_workers_and_records_exact_quota(
    monkeypatch, tmp_path
) -> None:
    worker_indices = []

    class EvalOnlyDriver(_Driver):
        def __init__(self, functions, parallel):
            environments = [function() for function in functions]
            worker_indices.extend(environment.index for environment in environments)
            super().__init__(functions, parallel)

    output = tmp_path / "heldout_eval"
    logger = _RecordingLogger(output)
    _Checkpoint.loaded.clear()
    monkeypatch.setattr(evaluation.embodied, "Driver", EvalOnlyDriver)
    monkeypatch.setattr(evaluation.elements, "Checkpoint", _Checkpoint)

    evaluation.eval_only(
        _Agent,
        lambda index: SimpleNamespace(index=index),
        lambda: logger,
        SimpleNamespace(
            from_checkpoint="/checkpoints/step50000",
            eval_eps=128,
            envs=4,
            eval_worker_offset=100_000,
            eval_policy_mode="eval",
            debug=True,
            logdir=str(output),
        ),
    )

    assert worker_indices == [100_000, 100_001, 100_002, 100_003]
    assert _Checkpoint.loaded == [
        ("/checkpoints/step50000", ["agent"]),
    ]
    records = [
        json.loads(line)
        for line in (output / "evaluation_episodes.jsonl").read_text().splitlines()
    ]
    assert len(records) == 128
    assert [record["episode"] for record in records] == list(range(128))
    assert {record["metadata"]["worker_index"] for record in records} == {
        100_000,
        100_001,
        100_002,
        100_003,
    }
    for worker in range(4):
        worker_records = [
            record for record in records if record["metadata"]["worker"] == worker
        ]
        assert len(worker_records) == 32
        assert [
            record["metadata"]["worker_episode"] for record in worker_records
        ] == list(range(32))

    summary = json.loads((output / "evaluation_summary.json").read_text())
    assert summary["episodes"] == 128
    assert len(summary["returns"]) == 128
    assert len(summary["team_returns"]) == 128
    assert len(summary["per_agent_returns"]) == 128
    assert summary["wins"] == 32
    assert summary["evaluation_protocol"] == {
        "episodes": 128,
        "envs": 4,
        "worker_indices": [100_000, 100_001, 100_002, 100_003],
        "worker_offset": 100_000,
        "policy_mode": "eval",
    }

    aggregate = next(
        index
        for index, event in enumerate(logger.events)
        if event[:2] == ("add", "final_eval")
    )
    assert logger.events[aggregate][2]["episodes"] == 128
    assert logger.events[aggregate][2]["wins"] == 32
    assert logger.events[aggregate + 1] == ("write",)
    assert logger.events[-1] == ("close", True, True)
    assert aggregate < len(logger.events) - 1


def test_eval_only_preserves_zero_worker_offset(monkeypatch, tmp_path) -> None:
    worker_indices = []

    class EvalOnlyDriver(_Driver):
        def __init__(self, functions, parallel):
            environments = [function() for function in functions]
            worker_indices.extend(environment.index for environment in environments)
            super().__init__(functions, parallel)

    output = tmp_path / "default_offset_eval"
    monkeypatch.setattr(evaluation.embodied, "Driver", EvalOnlyDriver)
    monkeypatch.setattr(evaluation.elements, "Checkpoint", _Checkpoint)

    evaluation.eval_only(
        _Agent,
        lambda index: SimpleNamespace(index=index),
        lambda: _RecordingLogger(output),
        SimpleNamespace(
            from_checkpoint="/checkpoints/step50000",
            eval_eps=2,
            envs=2,
            eval_worker_offset=0,
            eval_policy_mode="eval",
            debug=True,
            logdir=str(output),
        ),
    )

    assert worker_indices == [0, 1]


def test_eval_only_rejects_negative_worker_offset() -> None:
    with pytest.raises(ValueError, match="worker offset must be nonnegative"):
        evaluation.eval_only(
            _Agent,
            lambda index: SimpleNamespace(index=index),
            lambda: object(),
            SimpleNamespace(
                from_checkpoint="/checkpoints/step50000",
                eval_eps=1,
                envs=1,
                eval_worker_offset=-1,
                eval_policy_mode="eval",
                debug=True,
                logdir="unused",
            ),
        )
