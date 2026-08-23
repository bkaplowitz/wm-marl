from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from dreamarl.envs.smac import SMACEnv


class _FakeSMAC:
    def __init__(self):
        self.count = 0
        self.actions = None
        self.n_actions_no_attack = 2
        self.reward_death_value = 10.0
        self.reward_win = 20.0
        self.reward_scale = True
        self.reward_scale_rate = 20.0
        self.max_reward = 40.0
        self.reward_sparse = False
        self.agents = {}
        self.enemies = {}
        self.previous_ally_units = None
        self.previous_enemy_units = None

    def get_env_info(self):
        return {
            "n_agents": 3,
            "n_enemies": 2,
            "obs_shape": 5,
            "n_actions": 4,
        }

    def reset(self):
        self.count = 0
        self.agents = {
            index: SimpleNamespace(health=10.0, shield=0.0) for index in range(3)
        }
        self.enemies = {
            0: SimpleNamespace(health=10.0, shield=5.0),
            1: SimpleNamespace(health=10.0, shield=0.0),
        }

    def get_obs(self):
        return [np.full(5, self.count + index, np.float32) for index in range(3)]

    def get_avail_actions(self):
        if self.count:
            return [[1, 1, 0, 0], [1, 0, 0, 0], [0, 1, 1, 1]]
        return np.ones((3, 4), np.int32)

    def step(self, actions):
        self.actions = actions
        self.previous_ally_units = copy.deepcopy(self.agents)
        self.previous_enemy_units = copy.deepcopy(self.enemies)
        self.agents[2].health = 0.0
        self.enemies[0].health = 8.0
        self.enemies[0].shield = 0.0
        self.enemies[1].health = 0.0
        self.count += 1
        return 2.5, True, {
            "battle_won": True,
            "dead_allies": 1,
            "dead_enemies": 1,
        }

    def close(self):
        pass


def test_smac_adapter_preserves_team_reward_masks_and_win_metric() -> None:
    backend = _FakeSMAC()
    env = SMACEnv("unused", sc2_env=backend)
    first = env.step({"reset": True, "action": np.zeros(3, np.int32)})
    assert first["observation"].shape == (3, 5)
    np.testing.assert_array_equal(first["reward"], np.zeros(3, np.float32))
    np.testing.assert_array_equal(first["agent_alive"], [True, True, True])
    np.testing.assert_array_equal(first["controllable_alive"], [True, True, True])

    final = env.step(
        {"reset": False, "action": np.array([1, 0, 2], np.int32)}
    )
    assert backend.actions == [1, 0, 2]
    np.testing.assert_array_equal(final["reward"], [2.5, 2.5, 2.5])
    np.testing.assert_array_equal(final["agent_alive"], [True, True, True])
    np.testing.assert_array_equal(final["controllable_alive"], [True, False, True])
    assert final["is_last"] and final["is_terminal"]
    assert final["log/battle_won"] == 1.0
    assert final["log/dead_allies"] == 1.0
    assert final["log/dead_enemies"] == 1.0
    assert final["log/legacy_reward"] == 2.5
    assert final["log/enemy_health_damage"] == 12.0
    assert final["log/enemy_shield_damage"] == 5.0
    assert final["log/enemy_shield_regen"] == 0.0
    assert final["log/enemy_damage"] == 17.0
    assert final["log/enemy_deaths_step"] == 1.0
    assert final["log/ally_deaths_step"] == 1.0
    assert final["log/enemy_survivors"] == 1.0
    assert final["log/ally_survivors"] == 2.0
    assert final["log/corrected_reward"] == 23.5
    assert final["log/action_noop_count"] == 1.0
    assert final["log/action_stop_count"] == 1.0
    assert final["log/action_attack_count"] == 1.0
    assert final["log/attack_target_0_count"] == 1.0


def test_smac_adapter_rejects_unavailable_actions() -> None:
    backend = _FakeSMAC()
    env = SMACEnv("unused", sc2_env=backend)
    env.step({"reset": True, "action": np.zeros(3, np.int32)})
    backend.count = 1
    with pytest.raises(ValueError, match="unavailable actions"):
        env.step({"reset": False, "action": np.array([2, 0, 1], np.int32)})
