from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from dreamarl.envs.meltingpot import (
    BENCHMARK_SUBSTRATES,
    MeltingPotEnv,
)


@dataclass(frozen=True)
class _Space:
    shape: tuple[int, ...]
    n: int | None = None


class _ObservationSpace:
    def __init__(self):
        self.spaces = {"RGB": _Space((8, 8, 3))}


class _ParallelEnv:
    possible_agents = ("player_0", "player_1")

    def __init__(self):
        self.agents = list(self.possible_agents)
        self.count = 0

    def observation_space(self, agent):
        del agent
        return _ObservationSpace()

    def action_space(self, agent):
        del agent
        return _Space((), 4)

    def reset(self, seed=None):
        del seed
        self.count = 0
        return self._observations(), {}

    def step(self, actions):
        assert actions == {"player_0": 1, "player_1": 2}
        self.count += 1
        observations = self._observations()
        rewards = {"player_0": 1.0, "player_1": 3.0}
        terminated = {agent: self.count == 2 for agent in self.possible_agents}
        truncated = {agent: False for agent in self.possible_agents}
        return observations, rewards, terminated, truncated, {}

    def close(self):
        pass

    def _observations(self):
        return {
            agent: {"RGB": np.full((8, 8, 3), index, np.uint8)}
            for index, agent in enumerate(self.possible_agents)
        }


def test_meltingpot_adapter_preserves_agent_geometry_and_benchmark_score() -> None:
    env = MeltingPotEnv("unused", size=(4, 4), seed=7, parallel_env=_ParallelEnv())
    assert env.num_agents == 2
    assert env.obs_space["image"].shape == (2, 4, 4, 3)
    assert env.obs_space["agent_present"].shape == (2,)
    assert env.obs_space["agent_alive"].shape == (2,)
    assert env.obs_space["action_mask"].shape == (2, 4)
    assert env.act_space["action"].shape == (2,)

    first = env.step({"reset": True, "action": np.zeros(2, np.int32)})
    assert first["is_first"]
    np.testing.assert_array_equal(first["reward"], [0.0, 0.0])
    np.testing.assert_array_equal(first["agent_present"], [True, True])
    np.testing.assert_array_equal(first["agent_alive"], [True, True])
    np.testing.assert_array_equal(first["action_mask"], np.ones((2, 4), bool))
    assert first["image"].dtype == np.uint8

    step = env.step({"reset": False, "action": np.array([1, 2], np.int32)})
    np.testing.assert_array_equal(step["reward"], [1.0, 3.0])
    assert step["log/reward_min"] == 1.0
    assert step["log/reward_max"] == 3.0
    assert step["log/reward_std"] == 1.0
    assert not step["is_last"]


def test_all_registered_meltingpot_benchmarks_reset_and_step() -> None:
    pytest.importorskip("meltingpot")
    for substrate in BENCHMARK_SUBSTRATES:
        env = MeltingPotEnv(substrate, size=(16, 16), max_cycles=2, seed=0)
        try:
            first = env.step(
                {
                    "reset": True,
                    "action": np.zeros(env.num_agents, np.int32),
                }
            )
            assert first["image"].shape == (env.num_agents, 16, 16, 3)
            step = env.step(
                {
                    "reset": False,
                    "action": np.zeros(env.num_agents, np.int32),
                }
            )
            assert np.isfinite(step["reward"]).all()
        finally:
            env.close()


def test_meltingpot_same_seed_trajectory_documents_backend_limit() -> None:
    pytest.importorskip("meltingpot")
    envs = [
        MeltingPotEnv(
            "externality_mushrooms__dense",
            size=(16, 16),
            max_cycles=3,
            seed=123,
        )
        for _ in range(2)
    ]
    mismatch = None
    try:
        for step in range(3):
            action = {
                "reset": step == 0,
                "action": np.zeros(envs[0].num_agents, np.int32),
            }
            observations = [env.step(action) for env in envs]
            for key in ("image", "reward", "is_first", "is_last", "is_terminal"):
                if not np.array_equal(observations[0][key], observations[1][key]):
                    differing = int(
                        np.count_nonzero(observations[0][key] != observations[1][key])
                    )
                    mismatch = f"step={step}, field={key}, differing_values={differing}"
                    break
            if mismatch is not None:
                break
    finally:
        for env in envs:
            env.close()

    if mismatch is not None:
        pytest.xfail(
            "Pinned Lab2D does not guarantee identical trajectories for identical "
            f"construction seeds and actions ({mismatch}); Shimmy reset(seed) is "
            "documented as ignored."
        )
