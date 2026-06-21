from __future__ import annotations

import numpy as np
import pytest

from world_marl.envs.dmc_adapter import DMCVectorAdapter, dmc_env_name


class _Spec:
    def __init__(self, shape, minimum=None, maximum=None):
        self.shape = tuple(shape)
        self.minimum = minimum
        self.maximum = maximum


class _TimeStep:
    def __init__(self, observation, reward=0.0, last=False):
        self.observation = observation
        self.reward = reward
        self._last = last

    def last(self):
        return self._last


class _FakeDMCEnv:
    def __init__(self, seed: int):
        self.seed = seed
        self.count = 0
        self.closed = False

    def observation_spec(self):
        return {
            "position": _Spec((2,)),
            "velocity": _Spec((1,)),
        }

    def action_spec(self):
        return _Spec((2,), minimum=np.asarray([-1.0, -2.0]), maximum=np.asarray([1.0, 2.0]))

    def reset(self):
        self.count = 0
        return _TimeStep(self._observation(), reward=0.0, last=False)

    def step(self, action):
        self.count += 1
        return _TimeStep(
            self._observation(),
            reward=float(np.asarray(action).sum()),
            last=self.count >= 2,
        )

    def close(self):
        self.closed = True

    def _observation(self):
        return {
            "position": np.asarray([self.seed, self.count], dtype=np.float32),
            "velocity": np.asarray([self.count + 0.5], dtype=np.float32),
        }


def test_dmc_env_name_parses_domain_task():
    assert dmc_env_name("dmc:cartpole/swingup") == "cartpole/swingup"


@pytest.mark.parametrize("num_workers", [1, 2])
def test_dmc_adapter_reset_step_and_completion(num_workers):
    adapter = DMCVectorAdapter(
        "fake/task",
        num_envs=2,
        max_cycles=5,
        seed=10,
        env_factory=lambda seed: _FakeDMCEnv(seed),
        num_workers=num_workers,
    )
    try:
        observations = adapter.reset()
        actions = adapter.sample_actions(np.random.default_rng(0))
        first = adapter.step(actions)
        second = adapter.step(np.zeros((2, 1, 2), dtype=np.float32))

        assert adapter.num_agents == 1
        assert adapter.action_dim == 2
        assert adapter.observation_shape == (3,)
        assert observations.shape == (2, 1, 3)
        assert actions.shape == (2, 1, 2)
        assert first.observations.shape == (2, 1, 3)
        assert first.rewards.shape == (2, 1)
        assert second.dones.tolist() == [[1.0], [1.0]]
        assert len(second.completed_returns) == 2
        assert second.completed_lengths == (2, 2)
    finally:
        adapter.close()
