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


class _FakePhysics:
    def __init__(self, seed: int):
        self.seed = seed

    def render(self, *, height, width, camera_id):
        return np.full((height, width, 3), self.seed + camera_id, dtype=np.uint8)


class _FakeDMCEnv:
    def __init__(self, seed: int):
        self.seed = seed
        self.count = 0
        self.closed = False
        self.physics = _FakePhysics(seed)

    def observation_spec(self):
        return {
            "position": _Spec((2,)),
            "velocity": _Spec((1,)),
        }

    def action_spec(self):
        return _Spec(
            (2,), minimum=np.asarray([-1.0, -2.0]), maximum=np.asarray([1.0, 2.0])
        )

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


def test_dmc_adapter_renders_one_vector_member():
    adapter = DMCVectorAdapter(
        "fake/task",
        num_envs=2,
        seed=10,
        env_factory=lambda seed: _FakeDMCEnv(seed),
    )
    try:
        frame = adapter.render(1, height=12, width=16, camera_id=2)
    finally:
        adapter.close()

    assert frame.shape == (12, 16, 3)
    assert frame.dtype == np.uint8
    assert np.all(frame == 13)


def test_dmc_adapter_can_reset_selected_vector_members():
    adapter = DMCVectorAdapter(
        "fake/task",
        num_envs=2,
        max_cycles=5,
        seed=10,
        env_factory=lambda seed: _FakeDMCEnv(seed),
    )
    try:
        adapter.reset()
        adapter.step(np.zeros((2, 1, 2), dtype=np.float32))
        reset_observations = adapter.reset_indices(np.asarray([0]))
        following = adapter.step(np.zeros((2, 1, 2), dtype=np.float32))
    finally:
        adapter.close()

    assert reset_observations.shape == (1, 1, 3)
    np.testing.assert_array_equal(
        reset_observations[0, 0],
        np.asarray([10.0, 0.0, 0.5]),
    )
    assert following.dones.tolist() == [[0.0], [1.0]]


class _FakePhysicsData:
    def __init__(self):
        self.time = 0.0


class _StatefulFakePhysics(_FakePhysics):
    def __init__(self, seed: int):
        super().__init__(seed)
        self.data = _FakePhysicsData()
        self.state = np.asarray([float(seed), 0.0], dtype=np.float64)

    def get_state(self):
        return self.state.copy()

    def set_state(self, state):
        self.state = np.asarray(state, dtype=np.float64).copy()

    def forward(self):
        return None


class _StatefulFakeDMCEnv(_FakeDMCEnv):
    def __init__(self, seed: int):
        super().__init__(seed)
        self.physics = _StatefulFakePhysics(seed)
        self._task = type("Task", (), {})()
        self._task._random = np.random.RandomState(seed)
        self._step_count = 0
        self._reset_next_step = False


def test_dmc_adapter_state_snapshot_round_trip(tmp_path):
    adapter = DMCVectorAdapter(
        "fake/task",
        num_envs=2,
        max_cycles=5,
        seed=10,
        env_factory=lambda seed: _StatefulFakeDMCEnv(seed),
    )
    restored = DMCVectorAdapter(
        "fake/task",
        num_envs=2,
        max_cycles=5,
        seed=10,
        env_factory=lambda seed: _StatefulFakeDMCEnv(seed),
    )
    try:
        adapter.reset()
        adapter._envs[0].physics.state[:] = [3.0, 4.0]
        adapter._envs[0].physics.data.time = 7.5
        adapter._envs[0]._step_count = 13
        adapter._envs[0]._reset_next_step = True
        adapter._episode_returns[0, 0] = 42.0
        adapter._episode_lengths[0] = 17
        adapter._envs[0]._task._random.uniform()
        expected_random = adapter._envs[0]._task._random.uniform()
        adapter._envs[0]._task._random.set_state(
            np.random.RandomState(10).get_state()
        )
        adapter._envs[0]._task._random.uniform()
        snapshot = tmp_path / "dmc_state.npz"
        adapter.save_state_npz(snapshot)

        restored.load_state_npz(snapshot)

        np.testing.assert_array_equal(
            restored._envs[0].physics.state,
            adapter._envs[0].physics.state,
        )
        assert restored._envs[0].physics.data.time == 7.5
        assert restored._envs[0]._step_count == 13
        assert restored._envs[0]._reset_next_step is True
        assert restored._episode_returns[0, 0] == 42.0
        assert restored._episode_lengths[0] == 17
        assert restored._envs[0]._task._random.uniform() == expected_random
    finally:
        adapter.close()
        restored.close()
