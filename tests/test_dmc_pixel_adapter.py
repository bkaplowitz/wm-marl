from __future__ import annotations

import numpy as np
import pytest

from world_marl.envs.dmc_pixel_adapter import (
    DMCPixelAdapter,
    dmc_pixel_env_name,
)


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


class _FakePixelDMCEnv:
    def __init__(self, seed: int, *, height: int = 6, width: int = 8):
        self.seed = seed
        self.height = height
        self.width = width
        self.count = 0
        self.closed = False

    def observation_spec(self):
        return {"pixels": _Spec((self.height, self.width, 3))}

    def action_spec(self):
        return _Spec(
            (2,), minimum=np.asarray([-1.0, -2.0]), maximum=np.asarray([1.0, 2.0])
        )

    def reset(self):
        self.count = 0
        return _TimeStep(self._observation(), reward=None, last=False)

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
        value = np.uint8(min(255, self.seed + self.count))
        return {"pixels": np.full((self.height, self.width, 3), value, np.uint8)}


def test_dmc_pixel_env_name_parses_domain_task():
    assert dmc_pixel_env_name("dmc-pixels:point_mass/easy") == "point_mass/easy"
    with pytest.raises(ValueError, match="dmc-pixels:<domain>/<task>"):
        dmc_pixel_env_name("dmc-pixels:point_mass")


@pytest.mark.parametrize("num_workers", [1, 2])
def test_dmc_pixel_adapter_preserves_environment_semantics(num_workers):
    created = []

    def factory(seed):
        env = _FakePixelDMCEnv(seed)
        created.append(env)
        return env

    adapter = DMCPixelAdapter(
        "fake/task",
        num_envs=2,
        max_cycles=5,
        seed=10,
        image_size=6,
        env_factory=factory,
        num_workers=num_workers,
    )
    try:
        observations = adapter.reset()
        actions = adapter.sample_actions(np.random.default_rng(0))
        first = adapter.step(actions)
        second = adapter.step(np.zeros((2, 1, 2), dtype=np.float32))

        assert adapter.observation_shape == (6, 8, 3)
        assert adapter.action_shape == (2,)
        assert adapter.action_dim == 2
        np.testing.assert_array_equal(adapter.action_low, [-1.0, -2.0])
        np.testing.assert_array_equal(adapter.action_high, [1.0, 2.0])
        assert observations.shape == (2, 1, 6, 8, 3)
        assert observations.dtype == np.float32
        assert float(observations.min()) >= 0.0
        assert float(observations.max()) <= 1.0
        assert actions.shape == (2, 1, 2)
        np.testing.assert_allclose(first.rewards[:, 0], actions[:, 0].sum(axis=-1))
        assert second.dones.tolist() == [[1.0], [1.0]]
        assert second.completed_lengths == (2, 2)
        assert len(second.completed_returns) == 2
        assert second.observations.shape == (2, 1, 6, 8, 3)
        assert adapter.environment_metadata == {
            "environment_backend": "dm_control",
            "observation_mode": "pixels",
            "dmc_domain": "fake",
            "dmc_task": "task",
            "image_height": 6,
            "image_width": 8,
            "camera_id": 0,
            "frame_stack": 1,
            "action_repeat": 1,
            "flatten": False,
        }
    finally:
        adapter.close()

    assert all(env.closed for env in created)


def test_dmc_pixel_adapter_marks_adapter_time_limit_as_truncation():
    adapter = DMCPixelAdapter(
        "fake/task",
        num_envs=1,
        max_cycles=1,
        seed=0,
        env_factory=lambda seed: _FakePixelDMCEnv(seed),
    )
    try:
        adapter.reset()
        step = adapter.step(np.zeros((1, 1, 2), dtype=np.float32))
        assert step.dones.tolist() == [[1.0]]
        assert step.infos[0]["terminated"] is False
        assert step.infos[0]["truncated"] is True
    finally:
        adapter.close()


def test_frame_stack_orders_newest_last_and_fills_on_reset():
    adapter = DMCPixelAdapter(
        "point_mass/easy",
        num_envs=1,
        seed=10,
        env_factory=lambda seed: _FakePixelDMCEnv(seed),
        frame_stack=2,
    )
    obs = adapter.reset()
    assert adapter.raw_observation_shape == (6, 8, 6)
    assert adapter.observation_shape == (6, 8, 6)
    assert obs.shape == (1, 1, 6, 8, 6)
    np.testing.assert_array_equal(obs[0, 0, ..., :3], obs[0, 0, ..., 3:])
    assert float(obs[0, 0, 0, 0, 0]) == pytest.approx(10 / 255.0)

    step = adapter.step(np.zeros((1, 1, 2), np.float32))
    stacked = step.observations[0, 0]
    assert float(stacked[0, 0, 0]) == pytest.approx(10 / 255.0)  # oldest first
    assert float(stacked[0, 0, 3]) == pytest.approx(11 / 255.0)  # newest last


def test_action_repeat_sums_rewards_and_stops_at_episode_end():
    adapter = DMCPixelAdapter(
        "point_mass/easy",
        num_envs=1,
        seed=0,
        env_factory=lambda seed: _FakePixelDMCEnv(seed),
        action_repeat=3,
        auto_reset=False,
    )
    adapter.reset()
    actions = np.full((1, 1, 2), 0.5, np.float32)
    step = adapter.step(actions)
    # fake env ends at count >= 2, so only 2 of the 3 sub-steps run
    assert step.rewards[0, 0] == pytest.approx(2.0)
    assert step.dones[0, 0] == 1.0
    assert adapter._envs[0].count == 2


def test_auto_reset_refills_frame_stack_with_reset_frame():
    adapter = DMCPixelAdapter(
        "point_mass/easy",
        num_envs=1,
        seed=10,
        env_factory=lambda seed: _FakePixelDMCEnv(seed),
        frame_stack=2,
        action_repeat=2,
    )
    adapter.reset()
    step = adapter.step(np.zeros((1, 1, 2), np.float32))  # episode done, auto-reset
    assert step.dones[0, 0] == 1.0
    stacked = step.observations[0, 0]
    np.testing.assert_array_equal(stacked[..., :3], stacked[..., 3:])
    assert float(stacked[0, 0, 0]) == pytest.approx(10 / 255.0)  # reset frame


def test_flatten_emits_flat_float_observations():
    adapter = DMCPixelAdapter(
        "point_mass/easy",
        num_envs=2,
        seed=0,
        env_factory=lambda seed: _FakePixelDMCEnv(seed),
        frame_stack=2,
        flatten=True,
    )
    assert adapter.observation_shape == (6 * 8 * 6,)
    assert adapter.raw_observation_shape == (6, 8, 6)
    obs = adapter.reset()
    assert obs.shape == (2, 1, 6 * 8 * 6)
    assert obs.dtype == np.float32
    step = adapter.step(np.zeros((2, 1, 2), np.float32))
    assert step.observations.shape == (2, 1, 6 * 8 * 6)


def test_invalid_stack_and_repeat_rejected():
    with pytest.raises(ValueError, match="frame_stack"):
        DMCPixelAdapter(
            "point_mass/easy",
            env_factory=lambda seed: _FakePixelDMCEnv(seed),
            frame_stack=0,
        )
    with pytest.raises(ValueError, match="action_repeat"):
        DMCPixelAdapter(
            "point_mass/easy",
            env_factory=lambda seed: _FakePixelDMCEnv(seed),
            action_repeat=0,
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    "env_id",
    [
        "point_mass/easy",
        "point_mass/hard",
        "cartpole/swingup",
        "finger/spin",
    ],
)
def test_official_dmc_benchmark_tasks_render_nonblank_hwc_pixels(env_id):
    pytest.importorskip("dm_control")
    adapter = DMCPixelAdapter(
        env_id,
        num_envs=1,
        max_cycles=4,
        seed=0,
        image_size=32,
    )
    try:
        observations = adapter.reset()
        step = adapter.step(np.zeros((1, 1, adapter.action_dim), dtype=np.float32))

        assert observations.shape == (1, 1, 32, 32, 3)
        assert observations.dtype == np.float32
        assert float(observations.std()) > 0.0
        assert np.isfinite(step.rewards).all()
        assert adapter.environment_metadata["environment_backend"] == "dm_control"
        domain, task = env_id.split("/", 1)
        assert adapter.environment_metadata["dmc_domain"] == domain
        assert adapter.environment_metadata["dmc_task"] == task
    finally:
        adapter.close()
