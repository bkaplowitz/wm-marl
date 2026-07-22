from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from world_marl.envs.dmc_adapter import DMCVectorAdapter
from world_marl.scripts import eval_jepa_wm, train_dmc_jepa


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


class _Physics:
    def __init__(self, env):
        self.env = env

    def render(self, *, height, width, camera_id):
        return np.full((height, width, 3), self.env.count, dtype=np.uint8)


class _Env:
    def __init__(self):
        self.count = 0
        self.physics = _Physics(self)

    def observation_spec(self):
        return {"observation": _Spec((1,))}

    def action_spec(self):
        return _Spec((1,), minimum=np.asarray([-1.0]), maximum=np.asarray([1.0]))

    def reset(self):
        self.count = 0
        return _TimeStep({"observation": np.asarray([0.0])})

    def step(self, action):
        self.count += 1
        return _TimeStep(
            {"observation": np.asarray([self.count], dtype=np.float32)},
            reward=1.0,
            last=self.count == 2,
        )


class _VideoLogger:
    def __init__(self, path):
        self.path = path
        self.call = None

    def write_video(self, filename, frames, *, fps, key, caption):
        self.call = {
            "filename": filename,
            "frames": frames,
            "fps": fps,
            "key": key,
            "caption": caption,
        }
        return self.path


def test_policy_evaluation_records_only_first_episode(monkeypatch, tmp_path):
    adapter = DMCVectorAdapter(
        "fake/task",
        num_envs=1,
        env_factory=lambda seed: _Env(),
    )
    monkeypatch.setattr(
        train_dmc_jepa,
        "_make_vector_adapter",
        lambda args, seed, num_envs: adapter,
    )
    monkeypatch.setattr(
        train_dmc_jepa,
        "select_continuous_actions",
        lambda *args, **kwargs: np.zeros((1, 1), dtype=np.float32),
    )
    args = SimpleNamespace(
        quiet=True,
        wandb_video_size=8,
        wandb_video_camera=0,
        wandb_video_frame_stride=1,
        wandb_video_fps=10,
        failure_return_threshold=0.5,
        success_return_threshold=1.5,
    )
    logger = _VideoLogger(tmp_path / "eval.mp4")

    result = train_dmc_jepa._evaluate_continuous_policy(
        args,
        state=None,
        config=None,
        seed=0,
        num_envs=1,
        episodes=1,
        action_low=jnp.asarray([-1.0]),
        action_high=jnp.asarray([1.0]),
        desc="video eval",
        video_logger=logger,
        video_filename="videos/eval.mp4",
        video_key="videos/eval",
    )

    assert result["mean_return"] == 2.0
    assert result["video_path"] == str(tmp_path / "eval.mp4")
    assert logger.call["filename"] == "videos/eval.mp4"
    assert logger.call["key"] == "videos/eval"
    assert len(logger.call["frames"]) == 2


def test_world_model_evaluator_ignores_removed_checkpoint_config_keys():
    config, ignored = eval_jepa_wm._jepa_config_from_metadata(
        {
            "jepa_config": {
                "observation_dim": 4,
                "action_dim": 2,
                "clip_imagined_rewards": False,
                "imagined_reward_min": 0.0,
                "imagined_reward_max": 1.0,
            }
        }
    )

    assert config.observation_dim == 4
    assert config.action_dim == 2
    assert ignored == [
        "clip_imagined_rewards",
        "imagined_reward_max",
        "imagined_reward_min",
    ]


def test_world_model_evaluator_collects_with_explicit_terminal_contract(monkeypatch):
    class FakeAdapter:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.num_envs = 1
            self.action_low = np.asarray([-1.0], dtype=np.float32)
            self.action_high = np.asarray([1.0], dtype=np.float32)

        def reset(self):
            return np.zeros((1, 1, 2), dtype=np.float32)

        def sample_actions(self, rng):
            del rng
            return np.zeros((1, 1, 1), dtype=np.float32)

        def step(self, actions):
            del actions
            return SimpleNamespace(
                observations=np.zeros((1, 1, 2), dtype=np.float32),
                rewards=np.ones((1, 1), dtype=np.float32),
                dones=np.ones((1, 1), dtype=np.float32),
                is_last=np.ones((1, 1), dtype=np.float32),
                is_terminal=np.zeros((1, 1), dtype=np.float32),
            )

        def close(self):
            return None

    monkeypatch.setattr(eval_jepa_wm, "DMCVectorAdapter", FakeAdapter)
    args = SimpleNamespace(
        num_envs=1,
        max_cycles=2,
        env_workers=1,
        collect_steps=2,
        collect_policy="random",
        quiet=True,
    )
    config = SimpleNamespace(observation_dim=2, action_dim=1)

    replay, _ = eval_jepa_wm.collect_dmc_replay(
        args,
        state=None,
        config=config,
        env="dmc:fake/task",
        seed=0,
    )

    np.testing.assert_array_equal(replay.is_last[:2, 0], np.ones(2))
    np.testing.assert_array_equal(replay.is_terminal[:2, 0], np.zeros(2))
