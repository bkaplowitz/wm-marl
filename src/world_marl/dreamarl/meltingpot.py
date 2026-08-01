"""Melting Pot environment contract for DreaMARL.

The adapter exposes homogeneous local observations and actions with a leading
agent axis while retaining joint episode boundaries. The scalar environment
reward is the arithmetic mean of the agents' rewards, so its episode sum is
the average per-agent return reported by the Melting Pot benchmark.
"""

from __future__ import annotations

import functools

import elements
import embodied
import numpy as np

from world_marl.envs.meltingpot_adapter import make_meltingpot_env


BENCHMARK_SUBSTRATES = (
    "chicken_in_the_matrix__arena",
    "coop_mining",
    "externality_mushrooms__dense",
    "gift_refinements",
    "pure_coordination_in_the_matrix__repeated",
    "rationalizable_coordination_in_the_matrix__repeated",
    "stag_hunt_in_the_matrix__arena",
)


class MeltingPotEnv(embodied.Env):
    """Expose one Melting Pot substrate through the DreaMARL tensor contract."""

    def __init__(
        self,
        substrate: str,
        *,
        size: tuple[int, int] = (64, 64),
        max_cycles: int = 1000,
        seed: int | None = None,
        parallel_env=None,
    ):
        self._env = parallel_env or make_meltingpot_env(
            substrate, max_cycles=max_cycles
        )
        self._agents = tuple(self._env.possible_agents)
        if not self._agents:
            raise ValueError("Melting Pot must expose at least one agent")
        self.num_agents = len(self._agents)
        self._size = tuple(int(value) for value in size)
        if len(self._size) != 2 or min(self._size) < 1:
            raise ValueError(f"invalid image size: {size}")
        self._seed = seed
        self._needs_reset = True
        self._action_dim, self._image_shape = self._validate_spaces()

    @functools.cached_property
    def obs_space(self):
        return {
            "image": elements.Space(
                np.uint8,
                (self.num_agents, *self._size, self._image_shape[-1]),
                0,
                256,
            ),
            "reward": elements.Space(np.float32, ()),
            "is_first": elements.Space(bool, ()),
            "is_last": elements.Space(bool, ()),
            "is_terminal": elements.Space(bool, ()),
            "log/reward_min": elements.Space(np.float32, ()),
            "log/reward_max": elements.Space(np.float32, ()),
            "log/reward_std": elements.Space(np.float32, ()),
        }

    @functools.cached_property
    def act_space(self):
        return {
            "action": elements.Space(
                np.int32, (self.num_agents,), 0, self._action_dim
            ),
            "reset": elements.Space(bool, (), 0, 2),
        }

    def step(self, action):
        if bool(np.asarray(action["reset"])) or self._needs_reset:
            return self._reset()
        actions = np.asarray(action["action"], np.int32)
        if actions.shape != (self.num_agents,):
            raise ValueError(
                f"expected actions shaped {(self.num_agents,)}, got {actions.shape}"
            )
        observations, rewards, terminations, truncations, _ = self._env.step(
            {
                agent: int(actions[index])
                for index, agent in enumerate(self._agents)
            }
        )
        terminal = self._joint_flag(terminations, "termination")
        truncated = self._joint_flag(truncations, "truncation")
        self._needs_reset = terminal or truncated
        return self._observation(
            observations,
            rewards,
            is_first=False,
            is_last=self._needs_reset,
            is_terminal=terminal,
        )

    def close(self):
        return self._env.close()

    def _reset(self):
        observations, _ = self._env.reset(seed=self._seed)
        self._seed = None
        self._needs_reset = False
        rewards = {agent: 0.0 for agent in self._agents}
        return self._observation(
            observations,
            rewards,
            is_first=True,
            is_last=False,
            is_terminal=False,
        )

    def _observation(
        self,
        observations,
        rewards,
        *,
        is_first: bool,
        is_last: bool,
        is_terminal: bool,
    ):
        reward_values = np.asarray(
            [rewards.get(agent, 0.0) for agent in self._agents], np.float32
        )
        images = np.stack(
            [
                self._resize_nearest(
                    np.asarray(observations[agent]["RGB"], np.uint8)
                )
                for agent in self._agents
            ]
        )
        return {
            "image": images,
            "reward": np.float32(reward_values.mean()),
            "is_first": np.bool_(is_first),
            "is_last": np.bool_(is_last),
            "is_terminal": np.bool_(is_terminal),
            "log/reward_min": np.float32(reward_values.min()),
            "log/reward_max": np.float32(reward_values.max()),
            "log/reward_std": np.float32(reward_values.std()),
        }

    def _validate_spaces(self):
        action_dims = set()
        image_shapes = set()
        for agent in self._agents:
            action_space = self._env.action_space(agent)
            if not hasattr(action_space, "n"):
                raise TypeError("Melting Pot actions must be homogeneous and discrete")
            action_dims.add(int(action_space.n))
            observation_space = self._env.observation_space(agent)
            if not hasattr(observation_space, "spaces"):
                raise TypeError("Melting Pot observations must be dictionaries")
            image_space = observation_space.spaces.get("RGB")
            if image_space is None or len(image_space.shape) != 3:
                raise ValueError("Melting Pot observations must contain RGB images")
            image_shapes.add(tuple(int(value) for value in image_space.shape))
        if len(action_dims) != 1 or len(image_shapes) != 1:
            raise ValueError(
                "shared DreaMARL modules require homogeneous per-agent spaces"
            )
        return action_dims.pop(), image_shapes.pop()

    def _joint_flag(self, values, name: str) -> bool:
        flags = [bool(values.get(agent, False)) for agent in self._agents]
        if len(set(flags)) != 1:
            raise ValueError(f"Melting Pot {name} flags are not joint: {values}")
        return flags[0]

    def _resize_nearest(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] == self._size:
            return image
        rows = np.linspace(0, image.shape[0] - 1, self._size[0]).astype(np.int32)
        columns = np.linspace(0, image.shape[1] - 1, self._size[1]).astype(np.int32)
        return image[rows][:, columns]
