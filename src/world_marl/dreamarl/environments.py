"""Environment contracts for the agent-axis-native DreaMARL learner."""

from __future__ import annotations

import functools

import elements
import embodied
import numpy as np

from .axes import GLOBAL_OBSERVATION_KEYS


class SingletonAgentEnv(embodied.Env):
    """Expose an unchanged single-agent environment with an agent axis.

    This adapter changes shapes only. It makes no random calls and leaves
    rewards, lifecycle flags, observations, and actions byte-identical after
    removing the singleton axis again.
    """

    num_agents = 1

    def __init__(self, env):
        self._env = env

    @functools.cached_property
    def obs_space(self):
        return {
            key: (
                space
                if key in GLOBAL_OBSERVATION_KEYS or key.startswith("log/")
                else _add_agent_axis(space, self.num_agents)
            )
            for key, space in self._env.obs_space.items()
        }

    @functools.cached_property
    def act_space(self):
        return {
            key: (space if key == "reset" else _add_agent_axis(space, self.num_agents))
            for key, space in self._env.act_space.items()
        }

    def step(self, action):
        local_action = {
            key: value if key == "reset" else np.asarray(value)[0]
            for key, value in action.items()
        }
        observation = self._env.step(local_action)
        return {
            key: (
                value
                if key in GLOBAL_OBSERVATION_KEYS or key.startswith("log/")
                else np.asarray(value)[None]
            )
            for key, value in observation.items()
        }

    def close(self):
        return self._env.close()


def _add_agent_axis(space, num_agents: int):
    shape = (num_agents, *space.shape)
    low = None if space.low is None else np.broadcast_to(space.low, shape)
    high = None if space.high is None else np.broadcast_to(space.high, shape)
    return elements.Space(space.dtype, shape, low, high)
