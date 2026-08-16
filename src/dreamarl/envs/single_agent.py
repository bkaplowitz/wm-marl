"""Expose a conventional environment through the DreaMARL agent-axis contract."""

from __future__ import annotations

import elements
import embodied
import numpy as np

from ..marl.spaces import add_agent_axis


class SingletonAgentEnv(embodied.Env):
    """Add an identity agent axis without changing environment semantics."""

    _GLOBAL_OBSERVATIONS = frozenset({"is_first", "is_last", "is_terminal"})

    def __init__(self, env: embodied.Env):
        self._env = env
        self.num_agents = 1

    @property
    def obs_space(self):
        spaces = {
            key: (
                space
                if key in self._GLOBAL_OBSERVATIONS or key.startswith("log/")
                else add_agent_axis(space, self.num_agents)
            )
            for key, space in self._env.obs_space.items()
        }
        spaces.update(
            agent_present=elements.Space(bool, (1,)),
            agent_alive=elements.Space(bool, (1,)),
        )
        return spaces

    @property
    def act_space(self):
        return {
            key: space if key == "reset" else add_agent_axis(space, self.num_agents)
            for key, space in self._env.act_space.items()
        }

    def step(self, action):
        local_action = {
            key: value if key == "reset" else self._remove_agent_axis(key, value)
            for key, value in action.items()
        }
        observation = self._env.step(local_action)
        result = {
            key: (
                value
                if key in self._GLOBAL_OBSERVATIONS or key.startswith("log/")
                else np.expand_dims(value, 0)
            )
            for key, value in observation.items()
        }
        result.update(
            agent_present=np.ones((1,), bool),
            agent_alive=np.ones((1,), bool),
        )
        return result

    def close(self):
        return self._env.close()

    @staticmethod
    def _remove_agent_axis(name, value):
        value = np.asarray(value)
        if not value.shape or value.shape[0] != 1:
            raise ValueError(
                f"{name!r} must expose singleton leading agent axis, got {value.shape}"
            )
        return value[0]
