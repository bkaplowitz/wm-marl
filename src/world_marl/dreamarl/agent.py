"""Agent-axis-native DreaMARL model built from the exact M3 modules.

There is one implementation for every agent count. Environment transitions
and replay retain an explicit agent axis, while the unchanged M3 neural
modules consume a folded ``batch * agent`` axis with shared parameters. For a
singleton agent axis every fold and unfold is an identity reshape, making the
model algebraically identical to M3 without selecting a separate code path.
"""

from __future__ import annotations

import elements
import numpy as np

from .axes import (
    GLOBAL_OBSERVATION_KEYS,
    GLOBAL_REPLAY_KEYS,
    broadcast_global_batch,
    broadcast_global_sequence,
    fold_agent_batch,
    fold_agent_sequence,
    fold_tree_batch,
    unfold_agent_sequence,
    unfold_tree_batch,
)
from .m3.agent import Agent as M3Agent


class Agent(M3Agent):
    """The exact M3 learner lifted over a shared explicit agent axis."""

    banner = [
        r"---  ____                 __  ___    _    ____  _     ---",
        r"--- |  _ \ _ __ ___  __ _|  \/  |  / \  |  _ \| |    ---",
        r"--- | | | | '__/ _ \/ _` | |\/| | / _ \ | |_) | |    ---",
        r"--- | |_| | | |  __/ (_| | |  | |/ ___ \|  _ <| |___ ---",
        r"--- |____/|_|  \___|\__,_|_|  |_/_/   \_\_| \_\_____|---",
    ]

    def __init__(self, obs_space, act_space, config):
        self.num_agents = int(config.num_agents)
        if self.num_agents < 1:
            raise ValueError("num_agents must be positive")
        self.joint_obs_space = obs_space
        self.joint_act_space = act_space
        local_obs_space = {
            key: (
                space
                if key in GLOBAL_OBSERVATION_KEYS
                else _remove_agent_axis(key, space, self.num_agents)
            )
            for key, space in obs_space.items()
        }
        local_act_space = {
            key: _remove_agent_axis(key, space, self.num_agents)
            for key, space in act_space.items()
        }
        super().__init__(local_obs_space, local_act_space, config)

    @property
    def ext_space(self):
        spaces = super().ext_space
        return {
            key: (
                space
                if key in GLOBAL_REPLAY_KEYS
                else _add_agent_axis(space, self.num_agents)
            )
            for key, space in spaces.items()
        }

    def init_policy(self, batch_size):
        carry = super().init_policy(batch_size * self.num_agents)
        return unfold_tree_batch(carry, self.num_agents)

    def init_train(self, batch_size):
        return self.init_policy(batch_size)

    def init_report(self, batch_size):
        return self.init_policy(batch_size)

    def policy(self, carry, obs, mode="train"):
        flat_carry = fold_tree_batch(carry, self.num_agents)
        flat_obs = {
            key: (
                broadcast_global_batch(value, self.num_agents)
                if key in GLOBAL_OBSERVATION_KEYS
                else fold_agent_batch(value, self.num_agents)
            )
            for key, value in obs.items()
        }
        flat_carry, flat_actions, flat_outputs = super().policy(
            flat_carry, flat_obs, mode
        )
        return (
            unfold_tree_batch(flat_carry, self.num_agents),
            unfold_tree_batch(flat_actions, self.num_agents),
            unfold_tree_batch(flat_outputs, self.num_agents),
        )

    def train(self, carry, data):
        flat_carry = fold_tree_batch(carry, self.num_agents)
        flat_data = self._fold_replay(data)
        flat_carry, outputs, metrics = super().train(flat_carry, flat_data)
        if "replay" in outputs:
            outputs = {
                **outputs,
                "replay": self._unfold_replay_updates(outputs["replay"]),
            }
        return unfold_tree_batch(flat_carry, self.num_agents), outputs, metrics

    def report(self, carry, data):
        flat_carry = fold_tree_batch(carry, self.num_agents)
        flat_carry, metrics = super().report(
            flat_carry, self._fold_replay(data)
        )
        return unfold_tree_batch(flat_carry, self.num_agents), metrics

    def _fold_replay(self, data):
        per_agent = set(self.joint_obs_space) - GLOBAL_OBSERVATION_KEYS
        per_agent.update(self.joint_act_space)
        per_agent.update(
            set(data)
            - set(self.joint_obs_space)
            - set(self.joint_act_space)
            - GLOBAL_REPLAY_KEYS
        )
        return {
            key: (
                fold_agent_sequence(value, self.num_agents)
                if key in per_agent
                else broadcast_global_sequence(value, self.num_agents)
            )
            for key, value in data.items()
        }

    def _unfold_replay_updates(self, updates):
        result = {}
        for key, value in updates.items():
            restored = unfold_agent_sequence(value, self.num_agents)
            # Step IDs address joint replay rows. They were broadcast only so
            # each shared M3 trajectory could execute the unchanged replay
            # context logic; replay.update() must receive one joint ID again.
            result[key] = restored[:, :, 0] if key == "stepid" else restored
        return result


def _remove_agent_axis(name: str, space, num_agents: int):
    if not space.shape or space.shape[0] != num_agents:
        raise ValueError(
            f"{name!r} must expose leading agent axis {num_agents}, "
            f"got shape {space.shape}"
        )
    return elements.Space(
        space.dtype,
        space.shape[1:],
        _remove_bound_axis(space.low),
        _remove_bound_axis(space.high),
    )


def _add_agent_axis(space, num_agents: int):
    shape = (num_agents, *space.shape)
    low = None if space.low is None else np.broadcast_to(space.low, shape)
    high = None if space.high is None else np.broadcast_to(space.high, shape)
    return elements.Space(space.dtype, shape, low, high)


def _remove_bound_axis(bound):
    if bound is None:
        return None
    return np.asarray(bound)[0]
