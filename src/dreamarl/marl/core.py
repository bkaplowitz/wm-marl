"""Agent-axis runtime for the shared joint-conditioned DreaMARL learner.

The public data contract retains team identity while local actors and critics
remain parameter-shared. The temporal model advances every active agent from
the synchronized team of latent-action pairs. For ``A=1``, the peer context is
empty and the implementation is exactly the locked single-agent learner.
"""

from __future__ import annotations

import elements
import jax
import jax.numpy as jnp

from ..agent import Agent as LocalAgent
from .axes import TeamAxis
from .spaces import (
    add_agent_axis,
    local_action_spaces,
    local_observation_spaces,
    report_rows,
)


class TeamAxisAdapter:
    """Apply a shared local learner without discarding team identity."""

    @property
    def ext_space(self):
        return {
            key: (
                space
                if key in {"consec", "stepid"}
                else add_agent_axis(space, self.team.size)
            )
            for key, space in super().ext_space.items()
        }

    def init_policy(self, batch_size):
        return self.team.unfold_tree_batch(
            super().init_policy(batch_size * self.team.size)
        )

    def init_train(self, batch_size):
        return self.init_policy(batch_size)

    def init_report(self, batch_size):
        return self.init_policy(batch_size)

    def report_rows(self, batch_size):
        return report_rows(batch_size, self.team.size)

    def policy(self, carry, obs, mode="train"):
        local_carry = self.team.fold_tree_batch(carry)
        local_obs = self.team.local_policy_data(obs)
        local_carry, action, output = super().policy(local_carry, local_obs, mode)
        return (
            self.team.unfold_tree_batch(local_carry),
            self.team.unfold_tree_batch(action),
            self.team.unfold_tree_batch(output),
        )

    def train(self, carry, data):
        local_carry = self.team.fold_tree_batch(carry)
        local_data = self.team.local_sequence_data(data)
        local_carry, output, metrics = super().train(local_carry, local_data)
        if "replay" in output:
            output = dict(
                output,
                replay=self.team.unfold_replay_updates(output["replay"]),
            )
        return self.team.unfold_tree_batch(local_carry), output, metrics

    def report(self, carry, data):
        local_carry = self.team.fold_tree_batch(carry)
        local_data = self.team.local_sequence_data(data)
        local_carry, metrics = super().report(local_carry, local_data)
        return self.team.unfold_tree_batch(local_carry), metrics


class MARLCore(TeamAxisAdapter, LocalAgent):
    """Shared local control with joint-action-conditioned world dynamics."""

    def __init__(self, obs_space, act_space, config):
        self.team = TeamAxis(int(config.num_agents))
        local_obs_space = local_observation_spaces(obs_space, self.team.size)
        local_act_space = local_action_spaces(act_space, self.team.size)
        super().__init__(
            local_obs_space,
            local_act_space,
            config,
        )

    def observe_dynamics(self, carry, tokens, action, reset, obs, training, single):
        return self.dyn.observe(
            carry,
            tokens,
            action,
            reset,
            training,
            single=single,
            active=self._active(obs),
        )

    def dynamics_loss(self, carry, tokens, actions, reset, obs, training):
        return self.dyn.loss(
            carry,
            tokens,
            actions,
            reset,
            training,
            active=self._active(obs),
        )

    def dynamics_replay_entry_space(self):
        return dict(super().dynamics_replay_entry_space(), active=elements.Space(bool))

    def policy_dynamics_replay_entries(self, entries):
        return dict(
            super().policy_dynamics_replay_entries(entries),
            active=entries["active"],
        )

    def dynamics_replay_entries(self, entries):
        return dict(super().dynamics_replay_entries(entries), active=entries["active"])

    def truncate_dynamics_replay(self, entries, carry):
        return self.dyn.truncate(entries, carry, active=entries["active"])

    def imagination_starts(
        self,
        dyn_entries,
        dyn_carry,
        repfeat,
        obs,
        starts_count,
    ):
        starts, first, _ = super().imagination_starts(
            dyn_entries, dyn_carry, repfeat, obs, starts_count
        )
        grouped = self.team.group_tree_starts(starts, starts_count)
        starts = self.team.fold_tree_batch(grouped)
        first = self.team.fold_tree_batch(
            self.team.group_tree_starts(first, starts_count)
        )
        active = self._active(obs)[:, -starts_count:].reshape((-1,))
        active = self.team.fold_batch(self.team.group_starts(active, starts_count))
        return starts, first, (starts_count, active)

    def imagine(self, starts, policy, horizon, training, context=None):
        starts_count, active = context
        return self.dyn.imagine(
            starts,
            policy,
            horizon,
            training,
            active=active,
        )

    def restore_imagination_results(self, losses, outputs, context=None):
        starts_count, _ = context

        def restore(value):
            grouped = self.team.unfold_batch(value)
            return self.team.ungroup_starts(grouped, starts_count)

        return jax.tree.map(restore, (losses, outputs))

    def report_imagination(self, carry, actions, length, training):
        return self.dyn.imagine(
            carry,
            actions,
            length,
            training,
        )

    @staticmethod
    def _active(obs):
        active = jnp.ones_like(obs["is_first"], bool)
        for key in ("agent_present", "agent_alive"):
            if key in obs:
                active &= obs[key].astype(bool)
        return active


__all__ = ["MARLCore", "TeamAxisAdapter"]
