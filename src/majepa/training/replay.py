"""Recurrent replay-context handling for the shared-local learner."""

from ..world_model import replay_entries


class ReplayMixin:
    def _replay_observations(self, data):
        obs = {key: data[key] for key in self.obs_space}
        return obs

    def dynamics_replay_entry_space(self):
        return self.dyn.entry_space

    def policy_dynamics_replay_entries(self, entries):
        return replay_entries(entries)

    def dynamics_replay_entries(self, entries):
        return replay_entries(entries)
