"""Recurrent replay-context handling for the shared-local learner."""

import elements
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp


class ReplayMixin:
    def dynamics_replay_entry_space(self):
        return self.dyn.entry_space

    def policy_dynamics_replay_entries(self, entries):
        return self.world_model.replay_entries(entries)

    def dynamics_replay_entries(self, entries):
        return self.world_model.replay_entries(entries)

    def truncate_dynamics_replay(self, entries, carry):
        return self.dyn.truncate(entries, carry)

    def _apply_replay_context(self, carry, data):
        enc_carry, dyn_carry, dec_carry, prevact = carry
        carry = (enc_carry, dyn_carry, dec_carry)
        stepid = data["stepid"]
        obs = {key: data[key] for key in self.obs_space}

        def prepend(initial, sequence):
            return jnp.concatenate([initial[:, None], sequence[:, :-1]], 1)

        prevact = {key: prepend(prevact[key], data[key]) for key in self.act_space}
        if not self.config.replay_context:
            return carry, obs, prevact, stepid

        context = self.config.replay_context
        nested = elements.tree.nestdict(data)
        entries = [nested.get(key, {}) for key in ("enc", "dyn", "dec")]

        def lhs(xs):
            return jax.tree.map(lambda value: value[:, :context], xs)

        def rhs(xs):
            return jax.tree.map(lambda value: value[:, context:], xs)

        replay_carry = (
            self.enc.truncate(lhs(entries[0]), enc_carry),
            self.truncate_dynamics_replay(lhs(entries[1]), dyn_carry),
            (
                self.dec.truncate(lhs(entries[2]), dec_carry)
                if self.dec is not None
                else {}
            ),
        )
        replay_obs = {key: rhs(data[key]) for key in self.obs_space}
        replay_prevact = {key: data[key][:, context - 1 : -1] for key in self.act_space}
        replay_stepid = rhs(stepid)

        first_chunk = data["consec"][:, 0] == 0
        return jax.tree.map(
            lambda normal, replay: nn.where(first_chunk, replay, normal),
            (carry, rhs(obs), rhs(prevact), rhs(stepid)),
            (replay_carry, replay_obs, replay_prevact, replay_stepid),
        )
