"""Reconstruct local and joint histories from replay."""

from __future__ import annotations

import elements
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp


class TeamReplay:
    """Reconstruct local and joint histories from replay."""

    def _apply_replay_context(self, carry, data):
        if not self.ctde_enabled or not self.config.replay_context:
            return super()._apply_replay_context(carry, data)

        enc_carry, dyn_carry, dec_carry, prevact = carry
        normal_carry = (enc_carry, dyn_carry, dec_carry)
        stepid = data["stepid"]
        obs = self._replay_observations(data)

        def prepend(initial, sequence):
            return jnp.concatenate([initial[:, None], sequence[:, :-1]], 1)

        shifted_prevact = {
            key: prepend(prevact[key], data[key]) for key in self.act_space
        }
        context = int(self.config.replay_context)
        nested = elements.tree.nestdict(data)
        enc_entries = nested.get("enc", {})
        dyn_entries = nested.get("dyn", {})

        def lhs(tree):
            return jax.tree.map(lambda value: value[:, :context], tree)

        def rhs(tree):
            return jax.tree.map(lambda value: value[:, context:], tree)

        prefix_dyn = lhs(dyn_entries)
        prefix_active = prefix_dyn.get(
            "active", self._active({key: lhs(value) for key, value in obs.items()})
        )
        replay_dyn_carry, replay_features = self.dyn.replay_sequence(
            prefix_dyn,
            carry=dyn_carry,
            active=prefix_active,
        )
        replay_carry = (
            self.enc.truncate(lhs(enc_entries), enc_carry),
            replay_dyn_carry,
            {},
        )
        replay_obs = rhs(self._replay_observations(data))
        replay_prevact = {key: data[key][:, context - 1 : -1] for key in self.act_space}
        replay_stepid = rhs(stepid)
        first_chunk = data["consec"][:, 0] == 0
        selected = jax.tree.map(
            lambda normal, replay: nn.where(first_chunk, replay, normal),
            (
                normal_carry,
                rhs(obs),
                rhs(shifted_prevact),
                rhs(stepid),
            ),
            (replay_carry, replay_obs, replay_prevact, replay_stepid),
        )
        selected_carry, selected_obs, selected_prevact, selected_stepid = selected
        burnin = {
            "state": jax.lax.stop_gradient(self.feat2tensor(replay_features)),
            "action": lhs(data[self.ctde_action_key]).astype(jnp.int32),
            "present": lhs(self._present(obs)).astype(bool),
            "controllable_alive": lhs(self._controllable(obs)).astype(bool),
            "is_first": lhs(obs["is_first"]).astype(bool),
            "position": prefix_dyn["position"].astype(jnp.int32),
        }
        selected_obs = dict(selected_obs, _ctde_burnin=burnin)
        return selected_carry, selected_obs, selected_prevact, selected_stepid

    def behavior_replay_burnin_observation(
        self,
        suffix_obs,
        prefix_features,
        prefix_dyn_entries,
        prefix_obs,
        prefix_prevact,
        prefix_action,
    ):
        suffix_obs = super().behavior_replay_burnin_observation(
            suffix_obs,
            prefix_features,
            prefix_dyn_entries,
            prefix_obs,
            prefix_prevact,
            prefix_action,
        )
        burnin = {
            "state": jax.lax.stop_gradient(self.feat2tensor(prefix_features)),
            "action": prefix_action[self.ctde_action_key].astype(jnp.int32),
            "present": self._present(prefix_obs).astype(bool),
            "controllable_alive": self._controllable(prefix_obs).astype(bool),
            "is_first": prefix_obs["is_first"].astype(bool),
            "position": prefix_dyn_entries["position"].astype(jnp.int32),
        }
        return dict(suffix_obs, _ctde_burnin=burnin)

    def behavior_dynamics_entries(self, entries, obs):
        entries = super().behavior_dynamics_entries(entries, obs)
        return dict(
            entries,
            ctde_joint_carry=self._ctde_joint_burnin(entries, obs),
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

    def _ctde_joint_burnin(self, dyn_entries, obs):
        burnin = obs.get("_ctde_burnin")
        if burnin is None:
            grouped_position = self.team.unfold_sequence(dyn_entries["position"])
            grouped_first = self.team.unfold_sequence(obs["is_first"]).any(axis=-1)
            previous_position = grouped_position[:, 0] - 1
            previous_position = jnp.where(
                grouped_first[:, :1],
                -jnp.ones_like(previous_position),
                previous_position,
            )
            return self.ctde_joint.initial(
                grouped_position.shape[0],
                self.team.size,
                previous_position,
            )

        grouped_state = self.team.unfold_sequence(burnin["state"])
        grouped_action = self.team.unfold_sequence(burnin["action"])
        grouped_present = self.team.unfold_sequence(burnin["present"])
        grouped_alive = self.team.unfold_sequence(burnin["controllable_alive"])
        grouped_first = self.team.unfold_sequence(burnin["is_first"]).any(axis=-1)
        grouped_position = self.team.unfold_sequence(burnin["position"])
        previous_position = grouped_position[:, 0] - 1
        previous_position = jnp.where(
            grouped_first[:, :1],
            -jnp.ones_like(previous_position),
            previous_position,
        )
        cache = self.ctde_joint.initial(
            grouped_state.shape[0],
            self.team.size,
            previous_position,
        )
        cache, _, _ = self.ctde_joint.sequence(
            cache,
            grouped_state,
            grouped_action,
            grouped_present,
            grouped_alive,
            grouped_first,
            training=False,
        )
        return jax.lax.stop_gradient(cache)

    def dynamics_loss(self, carry, tokens, actions, reset, obs, training):
        result = self.dyn.loss(
            carry,
            tokens,
            actions,
            reset,
            training,
            active=self._active(obs),
        )
        carry, entries, losses, features, metrics, auxiliary = result
        entries = dict(
            entries,
            ctde_joint_carry=self._ctde_joint_burnin(entries, obs),
        )
        return carry, entries, losses, features, metrics, auxiliary

    def dynamics_replay_entry_space(self):
        return dict(
            super().dynamics_replay_entry_space(),
            active=elements.Space(bool),
        )

    def policy_dynamics_replay_entries(self, entries):
        return dict(
            super().policy_dynamics_replay_entries(entries),
            active=entries["active"],
        )

    def dynamics_replay_entries(self, entries):
        return dict(
            super().dynamics_replay_entries(entries),
            active=entries["active"],
        )

    def truncate_dynamics_replay(self, entries, carry):
        return self.dyn.truncate(entries, carry, active=entries["active"])
