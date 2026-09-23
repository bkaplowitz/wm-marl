"""Joint latent rollouts consumed by the decentralized PPO actor."""

from __future__ import annotations

import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj

from ..training.ctde import (
    imagined_action_mask,
    shared_team_outcomes,
)


class JointImagination:
    """Joint latent rollouts consumed by the decentralized PPO actor."""

    def imagination_starts(
        self,
        dyn_entries,
        dyn_carry,
        repfeat,
        obs,
        prevact,
        starts_count,
    ):
        batch = dyn_entries["deter"].shape[0]
        starts = self.dyn.starts(dyn_entries, dyn_carry, starts_count)
        first = jax.tree.map(
            lambda value: value[:, -starts_count:].reshape(
                (batch * starts_count, 1, *value.shape[2:])
            ),
            repfeat,
        )
        grouped = self.team.group_tree_starts(starts, starts_count)
        starts = self.team.fold_tree_batch(grouped)
        first = self.team.fold_tree_batch(
            self.team.group_tree_starts(first, starts_count)
        )
        active = self._active(obs)[:, -starts_count:].reshape((-1,))
        grouped_active = self.team.group_starts(active, starts_count).astype(bool)
        last = obs["is_last"][:, -starts_count:].reshape((-1,))
        grouped_last = self.team.group_starts(last, starts_count).astype(bool)
        grouped_active &= ~grouped_last
        present = self._present(obs)[:, -starts_count:].reshape((-1,))
        grouped_present = self.team.group_starts(present, starts_count).astype(bool)
        alive = self._controllable(obs)[:, -starts_count:].reshape((-1,))
        grouped_alive = self.team.group_starts(alive, starts_count).astype(bool)
        grouped_present &= ~grouped_last
        grouped_alive &= ~grouped_last
        action_mask = obs["action_mask"][:, -starts_count:]
        action_mask = action_mask.reshape((-1, *action_mask.shape[2:]))
        grouped_mask = self.team.group_starts(action_mask, starts_count).astype(bool)
        noop = jnp.zeros_like(grouped_mask).at[..., 0].set(True)
        grouped_mask = jnp.where(grouped_alive[..., None], grouped_mask, noop)
        grouped_state = self.team.unfold_sequence(self.feat2tensor(repfeat))
        grouped_action = self.team.unfold_sequence(
            prevact[self.ctde_action_key]
        ).astype(jnp.int32)
        grouped_history_present = self.team.unfold_sequence(self._present(obs)).astype(
            bool
        )
        grouped_history_alive = self.team.unfold_sequence(
            self._controllable(obs)
        ).astype(bool)
        grouped_first = self.team.unfold_sequence(obs["is_first"]).any(axis=-1)
        batch = grouped_state.shape[0]
        initial_joint = dyn_entries["ctde_joint_carry"]
        fresh_joint = self.ctde_joint.initial(batch, self.team.size)
        if grouped_state.shape[1] > 1:
            _, _, snapshots = self.ctde_joint.sequence(
                initial_joint,
                grouped_state[:, :-1],
                grouped_action[:, 1:],
                grouped_history_present[:, :-1],
                grouped_history_alive[:, :-1],
                grouped_first[:, :-1],
                training=False,
            )
            joint_history = {
                key: jnp.concatenate(
                    [initial_joint[key][:, None], snapshots[key]], axis=1
                )
                for key in initial_joint
            }
        else:
            joint_history = {
                key: value[:, None] for key, value in initial_joint.items()
            }
        folded_first = self.team.fold_sequence(
            jnp.broadcast_to(
                grouped_first[:, :, None],
                (*grouped_first.shape, self.team.size),
            )
        )
        joint_history = {
            key: jnp.where(
                folded_first.reshape((*folded_first.shape, *((1,) * (value.ndim - 2)))),
                jnp.broadcast_to(fresh_joint[key][:, None], value.shape),
                value,
            )
            for key, value in joint_history.items()
        }
        joint_carry = {
            key: self.team.fold_batch(
                self.team.group_starts(
                    value[:, -starts_count:].reshape((-1, *value.shape[2:])),
                    starts_count,
                )
            )
            for key, value in joint_history.items()
        }
        return (
            starts,
            first,
            {
                "starts_count": starts_count,
                "present": grouped_present,
                "controllable_alive": grouped_alive,
                "action_mask": grouped_mask,
                "joint_carry": joint_carry,
                "reset": grouped_first[:, -starts_count:].reshape(-1),
            },
        )
        active = self.team.fold_batch(grouped_active)
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

    def _ctde_complete(self, cache, deter, prediction):
        """Complete a local temporal step using the configured joint simulator."""
        return self.dyn.complete_from_observation(
            cache, deter, self.team.fold_batch(prediction["embedding"]), sample=True
        )

    def imagine_with_aux(self, starts, policy, horizon, training, context=None):
        del policy
        grouped_carry = nn.cast(self.team.unfold_tree_batch(starts))
        present = context["present"].astype(bool)
        alive = context["controllable_alive"].astype(bool)
        action_mask = context["action_mask"].astype(bool)
        teams, agents = present.shape
        central_carry = context["joint_carry"]
        reset = context["reset"].astype(bool)

        def transition(state, _):
            (
                local_carry,
                joint_carry,
                current_present,
                current_alive,
                current_mask,
                current_reset,
            ) = state
            local_features = {
                "deter": local_carry["deter"],
                "stoch": local_carry["stoch"],
            }
            folded_features = self.team.fold_tree_batch(local_features)
            folded_mask = self.team.fold_batch(current_mask)
            distribution = self.policy_distribution(
                self.feat2tensor(folded_features),
                1,
                action_mask=folded_mask,
            )
            # The standard CTDE path has exactly one categorical action. Use
            # its existing draw key for a separate availability substream, so
            # enabling the treatment does not shift other learner RNG draws.
            action_seed = nj.seed()
            folded_action = {
                self.ctde_action_key: distribution[self.ctde_action_key].sample(
                    action_seed
                )
            }
            grouped_action = self.team.unfold_batch(folded_action[self.ctde_action_key])

            folded_carry = self.team.fold_tree_batch(local_carry)
            folded_present = self.team.fold_batch(current_present)
            local_cache, deter = self.dyn.advance(
                folded_carry,
                folded_action,
                training,
                active=folded_present,
            )
            grouped_state = self.team.unfold_batch(self.feat2tensor(folded_features))
            joint_carry, prediction = self.ctde_joint.step(
                joint_carry,
                grouped_state,
                grouped_action,
                current_present,
                current_alive,
                current_reset,
                training=False,
            )
            folded_next, next_features = self._ctde_complete(
                local_cache, deter, prediction
            )
            next_carry = self.team.unfold_tree_batch(folded_next)
            next_features = self.team.unfold_tree_batch(next_features)

            hidden = prediction["hidden"]
            alive_probability = self.ctde_alive(hidden, 2).prob(1)
            next_present = current_present
            next_alive = jax.lax.stop_gradient(
                current_alive & next_present & (alive_probability >= 0.5)
            )
            reward, continuation = shared_team_outcomes(
                self.ctde_rew(hidden, 2).pred(),
                self.ctde_con(hidden, 2).prob(1),
                current_present,
                current_alive,
                next_alive,
            )
            mask_output = self.actmask(self.feat2tensor(next_features), 2)
            mask_probability = jax.nn.sigmoid(mask_output.output.logit)
            # Store the realized mask in auxiliary below. PPO must condition on
            # that same mask in every epoch, never redraw it for likelihoods.
            next_mask = imagined_action_mask(
                mask_probability,
                next_alive,
                jax.random.fold_in(action_seed, 0x4D41534B)
                if self.ctde_imagination_mask_sampling == "bernoulli"
                else None,
            )

            next_state = (
                next_carry,
                joint_carry,
                next_present,
                next_alive,
                next_mask,
                jnp.zeros_like(current_reset),
            )
            outputs = (
                next_features,
                self.team.unfold_tree_batch(folded_action),
                reward,
                continuation,
                next_mask,
                next_present,
                next_alive,
            )
            return next_state, outputs

        state = (
            grouped_carry,
            central_carry,
            present,
            alive,
            action_mask,
            reset,
        )
        state, outputs = nj.scan(
            transition,
            state,
            (),
            horizon,
            axis=1,
        )
        (
            next_features,
            actions,
            rewards,
            continuations,
            masks,
            present_sequence,
            alive_sequence,
        ) = outputs
        local_carry = self.team.fold_tree_batch(state[0])
        features = jax.tree.map(self.team.fold_sequence, next_features)
        actions = jax.tree.map(self.team.fold_sequence, actions)
        discount = 1.0 - 1.0 / float(self.config.horizon)
        root_reward = jnp.zeros((teams, 1, agents), jnp.float32)
        root_continuation = jnp.full(
            (teams, 1, agents),
            discount if self.config.contdisc else 1.0,
            jnp.float32,
        )
        auxiliary = {
            "reward": jnp.concatenate([root_reward, rewards], axis=1),
            "continuation": jnp.concatenate([root_continuation, continuations], axis=1),
            "action_mask": jnp.concatenate([action_mask[:, None], masks], axis=1),
            "present": jnp.concatenate([present[:, None], present_sequence], axis=1),
            "controllable_alive": jnp.concatenate(
                [alive[:, None], alive_sequence], axis=1
            ),
        }
        return local_carry, features, actions, auxiliary

    def imagination_policy_distribution(self, policy_inputs, auxiliary):
        action_mask = self.team.fold_sequence(auxiliary["action_mask"])
        return self.policy_distribution(policy_inputs, 2, action_mask=action_mask)

    def imagination_action_mask(self, auxiliary):
        action_mask = self.team.fold_sequence(auxiliary["action_mask"]).astype(bool)
        if action_mask.shape[1] < 2:
            raise ValueError("CTDE PPO action mask must include at least one decision")
        return action_mask[:, :-1]

    def imagination_reward_continuation(self, local_inputs, auxiliary):
        del local_inputs
        return (
            self.team.fold_sequence(auxiliary["reward"]),
            self.team.fold_sequence(auxiliary["continuation"]),
        )

    def imagination_state_validity(self, context, horizon, auxiliary=None):
        """Mask PPO decisions after an agent becomes uncontrollable."""

        if auxiliary is None:
            raise ValueError("CTDE PPO imagination requires predicted activity")
        present = auxiliary["present"].astype(bool)
        controllable = auxiliary["controllable_alive"]
        if not jnp.issubdtype(controllable.dtype, jnp.bool_):
            controllable = controllable >= 0.5
        validity = present & controllable.astype(bool)
        folded = self.team.fold_sequence(validity)
        expected = horizon + 1
        if folded.shape[1] != expected:
            raise ValueError(
                "CTDE PPO validity must include root and bootstrap states: "
                f"expected {expected}, got {folded.shape[1]}"
            )
        return folded

    def imagination_bootstrap_validity(self, context, horizon, auxiliary=None):
        """Keep shared team returns alive after a focal unit dies in SMAC."""

        if auxiliary is None:
            raise ValueError("CTDE PPO imagination requires predicted roster")
        present = self.team.fold_sequence(auxiliary["present"].astype(bool))
        if present.shape[1] != horizon + 1:
            raise ValueError("CTDE bootstrap roster must include root and final state")
        return present

    def imagination_behavior_metrics(self, actions, validity, auxiliary=None):
        """Summarize the actions that actually drive CTDE imagination."""

        del auxiliary
        action = actions[self.ctde_action_key]
        weight = (
            jnp.ones_like(action, jnp.float32)
            if validity is None
            else validity[:, : action.shape[1]].astype(jnp.float32)
        )
        count = jnp.maximum(weight.sum(), 1.0)

        def fraction(selected):
            return (weight * selected.astype(jnp.float32)).sum() / count

        attack_start = min(6, self.ctde_action_count)
        return {
            "imagined_action/bernoulli_availability": jnp.asarray(
                self.ctde_imagination_mask_sampling == "bernoulli", jnp.float32
            ),
            "imagined_action/noop_fraction": fraction(action == 0),
            "imagined_action/stop_fraction": fraction(action == 1),
            "imagined_action/move_fraction": fraction(
                (action >= 2) & (action < attack_start)
            ),
            "imagined_action/attack_fraction": fraction(action >= attack_start),
        }

    def imagination_critic_context(self, features, context, auxiliary=None):
        if auxiliary is None:
            raise ValueError("CTDE critic requires imagined activity")
        metrics = {}
        return {
            "present": auxiliary["present"],
            "controllable_alive": auxiliary["controllable_alive"],
        }, metrics

    @staticmethod
    def _active(obs):
        active = jnp.ones_like(obs["is_first"], bool)
        for key in ("agent_present", "agent_alive"):
            if key in obs:
                active &= obs[key].astype(bool)
        return active

    @staticmethod
    def _present(obs):
        if "agent_present" in obs:
            return obs["agent_present"].astype(bool)
        return jnp.ones_like(obs["is_first"], bool)

    @classmethod
    def _controllable(cls, obs):
        if "controllable_alive" in obs:
            return cls._present(obs) & obs["controllable_alive"].astype(bool)
        return cls._active(obs)
