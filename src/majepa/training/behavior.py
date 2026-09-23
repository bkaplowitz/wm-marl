"""Reconstruct independent behavior roots and build frozen PPO targets."""

import elements
import jax
import jax.numpy as jnp

from .common import concat, sample, sg
from .ppo import (
    clipped_policy_objective,
    generalized_advantage_estimate,
    masked_weighted_mean,
    normalize_advantage,
    value_objective,
)
from .replay_value import replay_lambda_return


class BehaviorLearning:
    """Reconstruct independent behavior roots and build frozen PPO targets."""

    def _apply_behavior_replay_context(self, data):
        """Burn in an independent replay prefix with current world weights."""

        context = int(self.config.replay_context)
        batch, total_length = data["is_first"].shape
        if context < 1 or total_length <= context:
            raise ValueError(
                "behavior replay must contain a non-empty prefix and suffix, got "
                f"context={context}, length={total_length}"
            )
        enc_carry, dyn_carry, dec_carry, initial_prevact = self._local_initial(batch)
        obs = self._replay_observations(data)

        def prepend(initial, sequence):
            return jnp.concatenate([initial[:, None], sequence[:, :-1]], axis=1)

        prevact = {
            key: prepend(initial_prevact[key], data[key]) for key in self.act_space
        }
        nested = elements.tree.nestdict(data)
        stored_dyn = nested.get("dyn", {})
        if "position" not in stored_dyn:
            raise ValueError(
                "behavior replay burn-in requires stored Transformer positions"
            )
        position = stored_dyn["position"][:, 0].astype(jnp.int32) - 1
        position = jnp.where(obs["is_first"][:, 0], -jnp.ones_like(position), position)
        dyn_carry = dict(dyn_carry, position=position)

        prefix_obs = {key: value[:, :context] for key, value in obs.items()}
        prefix_prevact = {key: value[:, :context] for key, value in prevact.items()}
        reset = prefix_obs["is_first"]
        enc_carry, _, tokens = self.enc(enc_carry, prefix_obs, reset, training=False)
        dyn_carry, dyn_entries, prefix_features, _ = self.observe_dynamics(
            dyn_carry,
            tokens,
            prefix_prevact,
            reset,
            prefix_obs,
            training=False,
            single=False,
        )
        carry = jax.tree.map(sg, (enc_carry, dyn_carry, dec_carry))
        suffix_obs = {key: value[:, context:] for key, value in obs.items()}
        suffix_obs = self.behavior_replay_burnin_observation(
            suffix_obs,
            jax.tree.map(sg, prefix_features),
            jax.tree.map(sg, dyn_entries),
            prefix_obs,
            prefix_prevact,
            {key: data[key][:, :context] for key in self.act_space},
        )
        suffix_prevact = {key: data[key][:, context - 1 : -1] for key in self.act_space}
        suffix_stepid = data["stepid"][:, context:]
        return carry, suffix_obs, suffix_prevact, suffix_stepid

    def behavior_replay_burnin_observation(
        self,
        suffix_obs,
        prefix_features,
        prefix_dyn_entries,
        prefix_obs,
        prefix_prevact,
        prefix_action,
    ):
        del (
            prefix_features,
            prefix_dyn_entries,
            prefix_obs,
            prefix_prevact,
            prefix_action,
        )
        return suffix_obs

    def _behavior_model_states(self, carry, obs, prevact):
        """Infer mode-stable current-weight states without world losses.

        The maintained local encoder and dynamics have no train/eval-only
        stochastic layers. Evaluation mode therefore retains posterior sampling
        while avoiding training-mode variation in the independent uniform-root
        reconstruction.
        """

        enc_carry, dyn_carry, dec_carry = carry
        reset = obs["is_first"]
        enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training=False)
        dyn_carry, dyn_entries, repfeat, _ = self.observe_dynamics(
            dyn_carry,
            tokens,
            prevact,
            reset,
            obs,
            training=False,
            single=False,
        )
        dyn_entries = self.behavior_dynamics_entries(dyn_entries, obs)
        dec_entries = {}
        return jax.tree.map(
            sg,
            (
                (enc_carry, dyn_carry, dec_carry),
                (enc_entries, dyn_entries, dec_entries),
                repfeat,
            ),
        )

    def behavior_dynamics_entries(self, entries, obs):
        del obs
        return entries

    def _prepare_ppo_batch(self, behavior_data, entropy_coefficient):
        """Create one immutable PPO batch after the JEPA model update."""

        behavior_carry, obs, prevact, _ = self._apply_behavior_replay_context(
            behavior_data
        )
        behavior_carry, behavior_entries, repfeat = self._behavior_model_states(
            behavior_carry,
            obs,
            prevact,
        )
        _, dyn_carry, _ = behavior_carry
        _, dyn_entries, _ = behavior_entries
        _, length = obs["is_first"].shape
        starts_count = min(self.config.imag_last or length, length)
        horizon = int(self.config.imag_length)
        starts, first, imagination_context = self.imagination_starts(
            dyn_entries,
            dyn_carry,
            repfeat,
            obs,
            prevact,
            starts_count,
        )

        def policyfn(features):
            inputs = self.feat2tensor(features)
            return sample(self.policy_distribution(inputs, 1))

        _, imagined_features, actions, auxiliary = self.imagine_with_aux(
            starts,
            policyfn,
            horizon,
            False,
            imagination_context,
        )
        first, imagined_features, actions, auxiliary = jax.tree.map(
            sg, (first, imagined_features, actions, auxiliary)
        )
        features = concat([first, imagined_features], 1)
        policy_features = self.imagination_policy_features(features)
        policy_inputs = self.feat2tensor(policy_features)
        local_inputs = self.feat2tensor(features)
        critic_context, critic_metrics = self.imagination_critic_context(
            features,
            imagination_context,
            auxiliary,
        )
        if critic_context is None:
            raise ValueError("MA-JEPA PPO requires centralized critic context")

        policy = self.imagination_policy_distribution(policy_inputs, auxiliary)
        action_key = self.action_mask_key
        if action_key is None or set(actions) != {action_key}:
            raise ValueError("MA-JEPA PPO requires exactly one masked action")
        old_logits = policy[action_key].logits[:, :-1]
        action = actions[action_key].astype(jnp.int32)
        if action.shape != old_logits.shape[:-1]:
            raise ValueError(
                "imagined actions and policy logits are misaligned: "
                f"{action.shape} and {old_logits.shape}"
            )

        reward, continuation = self.imagination_reward_continuation(
            local_inputs,
            auxiliary,
        )
        decision_state_valid = self.imagination_state_validity(
            imagination_context,
            horizon,
            auxiliary,
        )
        state_valid = self.imagination_bootstrap_validity(
            imagination_context,
            horizon,
            auxiliary,
        )
        # Build the live critic first. SlowModel initializes itself by copying
        # its source parameters and therefore requires the source to exist.
        current_value = self.critic(
            features,
            2,
            slow=False,
            context=critic_context,
        ).pred()
        target_value = self.critic(
            features,
            2,
            slow=True,
            context=critic_context,
        ).pred()
        predicted_action_mask = self.imagination_action_mask(auxiliary)
        if predicted_action_mask.shape != old_logits.shape:
            raise ValueError(
                "PPO action mask and logits are misaligned: "
                f"{predicted_action_mask.shape} and {old_logits.shape}"
            )
        # The applied policy is the source of truth because action masking has
        # a deterministic nonempty fallback. Freezing this effective support
        # guarantees that every PPO epoch assigns finite probability to every
        # action that could have been sampled by the behavior snapshot.
        action_mask = old_logits > -1e20
        sampled_legal = jnp.take_along_axis(
            action_mask,
            action[..., None],
            axis=-1,
        )[..., 0]
        target_return, advantage, valid, trajectory_weight = (
            generalized_advantage_estimate(
                reward,
                continuation,
                target_value,
                state_valid,
                decision_state_valid=decision_state_valid,
                lam=float(self.config.ppo.lam),
            )
        )
        advantage_scale = jnp.asarray(1.0, jnp.float32)
        if self.ppo_return_norm is None:
            advantage = normalize_advantage(advantage, valid, trajectory_weight)
        else:
            _, advantage_scale = self.ppo_return_norm(target_return, True, valid)
            advantage = jnp.where(valid, advantage / sg(advantage_scale), 0.0)

        def decisions(tree):
            return jax.tree.map(lambda value: value[:, :-1], tree)

        batch = sg(
            {
                "policy_inputs": policy_inputs[:, :-1],
                "critic_features": decisions(features),
                "critic_context": decisions(critic_context),
                "action": action,
                "action_mask": action_mask,
                "old_logits": old_logits,
                "advantage": advantage,
                "target_return": target_return,
                "valid": valid,
                "critic_valid": state_valid[:, :-1],
                "trajectory_weight": trajectory_weight,
                "entropy_coefficient": entropy_coefficient,
            }
        )
        if float(self.config.ppo.replay_value_scale):
            batch["replay_value"] = self._prepare_replay_value_batch(
                repfeat, obs, target_return[:, 0], starts_count
            )
        metrics = {
            "ppo/return_percentile_scale": advantage_scale,
            **critic_metrics,
            **self.imagination_interface_metrics(features, policy_features),
            **self.imagination_behavior_metrics(actions, valid, auxiliary),
            "replay_views/behavior_reward_mean": sg(
                obs["reward"].astype(jnp.float32).mean()
            ),
            "replay_views/behavior_rows": jnp.asarray(
                obs["is_first"].shape[0], jnp.float32
            ),
            "replay_views/behavior_length": jnp.asarray(length, jnp.float32),
            "ppo/batch_reward": masked_weighted_mean(
                reward[:, 1:], valid, trajectory_weight
            ),
            "ppo/batch_return": masked_weighted_mean(
                target_return, valid, trajectory_weight
            ),
            "ppo/batch_target_value": masked_weighted_mean(
                target_value[:, :-1], valid, trajectory_weight
            ),
            "ppo/batch_current_value": masked_weighted_mean(
                current_value[:, :-1], valid, trajectory_weight
            ),
            "ppo/batch_valid_fraction": valid.astype(jnp.float32).mean(),
            "ppo/batch_effective_weight": trajectory_weight.mean(),
            "ppo/batch_illegal_action_fraction": (
                valid.astype(jnp.float32) * (~sampled_legal).astype(jnp.float32)
            ).sum()
            / jnp.maximum(valid.astype(jnp.float32).sum(), 1.0),
            "ppo/batch_support_fallback_fraction": (
                action_mask != predicted_action_mask
            )
            .any(axis=-1)
            .astype(jnp.float32)
            .mean(),
        }
        return batch, metrics

    def _prepare_replay_value_batch(self, features, obs, root_return, starts_count):
        """Anchor the critic in real rewards with fresh imagined bootstraps.

        Replay actions may be off policy. This is an auxiliary critic target,
        never a PPO actor sample; its weight is configured independently. The
        imagined root returns supply current-policy bootstraps, as in the
        replay-value objective used before the PPO migration.
        """

        features = jax.tree.map(lambda value: value[:, -starts_count:], features)
        selected = {
            key: obs[key][:, -starts_count:]
            for key in ("reward", "is_first", "is_last", "is_terminal", "agent_present")
        }
        # Imagination is ordered [team, start, agent]; replay is [team, agent,
        # time]. A plain reshape silently assigns other agents' bootstraps.
        bootstrap = self.team.ungroup_starts(
            self.team.unfold_batch(root_return), starts_count
        ).reshape(selected["reward"].shape)
        context = {
            "present": self.team.unfold_sequence(selected["agent_present"]),
            "controllable_alive": self.team.unfold_sequence(
                self._controllable(obs)[:, -starts_count:]
            ),
        }
        # Imagination excludes all is_last roots. A nonterminal truncation
        # must instead bootstrap from its real final observation and roster.
        factual_value = self.critic(features, 2, slow=True, context=context).pred()
        bootstrap = jnp.where(
            selected["is_last"] & ~selected["is_terminal"], factual_value, bootstrap
        )
        targets, valid = replay_lambda_return(
            selected["reward"],
            selected["is_first"],
            selected["is_last"],
            selected["is_terminal"],
            selected["agent_present"],
            bootstrap,
            discount=1.0 - 1.0 / float(self.config.horizon),
            lam=float(self.config.ppo.replay_value_lam),
        )
        return sg(
            {
                "features": jax.tree.map(lambda value: value[:, :-1], features),
                "context": jax.tree.map(lambda value: value[:, :-1], context),
                "target_return": targets,
                "valid": valid,
            }
        )

    def _ppo_actor_loss(self, batch):
        policy = self.policy_distribution(
            batch["policy_inputs"],
            2,
            action_mask=batch["action_mask"],
        )
        new_logits = policy[self.action_mask_key].logits
        return clipped_policy_objective(
            new_logits,
            batch["old_logits"],
            batch["action"],
            batch["advantage"],
            batch["valid"],
            batch["trajectory_weight"],
            clip_epsilon=float(self.config.ppo.clip_epsilon),
            entropy_coefficient=batch["entropy_coefficient"],
            normalize_entropy=False,
        )

    def _ppo_critic_loss(self, batch):
        value = self.critic(
            batch["critic_features"],
            2,
            slow=False,
            context=batch["critic_context"],
        )
        loss, metrics = value_objective(
            value,
            batch["target_return"],
            batch["critic_valid"],
            batch["trajectory_weight"],
        )
        slowreg = float(self.config.ppo.get("critic_slowreg", 0.0))
        if slowreg:
            slow_prediction = sg(
                self.critic(
                    batch["critic_features"],
                    2,
                    slow=True,
                    context=batch["critic_context"],
                ).pred()
            )
            anchor_loss, _ = value_objective(
                value,
                slow_prediction,
                batch["critic_valid"],
                batch["trajectory_weight"],
            )
            loss += slowreg * anchor_loss
            metrics["slow_anchor_loss"] = anchor_loss
        if "replay_value" in batch:
            replay = batch["replay_value"]
            replay_value = self.critic(
                replay["features"], 2, slow=False, context=replay["context"]
            )
            replay_loss, replay_metrics = value_objective(
                replay_value,
                replay["target_return"],
                replay["valid"],
                jnp.ones_like(replay["target_return"]),
            )
            if slowreg:
                slow_prediction = sg(
                    self.critic(
                        replay["features"],
                        2,
                        slow=True,
                        context=replay["context"],
                    ).pred()
                )
                anchor_loss, _ = value_objective(
                    replay_value,
                    slow_prediction,
                    replay["valid"],
                    jnp.ones_like(replay["target_return"]),
                )
                replay_loss += slowreg * anchor_loss
                replay_metrics["slow_anchor_loss"] = anchor_loss
            loss += float(self.config.ppo.replay_value_scale) * replay_loss
            metrics.update(
                {f"replay_{key}": value for key, value in replay_metrics.items()}
            )
            metrics.update(
                {f"factual_{k}": v for k, v in replay.get("trace_metrics", {}).items()}
            )
        metrics["total_loss"] = loss
        return loss, metrics

    def imagination_policy_features(self, features):
        return features

    def imagination_interface_metrics(self, model_features, policy_features):
        del model_features, policy_features
        return {}
