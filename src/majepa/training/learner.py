"""World-model and imagined PPO training orchestration."""

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from ..models.heads import binary_vector_loss
from .common import concat, f32, sample, sg
from .ppo import (
    clipped_policy_objective,
    generalized_advantage_estimate,
    masked_weighted_mean,
    normalize_advantage,
    scheduled_entropy_coefficient,
    value_objective,
)
from .representation import (
    embedding_prediction_loss,
    embedding_std,
    mask_image_patches,
    masked_spatial_loss,
    sigreg_loss,
    spatial_patch_mask,
)


def masked_mean(value, valid, *, alignment="tail"):
    """Average a per-transition loss over valid local agent transitions."""

    if alignment == "tail":
        weight = valid[:, -value.shape[1] :]
    elif alignment == "replay_value":
        start = valid.shape[1] - value.shape[1] - 1
        weight = valid[:, start : start + value.shape[1]]
    elif alignment == "exact":
        if valid.shape != value.shape:
            raise ValueError(
                f"validity {valid.shape} does not match loss {value.shape}"
            )
        weight = valid
    else:
        raise ValueError(f"unknown validity alignment: {alignment!r}")
    weight = weight.astype(jnp.float32)
    return (value * weight).mean() / jnp.maximum(weight.mean(), 1e-8)


class LearnerMixin:
    def train(self, carry, data, behavior_data=None):
        if not self.two_branch_replay or behavior_data is None:
            raise ValueError(
                "MA-JEPA PPO requires independent world and behavior replay batches"
            )
        if int(self.config.replay_context) < 1:
            raise ValueError("MA-JEPA PPO requires replay_context burn-in")
        if not hasattr(self.opt, "step_group"):
            raise ValueError("MA-JEPA PPO requires the separated CTDE optimizer")

        ppo_active = self._ppo_schedule(data)
        carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
        metrics, (carry, entries, outs, mets) = self.opt(
            self.loss,
            carry,
            obs,
            prevact,
            training=True,
            has_aux=True,
            skip_groups=("actor", "critic"),
        )
        metrics.update(mets)

        # This is deliberately after the world-model optimizer step. PPO sees
        # the newest JEPA dynamics, and its immutable behavior snapshot cannot
        # be invalidated by a simultaneous teammate/world update.
        entropy_coefficient = self._ppo_entropy_coefficient(data)
        ppo_batch, batch_metrics = self._prepare_ppo_batch(
            behavior_data,
            entropy_coefficient,
        )
        metrics.update(batch_metrics)
        actor_epochs = []
        critic_epochs = []
        for _ in range(int(self.config.ppo.epochs)):
            actor_optimizer, actor_metrics = self.opt.step_group(
                "actor",
                self._ppo_actor_loss,
                ppo_batch,
                has_aux=True,
                active=ppo_active,
            )
            critic_optimizer, critic_metrics = self.opt.step_group(
                "critic",
                self._ppo_critic_loss,
                ppo_batch,
                has_aux=True,
                active=ppo_active,
            )
            actor_epochs.append(actor_metrics)
            critic_epochs.append(critic_metrics)
        metrics.update(actor_optimizer)
        metrics.update(critic_optimizer)
        metrics.update(self._ppo_epoch_metrics("actor", actor_epochs))
        metrics.update(self._ppo_epoch_metrics("critic", critic_epochs))
        metrics["ppo/epochs"] = jnp.asarray(self.config.ppo.epochs, jnp.float32)
        metrics["ppo/active"] = ppo_active.astype(jnp.float32)
        metrics["ppo/entropy_coefficient"] = entropy_coefficient

        self._update_slow_models(ppo_active)
        if self.slowenc is not None:
            self.slowenc.update()
        if self.ppo_start_step:
            environment_step = data["_environment_step"].reshape(-1)[0]
            metrics.update(
                {
                    "schedule/environment_step": environment_step,
                    "schedule/ppo_start_step": jnp.asarray(
                        self.ppo_start_step, jnp.int32
                    ),
                    "schedule/world_model_active": jnp.asarray(1.0, jnp.float32),
                    "schedule/ppo_active": ppo_active.astype(jnp.float32),
                    "schedule/world_only_active": (~ppo_active).astype(jnp.float32),
                }
            )
        outs = {}
        if self.config.replay_context:
            replay_entries = dict(
                stepid=stepid,
                enc=entries[0],
                dyn=self.dynamics_replay_entries(entries[1]),
            )
            if self.dec is not None:
                replay_entries["dec"] = entries[2]
            updates = elements.tree.flatdict(replay_entries)
            outs["replay"] = updates
        carry = (*carry, {key: data[key][:, -1] for key in self.act_space})
        return carry, outs, metrics

    @staticmethod
    def _ppo_epoch_metrics(group, epochs):
        metrics = {
            f"ppo/{group}/{key}": jnp.stack([epoch[key] for epoch in epochs]).mean()
            for key in epochs[0]
        }
        metrics.update(
            {f"ppo/{group}/final_{key}": value for key, value in epochs[-1].items()}
        )
        return metrics

    def _ppo_schedule(self, data):
        """Return whether proximal behavior updates are past their warm-up."""

        start = int(self.ppo_start_step)
        if not start:
            return jnp.asarray(True)
        if "_environment_step" not in data:
            raise ValueError("PPO warm-up requires _environment_step")
        environment_step = data["_environment_step"]
        if environment_step.ndim != 2:
            raise ValueError(
                "folded _environment_step must be [B*A,T], got "
                f"{environment_step.shape}"
            )
        return environment_step.reshape(-1)[0].astype(jnp.int32) >= start

    def _ppo_entropy_coefficient(self, data):
        schedule = self.config.ppo.entropy_schedule
        if not bool(schedule.enabled):
            return jnp.asarray(self.config.ppo.entropy_coefficient, jnp.float32)
        if "_environment_step" not in data:
            raise ValueError("PPO entropy annealing requires _environment_step")
        environment_step = data["_environment_step"]
        if environment_step.ndim != 2:
            raise ValueError(
                "folded _environment_step must be [B*A,T], got "
                f"{environment_step.shape}"
            )
        return scheduled_entropy_coefficient(
            environment_step.reshape(-1)[0],
            initial=float(schedule.initial),
            final=float(schedule.final),
            decay_steps=int(schedule.decay_steps),
            schedule=str(schedule.schedule),
        )

    def loss(
        self,
        carry,
        obs,
        prevact,
        training,
    ):
        model_carry, entries, tokens, repfeat, losses, metrics, target_tokens = (
            self._world_model_terms(carry, obs, prevact, training)
        )
        enc_carry, dyn_carry, dec_carry = model_carry
        enc_entries, dyn_entries, dec_entries = entries
        valid = self.validity(obs)

        extra_losses, extra_metrics = self.additional_world_model_losses(
            tokens,
            repfeat,
            dyn_entries,
            target_tokens,
            obs,
            prevact,
            training,
        )
        losses.update(extra_losses)
        metrics.update(extra_metrics)

        reduced = {key: masked_mean(value, valid) for key, value in losses.items()}
        metrics.update({f"loss/{key}": value for key, value in reduced.items()})
        loss = sum(value * self.scales[key] for key, value in reduced.items())
        metrics["replay_views/world_reward_mean"] = sg(
            obs["reward"].astype(jnp.float32).mean()
        )
        metrics["replay_views/world_loss"] = sg(loss)
        carry = (enc_carry, dyn_carry, dec_carry)
        entries = (enc_entries, dyn_entries, dec_entries)
        outs = {"tokens": tokens, "repfeat": repfeat, "losses": losses}
        if target_tokens is not None:
            outs["target_tokens"] = target_tokens
        return loss, (carry, entries, outs, metrics)

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
        obs = {key: data[key] for key in self.obs_space}

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
        if self.dec is not None:
            dec_carry, _, _ = self.dec(
                dec_carry,
                prefix_features,
                reset,
                training=False,
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
        if self.dec is not None:
            dec_carry, dec_entries, _ = self.dec(
                dec_carry, repfeat, reset, training=False
            )
        else:
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
        state_valid = self.imagination_state_validity(
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
                lam=float(self.config.ppo.lam),
            )
        )
        advantage = normalize_advantage(
            advantage,
            valid,
            trajectory_weight,
        )

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
                "trajectory_weight": trajectory_weight,
                "entropy_coefficient": entropy_coefficient,
            }
        )
        metrics = {
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
            normalize_entropy=bool(
                self.config.ppo.entropy_schedule.enabled
                and self.config.ppo.entropy_schedule.normalize
            ),
        )

    def _ppo_critic_loss(self, batch):
        value = self.critic(
            batch["critic_features"],
            2,
            slow=False,
            context=batch["critic_context"],
        )
        return value_objective(
            value,
            batch["target_return"],
            batch["valid"],
            batch["trajectory_weight"],
        )

    def _update_slow_models(self, ppo_active):
        self._gated_slow_update(self.slowval, ppo_active)

    @staticmethod
    def _gated_slow_update(model, active):
        """Update a slow model without advancing any state while disabled."""

        model._initonce()
        old_values = dict(model.model.values)
        old_count = model.count.read()
        model.update()
        for key, new_value in model.model.values.items():
            model.model.write(key, jnp.where(active, new_value, old_values[key]))
        model.count.write(jnp.where(active, model.count.read(), old_count))

    def additional_world_model_losses(
        self,
        tokens,
        repfeat,
        dyn_entries,
        target_tokens,
        obs,
        prevact,
        training,
    ):
        del tokens, repfeat, dyn_entries, target_tokens, obs, prevact, training
        return {}, {}

    def representation_prediction_branches(self, repfeat, dynamics_aux):
        """Return predictive states that share the maintained JEPA targets."""

        del dynamics_aux
        return {"model": repfeat}

    def dynamics_loss(self, carry, tokens, actions, reset, obs, training):
        del obs
        return self.dyn.loss(carry, tokens, actions, reset, training)

    def imagination_starts(
        self,
        dyn_entries,
        dyn_carry,
        repfeat,
        obs,
        prevact,
        starts_count,
    ):
        del obs, prevact
        batch = dyn_entries["deter"].shape[0]
        starts = self.dyn.starts(dyn_entries, dyn_carry, starts_count)
        first = jax.tree.map(
            lambda value: value[:, -starts_count:].reshape(
                (batch * starts_count, 1, *value.shape[2:])
            ),
            repfeat,
        )
        return starts, first, None

    def imagine(self, starts, policy, horizon, training, context=None):
        del context
        return self.dyn.imagine(starts, policy, horizon, training)

    def imagine_with_aux(self, starts, policy, horizon, training, context=None):
        carry, features, actions = self.imagine(
            starts, policy, horizon, training, context
        )
        return carry, features, actions, None

    def imagination_policy_distribution(self, policy_inputs, auxiliary):
        del auxiliary
        return self.policy_distribution(policy_inputs, 2)

    def imagination_action_mask(self, auxiliary):
        del auxiliary
        raise ValueError("PPO imagination requires an exact categorical action mask")

    def imagination_reward_continuation(self, local_inputs, auxiliary):
        del auxiliary
        return self.rew(local_inputs, 2).pred(), self.con(local_inputs, 2).prob(1)

    def imagination_policy_features(self, features):
        return features

    def imagination_interface_metrics(self, model_features, policy_features):
        del model_features, policy_features
        return {}

    def imagination_state_validity(self, context, horizon, auxiliary=None):
        del context, horizon, auxiliary
        raise ValueError("PPO imagination requires explicit state validity")

    def imagination_behavior_metrics(self, actions, validity, auxiliary=None):
        del actions, validity, auxiliary
        return {}

    def imagination_critic_context(self, features, context, auxiliary=None):
        del features, context, auxiliary
        return None, {}

    @staticmethod
    def validity(obs):
        valid = jnp.ones_like(obs["is_first"], dtype=jnp.float32)
        for key in ("agent_present", "agent_alive"):
            if key in obs:
                valid *= obs[key].astype(jnp.float32)
        return valid

    def _world_model_terms(self, carry, obs, prevact, training):
        enc_carry, dyn_carry, dec_carry = carry
        reset = obs["is_first"]
        batch, length = reset.shape
        losses = {}
        metrics = {}
        enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training)
        dyn_carry, dyn_entries, dyn_losses, repfeat, dyn_metrics, dynamics_aux = (
            self.dynamics_loss(dyn_carry, tokens, prevact, reset, obs, training)
        )
        losses.update(dyn_losses)
        metrics.update(dyn_metrics)
        valid = self.validity(obs)
        if self.sigreg:
            regularizer = sigreg_loss(
                tokens,
                nj.seed(),
                knots=int(self.config.sigreg.knots),
                num_proj=int(self.config.sigreg.num_proj),
                aggregation=str(self.config.sigreg.aggregation),
                team_size=int(self.config.num_agents),
                valid=valid,
            )
            losses["sigreg"] = jnp.broadcast_to(regularizer, (batch, length))
            metrics["sigreg/embedding_std"] = embedding_std(tokens)
        if getattr(self, "actmask", None) is not None:
            policy_input = self.feat2tensor(repfeat)
            losses["action_mask"] = binary_vector_loss(
                self.actmask(policy_input, 2),
                obs["action_mask"],
                str(getattr(self.config, "action_mask_reduction", "sum")),
            )
        target_tokens = None
        if self.dec is not None:
            dec_carry, dec_entries, reconstructions = self.dec(
                dec_carry, repfeat, reset, training
            )
        else:
            dec_entries = {}
            reconstructions = {}
        has_embedding_objective = (
            self.posterior_jepa or self.dynamics_jepa or self.spatial_jepa
        )
        if has_embedding_objective:
            if self.slowenc is not None:
                _, _, target_tokens = self.slowenc(
                    self.target_enc.initial(batch),
                    obs,
                    reset,
                    training=False,
                )
            else:
                target_tokens = tokens
            stop_target = self.embedding_target == "ema"

            branches = self.representation_prediction_branches(repfeat, dynamics_aux)

            def add_embedding_loss(key, feature_fn, predictor_name="pred"):
                branch_losses = []
                branch_cosines = []
                branch_mses = []
                branch_norms = []
                for branch, features in branches.items():
                    raw_prediction = self.dyn.predictor(
                        feature_fn(features), name=predictor_name
                    )
                    loss, cosine, mse = embedding_prediction_loss(
                        raw_prediction,
                        target_tokens,
                        distance=self.embedding_loss,
                        stop_target=stop_target,
                    )
                    branch_losses.append(loss)
                    branch_cosines.append(cosine.mean())
                    branch_mses.append(mse.mean())
                    branch_norms.append(jnp.linalg.norm(raw_prediction, axis=-1).mean())
                    if len(branches) > 1:
                        metrics[f"{key}/{branch}_cosine"] = cosine.mean()
                        metrics[f"{key}/{branch}_mse"] = mse.mean()
                losses[key] = jnp.stack(branch_losses).mean(0)
                metrics[f"{key}/cosine"] = jnp.stack(branch_cosines).mean()
                metrics[f"{key}/mse"] = jnp.stack(branch_mses).mean()
                metrics[f"{key}/pred_norm"] = jnp.stack(branch_norms).mean()
                metrics[f"{key}/target_std"] = (
                    target_tokens.astype(jnp.float32).std(axis=(0, 1)).mean()
                )

            if self.posterior_jepa:
                add_embedding_loss("posterior_jepa", self.feat2tensor)
            if self.dynamics_jepa:
                add_embedding_loss(
                    "dynamics_jepa", lambda features: features["deter"], "dynpred"
                )
            if self.spatial_jepa:
                assert self.target_enc is not None
                grid_height, grid_width, _ = self.enc.image_grid_shape()
                spatial_target = self.target_enc.spatial_tokens(target_tokens)
                mask = spatial_patch_mask(
                    nj.seed(),
                    reset.shape,
                    (grid_height, grid_width),
                    float(self.config.spatial_jepa.mask_ratio),
                )
                masked_obs = dict(obs)
                for key in self.enc.imgkeys:
                    masked_obs[key] = mask_image_patches(
                        obs[key],
                        mask,
                        fill_value=int(self.config.spatial_jepa.fill_value),
                    )
                _, _, masked_tokens = self.enc(
                    self.enc.initial(batch),
                    masked_obs,
                    reset,
                    training,
                )
                context = jnp.concatenate([repfeat["deter"], masked_tokens], axis=-1)
                raw_prediction = self.dyn.predictor(context, name="spatialpred")
                prediction = self.enc.spatial_tokens(raw_prediction)
                (
                    losses["spatial_jepa"],
                    spatial_cosine,
                    mask_fraction,
                ) = masked_spatial_loss(
                    prediction,
                    spatial_target,
                    mask,
                )
                metrics["spatial_jepa/cosine"] = spatial_cosine
                metrics["spatial_jepa/mask_fraction"] = mask_fraction
                metrics["spatial_jepa/target_std"] = (
                    target_tokens.astype(jnp.float32).std(axis=(0, 1)).mean()
                )
        model_inp = self.feat2tensor(repfeat)
        inp = sg(model_inp, skip=self.config.reward_grad)
        losses["rew"] = self.rew(inp, 2).loss(obs["reward"])
        continuation = f32(~obs["is_terminal"])
        if self.config.contdisc:
            continuation *= 1 - 1 / self.config.horizon
        losses["con"] = self.con(model_inp, 2).loss(continuation)
        if self.dec is not None:
            for key, reconstruction in reconstructions.items():
                space = self.obs_space[key]
                value = obs[key]
                target = (
                    f32(value) / 255
                    if space.dtype == np.uint8 and len(space.shape) == 3
                    else value
                )
                losses[key] = reconstruction.loss(sg(target))
        return (
            (enc_carry, dyn_carry, dec_carry),
            (enc_entries, dyn_entries, dec_entries),
            tokens,
            repfeat,
            losses,
            metrics,
            target_tokens,
        )
