"""World-model and actor-critic training orchestration."""

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from ..models.heads import binary_vector_loss
from .common import concat, f32, prefix, sample, sg
from .objectives import imag_loss, repl_loss
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


def value_fit_metrics(prediction, target, valid):
    """Return live-sample calibration metrics for a value prediction."""

    weight = jnp.ones_like(target) if valid is None else valid.astype(jnp.float32)
    count = jnp.maximum(weight.sum(), 1.0)
    error = prediction - target
    selected = weight.astype(bool)
    masked_error = jnp.where(selected, error, 0.0)
    masked_target = jnp.where(selected, target, 0.0)
    error_mean = masked_error.sum() / count
    target_mean = masked_target.sum() / count
    error_variance = (
        jnp.where(selected, jnp.square(error - error_mean), 0.0).sum() / count
    )
    target_variance = (
        jnp.where(selected, jnp.square(target - target_mean), 0.0).sum() / count
    )
    return {
        "rmse": jnp.sqrt(jnp.where(selected, jnp.square(error), 0.0).sum() / count),
        "bias": error_mean,
        "explained_variance": 1.0 - error_variance / jnp.maximum(target_variance, 1e-8),
    }


class LearnerMixin:
    def train(self, carry, data, behavior_data=None):
        if self.two_branch_replay:
            if behavior_data is None:
                raise ValueError(
                    "recent_world_uniform_behavior requires a behavior replay batch"
                )
            if int(self.config.replay_context) < 1:
                raise ValueError(
                    "recent_world_uniform_behavior requires replay_context burn-in"
                )
        elif behavior_data is not None:
            raise ValueError(
                "behavior replay batch supplied outside recent_world_uniform_behavior"
            )
        behavior_active = self._actor_critic_schedule(data)
        carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
        if self.two_branch_replay:
            optimizer_kwargs = {}
            loss_kwargs = {}
            if behavior_active is not None:
                optimizer_kwargs["active_groups"] = {
                    "actor": behavior_active,
                    "critic": behavior_active,
                }
                loss_kwargs["behavior_active"] = behavior_active
            metrics, (carry, entries, outs, mets) = self.opt(
                self.two_branch_loss,
                carry,
                obs,
                prevact,
                behavior_data=behavior_data,
                training=True,
                has_aux=True,
                **loss_kwargs,
                **optimizer_kwargs,
            )
        else:
            if behavior_active is not None:
                raise ValueError(
                    "actor/critic warm-start gating requires separated CTDE replay"
                )
            metrics, (carry, entries, outs, mets) = self.opt(
                self.loss,
                carry,
                obs,
                prevact,
                training=True,
                has_aux=True,
            )
        metrics.update(mets)
        self._update_slow_models(behavior_active)
        if self.slowenc is not None:
            self.slowenc.update()
        if behavior_active is not None:
            environment_step = data["_environment_step"].reshape(-1)[0]
            metrics.update(
                {
                    "schedule/environment_step": environment_step,
                    "schedule/actor_critic_start_step": jnp.asarray(
                        self.actor_critic_start_step, jnp.int32
                    ),
                    "schedule/world_model_active": jnp.asarray(1.0, jnp.float32),
                    "schedule/actor_active": behavior_active.astype(jnp.float32),
                    "schedule/critic_active": behavior_active.astype(jnp.float32),
                    "schedule/world_only_active": (~behavior_active).astype(
                        jnp.float32
                    ),
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

    def _actor_critic_schedule(self, data):
        """Return the dynamic behavior-group gate, or None for exact control."""

        start = int(self.actor_critic_start_step)
        if not start:
            return None
        if "_environment_step" not in data:
            raise ValueError("warm-start training requires _environment_step")
        environment_step = data["_environment_step"]
        if environment_step.ndim != 2:
            raise ValueError(
                "folded _environment_step must be [B*A,T], got "
                f"{environment_step.shape}"
            )
        return environment_step.reshape(-1)[0].astype(jnp.int32) >= start

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

        behavior_losses, behavior_metrics = self._behavior_terms(
            dyn_carry,
            dyn_entries,
            repfeat,
            obs,
            prevact,
            training,
            detach_world=False,
        )
        losses.update(behavior_losses)
        metrics.update(behavior_metrics)

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

        metrics.update(
            {
                f"loss/{key}": masked_mean(
                    value,
                    valid,
                    alignment="replay_value" if key == "repval" else "tail",
                )
                for key, value in losses.items()
            }
        )
        loss = sum(
            masked_mean(
                value,
                valid,
                alignment="replay_value" if key == "repval" else "tail",
            )
            * self.scales[key]
            for key, value in losses.items()
        )
        carry = (enc_carry, dyn_carry, dec_carry)
        entries = (enc_entries, dyn_entries, dec_entries)
        outs = {"tokens": tokens, "repfeat": repfeat, "losses": losses}
        if target_tokens is not None:
            outs["target_tokens"] = target_tokens
        return loss, (carry, entries, outs, metrics)

    def two_branch_loss(
        self,
        carry,
        obs,
        prevact,
        behavior_data,
        training,
        behavior_active=None,
    ):
        """Train world and behavior modules from independent replay views."""

        (
            model_carry,
            entries,
            tokens,
            repfeat,
            world_losses,
            metrics,
            target_tokens,
        ) = self._world_model_terms(carry, obs, prevact, training)
        enc_carry, dyn_carry, dec_carry = model_carry
        enc_entries, dyn_entries, dec_entries = entries
        extra_losses, extra_metrics = self.additional_world_model_losses(
            tokens,
            repfeat,
            dyn_entries,
            target_tokens,
            obs,
            prevact,
            training,
        )
        world_losses.update(extra_losses)
        metrics.update(extra_metrics)

        behavior_carry, behavior_obs, behavior_prevact, _ = (
            self._apply_behavior_replay_context(behavior_data)
        )
        behavior_carry, behavior_entries, behavior_repfeat = (
            self._behavior_model_states(
                behavior_carry,
                behavior_obs,
                behavior_prevact,
            )
        )
        _, behavior_dyn_carry, _ = behavior_carry
        _, behavior_dyn_entries, _ = behavior_entries
        behavior_losses, behavior_metrics = self._behavior_terms(
            behavior_dyn_carry,
            behavior_dyn_entries,
            behavior_repfeat,
            behavior_obs,
            behavior_prevact,
            training,
            detach_world=True,
            behavior_active=behavior_active,
        )
        overlap = set(world_losses).intersection(behavior_losses)
        if overlap:
            raise ValueError(
                "world and behavior replay losses must have disjoint ownership: "
                f"{sorted(overlap)}"
            )
        metrics.update(behavior_metrics)

        world_valid = self.validity(obs)
        behavior_valid = self.validity(behavior_obs)

        def reduced(losses, valid):
            return {
                key: masked_mean(
                    value,
                    valid,
                    alignment="replay_value" if key == "repval" else "tail",
                )
                for key, value in losses.items()
            }

        world_reduced = reduced(world_losses, world_valid)
        behavior_reduced = reduced(behavior_losses, behavior_valid)
        metrics.update({f"loss/{key}": value for key, value in world_reduced.items()})
        metrics.update(
            {f"loss/{key}": value for key, value in behavior_reduced.items()}
        )
        world_loss = sum(
            value * self.scales[key] for key, value in world_reduced.items()
        )
        behavior_loss = sum(
            value * self.scales[key] for key, value in behavior_reduced.items()
        )
        world_loss = jnp.asarray(world_loss, jnp.float32)
        behavior_loss = jnp.asarray(behavior_loss, jnp.float32)
        metrics.update(
            {
                "replay_views/world_reward_mean": sg(
                    obs["reward"].astype(jnp.float32).mean()
                ),
                "replay_views/behavior_reward_mean": sg(
                    behavior_obs["reward"].astype(jnp.float32).mean()
                ),
                "replay_views/world_rows": jnp.asarray(
                    obs["is_first"].shape[0], jnp.float32
                ),
                "replay_views/behavior_rows": jnp.asarray(
                    behavior_obs["is_first"].shape[0], jnp.float32
                ),
                "replay_views/world_length": jnp.asarray(
                    obs["is_first"].shape[1], jnp.float32
                ),
                "replay_views/behavior_length": jnp.asarray(
                    behavior_obs["is_first"].shape[1], jnp.float32
                ),
                "replay_views/world_loss": sg(world_loss),
                "replay_views/behavior_loss": sg(behavior_loss),
            }
        )

        carry = (enc_carry, dyn_carry, dec_carry)
        entries = (enc_entries, dyn_entries, dec_entries)
        outs = {
            "tokens": tokens,
            "repfeat": repfeat,
            "world_losses": world_losses,
            "behavior_losses": behavior_losses,
        }
        if target_tokens is not None:
            outs["target_tokens"] = target_tokens
        return world_loss + behavior_loss, (carry, entries, outs, metrics)

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

    def _behavior_terms(
        self,
        dyn_carry,
        dyn_entries,
        repfeat,
        obs,
        prevact,
        training,
        *,
        detach_world,
        behavior_active=None,
    ):
        """Compute actor and critic objectives from one explicit replay view."""

        if detach_world:
            dyn_carry, dyn_entries, repfeat = jax.tree.map(
                sg, (dyn_carry, dyn_entries, repfeat)
            )
        batch, length = obs["is_first"].shape
        losses = {}
        metrics = {}
        starts_count = min(self.config.imag_last or length, length)
        horizon = self.config.imag_length
        starts, first, imagination_context = self.imagination_starts(
            dyn_entries,
            dyn_carry,
            repfeat,
            obs,
            prevact,
            starts_count,
        )

        def policyfn(feat):
            tensor = self.feat2tensor(feat)
            return sample(self.policy_distribution(tensor, 1))

        _, imgfeat, imgprevact, imagination_aux = self.imagine_with_aux(
            starts,
            policyfn,
            horizon,
            training,
            imagination_context,
        )
        if detach_world:
            first, imgfeat, imgprevact, imagination_aux = jax.tree.map(
                sg, (first, imgfeat, imgprevact, imagination_aux)
            )
        imgfeat = concat(
            [
                sg(first, skip=self.config.ac_grads and not detach_world),
                sg(imgfeat, skip=self.config.ac_grads and not detach_world),
            ],
            1,
        )
        policyfeat = self.imagination_policy_features(imgfeat)
        metrics.update(self.imagination_interface_metrics(imgfeat, policyfeat))
        lastact = self.imagination_last_action(
            policyfeat,
            imagination_aux,
            policyfn,
        )
        lastact = jax.tree.map(lambda value: value[:, None], lastact)
        imgact = concat([imgprevact, lastact], 1)
        local_inp = self.feat2tensor(imgfeat)
        policy_inp = self.feat2tensor(policyfeat)
        critic_context, critic_metrics = self.imagination_critic_context(
            imgfeat, imagination_context, imagination_aux
        )
        metrics.update(critic_metrics)
        value = self.critic(imgfeat, 2, slow=False, context=critic_context)
        slowvalue = self.critic(imgfeat, 2, slow=True, context=critic_context)
        local_only_value = None
        if critic_context is not None and not training:
            local_only_value = self.critic(
                imgfeat,
                2,
                slow=False,
                context=jax.tree.map(jnp.zeros_like, critic_context),
            )
        imagination_valid = self.imagination_validity(
            imagination_context, horizon, imagination_aux
        )
        if behavior_active is not None:
            if imagination_valid is None:
                imagination_valid = jnp.ones(
                    imgprevact[next(iter(imgprevact))].shape[:2], bool
                )
            imagination_valid = jnp.where(
                behavior_active,
                imagination_valid,
                jnp.zeros_like(imagination_valid),
            )
        metrics.update(
            self.imagination_behavior_metrics(
                imgprevact,
                imagination_valid,
                imagination_aux,
            )
        )
        policy = self.imagination_policy_distribution(
            policy_inp,
            imagination_aux,
        )
        imagined_reward, imagined_continuation = self.imagination_reward_continuation(
            local_inp,
            imagination_aux,
        )
        if detach_world:
            imagined_reward, imagined_continuation = jax.tree.map(
                sg, (imagined_reward, imagined_continuation)
            )
        imagined_losses, imgloss_out, imagined_metrics = imag_loss(
            imgact,
            imagined_reward,
            imagined_continuation,
            policy,
            value,
            slowvalue,
            self.retnorm,
            self.valnorm,
            self.advnorm,
            update=training,
            valid=imagination_valid,
            contdisc=self.config.contdisc,
            horizon=self.config.horizon,
            **self.config.imag_loss,
        )
        if local_only_value is not None:
            local_metrics = value_fit_metrics(
                local_only_value.pred()[:, :-1],
                imgloss_out["ret"],
                None
                if imagination_valid is None
                else imagination_valid[:, : imgloss_out["ret"].shape[1]],
            )
            metrics.update(
                {
                    f"central_critic/local_only_imag_{key}": metric
                    for key, metric in local_metrics.items()
                }
            )
        imagined_losses, imgloss_out = self.restore_imagination_results(
            imagined_losses,
            imgloss_out,
            imagination_context,
        )
        losses.update(
            {
                key: value.mean(1).reshape((batch, starts_count))
                for key, value in imagined_losses.items()
            }
        )
        metrics.update(imagined_metrics)

        if self.config.repval_loss:
            feat = repfeat
            last, term, rew = [obs[key] for key in ("is_last", "is_terminal", "reward")]
            boot = imgloss_out["ret"][:, 0].reshape(batch, starts_count)
            feat, last, term, rew, boot = jax.tree.map(
                lambda value: value[:, -starts_count:],
                (feat, last, term, rew, boot),
            )
            critic_context, critic_metrics = self.replay_critic_context(
                feat, obs, starts_count
            )
            metrics.update(critic_metrics)
            value = self.critic(feat, 2, slow=False, context=critic_context)
            slowvalue = self.critic(feat, 2, slow=True, context=critic_context)
            local_only_value = None
            if critic_context is not None and not training:
                local_only_value = self.critic(
                    feat,
                    2,
                    slow=False,
                    context=jax.tree.map(jnp.zeros_like, critic_context),
                )
            replay_valid = self.replay_value_validity(obs)[:, -starts_count:-1]
            if behavior_active is not None:
                replay_valid = jnp.where(
                    behavior_active,
                    replay_valid,
                    jnp.zeros_like(replay_valid),
                )
            replay_losses, replay_out, replay_metrics = repl_loss(
                last,
                term,
                rew,
                boot,
                value,
                slowvalue,
                self.valnorm,
                update=training,
                valid=replay_valid,
                horizon=self.config.horizon,
                **self.config.repl_loss,
            )
            losses.update(replay_losses)
            metrics.update(prefix(replay_metrics, "reploss"))
            if local_only_value is not None:
                local_metrics = value_fit_metrics(
                    local_only_value.pred()[:, :-1],
                    replay_out["ret"],
                    replay_valid[:, : replay_out["ret"].shape[1]],
                )
                metrics.update(
                    {
                        f"central_critic/local_only_replay_{key}": metric
                        for key, metric in local_metrics.items()
                    }
                )
        return losses, metrics

    def _update_slow_models(self, behavior_active=None):
        if behavior_active is None:
            self.slowval.update()
        else:
            self._gated_slow_update(self.slowval, behavior_active)

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

    def imagination_last_action(self, policy_features, auxiliary, policyfn):
        del auxiliary
        return policyfn(jax.tree.map(lambda value: value[:, -1], policy_features))

    def imagination_policy_distribution(self, policy_inputs, auxiliary):
        del auxiliary
        return self.policy_distribution(policy_inputs, 2)

    def imagination_reward_continuation(self, local_inputs, auxiliary):
        del auxiliary
        return self.rew(local_inputs, 2).pred(), self.con(local_inputs, 2).prob(1)

    def restore_imagination_results(self, losses, outputs, context=None):
        del context
        return losses, outputs

    def imagination_policy_features(self, features):
        return features

    def imagination_interface_metrics(self, model_features, policy_features):
        del model_features, policy_features
        return {}

    def imagination_validity(self, context, horizon, auxiliary=None):
        del context, auxiliary
        return None

    def imagination_behavior_metrics(self, actions, validity, auxiliary=None):
        del actions, validity, auxiliary
        return {}

    def imagination_critic_context(self, features, context, auxiliary=None):
        del features, context, auxiliary
        return None, {}

    def replay_critic_context(self, features, obs, starts_count):
        del features, obs, starts_count
        return None, {}

    @staticmethod
    def validity(obs):
        valid = jnp.ones_like(obs["is_first"], dtype=jnp.float32)
        for key in ("agent_present", "agent_alive"):
            if key in obs:
                valid *= obs[key].astype(jnp.float32)
        return valid

    def replay_value_validity(self, obs):
        return self.validity(obs)

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
