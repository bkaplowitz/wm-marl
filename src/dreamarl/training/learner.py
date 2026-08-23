"""World-model and actor-critic training orchestration."""

import elements
import embodied.jax.outs as jaxouts
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

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
    def train(self, carry, data):
        environment_step = data.pop("_environment_step", None)
        if environment_step is not None:
            environment_step = environment_step.reshape((-1,))[0]
        if getattr(self, "jecc_pretrain_only", False):
            return self.jecc_pretrain(carry, data)
        carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
        metrics, (carry, entries, outs, mets) = self.opt(
            self.loss,
            carry,
            obs,
            prevact,
            training=True,
            environment_step=environment_step,
            has_aux=True,
        )
        metrics.update(mets)
        if self.behavior_objective == "ppo":
            ppo_batch = outs.pop("ppo_batch")
            for _ in range(int(self.config.ppo.epochs)):
                actor_metrics = self.actor_opt(self.ppo_actor_loss, ppo_batch)
            metrics.update(actor_metrics)
            metrics.update(self.ppo_actor_metrics(ppo_batch))
            metrics["ppo/epochs"] = jnp.asarray(self.config.ppo.epochs, jnp.float32)
        self._update_slow_models()
        if self.slowenc is not None:
            self.slowenc.update()
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

    def loss(self, carry, obs, prevact, training, environment_step=None):
        model_carry, entries, tokens, repfeat, losses, metrics, target_tokens = (
            self._world_model_terms(carry, obs, prevact, training)
        )
        enc_carry, dyn_carry, dec_carry = model_carry
        enc_entries, dyn_entries, dec_entries = entries
        batch, length = obs["is_first"].shape
        valid = self.validity(obs)

        early_extensions = self.additional_losses_before_imagination()

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
        imgfeat = concat(
            [
                sg(first, skip=self.config.ac_grads),
                sg(imgfeat, skip=self.config.ac_grads),
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
        policy = self.imagination_policy_distribution(
            policy_inp,
            imagination_aux,
        )
        if early_extensions:
            # Initialize training-only modules only after the complete B0 path
            # has initialized and sampled its imagination. This preserves B0's
            # parameter/RNG order while keeping extension parameters outside
            # the counterfactual map below.
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
        advantage_transform = self.actor_advantage_transform(
            imgfeat,
            imgact,
            policy,
            imagination_context,
            environment_step,
            training,
            starts,
        )
        imagined_reward, imagined_continuation = (
            self.imagination_reward_continuation(
                local_inp,
                imagination_aux,
            )
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
            advantage_transform=advantage_transform,
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
        ppo_imag_out = imgloss_out
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
            feat = sg(repfeat, skip=self.config.repval_grad)
            last, term, rew = [obs[key] for key in ("is_last", "is_terminal", "reward")]
            boot = imgloss_out["ret"][:, 0].reshape(batch, starts_count)
            feat, last, term, rew, boot = jax.tree.map(
                lambda value: value[:, -starts_count:],
                (feat, last, term, rew, boot),
            )
            local_inp = self.feat2tensor(feat)
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
            replay_valid = self.replay_value_validity(obs)[
                :, -starts_count:-1
            ]
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

        # Initialize and evaluate extension modules only after the locked local
        # model, actor, and critic have consumed their normal RNG sequence.
        if not early_extensions:
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
        if self.behavior_objective == "ppo" and training:
            action_key = self.action_mask_key
            if action_key is None or len(self.act_space) != 1:
                raise ValueError("imagined PPO requires one masked categorical action")
            raw_policy = self.pol(policy_inp, 2)[action_key]
            old_policy = policy[action_key]
            valid_actor = (
                jnp.ones_like(ppo_imag_out["adv_normed"], dtype=jnp.float32)
                if imagination_valid is None
                else imagination_valid[:, : ppo_imag_out["adv_normed"].shape[1]]
            )
            outs["ppo_batch"] = jax.tree.map(
                jax.lax.stop_gradient,
                {
                    "features": policy_inp[:, :-1],
                    "actions": imgact[action_key][:, :-1],
                    "old_logits": old_policy.logits[:, :-1],
                    "mask_bias": (
                        old_policy.logits[:, :-1] - raw_policy.logits[:, :-1]
                    ),
                    "advantages": ppo_imag_out["adv_normed"],
                    "weights": ppo_imag_out["weight"],
                    "valid": valid_actor,
                },
            )
        if target_tokens is not None:
            outs["target_tokens"] = target_tokens
        return loss, (carry, entries, outs, metrics)

    def _ppo_distribution(self, batch):
        action_key = self.action_mask_key
        raw = self.pol(batch["features"], 2)[action_key]
        return jaxouts.Categorical(raw.logits + batch["mask_bias"])

    def ppo_actor_loss(self, batch):
        distribution = self._ppo_distribution(batch)
        new_logp = distribution.logp(batch["actions"])
        old = jaxouts.Categorical(batch["old_logits"])
        old_logp = old.logp(batch["actions"])
        ratio = jnp.exp(jnp.clip(new_logp - old_logp, -20.0, 20.0))
        clip = float(self.config.ppo.clip)
        clipped_ratio = jnp.clip(ratio, 1.0 - clip, 1.0 + clip)
        advantage = batch["advantages"]
        surrogate = jnp.minimum(ratio * advantage, clipped_ratio * advantage)
        entropy = distribution.entropy()
        per_item = batch["weights"] * -(
            surrogate + float(self.config.imag_loss.actent) * entropy
        )
        valid = batch["valid"].astype(jnp.float32)
        return (per_item * valid).sum() / jnp.maximum(valid.sum(), 1.0)

    def ppo_actor_metrics(self, batch):
        distribution = self._ppo_distribution(batch)
        old = jaxouts.Categorical(batch["old_logits"])
        new_logp = distribution.logp(batch["actions"])
        old_logp = old.logp(batch["actions"])
        ratio = jnp.exp(jnp.clip(new_logp - old_logp, -20.0, 20.0))
        valid = batch["valid"].astype(jnp.float32)
        denominator = jnp.maximum(valid.sum(), 1.0)

        def mean(value):
            return (value * valid).sum() / denominator

        clip = float(self.config.ppo.clip)
        return {
            "ppo/kl": mean(old.kl(distribution)),
            "ppo/approx_kl": mean(old_logp - new_logp),
            "ppo/clip_fraction": mean(jnp.abs(ratio - 1.0) > clip),
            "ppo/ratio_mean": mean(ratio),
            "ppo/selected_probability": mean(jnp.exp(new_logp)),
            "ppo/entropy": mean(distribution.entropy()),
        }

    def _update_slow_models(self):
        self.slowval.update()

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

    def additional_losses_before_imagination(self):
        """Whether extension modules must be initialized before actor credit."""

        return False

    def actor_advantage_transform(
        self,
        imgfeat,
        imgact,
        policy,
        imagination_context,
        environment_step,
        training,
        dynamics_starts=None,
    ):
        """Return an optional transform of B0's normalized actor advantage."""

        del (
            imgfeat,
            imgact,
            policy,
            imagination_context,
            environment_step,
            training,
            dynamics_starts,
        )
        return None

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
            losses["action_mask"] = self.actmask(policy_input, 2).loss(
                obs["action_mask"].astype(bool)
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
                topology = str(self.config.spatial_jepa.topology)
                if topology == "vjepa21_multiblock":
                    from ..ablations.representation import (
                        vjepa21_dense_token_loss,
                        vjepa21_multiblock_masks,
                    )

                    assert self.spatial_predictor is not None
                    spatial_length = min(16, length)
                    spatial_indices = jnp.arange(
                        length - spatial_length, length, dtype=jnp.int32
                    )
                    spatial_obs = jax.tree.map(
                        lambda value: value[:, spatial_indices], obs
                    )
                    spatial_reset = reset[:, spatial_indices]
                    spatial_target = self.target_enc.full_predictor_tokens(
                        spatial_obs, spatial_reset
                    )
                    contexts, targets = vjepa21_multiblock_masks(
                        nj.seed(), spatial_reset.shape, (grid_height, grid_width)
                    )
                    group_losses = []
                    group_cosines = []
                    group_fractions = []
                    for visible, mask in zip(contexts, targets):
                        context_tokens = self.enc.visible_spatial_tokens(
                            spatial_obs, spatial_reset, visible
                        )
                        prediction, context_prediction = self.spatial_predictor(
                            context_tokens, visible, mask
                        )
                        group_loss, group_cosine, group_fraction = (
                            vjepa21_dense_token_loss(
                                prediction,
                                context_prediction,
                                spatial_target,
                                mask,
                                visible,
                                context_weight=0.5,
                            )
                        )
                        group_losses.append(group_loss)
                        group_cosines.append(group_cosine)
                        group_fractions.append(group_fraction)
                    sampled_loss = jnp.stack(group_losses).mean(0)
                    losses["spatial_jepa"] = jnp.broadcast_to(
                        sampled_loss.mean(axis=1, keepdims=True),
                        (batch, length),
                    )
                    spatial_cosine = jnp.stack(group_cosines).mean()
                    mask_fraction = jnp.stack(group_fractions).mean()
                    metrics["spatial_jepa/frames_per_sequence"] = spatial_length
                else:
                    spatial_target = self.target_enc.spatial_tokens(target_tokens)
                    if topology == "fixed_count":
                        mask = spatial_patch_mask(
                            nj.seed(),
                            reset.shape,
                            (grid_height, grid_width),
                            float(self.config.spatial_jepa.mask_ratio),
                        )
                    else:
                        from ..ablations.representation import (
                            spatial_patch_mask as ablation_mask,
                        )

                        mask = ablation_mask(
                            nj.seed(),
                            reset.shape,
                            (grid_height, grid_width),
                            float(self.config.spatial_jepa.mask_ratio),
                            topology,
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
                    context = jnp.concatenate(
                        [repfeat["deter"], masked_tokens], axis=-1
                    )
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
