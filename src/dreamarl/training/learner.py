"""World-model and actor-critic training orchestration."""

import elements
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


def masked_mean(value, valid):
    """Average a per-transition loss over valid local agent transitions."""

    weight = valid[:, -value.shape[1] :].astype(jnp.float32)
    return (value * weight).mean() / jnp.maximum(weight.mean(), 1e-8)


class LearnerMixin:
    def train(self, carry, data):
        carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
        metrics, (carry, entries, outs, mets) = self.opt(
            self.loss, carry, obs, prevact, training=True, has_aux=True
        )
        metrics.update(mets)
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

    def loss(self, carry, obs, prevact, training):
        model_carry, entries, tokens, repfeat, losses, metrics, target_tokens = (
            self._world_model_terms(carry, obs, prevact, training)
        )
        enc_carry, dyn_carry, dec_carry = model_carry
        enc_entries, dyn_entries, dec_entries = entries
        batch, length = obs["is_first"].shape

        starts_count = min(self.config.imag_last or length, length)
        horizon = self.config.imag_length
        starts, first, imagination_context = self.imagination_starts(
            dyn_entries,
            dyn_carry,
            repfeat,
            obs,
            starts_count,
        )

        def policyfn(feat):
            return sample(self.pol(self.feat2tensor(feat), 1))

        _, imgfeat, imgprevact = self.imagine(
            starts,
            policyfn,
            horizon,
            training,
            imagination_context,
        )
        imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
        policyfeat = self.imagination_policy_features(imgfeat)
        metrics.update(self.imagination_interface_metrics(imgfeat, policyfeat))
        lastact = policyfn(jax.tree.map(lambda value: value[:, -1], policyfeat))
        lastact = jax.tree.map(lambda value: value[:, None], lastact)
        imgact = concat([imgprevact, lastact], 1)
        local_inp = self.feat2tensor(imgfeat)
        policy_inp = self.feat2tensor(policyfeat)
        value = self.critic(imgfeat, 2, slow=False)
        slowvalue = self.critic(imgfeat, 2, slow=True)
        imagined_losses, imgloss_out, imagined_metrics = imag_loss(
            imgact,
            self.rew(local_inp, 2).pred(),
            self.con(local_inp, 2).prob(1),
            self.pol(policy_inp, 2),
            value,
            slowvalue,
            self.retnorm,
            self.valnorm,
            self.advnorm,
            update=training,
            contdisc=self.config.contdisc,
            horizon=self.config.horizon,
            **self.config.imag_loss,
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
            feat = sg(repfeat, skip=self.config.repval_grad)
            last, term, rew = [obs[key] for key in ("is_last", "is_terminal", "reward")]
            boot = imgloss_out["ret"][:, 0].reshape(batch, starts_count)
            feat, last, term, rew, boot = jax.tree.map(
                lambda value: value[:, -starts_count:],
                (feat, last, term, rew, boot),
            )
            local_inp = self.feat2tensor(feat)
            value = self.critic(feat, 2, slow=False)
            slowvalue = self.critic(feat, 2, slow=True)
            replay_losses, _, replay_metrics = repl_loss(
                last,
                term,
                rew,
                boot,
                value,
                slowvalue,
                self.valnorm,
                update=training,
                horizon=self.config.horizon,
                **self.config.repl_loss,
            )
            losses.update(replay_losses)
            metrics.update(prefix(replay_metrics, "reploss"))

        # Initialize and evaluate extension modules only after the locked local
        # model, actor, and critic have consumed their normal RNG sequence.
        extra_losses, extra_metrics = self.additional_world_model_losses(
            repfeat,
            target_tokens,
            obs,
            prevact,
            training,
        )
        losses.update(extra_losses)
        metrics.update(extra_metrics)

        valid = jnp.ones_like(obs["is_first"], dtype=jnp.float32)
        for key in ("agent_present", "agent_alive"):
            if key in obs:
                valid *= obs[key].astype(jnp.float32)

        metrics.update(
            {f"loss/{key}": masked_mean(value, valid) for key, value in losses.items()}
        )
        loss = sum(
            masked_mean(value, valid) * self.scales[key]
            for key, value in losses.items()
        )

        carry = (enc_carry, dyn_carry, dec_carry)
        entries = (enc_entries, dyn_entries, dec_entries)
        outs = {"tokens": tokens, "repfeat": repfeat, "losses": losses}
        if target_tokens is not None:
            outs["target_tokens"] = target_tokens
        return loss, (carry, entries, outs, metrics)

    def _update_slow_models(self):
        self.slowval.update()

    def additional_world_model_losses(
        self,
        repfeat,
        target_tokens,
        obs,
        prevact,
        training,
    ):
        del repfeat, target_tokens, obs, prevact, training
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
        starts_count,
    ):
        del obs
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

    def restore_imagination_results(self, losses, outputs, context=None):
        del context
        return losses, outputs

    def imagination_policy_features(self, features):
        return features

    def imagination_interface_metrics(self, model_features, policy_features):
        del model_features, policy_features
        return {}

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
        if self.sigreg:
            regularizer = sigreg_loss(
                tokens,
                nj.seed(),
                knots=int(self.config.sigreg.knots),
                num_proj=int(self.config.sigreg.num_proj),
                aggregation=str(self.config.sigreg.aggregation),
            )
            losses["sigreg"] = jnp.broadcast_to(regularizer, (batch, length))
            metrics["sigreg/embedding_std"] = embedding_std(tokens)
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
                if topology == "vjepa_multiblock":
                    from ..ablations.representation import (
                        vjepa_multiblock_masks,
                        vjepa_token_loss,
                    )

                    assert self.spatial_predictor is not None
                    spatial_length = min(4, length)
                    spatial_indices = jnp.rint(
                        jnp.linspace(0, length - 1, spatial_length)
                    ).astype(jnp.int32)
                    spatial_obs = jax.tree.map(
                        lambda value: value[:, spatial_indices], obs
                    )
                    spatial_reset = reset[:, spatial_indices]
                    spatial_target = self.target_enc.spatial_tokens(
                        target_tokens[:, spatial_indices]
                    )
                    masks = vjepa_multiblock_masks(
                        nj.seed(), spatial_reset.shape, (grid_height, grid_width)
                    )
                    group_losses = []
                    group_cosines = []
                    group_fractions = []
                    for mask in masks:
                        visible = ~mask
                        context_tokens = self.enc.visible_spatial_tokens(
                            spatial_obs, spatial_reset, visible
                        )
                        prediction = self.spatial_predictor(
                            context_tokens, visible, mask
                        )
                        group_loss, group_cosine, group_fraction = vjepa_token_loss(
                            prediction, spatial_target, mask
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
