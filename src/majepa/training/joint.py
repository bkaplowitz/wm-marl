"""Factual joint JEPA, interface and outcome supervision."""

from __future__ import annotations

import embodied.jax.outs as jaxouts
import jax
import jax.numpy as jnp
import ninjax as nj

from ..models.heads import (
    binary_vector_loss,
)
from ..models.multistep_jepa import (
    isolated_creation_call,
)
from .direct_jepa import direct_jepa_losses
from .self_fed import frozen_posterior, mixed_posterior_kl, self_fed_losses


def joint_world_model_losses(
    self,
    online_tokens,
    repfeat,
    dyn_entries,
    target_tokens,
    obs,
    prevact,
    training,
):
    """Fit the authoritative joint transition on aligned factual replay."""

    if target_tokens is None:
        raise RuntimeError("CTDE requires EMA encoder targets")
    grouped_state = self.team.unfold_sequence(self.feat2tensor(repfeat))
    grouped_online = self.team.unfold_sequence(online_tokens)
    grouped_target = self.team.unfold_sequence(target_tokens)
    grouped_present = self.team.unfold_sequence(self._present(obs)).astype(bool)
    grouped_alive = self.team.unfold_sequence(self._controllable(obs)).astype(bool)
    grouped_first = self.team.unfold_sequence(obs["is_first"]).any(axis=-1)
    grouped_reward = self.team.unfold_sequence(obs["reward"])
    grouped_mask = self.team.unfold_sequence(obs["action_mask"]).astype(bool)
    grouped_action = self.team.unfold_sequence(prevact[self.ctde_action_key]).astype(
        jnp.int32
    )

    source_state = grouped_state[:, :-1]
    source_action = grouped_action[:, 1:]
    source_present = grouped_present[:, :-1]
    source_alive = grouped_alive[:, :-1]
    reset = grouped_first[:, :-1]
    cache = dyn_entries["ctde_joint_carry"]
    joint_args = (
        cache,
        source_state,
        source_action,
        source_present,
        source_alive,
        reset,
        training,
    )
    history_prediction = None
    if (
        self.config.marl.ctde.get("factual_jepa_history_gradient", False)
        and not nj.creating()
    ):
        # Identical dropout draws for both passes. The ordinary pass keeps
        # every existing objective's local-state boundary unchanged.
        seed = nj.seed()
        params = {
            key: value
            for key, value in nj.context().items()
            if key.startswith(self.ctde_joint.path + "/")
        }
        sequence = nj.pure(self.ctde_joint.sequence, nested=True)
        _, (_, prediction, snapshots) = sequence(
            params, *joint_args, seed=seed, create=False, modify=False
        )
        frozen = jax.tree.map(jax.lax.stop_gradient, params)
        frozen_args = (jax.lax.stop_gradient(cache), *joint_args[1:])
        _, (_, history_outputs, _) = sequence(
            frozen,
            *frozen_args,
            stop_state_gradient=False,
            seed=seed,
            create=False,
            modify=False,
        )
        history_prediction = history_outputs["embedding"].astype(jnp.float32)
    else:
        _, prediction, snapshots = self.ctde_joint.sequence(*joint_args)

    next_first = grouped_first[:, 1:]
    transition_valid = source_alive & grouped_present[:, 1:] & ~next_first[..., None]
    weight = transition_valid.astype(jnp.float32)
    count = jnp.maximum(weight.sum(), 1.0)
    normalized_weight = weight / jnp.maximum(weight.mean(), 1e-8)
    # SMAC broadcasts team reward and termination to its fixed roster.
    # Dead units retain the surviving team's return; their shared-signal
    # heads therefore need supervision after they become uncontrollable.
    team_valid = source_present & grouped_present[:, 1:] & ~next_first[..., None]
    team_weight = team_valid.astype(jnp.float32)
    team_count = jnp.maximum(team_weight.sum(), 1.0)
    normalized_team_weight = team_weight / jnp.maximum(team_weight.mean(), 1e-8)

    predicted_embedding = prediction["embedding"].astype(jnp.float32)
    ema_target = jax.lax.stop_gradient(grouped_target[:, 1:].astype(jnp.float32))
    online_target = jax.lax.stop_gradient(grouped_online[:, 1:].astype(jnp.float32))
    pred_unit = predicted_embedding / jnp.maximum(
        jnp.linalg.norm(predicted_embedding, axis=-1, keepdims=True), 1e-8
    )
    ema_unit = ema_target / jnp.maximum(
        jnp.linalg.norm(ema_target, axis=-1, keepdims=True), 1e-8
    )
    embedding_loss = 1.0 - jnp.sum(pred_unit * ema_unit, axis=-1)
    if history_prediction is not None:
        history_unit = history_prediction / jnp.maximum(
            jnp.linalg.norm(history_prediction, axis=-1, keepdims=True), 1e-8
        )
        history_loss = 1.0 - jnp.sum(history_unit * ema_unit, axis=-1)
        # Zero forward contribution: add only the input Jacobian of this
        # same factual cosine objective, with all joint parameters frozen.
        embedding_loss += history_loss - jax.lax.stop_gradient(history_loss)
    interface_error = jnp.abs(predicted_embedding - online_target)
    interface_loss = jnp.where(
        interface_error < 1.0,
        0.5 * jnp.square(interface_error),
        interface_error - 0.5,
    ).mean(axis=-1)

    hidden = prediction["hidden"]
    reward_loss = self.ctde_rew(hidden, 3).loss(grouped_reward[:, 1:])
    continuation = (~self.team.unfold_sequence(obs["is_terminal"])[:, 1:]).astype(
        jnp.float32
    )
    if self.config.contdisc:
        continuation *= 1.0 - 1.0 / float(self.config.horizon)
    continuation_loss = self.ctde_con(hidden, 3).loss(continuation)
    mask_target = grouped_mask[:, 1:]
    mask_output = self.ctde_mask(hidden, 3)
    mask_loss = binary_vector_loss(
        mask_output,
        mask_target,
        self.action_mask_reduction,
    )
    mask_binary = (
        mask_output.output if isinstance(mask_output, jaxouts.Agg) else mask_output
    )
    mask_prediction = mask_binary.logit >= 0.0
    mask_event_weight = weight[..., None]
    positive_weight = mask_event_weight * mask_target.astype(jnp.float32)
    negative_weight = mask_event_weight * (~mask_target).astype(jnp.float32)
    attack_selector = (jnp.arange(mask_target.shape[-1], dtype=jnp.int32) >= 6).astype(
        jnp.float32
    )
    attack_positive_weight = positive_weight * attack_selector

    def event_rate(matches, event_weight):
        return (matches.astype(jnp.float32) * event_weight).sum() / jnp.maximum(
            event_weight.sum(), 1.0
        )

    mask_positive_recall = event_rate(mask_prediction, positive_weight)
    mask_negative_specificity = event_rate(~mask_prediction, negative_weight)
    attack_mask_positive_recall = event_rate(mask_prediction, attack_positive_weight)
    attack_mask_target_rate = attack_positive_weight.sum() / jnp.maximum(
        (mask_event_weight * attack_selector).sum(), 1.0
    )
    attack_mask_prediction_rate = (
        mask_prediction.astype(jnp.float32) * mask_event_weight * attack_selector
    ).sum() / jnp.maximum((mask_event_weight * attack_selector).sum(), 1.0)
    alive_loss = self.ctde_alive(hidden, 3).loss(grouped_alive[:, 1:])
    alive_valid = (source_present) & ~next_first[..., None]
    alive_weight = alive_valid.astype(jnp.float32)
    alive_count = jnp.maximum(alive_weight.sum(), 1.0)
    normalized_alive_weight = alive_weight / jnp.maximum(alive_weight.mean(), 1e-8)

    def masked_metric(value):
        return (value.astype(jnp.float32) * weight).sum() / count

    def team_metric(value):
        return (value.astype(jnp.float32) * team_weight).sum() / team_count

    folded_prediction = self.team.fold_sequence(prediction["embedding"])
    folded_online = self.team.fold_sequence(grouped_online[:, 1:])
    folded_deter = self.team.fold_sequence(
        self.team.unfold_sequence(repfeat["deter"])[:, 1:]
    )
    predicted_logits = self.dyn.posterior(folded_prediction, folded_deter)
    factual_logits = jax.lax.stop_gradient(
        self.dyn.posterior(folded_online, folded_deter)
    )
    predicted_logprob = jax.nn.log_softmax(
        predicted_logits.astype(jnp.float32), axis=-1
    )
    factual_logprob = jax.nn.log_softmax(factual_logits.astype(jnp.float32), axis=-1)
    factual_prob = jnp.exp(factual_logprob)
    posterior_kl = jnp.sum(
        factual_prob * (factual_logprob - predicted_logprob), axis=(-1, -2)
    )
    posterior_kl = self.team.unfold_sequence(posterior_kl)

    metrics = {
        "ctde/embedding_cosine": 1.0 - masked_metric(embedding_loss),
        "ctde/interface_smooth_l1": masked_metric(interface_loss),
        "ctde/reward_loss": team_metric(reward_loss),
        "ctde/continuation_loss": team_metric(continuation_loss),
        "ctde/team_signal_valid_fraction": team_weight.mean(),
        "ctde/alive_loss": (alive_loss.astype(jnp.float32) * alive_weight).sum()
        / alive_count,
        "ctde/posterior_kl": masked_metric(posterior_kl),
        "ctde/valid_fraction": weight.mean(),
        "ctde/controllable_alive_fraction": source_alive.mean(),
    }

    metrics.update(
        {
            "ctde/action_mask_loss": masked_metric(mask_loss),
            "ctde/action_mask_positive_recall": mask_positive_recall,
            "ctde/action_mask_negative_specificity": mask_negative_specificity,
            "ctde/attack_mask_positive_recall": attack_mask_positive_recall,
            "ctde/attack_mask_target_rate": attack_mask_target_rate,
            "ctde/attack_mask_prediction_rate": attack_mask_prediction_rate,
        }
    )

    def folded(value):
        value = value * normalized_weight
        value = jnp.pad(value, ((0, 0), (0, 1), (0, 0)))
        value *= value.shape[1] / max(value.shape[1] - 1, 1)
        return self.team.fold_sequence(value)

    def folded_alive(value):
        value = value * normalized_alive_weight
        value = jnp.pad(value, ((0, 0), (0, 1), (0, 0)))
        value *= value.shape[1] / max(value.shape[1] - 1, 1)
        return self.team.fold_sequence(value)

    def folded_team(value):
        value = value * normalized_team_weight
        value = jnp.pad(value, ((0, 0), (0, 1), (0, 0)))
        value *= value.shape[1] / max(value.shape[1] - 1, 1)
        return self.team.fold_sequence(value)

    losses = {
        "ctde_embedding": folded(embedding_loss),
        "ctde_interface": folded(interface_loss),
        "ctde_reward": folded_team(reward_loss),
        "ctde_continuation": folded_team(continuation_loss),
        "ctde_alive": folded_alive(alive_loss),
    }
    losses["ctde_action_mask"] = folded(mask_loss)
    if self.ctde_posterior_alignment_scale > 0:
        # Freeze the local posterior and factual history/teacher. Credit only
        # the joint producer for the distribution the existing interface induces.
        logits = frozen_posterior(self.dyn, folded_prediction, folded_deter)
        kl = self.team.unfold_sequence(
            mixed_posterior_kl(logits, factual_logits, self.dyn.unimix)
        )
        losses["ctde_posterior_alignment"] = folded(kl)
        metrics["ctde/dense_posterior_alignment_kl"] = masked_metric(kl)
    if self.ctde_multistep_jepa_enabled:
        multistep_losses, multistep_metrics = direct_jepa_losses(
            self,
            prediction["hidden"],
            source_state,
            grouped_target,
            grouped_present,
            grouped_alive,
            grouped_first,
            grouped_mask,
            grouped_action,
            training=training,
        )
        losses.update(multistep_losses)
        metrics.update(multistep_metrics)
    if training and self.ctde_self_fed_enabled:
        auxiliary_losses, auxiliary_metrics = isolated_creation_call(
            self_fed_losses,
            864_025,
            self,
            online_tokens,
            repfeat,
            dyn_entries,
            target_tokens,
            obs,
            prevact,
        )
        losses.update(auxiliary_losses)
        metrics.update(auxiliary_metrics)
    return losses, metrics
