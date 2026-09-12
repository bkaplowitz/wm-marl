"""Agent-axis runtime for the maintained MA-JEPA learner.

The public data contract retains team identity while local actors and world
models remain parameter-shared and observation-local. Team trajectory grouping
supports synchronized imagination and training-only team objectives. The CTDE
configuration adds a joint JEPA simulator and central attention critic while
preserving the local executable actor boundary. Training uses synchronized team trajectories; executable actors remain local.
"""

from __future__ import annotations

import math

import elements
import embodied.jax
import embodied.jax.nets as nn
import embodied.jax.outs as jaxouts
import jax
import jax.numpy as jnp
import ninjax as nj

from ..agent import Agent as LocalAgent
from ..models.ctde import (
    CentralAttentionCritic,
    JointObservationJEPA,
)
from ..models.multistep_jepa import (
    ActionConditionedMultiStepJEPA,
    isolated_creation_call,
)
from ..models.heads import (
    binary_vector_loss,
)
from ..training.ctde import (
    imagined_action_mask,
    shared_team_outcomes,
)
from ..training.multistep_jepa import (
    aligned_action_windows,
    all_legal_same_focal_action_interventions,
    direct_multistep_objective,
)
from ..training.self_fed import self_fed_losses, mixed_posterior_kl, frozen_posterior
from ..models.target import CriticTarget
from .axes import (
    BEHAVIOR_REPLAY_PREFIX,
    TeamAxis,
    is_environment_field,
    split_prefixed_data,
)
from .spaces import (
    add_agent_axis,
    local_action_spaces,
    local_observation_spaces,
    report_rows,
)


class TeamAxisAdapter:
    """Apply a shared local learner without discarding team identity."""

    @property
    def ext_space(self):
        local_spaces = {
            key: (
                space
                if is_environment_field(key)
                else add_agent_axis(space, self.team.size)
            )
            for key, space in super().ext_space.items()
        }
        spaces = dict(local_spaces)
        # Learner-control inputs are stamped after replay sampling. They are
        # intentionally absent from independent replay views.
        replay_local_spaces = {
            key: value
            for key, value in local_spaces.items()
            if key != "_environment_step"
        }
        replay_view = {
            **self.public_obs_space,
            **self.public_act_space,
            **replay_local_spaces,
        }
        spaces.update(
            {
                f"{BEHAVIOR_REPLAY_PREFIX}{key}": space
                for key, space in replay_view.items()
            }
        )
        return spaces

    def init_policy(self, batch_size):
        return self.team.unfold_tree_batch(
            super().init_policy(batch_size * self.team.size)
        )

    def init_train(self, batch_size):
        return self.init_policy(batch_size)

    def init_report(self, batch_size):
        return self.init_policy(batch_size)

    def report_rows(self, batch_size):
        return report_rows(batch_size, self.team.size)

    def policy(self, carry, obs, mode="train"):
        local_carry = self.team.fold_tree_batch(carry)
        local_obs = self.team.local_policy_data(obs)
        local_carry, action, output = super().policy(local_carry, local_obs, mode)
        return (
            self.team.unfold_tree_batch(local_carry),
            self.team.unfold_tree_batch(action),
            self.team.unfold_tree_batch(output),
        )

    def train(self, carry, data):
        data, behavior = split_prefixed_data(data)
        if not behavior:
            raise ValueError(
                "recent_world_uniform_behavior requires an independent "
                f"{BEHAVIOR_REPLAY_PREFIX} batch"
            )
        local_carry = self.team.fold_tree_batch(carry)
        local_data = self.team.local_sequence_data(data)
        local_behavior = self.team.local_sequence_data(behavior) if behavior else None
        local_carry, output, metrics = super().train(
            local_carry,
            local_data,
            behavior_data=local_behavior,
        )
        if "replay" in output:
            output = dict(
                output,
                replay=self.team.unfold_replay_updates(output["replay"]),
            )
        return self.team.unfold_tree_batch(local_carry), output, metrics

    def report(self, carry, data):
        data, _ = split_prefixed_data(data)
        local_carry = self.team.fold_tree_batch(carry)
        local_data = self.team.local_sequence_data(data)
        local_carry, metrics = super().report(local_carry, local_data)
        return self.team.unfold_tree_batch(local_carry), metrics


class MARLCore(TeamAxisAdapter, LocalAgent):
    """Agent-axis learner with a permanently decentralized actor path."""

    def __init__(self, obs_space, act_space, config):
        marl = config.marl
        if str(marl.stage) != "ctde" or str(marl.execution) != "strict_decentralized":
            raise ValueError("MA-JEPA requires CTDE with decentralized execution")
        self.team = TeamAxis(int(config.num_agents))
        if self.team.size < 2:
            raise ValueError("MA-JEPA requires at least two agents")
        self.public_obs_space = dict(obs_space)
        self.public_act_space = dict(act_space)
        self_fed = marl.ctde.self_fed
        if int(self_fed.bptt_steps) != 2:
            raise ValueError("The maintained recurrent objective uses BPTT2")
        horizons = tuple(int(value) for value in self_fed.horizons)
        if (
            not horizons
            or horizons != tuple(sorted(set(horizons)))
            or min(horizons) < 2
            or max(horizons) > 15
            or int(self_fed.anchors) < 1
        ):
            raise ValueError(
                "Recurrent supervision needs sorted unique H2..15 and positive anchors"
            )
        for value in (self_fed.scale, self_fed.trajectory_kl_scale):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError("Recurrent loss scales must be finite and nonnegative")
        multistep = marl.ctde.multistep_jepa
        self.ctde_multistep_jepa_horizons = tuple(int(x) for x in multistep.horizons)
        self.ctde_multistep_jepa_max_horizon = int(multistep.max_horizon)
        self.ctde_multistep_jepa_decay = float(multistep.decay)
        self.ctde_multistep_jepa_action_margin = float(multistep.action_margin)
        self.ctde_multistep_jepa_action_scale = float(
            config.loss_scales.ctde_multistep_jepa_action
        )
        horizons = self.ctde_multistep_jepa_horizons
        if (
            not horizons
            or horizons != tuple(sorted(set(horizons)))
            or min(horizons) < 1
            or max(horizons) != self.ctde_multistep_jepa_max_horizon
        ):
            raise ValueError(
                "Multi-step JEPA horizons must be sorted unique positives ending at K"
            )
        if not 0 < self.ctde_multistep_jepa_decay <= 1:
            raise ValueError("Multi-step JEPA decay must be in (0, 1]")
        if (
            min(
                self.ctde_multistep_jepa_action_scale,
                self.ctde_multistep_jepa_action_margin,
            )
            < 0
        ):
            raise ValueError("Action-margin scale and margin must be nonnegative")
        if str(multistep.action_counterfactual_mode) != "all_legal_mean":
            raise ValueError("MA-JEPA uses all-legal action counterfactuals")
        if int(config.batch_length) <= max(2, self.ctde_multistep_jepa_max_horizon):
            raise ValueError("Replay suffix must exceed the JEPA prediction horizon")
        self.action_mask_reduction = str(config.action_mask_reduction)
        if self.action_mask_reduction != "balanced":
            raise ValueError("The maintained availability objective uses balanced BCE")
        if str(marl.ctde.imagination_mask_sampling) != "bernoulli":
            raise ValueError("Imagination requires sampled Bernoulli availability")
        self.ctde_posterior_alignment_scale = float(
            config.loss_scales.ctde_posterior_alignment
        )
        if (
            not math.isfinite(self.ctde_posterior_alignment_scale)
            or self.ctde_posterior_alignment_scale < 0
        ):
            raise ValueError("Posterior alignment scale must be finite and nonnegative")
        joint_burnin = int(marl.ctde.joint.context) * int(
            marl.ctde.joint.temporal_layers
        )
        if int(config.replay_context) < joint_burnin:
            raise ValueError(
                f"replay_context must cover the joint receptive field ({joint_burnin})"
            )
        super().__init__(
            local_observation_spaces(obs_space, self.team.size),
            local_action_spaces(act_space, self.team.size),
            config,
        )
        scale = float(self_fed.scale)
        for name in (
            "embedding",
            "interface",
            "reward",
            "continuation",
            "action_mask",
            "alive",
        ):
            self.scales[f"ctde_self_fed_{name}"] = scale * float(
                self.scales[f"ctde_{name}"]
            )
        if float(self_fed.trajectory_kl_scale):
            self.scales["ctde_self_fed_trajectory_kl"] = scale * float(
                self_fed.trajectory_kl_scale
            )

    def _make_value_models(self, scalar, config):
        cfg = config.marl.ctde.critic
        value = CentralAttentionCritic(
            width=int(cfg.width),
            heads=int(cfg.heads),
            layers=int(cfg.layers),
            ffup=int(cfg.ffup),
            dropout=float(cfg.dropout),
            act=str(cfg.act),
            norm=str(cfg.norm),
            winit=str(cfg.winit),
            value_layers=int(cfg.value_layers),
            value_units=int(cfg.value_units),
            bins=int(cfg.bins),
            outscale=float(cfg.outscale),
            name="ctde_val",
        )
        slowvalue = CriticTarget(
            CentralAttentionCritic(
                width=int(cfg.width),
                heads=int(cfg.heads),
                layers=int(cfg.layers),
                ffup=int(cfg.ffup),
                dropout=float(cfg.dropout),
                act=str(cfg.act),
                norm=str(cfg.norm),
                winit=str(cfg.winit),
                value_layers=int(cfg.value_layers),
                value_units=int(cfg.value_units),
                bins=int(cfg.bins),
                outscale=float(cfg.outscale),
                name="slowctde_val",
            ),
            source=value,
            **config.slowvalue,
        )
        return value, slowvalue

    def critic(self, features, bdims, *, slow=False, context=None):
        if bdims != 2 or context is None:
            raise ValueError("CTDE critic requires synchronized sequence activity")
        local_state = (
            self.feat2tensor(features) if isinstance(features, dict) else features
        )
        grouped_state = self.team.unfold_sequence(local_state)
        grouped_present = context["present"].astype(bool)
        grouped_alive = context["controllable_alive"]
        grouped_alive = grouped_alive.astype(bool)
        if (
            grouped_present.shape != grouped_state.shape[:3]
            or grouped_alive.shape != grouped_state.shape[:3]
        ):
            raise ValueError(
                "CTDE critic roster/liveness does not match grouped local states: "
                f"{grouped_present.shape}, {grouped_alive.shape} versus "
                f"{grouped_state.shape}"
            )
        value_head = self.slowval if slow else self.val
        distribution = value_head(
            grouped_state,
            grouped_present,
            grouped_alive,
            bdims=3,
        )
        logits = self.team.fold_sequence(distribution.logits)
        return jaxouts.TwoHot(logits, distribution.bins)

    def additional_modules(self):
        modules = list(super().additional_modules())
        cfg = self.config.marl.ctde
        discrete_actions = [
            key for key, space in self.act_space.items() if space.discrete
        ]
        if len(self.act_space) != 1 or len(discrete_actions) != 1:
            raise ValueError("CTDE requires exactly one categorical action")
        if self.action_mask_key is None:
            raise ValueError("CTDE requires an environment action mask")
        self.ctde_action_key = discrete_actions[0]
        action_space = self.act_space[self.ctde_action_key]
        self.ctde_action_low = int(action_space.low)
        self.ctde_action_count = int(action_space.high - action_space.low)
        common = dict(
            act=str(cfg.joint.act),
            norm=str(cfg.joint.norm),
            winit=str(cfg.joint.winit),
        )
        self.ctde_joint = JointObservationJEPA(
            self.ctde_action_count,
            self.ctde_action_low,
            self.enc_output_dim,
            width=int(cfg.joint.width),
            heads=int(cfg.joint.heads),
            agent_layers=int(cfg.joint.agent_layers),
            temporal_layers=int(cfg.joint.temporal_layers),
            context=int(cfg.joint.context),
            ffup=int(cfg.joint.ffup),
            dropout=float(cfg.joint.dropout),
            action_conditioning=str(cfg.joint.action_conditioning),
            latent_stoch=0,
            latent_classes=0,
            **common,
            name="ctde_joint",
        )
        scalar = elements.Space(jnp.float32, ())
        binary = elements.Space(bool, (), 0, 2)
        head = dict(
            layers=int(cfg.head.layers),
            units=int(cfg.head.units),
            act=str(cfg.head.act),
            norm=str(cfg.head.norm),
            winit=str(cfg.head.winit),
        )
        self.ctde_rew = embodied.jax.MLPHead(
            scalar,
            output="symexp_twohot",
            bins=int(cfg.head.bins),
            outscale=float(cfg.head.outscale),
            **head,
            name="ctde_rew",
        )
        self.ctde_con = embodied.jax.MLPHead(
            binary,
            output="binary",
            outscale=1.0,
            **head,
            name="ctde_con",
        )
        mask_space = self.obs_space["action_mask"]
        self.ctde_mask = embodied.jax.MLPHead(
            mask_space,
            output="binary",
            outscale=0.0,
            **head,
            name="ctde_mask",
        )
        self.ctde_alive = embodied.jax.MLPHead(
            binary,
            output="binary",
            outscale=1.0,
            **head,
            name="ctde_alive",
        )
        ctde_modules = [
            self.ctde_joint,
            self.ctde_rew,
            self.ctde_con,
            self.ctde_mask,
            self.ctde_alive,
        ]
        actor_modules = []
        multistep = cfg.multistep_jepa
        self.ctde_multistep_jepa = ActionConditionedMultiStepJEPA(
            self.ctde_action_count,
            self.ctde_action_low,
            self.enc_output_dim,
            self.ctde_multistep_jepa_horizons,
            self.ctde_multistep_jepa_max_horizon,
            width=int(multistep.width),
            layers=int(multistep.layers),
            units=int(multistep.units),
            act=str(multistep.act),
            norm=str(multistep.norm),
            winit=str(multistep.winit),
            name="ctde_multistep_jepa",
        )
        ctde_modules.append(self.ctde_multistep_jepa)
        self.ctde_modules = tuple(ctde_modules)
        self.ctde_actor_modules = tuple(actor_modules)
        modules.extend(self.ctde_modules)
        modules.extend(self.ctde_actor_modules)
        return modules

    def imagination_critic_context(self, features, context, auxiliary=None):
        if auxiliary is None:
            raise ValueError("CTDE critic requires imagined activity")
        metrics = {}
        return {
            "present": auxiliary["present"],
            "controllable_alive": auxiliary["controllable_alive"],
        }, metrics

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
        return self._ctde_replay_losses(
            tokens,
            repfeat,
            dyn_entries,
            target_tokens,
            obs,
            prevact,
            training,
        )

    def _ctde_replay_losses(
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
        grouped_action = self.team.unfold_sequence(
            prevact[self.ctde_action_key]
        ).astype(jnp.int32)

        source_state = grouped_state[:, :-1]
        source_action = grouped_action[:, 1:]
        source_present = grouped_present[:, :-1]
        source_alive = grouped_alive[:, :-1]
        reset = grouped_first[:, :-1]
        cache = dyn_entries["ctde_joint_carry"]
        _, prediction, snapshots = self.ctde_joint.sequence(
            cache,
            source_state,
            source_action,
            source_present,
            source_alive,
            reset,
            training,
        )

        next_first = grouped_first[:, 1:]
        transition_valid = (
            source_alive & grouped_present[:, 1:] & ~next_first[..., None]
        )
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
        attack_selector = (
            jnp.arange(mask_target.shape[-1], dtype=jnp.int32) >= 6
        ).astype(jnp.float32)
        attack_positive_weight = positive_weight * attack_selector

        def event_rate(matches, event_weight):
            return (matches.astype(jnp.float32) * event_weight).sum() / jnp.maximum(
                event_weight.sum(), 1.0
            )

        mask_positive_recall = event_rate(mask_prediction, positive_weight)
        mask_negative_specificity = event_rate(~mask_prediction, negative_weight)
        attack_mask_positive_recall = event_rate(
            mask_prediction, attack_positive_weight
        )
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
        factual_logprob = jax.nn.log_softmax(
            factual_logits.astype(jnp.float32), axis=-1
        )
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
            "ctde/action_mask_loss": masked_metric(mask_loss),
            "ctde/action_mask_positive_recall": mask_positive_recall,
            "ctde/action_mask_negative_specificity": mask_negative_specificity,
            "ctde/attack_mask_positive_recall": attack_mask_positive_recall,
            "ctde/attack_mask_target_rate": attack_mask_target_rate,
            "ctde/attack_mask_prediction_rate": attack_mask_prediction_rate,
            "ctde/alive_loss": (alive_loss.astype(jnp.float32) * alive_weight).sum()
            / alive_count,
            "ctde/posterior_kl": masked_metric(posterior_kl),
            "ctde/valid_fraction": weight.mean(),
            "ctde/controllable_alive_fraction": source_alive.mean(),
        }

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
            "ctde_action_mask": folded(mask_loss),
            "ctde_alive": folded_alive(alive_loss),
        }
        if self.ctde_posterior_alignment_scale > 0:
            # Freeze the local posterior and factual history/teacher. Credit only
            # the joint producer for the distribution the existing interface induces.
            logits = frozen_posterior(self.dyn, folded_prediction, folded_deter)
            kl = self.team.unfold_sequence(
                mixed_posterior_kl(logits, factual_logits, self.dyn.unimix)
            )
            losses["ctde_posterior_alignment"] = folded(kl)
            metrics["ctde/dense_posterior_alignment_kl"] = masked_metric(kl)
        multistep_losses, multistep_metrics = self._ctde_direct_multistep_jepa_loss(
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
        if training:
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
        if not training:
            metrics.update(
                self._ctde_self_fed_report(
                    repfeat,
                    dyn_entries,
                    snapshots,
                    online_tokens,
                    target_tokens,
                    obs,
                    prevact,
                )
            )
        return losses, metrics

    def _ctde_direct_multistep_jepa_loss(
        self,
        grouped_hidden,
        grouped_source_state,
        grouped_target,
        grouped_present,
        grouped_alive,
        grouped_first,
        grouped_mask,
        grouped_action,
        *,
        training,
    ):
        """Fit EMA futures from joint roots and focal replay-action tails."""
        max_horizon = self.ctde_multistep_jepa_max_horizon
        length = grouped_target.shape[1]
        roots = length - max_horizon
        expected = (grouped_target.shape[0], length - 1, self.team.size)
        if grouped_hidden.shape[:3] != expected:
            raise ValueError(
                "multi-step shared hidden is not aligned with factual replay: "
                f"{grouped_hidden.shape[:3]} versus {expected}"
            )
        if grouped_source_state.shape[:3] != expected:
            raise ValueError(
                "multi-step local roots are not aligned with factual replay: "
                f"{grouped_source_state.shape[:3]} versus {expected}"
            )

        action_windows, all_valid = aligned_action_windows(
            grouped_action[:, 1:],
            grouped_mask[:, :-1],
            grouped_present,
            grouped_alive,
            grouped_first,
            action_low=self.ctde_action_low,
            max_horizon=max_horizon,
        )
        horizons = self.ctde_multistep_jepa_horizons
        valid = {horizon: all_valid[horizon] for horizon in horizons}
        root_hidden = grouped_hidden[:, :roots]

        predictions = isolated_creation_call(
            self.ctde_multistep_jepa,
            0x4D534A50,
            root_hidden,
            action_windows,
        )
        if self.ctde_multistep_jepa_action_scale > 0.0:
            counterfactual_windows, distinct_tail = (
                all_legal_same_focal_action_interventions(
                    action_windows,
                    grouped_mask[:, :-1],
                    action_low=self.ctde_action_low,
                    horizons=horizons,
                )
            )
            counterfactual_predictions = {}
            counterfactual_valid = {}
            classes = grouped_mask.shape[-1]
            batch = root_hidden.shape[0]
            expanded_root = jnp.broadcast_to(
                root_hidden[:, None],
                (batch, classes, *root_hidden.shape[1:]),
            ).reshape((batch * classes, *root_hidden.shape[1:]))
            for horizon in horizons:
                candidate_valid = valid[horizon][..., None] & distinct_tail[horizon]
                counterfactual_valid[horizon] = candidate_valid
                if horizon == 1:
                    counterfactual_predictions[horizon] = jnp.broadcast_to(
                        jax.lax.stop_gradient(predictions[horizon])[..., None, :],
                        (
                            *predictions[horizon].shape[:-1],
                            classes,
                            predictions[horizon].shape[-1],
                        ),
                    )
                    continue
                window = counterfactual_windows[horizon]
                flat_window = jnp.transpose(window, (0, 3, 1, 2, 4)).reshape(
                    (batch * classes, *window.shape[1:3], window.shape[-1])
                )
                flat_prediction = self.ctde_multistep_jepa(
                    expanded_root,
                    flat_window,
                    selected_horizon=horizon,
                )[horizon]
                counterfactual_predictions[horizon] = jnp.transpose(
                    flat_prediction.reshape(
                        (batch, classes, *flat_prediction.shape[1:])
                    ),
                    (0, 2, 3, 1, 4),
                )
            counterfactual_enabled = jnp.asarray(1.0, jnp.float32)
        else:
            counterfactual_predictions = {
                horizon: jax.lax.stop_gradient(predictions[horizon])
                for horizon in horizons
            }
            counterfactual_valid = {
                horizon: jnp.zeros_like(valid[horizon]) for horizon in horizons
            }
            distinct_tail = {
                horizon: jnp.zeros_like(valid[horizon]) for horizon in horizons
            }
            counterfactual_enabled = jnp.asarray(0.0, jnp.float32)

        targets = {
            horizon: jax.lax.stop_gradient(grouped_target[:, horizon : horizon + roots])
            for horizon in horizons
        }
        root_losses, raw_metrics = direct_multistep_objective(
            predictions,
            targets,
            valid,
            counterfactual_predictions,
            counterfactual_valid,
            distinct_tail,
            horizons=horizons,
            decay=self.ctde_multistep_jepa_decay,
            action_margin=self.ctde_multistep_jepa_action_margin,
        )
        losses = {}
        for name, root_loss in root_losses.items():
            padded = jnp.pad(root_loss, ((0, 0), (0, max_horizon), (0, 0)))
            padded *= length / roots
            losses[f"ctde_multistep_jepa_{name}"] = self.team.fold_sequence(padded)

        metrics = {
            f"ctde/multistep_jepa_{key}": value for key, value in raw_metrics.items()
        }
        metrics.update(
            {
                "ctde/multistep_jepa_root_count": jnp.asarray(
                    grouped_target.shape[0] * roots * self.team.size,
                    jnp.float32,
                ),
                "ctde/multistep_jepa_max_horizon": jnp.asarray(
                    max_horizon, jnp.float32
                ),
                "ctde/multistep_jepa_belief_context_enabled": jnp.asarray(
                    float(False), jnp.float32
                ),
                "ctde/multistep_jepa_action_counterfactual_enabled": (
                    counterfactual_enabled
                ),
                "ctde/multistep_jepa_recent_training_view": jnp.asarray(
                    float(bool(training)), jnp.float32
                ),
            }
        )
        metrics["ctde/multistep_jepa_action_counterfactual_all_legal_enabled"] = (
            jnp.asarray(float(self.ctde_multistep_jepa_action_scale > 0.0), jnp.float32)
        )
        return losses, metrics

    def _apply_replay_context(self, carry, data):

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

    def imagination_starts(
        self,
        dyn_entries,
        dyn_carry,
        repfeat,
        obs,
        prevact,
        starts_count,
    ):
        starts, first, _ = super().imagination_starts(
            dyn_entries, dyn_carry, repfeat, obs, prevact, starts_count
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

    def _ctde_complete(self, cache, deter, prediction):
        """Complete a local temporal step using the configured joint simulator."""
        return self.dyn.complete_from_observation(
            cache, deter, self.team.fold_batch(prediction["embedding"]), sample=True
        )

    def imagine_with_aux(self, starts, horizon, training, context=None):
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
            # availability sampling does not shift other learner RNG draws.
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
            mask_output = self.ctde_mask(hidden, 2)
            mask_probability = jax.nn.sigmoid(mask_output.output.logit)
            # Store the realized mask in auxiliary below. PPO must condition on
            # that same mask in every epoch, never redraw it for likelihoods.
            next_mask = imagined_action_mask(
                mask_probability,
                next_alive,
                jax.random.fold_in(action_seed, 0x4D41534B),
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
                "bernoulli" == "bernoulli", jnp.float32
            ),
            "imagined_action/noop_fraction": fraction(action == 0),
            "imagined_action/stop_fraction": fraction(action == 1),
            "imagined_action/move_fraction": fraction(
                (action >= 2) & (action < attack_start)
            ),
            "imagined_action/attack_fraction": fraction(action >= attack_start),
        }

    def report_imagination(self, carry, actions, length, training):
        return self.dyn.imagine(
            carry,
            actions,
            length,
            training,
        )

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


__all__ = ["MARLCore", "TeamAxisAdapter"]
