"""Agent-axis runtime for the shared independent DreaMARL learner.

The public data contract retains team identity while local actors, critics, and
world models remain parameter-shared and observation-local. Team trajectory
grouping is retained for synchronized imagination and future training-only
team objectives. No peer tensor reaches the B0 execution path. For ``A=1``, the
implementation is exactly the locked single-agent learner.
"""

from __future__ import annotations

import elements
import embodied.jax
import jax
import jax.numpy as jnp
import ninjax as nj

from ..agent import Agent as LocalAgent
from ..models.team import (
    AgentContextEncoder,
    TeamActionConditioner,
    TeamContentPredictor,
    TeamSlotEncoder,
    TeamSlotPredictor,
    mask_active_agents,
    masked_agent_coverage_loss,
    scale_gradient,
    team_set_matching_loss,
    team_slot_jepa_loss,
    team_slot_regularization,
    team_utility_probe_metrics,
)
from .axes import TeamAxis
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
        return {
            key: (
                space
                if key in {"consec", "stepid"}
                else add_agent_axis(space, self.team.size)
            )
            for key, space in super().ext_space.items()
        }

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
        local_carry = self.team.fold_tree_batch(carry)
        local_data = self.team.local_sequence_data(data)
        local_carry, output, metrics = super().train(local_carry, local_data)
        if "replay" in output:
            output = dict(
                output,
                replay=self.team.unfold_replay_updates(output["replay"]),
            )
        return self.team.unfold_tree_batch(local_carry), output, metrics

    def report(self, carry, data):
        local_carry = self.team.fold_tree_batch(carry)
        local_data = self.team.local_sequence_data(data)
        local_carry, metrics = super().report(local_carry, local_data)
        return self.team.unfold_tree_batch(local_carry), metrics


class MARLCore(TeamAxisAdapter, LocalAgent):
    """B0/B1 learner with a permanently decentralized execution path."""

    def __init__(self, obs_space, act_space, config):
        marl = config.marl
        if str(marl.stage) not in {"b0", "b1"}:
            raise ValueError(f"unsupported MARL stage: {marl.stage!r}")
        if str(marl.execution) != "strict_decentralized":
            raise ValueError(f"unsupported execution contract: {marl.execution!r}")
        self.team = TeamAxis(int(config.num_agents))
        self.marl_stage = str(marl.stage)
        self.agent_jepa_enabled = self.marl_stage == "b1" and self.team.size > 1
        self.agent_jepa_future_enabled = self.agent_jepa_enabled and float(
            marl.agent_jepa.future_scale
        ) > 0.0
        local_obs_space = local_observation_spaces(obs_space, self.team.size)
        local_act_space = local_action_spaces(act_space, self.team.size)
        super().__init__(
            local_obs_space,
            local_act_space,
            config,
        )

    def additional_modules(self):
        modules = list(super().additional_modules())
        if not self.agent_jepa_enabled:
            return modules
        cfg = self.config.marl.agent_jepa
        kwargs = dict(
            slots=int(cfg.slots),
            width=int(cfg.width),
            heads=int(cfg.heads),
            layers=int(cfg.layers),
            ffup=int(cfg.ffup),
            act=str(cfg.act),
            norm=str(cfg.norm),
            winit=str(cfg.winit),
        )
        self.team_encoder = TeamSlotEncoder(**kwargs, name="team_encoder")
        self.target_team_encoder = TeamSlotEncoder(
            **kwargs, name="target_team_encoder"
        )
        self.slowteam = embodied.jax.SlowModel(
            self.target_team_encoder,
            source=self.team_encoder,
            rate=float(cfg.teacher_rate),
            every=int(cfg.teacher_every),
        )
        history_kwargs = dict(kwargs)
        history_kwargs.pop("layers")
        self.team_history_encoder = AgentContextEncoder(
            **history_kwargs, name="team_history_encoder"
        )
        self.team_predictor = TeamSlotPredictor(
            width=int(cfg.width),
            heads=int(cfg.heads),
            layers=int(cfg.predictor_layers),
            ffup=int(cfg.ffup),
            act=str(cfg.act),
            norm=str(cfg.norm),
            winit=str(cfg.winit),
            name="team_predictor",
        )
        self.team_content_predictor = TeamContentPredictor(
            self.enc_output_dim,
            hidden=int(cfg.predictor_hidden),
            act=str(cfg.act),
            norm=str(cfg.norm),
            winit=str(cfg.winit),
            name="team_content_predictor",
        )
        modules.extend(
            [
                self.team_encoder,
                self.team_history_encoder,
                self.team_predictor,
                self.team_content_predictor,
            ]
        )
        if not self.agent_jepa_future_enabled:
            return modules
        discrete_actions = [
            key for key, space in self.act_space.items() if space.discrete
        ]
        if len(self.act_space) != 1 or len(discrete_actions) != 1:
            raise ValueError(
                "B1 future team JEPA requires exactly one discrete action"
            )
        self.team_action_key = discrete_actions[0]
        action_space = self.act_space[self.team_action_key]
        self.team_action_low = int(action_space.low)
        self.team_action_count = int(action_space.high - action_space.low)
        self.team_action_conditioner = TeamActionConditioner(
            self.enc_output_dim,
            hidden=int(cfg.predictor_hidden),
            act=str(cfg.act),
            norm=str(cfg.norm),
            winit=str(cfg.winit),
            name="team_action_conditioner",
        )
        self.team_transition_encoder = TeamSlotEncoder(
            **kwargs, name="team_transition_encoder"
        )
        self.team_transition_predictor = TeamSlotPredictor(
            width=int(cfg.width),
            heads=int(cfg.heads),
            layers=int(cfg.predictor_layers),
            ffup=int(cfg.ffup),
            act=str(cfg.act),
            norm=str(cfg.norm),
            winit=str(cfg.winit),
            name="team_transition_predictor",
        )
        modules.extend(
            [
                self.team_action_conditioner,
                self.team_transition_encoder,
                self.team_transition_predictor,
            ]
        )
        return modules

    def _update_slow_models(self):
        super()._update_slow_models()
        if self.agent_jepa_enabled:
            self.slowteam.update()

    def additional_world_model_losses(
        self,
        tokens,
        repfeat,
        target_tokens,
        obs,
        prevact,
        training,
    ):
        if not self.agent_jepa_enabled:
            return {}, {}
        if target_tokens is None:
            raise RuntimeError("B1 agent JEPA requires EMA encoder targets")

        cfg = self.config.marl.agent_jepa
        local_grad_scale = float(cfg.local_grad_scale)
        online_members = self.team.unfold_sequence(tokens)
        online_members = scale_gradient(online_members, local_grad_scale)
        histories = self.team.unfold_sequence(self.feat2tensor(repfeat))
        histories = scale_gradient(histories, local_grad_scale)
        targets = self.team.unfold_sequence(target_tokens)
        active = self.team.unfold_sequence(self._active(obs))
        visible, hidden, eligible = mask_active_agents(
            active,
            nj.seed(),
            minimum=float(cfg.mask_min),
            maximum=float(cfg.mask_max),
        )

        # The online branch sees no content or local state from hidden agents.
        content_slots = self.team_encoder(online_members, visible, active)
        history_slots = self.team_history_encoder(histories, visible)
        predicted_slots = self.team_predictor(content_slots, history_slots)

        # The training-only teacher sees all active EMA local embeddings. Its
        # weights are an EMA of the online set encoder, never optimizer targets.
        target_slots = self.slowteam(targets, active, active)
        target_slots = jax.lax.stop_gradient(target_slots)
        team_loss, metrics = team_slot_jepa_loss(
            predicted_slots, target_slots, eligible
        )

        # Full online slots receive explicit anti-collapse pressure. The target
        # branch remains stop-gradient and follows this encoder through EMA.
        full_online_slots = self.team_encoder(online_members, active, active)
        regularizer_valid = active.any(axis=-1)
        predicted_content = self.team_content_predictor(predicted_slots)
        full_online_content = self.team_content_predictor(full_online_slots)
        predicted_set_loss, predicted_set_metrics = team_set_matching_loss(
            predicted_content,
            targets,
            active,
            eligible,
            temperature=float(cfg.matching_temperature),
            iterations=int(cfg.sinkhorn_iterations),
            name="predicted_set",
        )
        source_set_loss, source_set_metrics = team_set_matching_loss(
            full_online_content,
            targets,
            active,
            eligible,
            temperature=float(cfg.matching_temperature),
            iterations=int(cfg.sinkhorn_iterations),
            name="source_set",
        )
        hidden_coverage_loss, hidden_coverage_metrics = (
            masked_agent_coverage_loss(
                predicted_content,
                targets,
                active,
                hidden,
                eligible,
                temperature=float(cfg.matching_temperature),
            )
        )
        metrics.update(predicted_set_metrics)
        metrics.update(source_set_metrics)
        metrics.update(hidden_coverage_metrics)
        variance_loss, covariance_loss, regularizer_metrics = (
            team_slot_regularization(
                full_online_slots,
                regularizer_valid,
                target_std=float(cfg.slot_target_std),
            )
        )
        metrics.update(regularizer_metrics)

        future_composite = jnp.zeros_like(team_loss)
        if self.agent_jepa_future_enabled:
            # The replay action paired with state t is prevact[t + 1]. Preserve
            # that per-agent pairing before permutation-invariant team pooling.
            action = self.team.unfold_sequence(
                prevact[self.team_action_key]
            )[:, 1:]
            action = jax.nn.one_hot(
                action.astype(jnp.int32) - self.team_action_low,
                self.team_action_count,
                dtype=jnp.float32,
            )
            source_active = active[:, :-1]
            conditioned_members = self.team_action_conditioner(
                online_members[:, :-1],
                action,
                visible[:, :-1],
                source_active,
            )
            action_slots = self.team_transition_encoder(
                conditioned_members,
                source_active,
                source_active,
            )
            future_prediction = self.team_transition_predictor(
                predicted_slots[:, :-1], action_slots
            )
            reset = self.team.unfold_sequence(obs["is_first"]).any(
                axis=-1
            )[:, 1:]
            future_valid = (
                eligible[:, :-1]
                & source_active.any(axis=-1)
                & active[:, 1:].any(axis=-1)
                & ~reset
            )
            future_loss, future_metrics = team_slot_jepa_loss(
                future_prediction,
                target_slots[:, 1:],
                future_valid,
                name="future",
            )
            future_content = self.team_content_predictor(future_prediction)
            future_set_loss, future_set_metrics = team_set_matching_loss(
                future_content,
                targets[:, 1:],
                active[:, 1:],
                future_valid,
                temperature=float(cfg.matching_temperature),
                iterations=int(cfg.sinkhorn_iterations),
                name="future_set",
            )
            metrics.update(future_metrics)
            metrics.update(future_set_metrics)
            future_composite = (
                future_loss + float(cfg.future_set_scale) * future_set_loss
            )
            future_composite = jnp.pad(future_composite, ((0, 0), (0, 1)))
            future_composite *= future_composite.shape[1] / max(
                future_composite.shape[1] - 1, 1
            )
            metrics.update(
                {
                    "agent_jepa/future_loss": future_loss.mean(),
                    "agent_jepa/future_set_loss": future_set_loss.mean(),
                    "agent_jepa/future_valid_fraction": future_valid.mean(),
                }
            )
        if bool(getattr(cfg, "utility_probe", False)) and not training:
            action_keys = [
                key for key, space in self.act_space.items() if space.discrete
            ]
            if len(action_keys) != 1:
                raise ValueError(
                    "B1 utility probe requires exactly one discrete action"
                )
            action_key = action_keys[0]
            action_space = self.act_space[action_key]
            metrics.update(
                team_utility_probe_metrics(
                    {
                        "visible_content": content_slots,
                        "visible_history": history_slots,
                        "predicted_team": predicted_slots,
                        "teacher_team": target_slots,
                    },
                    self.team.unfold_sequence(prevact[action_key]),
                    self.team.unfold_sequence(obs["reward"]),
                    active,
                    hidden,
                    eligible,
                    self.team.unfold_sequence(obs["is_first"]).any(axis=-1),
                    action_count=int(action_space.high - action_space.low),
                )
            )

        weight = eligible.astype(jnp.float32)
        weight = weight / jnp.maximum(weight.mean(), 1e-8)
        regularizer = (
            float(cfg.variance_scale) * variance_loss
            + float(cfg.covariance_scale) * covariance_loss
        )
        k0_composite = (
            team_loss
            + float(cfg.predicted_set_scale) * predicted_set_loss
            + float(cfg.source_set_scale) * source_set_loss
            + float(cfg.hidden_coverage_scale) * hidden_coverage_loss
            + weight * regularizer
        )
        composite = (
            float(cfg.k0_scale) * k0_composite
            + float(cfg.future_scale) * future_composite
        )
        metrics.update(
            {
                "agent_jepa/team_loss": team_loss.mean(),
                "agent_jepa/predicted_set_loss": predicted_set_loss.mean(),
                "agent_jepa/source_set_loss": source_set_loss.mean(),
                "agent_jepa/hidden_coverage_loss": hidden_coverage_loss.mean(),
                "agent_jepa/regularizer": regularizer,
                "agent_jepa/k0_composite_loss": k0_composite.mean(),
                "agent_jepa/masked_fraction": hidden.sum()
                / jnp.maximum(active.sum(), 1),
            }
        )
        composite = jnp.broadcast_to(
            composite[:, :, None], (*composite.shape, self.team.size)
        )
        return {"agent_jepa": self.team.fold_sequence(composite)}, metrics

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

    def dynamics_loss(self, carry, tokens, actions, reset, obs, training):
        return self.dyn.loss(
            carry,
            tokens,
            actions,
            reset,
            training,
            active=self._active(obs),
        )

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

    def imagination_starts(
        self,
        dyn_entries,
        dyn_carry,
        repfeat,
        obs,
        starts_count,
    ):
        starts, first, _ = super().imagination_starts(
            dyn_entries, dyn_carry, repfeat, obs, starts_count
        )
        grouped = self.team.group_tree_starts(starts, starts_count)
        starts = self.team.fold_tree_batch(grouped)
        first = self.team.fold_tree_batch(
            self.team.group_tree_starts(first, starts_count)
        )
        active = self._active(obs)[:, -starts_count:].reshape((-1,))
        active = self.team.fold_batch(self.team.group_starts(active, starts_count))
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

    def restore_imagination_results(self, losses, outputs, context=None):
        starts_count, _ = context

        def restore(value):
            grouped = self.team.unfold_batch(value)
            return self.team.ungroup_starts(grouped, starts_count)

        return jax.tree.map(restore, (losses, outputs))

    def imagination_validity(self, context, horizon):
        _, active = context
        return jnp.broadcast_to(active[:, None], (active.shape[0], horizon))

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


__all__ = ["MARLCore", "TeamAxisAdapter"]
