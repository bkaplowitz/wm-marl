"""Agent-axis runtime for the maintained MA-JEPA learner.

The public data contract retains team identity while local actors and world
models remain parameter-shared and observation-local. Team trajectory grouping
supports synchronized imagination and training-only team objectives. The CTDE
configuration adds a joint JEPA simulator and central attention critic while
preserving the local executable actor boundary. For ``A=1``, the implementation
is exactly the canonical single-agent learner.
"""

from __future__ import annotations

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
    TeammateActionBelief,
    TeammateBeliefActorAdapter,
)
from ..models.multistep_jepa import (
    ActionConditionedMultiStepJEPA,
    TeammateActionPlanGRU,
    isolated_creation_call,
)
from ..models.heads import (
    apply_action_mask,
    apply_predicted_action_mask,
    apply_support_preserving_availability,
    balanced_binary_event_loss,
    binary_vector_loss,
)
from ..training.ctde import (
    detach_self_feed,
    gather_anchors,
    predicted_controllable_alive,
    sample_two_step_anchors,
    two_step_anchor_mask,
    two_step_objective,
)
from ..training.multistep_jepa import (
    aligned_action_windows,
    all_legal_same_focal_action_interventions,
    authoritative_action_binding_objective,
    direct_multistep_objective,
)
from ..training.common import sample
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
        if self.two_branch_replay:
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
        if self.two_branch_replay and not behavior:
            raise ValueError(
                "recent_world_uniform_behavior requires an independent "
                f"{BEHAVIOR_REPLAY_PREFIX} batch"
            )
        if behavior and not self.two_branch_replay:
            raise ValueError(
                f"unexpected {BEHAVIOR_REPLAY_PREFIX} batch for replay mode "
                f"{self.replay_sampling!r}"
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
        if str(marl.stage) not in {"local", "ctde"}:
            raise ValueError(f"unsupported MARL stage: {marl.stage!r}")
        if str(marl.execution) != "strict_decentralized":
            raise ValueError(f"unsupported execution contract: {marl.execution!r}")
        self.team = TeamAxis(int(config.num_agents))
        self.public_obs_space = dict(obs_space)
        self.public_act_space = dict(act_space)
        self.marl_stage = str(marl.stage)
        self.ctde_enabled = self.marl_stage == "ctde" and self.team.size > 1
        self.ctde_rollout_steps = (
            int(marl.ctde.rollout_steps) if self.ctde_enabled else 1
        )
        if self.ctde_rollout_steps not in {1, 2}:
            raise ValueError("CTDE rollout_steps must be 1 or 2")
        self.ctde_mask_calibration = bool(
            marl.ctde.mask_calibration.enabled if self.ctde_enabled else False
        )
        self.ctde_soft_liveness = bool(
            marl.ctde.mask_calibration.soft_liveness
            if self.ctde_mask_calibration
            else False
        )
        self.ctde_death_masking = bool(
            marl.ctde.death_masking.enabled if self.ctde_enabled else False
        )
        self.ctde_authoritative_action_binding = bool(
            marl.ctde.authoritative_action_binding.enabled
            if self.ctde_enabled
            else False
        )
        self.ctde_authoritative_action_binding_anchors = int(
            marl.ctde.authoritative_action_binding.anchors if self.ctde_enabled else 1
        )
        self.ctde_authoritative_action_binding_margin = float(
            marl.ctde.authoritative_action_binding.margin if self.ctde_enabled else 0.0
        )
        self.ctde_support_preserving = bool(
            marl.ctde.support_preserving.enabled if self.ctde_enabled else False
        )
        self.ctde_support_probability_floor = float(
            marl.ctde.support_preserving.probability_floor
            if self.ctde_enabled
            else 0.05
        )
        self.ctde_probabilistic_availability = (
            self.ctde_mask_calibration or self.ctde_support_preserving
        )
        if self.ctde_authoritative_action_binding_anchors < 1:
            raise ValueError("authoritative action binding anchors must be positive")
        if self.ctde_authoritative_action_binding_margin < 0.0:
            raise ValueError("authoritative action binding margin must be nonnegative")
        if (
            self.ctde_authoritative_action_binding
            and float(config.loss_scales.ctde_authoritative_action_binding) <= 0.0
        ):
            raise ValueError(
                "enabled authoritative action binding requires a positive loss scale"
            )
        if not 0.0 < self.ctde_support_probability_floor < 1.0:
            raise ValueError("support-preserving probability floor must be in (0, 1)")
        self.ctde_teammate_belief_enabled = bool(
            marl.ctde.teammate_belief.enabled if self.ctde_enabled else False
        )
        self.ctde_teammate_belief_logit_clip = (
            float(marl.ctde.teammate_belief.logit_clip)
            if self.ctde_teammate_belief_enabled
            else 1.0
        )
        if self.ctde_teammate_belief_logit_clip <= 0.0:
            raise ValueError("teammate belief logit clip must be positive")
        self.ctde_multistep_jepa_enabled = bool(
            marl.ctde.multistep_jepa.enabled if self.ctde_enabled else False
        )
        self.ctde_multistep_jepa_belief_context = bool(
            marl.ctde.multistep_jepa.belief_context
            if self.ctde_multistep_jepa_enabled
            else False
        )
        self.ctde_multistep_jepa_action_scale = (
            float(config.loss_scales.ctde_multistep_jepa_action)
            if self.ctde_multistep_jepa_enabled
            else 0.0
        )
        if self.ctde_multistep_jepa_action_scale < 0.0:
            raise ValueError("multi-step JEPA action loss scale must be nonnegative")
        if self.ctde_enabled:
            multistep_jepa = marl.ctde.multistep_jepa
            self.ctde_multistep_jepa_horizons = tuple(
                int(value) for value in multistep_jepa.horizons
            )
            self.ctde_multistep_jepa_max_horizon = int(multistep_jepa.max_horizon)
            self.ctde_multistep_jepa_decay = float(multistep_jepa.decay)
            self.ctde_multistep_jepa_action_margin = float(multistep_jepa.action_margin)
            self.ctde_multistep_jepa_action_counterfactual_mode = str(
                multistep_jepa.action_counterfactual_mode
            )
            self.ctde_multistep_jepa_plan_aggregation = str(
                multistep_jepa.plan_aggregation
            )
        else:
            self.ctde_multistep_jepa_horizons = (1, 2, 4, 8)
            self.ctde_multistep_jepa_max_horizon = 8
            self.ctde_multistep_jepa_decay = 0.75
            self.ctde_multistep_jepa_action_margin = 0.1
            self.ctde_multistep_jepa_action_counterfactual_mode = "all_legal_mean"
            self.ctde_multistep_jepa_plan_aggregation = "mean"
        if (
            not self.ctde_multistep_jepa_horizons
            or tuple(sorted(set(self.ctde_multistep_jepa_horizons)))
            != self.ctde_multistep_jepa_horizons
            or min(self.ctde_multistep_jepa_horizons) < 1
            or max(self.ctde_multistep_jepa_horizons)
            != self.ctde_multistep_jepa_max_horizon
        ):
            raise ValueError(
                "multi-step JEPA horizons must be sorted unique positives ending at K"
            )
        if not 0.0 < self.ctde_multistep_jepa_decay <= 1.0:
            raise ValueError("multi-step JEPA decay must be in (0, 1]")
        if self.ctde_multistep_jepa_action_margin < 0.0:
            raise ValueError("multi-step JEPA action margin must be nonnegative")
        if self.ctde_multistep_jepa_action_counterfactual_mode != "all_legal_mean":
            raise ValueError("MA-JEPA requires all-legal action counterfactuals")
        if self.ctde_multistep_jepa_plan_aggregation not in {
            "mean",
            "focal_attention",
        }:
            raise ValueError(
                "MA-JEPA teammate-plan aggregation must be 'mean' or "
                "'focal_attention'"
            )
        if (
            self.ctde_multistep_jepa_belief_context
            and not self.ctde_teammate_belief_enabled
        ):
            raise ValueError("multi-step belief context requires teammate belief v2")
        self.action_mask_reduction = str(
            getattr(config, "action_mask_reduction", "sum")
        )
        if self.action_mask_reduction not in {"sum", "mean", "balanced"}:
            raise ValueError(
                "action_mask_reduction must be 'sum', 'mean', or 'balanced'"
            )
        self.ctde_mask_horizons = (
            tuple(int(x) for x in marl.ctde.mask_calibration.horizons)
            if self.ctde_mask_calibration
            else ()
        )
        if self.ctde_mask_calibration and (
            not self.ctde_mask_horizons
            or min(self.ctde_mask_horizons) < 1
            or tuple(sorted(set(self.ctde_mask_horizons))) != self.ctde_mask_horizons
        ):
            raise ValueError(
                "CTDE mask calibration horizons must be sorted unique positives"
            )
        local_obs_space = local_observation_spaces(obs_space, self.team.size)
        local_act_space = local_action_spaces(act_space, self.team.size)
        super().__init__(
            local_obs_space,
            local_act_space,
            config,
        )
        if self.ctde_multistep_jepa_enabled:
            if not self.two_branch_replay:
                raise ValueError(
                    "multi-step JEPA is defined only on the recent world branch of "
                    "recent_world_uniform_behavior replay"
                )
            if int(config.batch_length) <= self.ctde_multistep_jepa_max_horizon:
                raise ValueError(
                    "multi-step JEPA batch_length must exceed max_horizon, got "
                    f"{config.batch_length} and "
                    f"{self.ctde_multistep_jepa_max_horizon}"
                )
        if self.two_branch_replay:
            if not self.ctde_enabled:
                raise ValueError(
                    "recent_world_uniform_behavior requires multi-agent CTDE"
                )
            if self.ctde_rollout_steps != 1 or self.ctde_mask_calibration:
                raise ValueError(
                    "recent_world_uniform_behavior supports only one-step factual "
                    "CTDE without mask calibration"
                )
            joint_burnin = int(marl.ctde.joint.context) * int(
                marl.ctde.joint.temporal_layers
            )
            if int(config.replay_context) < joint_burnin:
                raise ValueError(
                    "recent_world_uniform_behavior replay_context must cover the "
                    "joint Transformer's full temporal receptive field "
                    f"({joint_burnin}), got {config.replay_context}"
                )

    def _make_value_models(self, scalar, config):
        if self.ctde_enabled:
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
            slowvalue = embodied.jax.SlowModel(
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
        return super()._make_value_models(scalar, config)

    def critic(self, features, bdims, *, slow=False, context=None):
        if self.ctde_enabled:
            if bdims != 2 or context is None:
                raise ValueError("CTDE critic requires synchronized sequence activity")
            local_state = (
                self.feat2tensor(features) if isinstance(features, dict) else features
            )
            grouped_state = self.team.unfold_sequence(local_state)
            grouped_present = context["present"].astype(bool)
            grouped_alive = context["controllable_alive"]
            if not self.ctde_soft_liveness:
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
        return super().critic(features, bdims, slow=slow, context=context)

    def additional_modules(self):
        modules = list(super().additional_modules())
        if self.ctde_enabled:
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
            if self.ctde_teammate_belief_enabled:
                belief = cfg.teammate_belief
                self.ctde_teammate_belief = TeammateActionBelief(
                    self.team.size - 1,
                    self.ctde_action_count,
                    layers=int(belief.layers),
                    units=int(belief.units),
                    outscale=float(belief.outscale),
                    act=str(belief.act),
                    norm=str(belief.norm),
                    winit=str(belief.winit),
                    name="ctde_teammate_belief",
                )
                self.ctde_teammate_actor = TeammateBeliefActorAdapter(
                    self.ctde_action_count,
                    layers=int(belief.adapter_layers),
                    units=int(belief.adapter_units),
                    act=str(belief.act),
                    norm=str(belief.norm),
                    winit=str(belief.winit),
                    name="ctde_teammate_actor",
                )
                ctde_modules.append(self.ctde_teammate_belief)
                actor_modules.append(self.ctde_teammate_actor)
            if self.ctde_multistep_jepa_enabled:
                multistep = cfg.multistep_jepa
                if self.ctde_multistep_jepa_belief_context:
                    self.ctde_teammate_plan = TeammateActionPlanGRU(
                        self.ctde_action_count,
                        self.ctde_action_low,
                        self.team.size - 1,
                        self.ctde_multistep_jepa_max_horizon,
                        units=int(multistep.plan_units),
                        act=str(multistep.act),
                        norm=str(multistep.norm),
                        winit=str(multistep.winit),
                        name="ctde_teammate_plan",
                    )
                    ctde_modules.append(self.ctde_teammate_plan)
                self.ctde_multistep_jepa = ActionConditionedMultiStepJEPA(
                    self.ctde_action_count,
                    self.ctde_action_low,
                    self.enc_output_dim,
                    self.ctde_multistep_jepa_horizons,
                    self.ctde_multistep_jepa_max_horizon,
                    width=int(multistep.width),
                    layers=int(multistep.layers),
                    units=int(multistep.units),
                    plan_aggregation=str(multistep.plan_aggregation),
                    plan_attention_heads=int(multistep.plan_attention_heads),
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

    @property
    def policy_keys(self):
        if self.ctde_teammate_belief_enabled:
            return "^(enc|dyn|pol|ctde_teammate_belief|ctde_teammate_actor)/"
        return super().policy_keys

    def _teammate_peer_indices(self):
        return jnp.asarray(
            [
                [peer for peer in range(self.team.size) if peer != focal]
                for focal in range(self.team.size)
            ],
            jnp.int32,
        )

    @staticmethod
    def _isolated_creation_call(module, salt, *args, **kwargs):
        """Create treatment parameters without advancing the base RNG stream."""

        if not nj.creating():
            return module(*args, **kwargs)
        context = nj.context()
        outer_seed = context.seed
        outer_reserve = context.reserve
        if outer_seed is None:
            return module(*args, **kwargs)
        context.seed = jax.random.fold_in(outer_seed, int(salt))
        context.reserve = []
        try:
            return module(*args, **kwargs)
        finally:
            context.seed = outer_seed
            context.reserve = outer_reserve

    def _teammate_belief_logits(self, local_state, bdims):
        if not self.ctde_teammate_belief_enabled:
            raise RuntimeError("teammate belief is disabled")
        return self._isolated_creation_call(
            self.ctde_teammate_belief,
            0x54424C46,
            jax.lax.stop_gradient(local_state),
            bdims,
        )

    def _teammate_belief_context(self, logits):
        """Return bounded offset-invariant evidence, with uniform mapped to zero."""

        logits = logits.astype(jnp.float32)
        centered = logits - logits.mean(axis=-1, keepdims=True)
        context = (
            jnp.clip(
                centered,
                -self.ctde_teammate_belief_logit_clip,
                self.ctde_teammate_belief_logit_clip,
            )
            / self.ctde_teammate_belief_logit_clip
        )
        return jax.lax.stop_gradient(context)

    def _teammate_plan_context(self, logits):
        """Map plan logits to bounded, offset-invariant zero-uniform evidence."""

        probability = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        context = probability - 1.0 / self.ctde_action_count
        return jax.lax.stop_gradient(context)

    def _teammate_actor_residual(
        self,
        local_state,
        bdims,
        *,
        belief_logits=None,
        belief_context=None,
    ):
        if belief_context is None:
            if belief_logits is None:
                belief_logits = self._teammate_belief_logits(local_state, bdims)
            belief_context = self._teammate_belief_context(belief_logits)
        flat_context = belief_context.reshape((*belief_context.shape[:-2], -1))
        return self._isolated_creation_call(
            self.ctde_teammate_actor,
            0x54424144,
            jax.lax.stop_gradient(local_state),
            jax.lax.stop_gradient(flat_context),
            bdims,
        )

    @staticmethod
    def _add_categorical_residual(distribution, action_key, residual):
        previous = distribution[action_key]
        if not hasattr(previous, "raw_logits") or not hasattr(previous, "unimix"):
            raise TypeError(
                "teammate actor requires categorical raw-logit/unimix metadata"
            )
        raw_logits = previous.raw_logits + residual
        updated = jaxouts.Categorical(raw_logits, previous.unimix)
        updated.raw_logits = raw_logits
        updated.unimix = previous.unimix
        for name in ("minent", "maxent"):
            if hasattr(previous, name):
                setattr(updated, name, getattr(previous, name))
        return dict(distribution, **{action_key: updated})

    def _teammate_policy_before_mask(
        self,
        tensor,
        bdims,
        *,
        belief_context=None,
    ):
        base = self.pol(tensor, bdims=bdims)
        if not self.ctde_teammate_belief_enabled:
            return base, None
        residual = self._teammate_actor_residual(
            tensor, bdims, belief_context=belief_context
        )
        return (
            self._add_categorical_residual(base, self.ctde_action_key, residual),
            residual,
        )

    def policy_distribution(self, tensor, bdims, action_mask=None):
        if not self.ctde_teammate_belief_enabled:
            return super().policy_distribution(tensor, bdims, action_mask)
        policy, _ = self._teammate_policy_before_mask(tensor, bdims)
        if action_mask is None:
            output = self.actmask(tensor, bdims=bdims)
            binary = output.output if hasattr(output, "output") else output
            return apply_predicted_action_mask(
                policy,
                jax.lax.stop_gradient(binary.logit),
                self.action_mask_key,
            )
        return apply_action_mask(policy, action_mask, self.action_mask_key)

    def imagination_critic_context(self, features, context, auxiliary=None):
        if not self.ctde_enabled:
            return super().imagination_critic_context(features, context, auxiliary)
        if auxiliary is None:
            raise ValueError("CTDE critic requires imagined activity")
        metrics = {}
        if self.ctde_teammate_belief_enabled:
            local_state = self.feat2tensor(features)
            action_mask = self.team.fold_sequence(auxiliary["action_mask"])
            valid = auxiliary["present"].astype(jnp.float32)
            valid *= auxiliary["controllable_alive"].astype(jnp.float32)
            metrics.update(
                self._teammate_belief_policy_metrics(
                    local_state,
                    action_mask,
                    self.team.fold_sequence(valid),
                )
            )
        return {
            "present": auxiliary["present"],
            "controllable_alive": auxiliary["controllable_alive"],
        }, metrics

    def _teammate_belief_policy_metrics(self, local_state, action_mask, valid):
        """Measure causal belief influence without exposing oracle information."""

        bdims = 2
        base = self.pol(local_state, bdims=bdims)
        logits = self._teammate_belief_logits(local_state, bdims)
        context = self._teammate_belief_context(logits)
        residual = self._teammate_actor_residual(
            local_state, bdims, belief_context=context
        )
        learned = self._add_categorical_residual(base, self.ctde_action_key, residual)
        shuffled_context = (
            jnp.roll(context, 1, axis=-2) if context.shape[-2] > 1 else context
        )
        shuffled_residual = self._teammate_actor_residual(
            local_state, bdims, belief_context=shuffled_context
        )
        shuffled = self._add_categorical_residual(
            base, self.ctde_action_key, shuffled_residual
        )
        base = apply_action_mask(base, action_mask, self.ctde_action_key)
        learned = apply_action_mask(learned, action_mask, self.ctde_action_key)
        shuffled = apply_action_mask(shuffled, action_mask, self.ctde_action_key)
        base_logits = base[self.ctde_action_key].logits.astype(jnp.float32)
        learned_logits = learned[self.ctde_action_key].logits.astype(jnp.float32)
        shuffled_logits = shuffled[self.ctde_action_key].logits.astype(jnp.float32)

        def forward_kl(reference, candidate):
            reference_logprob = jax.nn.log_softmax(reference, axis=-1)
            candidate_logprob = jax.nn.log_softmax(candidate, axis=-1)
            return (
                jnp.exp(reference_logprob) * (reference_logprob - candidate_logprob)
            ).sum(axis=-1)

        def weighted_mean(value, weight=valid):
            weight = weight.astype(jnp.float32)
            return (value.astype(jnp.float32) * weight).sum() / jnp.maximum(
                weight.sum(), 1.0
            )

        zero_flip = jnp.argmax(learned_logits, axis=-1) != jnp.argmax(
            base_logits, axis=-1
        )
        shuffle_flip = jnp.argmax(learned_logits, axis=-1) != jnp.argmax(
            shuffled_logits, axis=-1
        )
        zero_kl = forward_kl(base_logits, learned_logits)
        shuffle_kl = forward_kl(learned_logits, shuffled_logits)
        centered_logits = logits.astype(jnp.float32) - logits.astype(jnp.float32).mean(
            axis=-1, keepdims=True
        )
        logit_rms = jnp.sqrt(jnp.square(centered_logits).mean(axis=(-1, -2)))
        context_norm = jnp.sqrt(jnp.square(context).sum(axis=(-1, -2)))
        residual_rms = jnp.sqrt(jnp.square(residual).mean(axis=-1))
        residual_max = jnp.abs(residual).max(axis=-1)
        shuffle_residual_rms = jnp.sqrt(jnp.square(shuffled_residual).mean(axis=-1))
        root_weight = valid[:, :1]
        future_weight = valid[:, 1:]
        root_logit_rms = weighted_mean(logit_rms[:, :1], root_weight)
        future_logit_rms = weighted_mean(logit_rms[:, 1:], future_weight)
        root_context_norm = weighted_mean(context_norm[:, :1], root_weight)
        future_context_norm = weighted_mean(context_norm[:, 1:], future_weight)
        root_residual_rms = weighted_mean(residual_rms[:, :1], root_weight)
        future_residual_rms = weighted_mean(residual_rms[:, 1:], future_weight)
        metrics = {
            "ctde/teammate_belief_policy_kl_vs_zero": weighted_mean(zero_kl),
            "ctde/teammate_belief_policy_flip_vs_zero": weighted_mean(zero_flip),
            "ctde/teammate_belief_policy_kl_vs_peer_shuffle": weighted_mean(shuffle_kl),
            "ctde/teammate_belief_policy_flip_vs_peer_shuffle": weighted_mean(
                shuffle_flip
            ),
            "ctde/teammate_belief_residual_rms": weighted_mean(residual_rms),
            "ctde/teammate_belief_residual_max": weighted_mean(residual_max),
            "ctde/teammate_belief_shuffle_residual_rms": weighted_mean(
                shuffle_residual_rms
            ),
            "ctde/teammate_belief_imagined_root_logit_rms": root_logit_rms,
            "ctde/teammate_belief_imagined_future_logit_rms": future_logit_rms,
            "ctde/teammate_belief_imagined_logit_rms_drift": (
                future_logit_rms - root_logit_rms
            ),
            "ctde/teammate_belief_imagined_root_context_norm": root_context_norm,
            "ctde/teammate_belief_imagined_future_context_norm": (future_context_norm),
            "ctde/teammate_belief_imagined_context_norm_drift": (
                future_context_norm - root_context_norm
            ),
            "ctde/teammate_belief_imagined_root_residual_rms": root_residual_rms,
            "ctde/teammate_belief_imagined_future_residual_rms": (future_residual_rms),
            "ctde/teammate_belief_imagined_residual_rms_drift": (
                future_residual_rms - root_residual_rms
            ),
        }
        for horizon in (1, 4, 8, 15):
            if horizon >= local_state.shape[1]:
                continue
            horizon_weight = valid[:, horizon : horizon + 1]

            def at_horizon(value):
                return weighted_mean(value[:, horizon : horizon + 1], horizon_weight)

            prefix = f"ctde/teammate_belief_h{horizon}"
            horizon_logit_rms = at_horizon(logit_rms)
            horizon_context_norm = at_horizon(context_norm)
            horizon_residual_rms = at_horizon(residual_rms)
            metrics.update(
                {
                    f"{prefix}_valid_fraction": horizon_weight.mean(),
                    f"{prefix}_valid_count": horizon_weight.sum(),
                    f"{prefix}_logit_rms": horizon_logit_rms,
                    f"{prefix}_context_norm": horizon_context_norm,
                    f"{prefix}_residual_rms": horizon_residual_rms,
                    f"{prefix}_policy_kl_vs_zero": at_horizon(zero_kl),
                    f"{prefix}_policy_flip_vs_zero": at_horizon(zero_flip),
                    f"{prefix}_policy_kl_vs_peer_shuffle": at_horizon(shuffle_kl),
                    f"{prefix}_policy_flip_vs_peer_shuffle": at_horizon(shuffle_flip),
                    f"{prefix}_logit_rms_drift_from_factual": (
                        horizon_logit_rms - root_logit_rms
                    ),
                    f"{prefix}_context_norm_drift_from_factual": (
                        horizon_context_norm - root_context_norm
                    ),
                    f"{prefix}_residual_rms_drift_from_factual": (
                        horizon_residual_rms - root_residual_rms
                    ),
                }
            )
        return metrics

    def replay_critic_context(self, features, obs, starts_count):
        if not self.ctde_enabled:
            return super().replay_critic_context(features, obs, starts_count)
        present = self._present(obs)[:, -starts_count:]
        alive = self._controllable(obs)[:, -starts_count:]
        return {
            "present": self.team.unfold_sequence(present).astype(bool),
            "controllable_alive": self.team.unfold_sequence(alive).astype(bool),
        }, {}

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
        if not self.ctde_enabled:
            return {}, {}
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
        alive_valid = (
            source_alive if self.ctde_soft_liveness else source_present
        ) & ~next_first[..., None]
        alive_weight = alive_valid.astype(jnp.float32)
        alive_count = jnp.maximum(alive_weight.sum(), 1.0)
        normalized_alive_weight = alive_weight / jnp.maximum(alive_weight.mean(), 1e-8)

        def masked_metric(value):
            return (value.astype(jnp.float32) * weight).sum() / count

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
            "ctde/reward_loss": masked_metric(reward_loss),
            "ctde/continuation_loss": masked_metric(continuation_loss),
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
            "ctde/treatment_death_masking": jnp.asarray(
                float(self.ctde_death_masking), jnp.float32
            ),
            "ctde/treatment_authoritative_action_binding": jnp.asarray(
                float(self.ctde_authoritative_action_binding), jnp.float32
            ),
            "ctde/treatment_support_preserving": jnp.asarray(
                float(self.ctde_support_preserving), jnp.float32
            ),
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

        losses = {
            "ctde_embedding": folded(embedding_loss),
            "ctde_interface": folded(interface_loss),
            "ctde_reward": folded(reward_loss),
            "ctde_continuation": folded(continuation_loss),
            "ctde_action_mask": folded(mask_loss),
            "ctde_alive": folded_alive(alive_loss),
        }
        if self.ctde_authoritative_action_binding:
            binding_loss, binding_metrics = (
                self._ctde_authoritative_action_binding_loss(
                    cache,
                    snapshots,
                    source_state,
                    source_action,
                    source_present,
                    source_alive,
                    reset,
                    grouped_target[:, 1:],
                    grouped_mask[:, :-1],
                    transition_valid,
                    grouped_present,
                )
            )
            losses["ctde_authoritative_action_binding"] = binding_loss
            metrics.update(binding_metrics)
        if self.ctde_teammate_belief_enabled:
            belief_loss, belief_metrics = self._ctde_teammate_belief_loss(
                source_state,
                source_action,
                grouped_action[:, :-1],
                source_present,
                source_alive,
                grouped_mask[:, :-1],
                reset,
                next_first,
            )
            losses["ctde_teammate_belief"] = belief_loss
            metrics.update(belief_metrics)
        if self.ctde_multistep_jepa_enabled:
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
        if self.ctde_mask_calibration:
            calibration_losses, calibration_metrics = (
                self._ctde_mask_calibration_losses(
                    dyn_entries,
                    grouped_present,
                    grouped_alive,
                    grouped_first,
                    grouped_mask,
                    grouped_action,
                )
            )
            losses.update(calibration_losses)
            metrics.update(calibration_metrics)
        if self.ctde_rollout_steps == 2:
            multistep_losses, multistep_metrics = self._ctde_multistep_losses(
                prediction,
                snapshots,
                repfeat,
                dyn_entries,
                grouped_online,
                grouped_target,
                grouped_present,
                grouped_alive,
                grouped_first,
                grouped_reward,
                self.team.unfold_sequence(obs["is_terminal"]),
                grouped_mask,
                grouped_action,
                self.team.unfold_sequence(self.validity(obs)).astype(bool),
            )
            losses.update(multistep_losses)
            metrics.update(multistep_metrics)
        return losses, metrics

    def _ctde_authoritative_action_binding_loss(
        self,
        initial_cache,
        snapshots,
        source_state,
        source_action,
        source_present,
        source_alive,
        reset,
        target_embedding,
        source_mask,
        transition_valid,
        destination_present,
    ):
        """Enumerate legal H1 actions through the simulator used by the actor."""

        batch, transitions, agents = source_action.shape
        classes = self.ctde_action_count
        anchor_count = min(
            self.ctde_authoritative_action_binding_anchors,
            batch * transitions,
        )
        anchor_mask = transition_valid.any(axis=-1)
        anchors = sample_two_step_anchors(nj.seed(), anchor_mask, anchor_count)

        grouped_initial = self.team.unfold_tree_batch(initial_cache)
        grouped_snapshots = jax.tree.map(self.team.unfold_sequence, snapshots)
        pre_transition_cache = jax.tree.map(
            lambda initial, history: jnp.concatenate(
                [initial[:, None], history[:, :-1]], axis=1
            ),
            grouped_initial,
            grouped_snapshots,
        )
        # Counterfactual candidates answer a same-history, current-action query.
        # Detaching the factual pre-transition cache prevents the ranking loss
        # from satisfying its margin by rewriting earlier temporal context.
        sampled_cache = detach_self_feed(gather_anchors(pre_transition_cache, anchors))
        sampled_state = gather_anchors(source_state, anchors)
        sampled_action = gather_anchors(source_action, anchors)
        sampled_present = gather_anchors(source_present, anchors)
        sampled_alive = gather_anchors(source_alive, anchors)
        sampled_reset = gather_anchors(reset, anchors)
        sampled_target = gather_anchors(target_embedding, anchors)
        sampled_mask = gather_anchors(source_mask, anchors)
        sampled_valid = gather_anchors(transition_valid, anchors)
        sampled_valid &= anchors.valid[:, None]

        scenario_shape = (anchor_count, agents, classes, agents)

        def expand_team(value):
            expanded = jnp.broadcast_to(
                value[:, None, None], (*scenario_shape, *value.shape[2:])
            )
            return expanded.reshape(
                (anchor_count * agents * classes * agents, *value.shape[2:])
            )

        expanded_cache = jax.tree.map(expand_team, sampled_cache)
        expanded_state = jnp.broadcast_to(
            sampled_state[:, None, None],
            (*scenario_shape, sampled_state.shape[-1]),
        ).reshape((anchor_count * agents * classes, agents, sampled_state.shape[-1]))
        base_action = jnp.broadcast_to(sampled_action[:, None, None], scenario_shape)
        candidate_action = (
            jnp.arange(classes, dtype=jnp.int32) + self.ctde_action_low
        )[None, None, :, None]
        focal = jnp.eye(agents, dtype=bool)[None, :, None, :]
        expanded_action = jnp.where(focal, candidate_action, base_action).reshape(
            (anchor_count * agents * classes, agents)
        )
        expanded_present = jnp.broadcast_to(
            sampled_present[:, None, None], scenario_shape
        ).reshape((anchor_count * agents * classes, agents))
        expanded_alive = jnp.broadcast_to(
            sampled_alive[:, None, None], scenario_shape
        ).reshape((anchor_count * agents * classes, agents))
        expanded_reset = jnp.broadcast_to(
            sampled_reset[:, None, None], (anchor_count, agents, classes)
        ).reshape(-1)

        _, candidate = self.ctde_joint.step(
            expanded_cache,
            expanded_state,
            expanded_action,
            expanded_present,
            expanded_alive,
            expanded_reset,
            training=False,
        )
        all_predictions = candidate["embedding"].reshape(
            (
                anchor_count,
                agents,
                classes,
                agents,
                candidate["embedding"].shape[-1],
            )
        )
        selector = jnp.eye(agents, dtype=all_predictions.dtype)[None, :, None, :, None]
        focal_predictions = (all_predictions * selector).sum(axis=3)
        sampled_loss, root_valid, raw_metrics = authoritative_action_binding_objective(
            focal_predictions,
            sampled_target,
            sampled_action,
            sampled_mask,
            sampled_valid,
            action_low=self.ctde_action_low,
            margin=self.ctde_authoritative_action_binding_margin,
        )

        outer_count = destination_present.astype(jnp.float32).sum()
        sample_count = root_valid.astype(jnp.float32).sum()
        scale = outer_count / jnp.maximum(sample_count, 1.0)
        replay_grid = jnp.zeros(destination_present.shape, jnp.float32)
        replay_grid = replay_grid.at[anchors.batch, anchors.time].add(
            sampled_loss * root_valid.astype(jnp.float32) * scale
        )
        metrics = {
            f"ctde/authoritative_action_binding_{key}": value
            for key, value in raw_metrics.items()
        }
        metrics.update(
            {
                "ctde/authoritative_action_binding_enabled": jnp.asarray(
                    1.0, jnp.float32
                ),
                "ctde/authoritative_action_binding_anchor_count": jnp.asarray(
                    anchor_count, jnp.float32
                ),
            }
        )
        return self.team.fold_sequence(replay_grid), metrics

    def _ctde_teammate_belief_loss(
        self,
        source_state,
        current_action,
        previous_action,
        present,
        controllable_alive,
        action_mask,
        current_first,
        next_first,
    ):
        """Predict factual peer ``a_t`` from stopped focal causal state ``s_t``."""

        if source_state.ndim != 4:
            raise ValueError(
                f"teammate belief source must be [B,T,A,F], got {source_state.shape}"
            )
        batch, length, agents = source_state.shape[:3]
        expected = (batch, length, agents)
        if any(
            value.shape != expected
            for value in (
                current_action,
                previous_action,
                present,
                controllable_alive,
            )
        ):
            raise ValueError("teammate belief replay labels are not time aligned")
        if action_mask.shape != (*expected, self.ctde_action_count):
            raise ValueError("teammate belief action masks do not match labels")
        if current_first.shape != (batch, length) or next_first.shape != (
            batch,
            length,
        ):
            raise ValueError("teammate belief reset masks are not time aligned")

        folded_state = self.team.fold_sequence(source_state)
        folded_logits = self._teammate_belief_logits(folded_state, bdims=2)
        logits = self.team.unfold_sequence(folded_logits)
        peer_indices = self._teammate_peer_indices()
        peer_action = jnp.take(current_action, peer_indices, axis=2)
        peer_previous_action = jnp.take(previous_action, peer_indices, axis=2)
        peer_present = jnp.take(present, peer_indices, axis=2)
        peer_alive = jnp.take(controllable_alive, peer_indices, axis=2)
        peer_action_mask = jnp.take(action_mask, peer_indices, axis=2)
        if logits.shape != (*peer_action.shape, self.ctde_action_count):
            raise ValueError(
                "teammate belief output/target shape mismatch: "
                f"{logits.shape} versus {peer_action.shape}"
            )

        target = peer_action.astype(jnp.int32) - self.ctde_action_low
        in_range = (target >= 0) & (target < self.ctde_action_count)
        safe_target = jnp.clip(target, 0, self.ctde_action_count - 1)
        target_legal = jnp.take_along_axis(
            peer_action_mask, safe_target[..., None], axis=-1
        )[..., 0]
        candidate = (
            controllable_alive[..., None] & peer_present & ~next_first[..., None, None]
        )
        valid = candidate & in_range & target_legal
        weight = valid.astype(jnp.float32)

        log_probability = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        nll = -jnp.take_along_axis(log_probability, safe_target[..., None], axis=-1)[
            ..., 0
        ]
        row_count = jnp.maximum(weight.sum(axis=-1), 1.0)
        row_loss = (nll * weight).sum(axis=-1) / row_count
        row_valid = (weight.sum(axis=-1) > 0).astype(jnp.float32)
        row_loss *= row_valid / jnp.maximum(row_valid.mean(), 1e-8)
        row_loss = jnp.pad(row_loss, ((0, 0), (0, 1), (0, 0)))
        row_loss *= row_loss.shape[1] / max(row_loss.shape[1] - 1, 1)
        folded_loss = self.team.fold_sequence(row_loss)

        probability = jnp.exp(log_probability)
        entropy = -(probability * log_probability).sum(axis=-1)
        top1 = jnp.argmax(logits, axis=-1) == safe_target
        action_onehot = jax.nn.one_hot(
            safe_target, self.ctde_action_count, dtype=jnp.float32
        )
        marginal_count = (action_onehot * weight[..., None]).sum(axis=(0, 1, 2, 3))
        marginal_probability = (marginal_count + 1.0) / (
            marginal_count.sum() + self.ctde_action_count
        )
        marginal_probability = jax.lax.stop_gradient(marginal_probability)
        marginal_nll = -jnp.log(jnp.take(marginal_probability, safe_target, axis=0))
        marginal_top1 = jnp.argmax(marginal_probability) == safe_target

        previous_target = peer_previous_action.astype(jnp.int32) - self.ctde_action_low
        previous_in_range = (previous_target >= 0) & (
            previous_target < self.ctde_action_count
        )
        safe_previous = jnp.clip(previous_target, 0, self.ctde_action_count - 1)
        repeat_valid = valid & previous_in_range & ~current_first[..., None, None]
        repeat_weight = repeat_valid.astype(jnp.float32)
        repeat_unimix = max(float(self.config.policy.unimix), 1e-6)
        repeat_probability = jax.nn.one_hot(
            safe_previous, self.ctde_action_count, dtype=jnp.float32
        )
        repeat_probability = (
            1.0 - repeat_unimix
        ) * repeat_probability + repeat_unimix / self.ctde_action_count
        repeat_nll = -jnp.log(
            jnp.take_along_axis(repeat_probability, safe_target[..., None], axis=-1)[
                ..., 0
            ]
        )
        repeat_top1 = safe_previous == safe_target

        def average(value, event_weight=weight):
            event_weight = event_weight.astype(jnp.float32)
            return (value.astype(jnp.float32) * event_weight).sum() / jnp.maximum(
                event_weight.sum(), 1.0
            )

        active_weight = weight * peer_alive.astype(jnp.float32)
        nonnoop_weight = weight * (safe_target != 0).astype(jnp.float32)
        attack_start = min(6, self.ctde_action_count)
        attack_weight = weight * (safe_target >= attack_start).astype(jnp.float32)
        centered_logits = logits.astype(jnp.float32) - logits.astype(jnp.float32).mean(
            axis=-1, keepdims=True
        )
        context = self._teammate_belief_context(logits)
        logit_rms = jnp.sqrt(jnp.square(centered_logits).mean(axis=-1))
        context_norm = jnp.sqrt(jnp.square(context).sum(axis=(-1, -2)))
        candidate_count = jnp.maximum(candidate.astype(jnp.float32).sum(), 1.0)
        belief_nll = average(nll)
        repeat_belief_nll = average(nll, repeat_weight)
        metrics = {
            "ctde/teammate_belief_nll": belief_nll,
            "ctde/teammate_belief_entropy": average(entropy),
            "ctde/teammate_belief_top1": average(top1),
            "ctde/teammate_belief_active_peer_nll": average(nll, active_weight),
            "ctde/teammate_belief_active_peer_top1": average(top1, active_weight),
            "ctde/teammate_belief_nonnoop_nll": average(nll, nonnoop_weight),
            "ctde/teammate_belief_nonnoop_top1": average(top1, nonnoop_weight),
            "ctde/teammate_belief_attack_nll": average(nll, attack_weight),
            "ctde/teammate_belief_attack_top1": average(top1, attack_weight),
            "ctde/teammate_belief_marginal_nll": average(marginal_nll),
            "ctde/teammate_belief_marginal_top1": average(marginal_top1),
            "ctde/teammate_belief_repeat_nll": average(repeat_nll, repeat_weight),
            "ctde/teammate_belief_repeat_top1": average(repeat_top1, repeat_weight),
            "ctde/teammate_belief_nll_gain_vs_marginal": (
                average(marginal_nll) - belief_nll
            ),
            "ctde/teammate_belief_nll_gain_vs_repeat": (
                average(repeat_nll, repeat_weight) - repeat_belief_nll
            ),
            "ctde/teammate_belief_factual_logit_rms": average(logit_rms),
            "ctde/teammate_belief_factual_context_norm": average(
                context_norm, row_valid
            ),
            "ctde/teammate_belief_target_count": weight.sum(),
            "ctde/teammate_belief_active_peer_count": active_weight.sum(),
            "ctde/teammate_belief_nonnoop_count": nonnoop_weight.sum(),
            "ctde/teammate_belief_attack_count": attack_weight.sum(),
            "ctde/teammate_belief_target_fraction": weight.mean(),
            "ctde/teammate_belief_target_legal_fraction": (
                (candidate & in_range & target_legal).astype(jnp.float32).sum()
                / candidate_count
            ),
            "ctde/teammate_belief_dead_peer_fraction": average(~peer_alive),
        }
        return folded_loss, metrics

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
        """Fit direct EMA futures with a stopped causal teammate-action plan."""

        if not self.ctde_multistep_jepa_enabled or not self.two_branch_replay:
            raise RuntimeError(
                "direct multi-step JEPA requires the recent world replay branch"
            )
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
        root_state = grouped_source_state[:, :roots]

        q0_logits = None
        q0_context = None
        plan_logits = None
        plan_context = None
        plan_active = None
        plan_loss = None
        plan_metrics = {}
        if self.ctde_multistep_jepa_belief_context:
            folded_root = self.team.fold_sequence(root_state)
            q0_logits = self.team.unfold_sequence(
                self._teammate_belief_logits(folded_root, bdims=2)
            )
            q0_context = self._teammate_belief_context(q0_logits)
            plan_logits = isolated_creation_call(
                self.ctde_teammate_plan,
                0x5442504C,
                root_state,
                action_windows,
                q0_logits,
                q0_context,
            )
            plan_context = self._teammate_plan_context(plan_logits)
            peer_indices = self._teammate_peer_indices()
            plan_active = jnp.stack(
                [
                    jnp.take(
                        (
                            grouped_present[:, step : step + roots]
                            & grouped_alive[:, step : step + roots]
                        ),
                        peer_indices,
                        axis=2,
                    )
                    for step in range(1, max_horizon)
                ],
                axis=3,
            )
            plan_loss, plan_metrics = self._ctde_teammate_plan_loss(
                plan_logits,
                q0_logits,
                grouped_action,
                grouped_mask,
                grouped_present,
                grouped_alive,
                grouped_first,
                all_valid,
                roots,
                length,
            )

        predictions = isolated_creation_call(
            self.ctde_multistep_jepa,
            0x4D534A50,
            root_hidden,
            action_windows,
            plan_context,
            plan_active,
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
            expanded_plan = (
                jnp.broadcast_to(
                    plan_context[:, None],
                    (batch, classes, *plan_context.shape[1:]),
                ).reshape((batch * classes, *plan_context.shape[1:]))
                if plan_context is not None
                else None
            )
            expanded_plan_active = (
                jnp.broadcast_to(
                    plan_active[:, None],
                    (batch, classes, *plan_active.shape[1:]),
                ).reshape((batch * classes, *plan_active.shape[1:]))
                if plan_active is not None
                else None
            )
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
                    expanded_plan,
                    expanded_plan_active,
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
        if plan_loss is not None:
            losses["ctde_teammate_plan"] = plan_loss

        metrics = {
            f"ctde/multistep_jepa_{key}": value for key, value in raw_metrics.items()
        }
        metrics.update(plan_metrics)
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
                    float(self.ctde_multistep_jepa_belief_context), jnp.float32
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
        if plan_context is not None:
            context_norm = jnp.sqrt(jnp.square(plan_context).sum(axis=(-1, -2)))
            metrics.update(
                {
                    "ctde/multistep_jepa_plan_context_rms": jnp.sqrt(
                        jnp.square(plan_context).mean()
                    ),
                    "ctde/multistep_jepa_plan_context_nonzero_fraction": (
                        context_norm > 1e-8
                    )
                    .astype(jnp.float32)
                    .mean(),
                }
            )
        if plan_context is not None and not training:
            zero_plan = jnp.zeros_like(plan_context)
            ablated_predictions = self.ctde_multistep_jepa(
                root_hidden,
                action_windows,
                zero_plan,
                plan_active,
            )
            for horizon in horizons:
                weight = valid[horizon].astype(jnp.float32)
                count = jnp.maximum(weight.sum(), 1.0)
                factual = predictions[horizon].astype(jnp.float32)
                ablated = ablated_predictions[horizon].astype(jnp.float32)
                delta = jnp.sqrt(jnp.square(factual - ablated).mean(axis=-1))
                metrics[f"ctde/multistep_jepa_h{horizon}_plan_ablation_delta_rms"] = (
                    delta * weight
                ).sum() / count
        return losses, metrics

    def _ctde_teammate_plan_loss(
        self,
        plan_logits,
        q0_logits,
        grouped_action,
        grouped_mask,
        grouped_present,
        grouped_alive,
        grouped_first,
        all_valid,
        roots,
        length,
    ):
        """Supervise q1..qK-1 with stopped factual future peer actions."""

        steps = self.ctde_multistep_jepa_max_horizon - 1
        expected = (
            grouped_action.shape[0],
            roots,
            self.team.size,
            steps,
            self.team.size - 1,
            self.ctde_action_count,
        )
        if plan_logits.shape != expected:
            raise ValueError(
                f"teammate plan logits {plan_logits.shape} do not match {expected}"
            )
        q0_expected = expected[:3] + expected[4:]
        if q0_logits.shape != q0_expected:
            raise ValueError(
                f"teammate q0 logits {q0_logits.shape} do not match {q0_expected}"
            )
        peer_indices = self._teammate_peer_indices()
        source_actions = grouped_action[:, 1:]
        nll_terms = []
        top1_terms = []
        weight_terms = []
        q0_nll_terms = []
        repeat_nll_terms = []
        repeat_weight_terms = []
        metrics = {}
        for step in range(1, steps + 1):
            peer_action = jnp.take(
                source_actions[:, step : step + roots], peer_indices, axis=2
            )
            peer_mask = jnp.take(
                grouped_mask[:, step : step + roots], peer_indices, axis=2
            )
            peer_present = jnp.take(
                grouped_present[:, step : step + roots], peer_indices, axis=2
            )
            peer_alive = jnp.take(
                grouped_alive[:, step : step + roots], peer_indices, axis=2
            )
            target = peer_action.astype(jnp.int32) - self.ctde_action_low
            in_range = (target >= 0) & (target < self.ctde_action_count)
            safe_target = jnp.clip(target, 0, self.ctde_action_count - 1)
            legal = jnp.take_along_axis(peer_mask, safe_target[..., None], axis=-1)[
                ..., 0
            ]
            valid = (
                all_valid[step][..., None]
                & peer_present
                & in_range
                & legal
                & ~grouped_first[:, step + 1 : step + roots + 1, None, None]
            )
            weight = valid.astype(jnp.float32)
            active_weight = weight * peer_alive.astype(jnp.float32)
            dead_weight = weight * (~peer_alive).astype(jnp.float32)
            logits = plan_logits[..., step - 1, :, :].astype(jnp.float32)
            log_probability = jax.nn.log_softmax(logits, axis=-1)
            nll = -jnp.take_along_axis(
                log_probability, safe_target[..., None], axis=-1
            )[..., 0]
            top1 = jnp.argmax(logits, axis=-1) == safe_target
            q0_log_probability = jax.nn.log_softmax(
                q0_logits.astype(jnp.float32), axis=-1
            )
            q0_nll = -jnp.take_along_axis(
                q0_log_probability, safe_target[..., None], axis=-1
            )[..., 0]
            root_peer_action = jnp.take(source_actions[:, :roots], peer_indices, axis=2)
            repeat_target = root_peer_action.astype(jnp.int32) - self.ctde_action_low
            repeat_in_range = (repeat_target >= 0) & (
                repeat_target < self.ctde_action_count
            )
            safe_repeat = jnp.clip(repeat_target, 0, self.ctde_action_count - 1)
            repeat_unimix = max(float(self.config.policy.unimix), 1e-6)
            repeat_probability = jax.nn.one_hot(
                safe_repeat, self.ctde_action_count, dtype=jnp.float32
            )
            repeat_probability = (
                1.0 - repeat_unimix
            ) * repeat_probability + repeat_unimix / self.ctde_action_count
            repeat_nll = -jnp.log(
                jnp.take_along_axis(
                    repeat_probability, safe_target[..., None], axis=-1
                )[..., 0]
            )
            repeat_weight = weight * repeat_in_range.astype(jnp.float32)
            count = jnp.maximum(weight.sum(), 1.0)
            active_count = jnp.maximum(active_weight.sum(), 1.0)
            dead_count = jnp.maximum(dead_weight.sum(), 1.0)
            metrics.update(
                {
                    f"ctde/teammate_plan_q{step}_nll": (nll * weight).sum() / count,
                    f"ctde/teammate_plan_q{step}_top1": (
                        top1.astype(jnp.float32) * weight
                    ).sum()
                    / count,
                    f"ctde/teammate_plan_q{step}_count": weight.sum(),
                    f"ctde/teammate_plan_q{step}_nll_gain_vs_q0": (
                        ((q0_nll - nll) * weight).sum() / count
                    ),
                    f"ctde/teammate_plan_q{step}_nll_gain_vs_root_repeat": (
                        ((repeat_nll - nll) * repeat_weight).sum()
                        / jnp.maximum(repeat_weight.sum(), 1.0)
                    ),
                    f"ctde/teammate_plan_q{step}_active_nll": (
                        nll * active_weight
                    ).sum()
                    / active_count,
                    f"ctde/teammate_plan_q{step}_active_top1": (
                        top1.astype(jnp.float32) * active_weight
                    ).sum()
                    / active_count,
                    f"ctde/teammate_plan_q{step}_active_count": active_weight.sum(),
                    f"ctde/teammate_plan_q{step}_dead_nll": (nll * dead_weight).sum()
                    / dead_count,
                    f"ctde/teammate_plan_q{step}_dead_top1": (
                        top1.astype(jnp.float32) * dead_weight
                    ).sum()
                    / dead_count,
                    f"ctde/teammate_plan_q{step}_dead_count": dead_weight.sum(),
                }
            )
            nll_terms.append(nll)
            top1_terms.append(top1)
            weight_terms.append(weight)
            q0_nll_terms.append(q0_nll)
            repeat_nll_terms.append(repeat_nll)
            repeat_weight_terms.append(repeat_weight)

        nll = jnp.stack(nll_terms, axis=3)
        top1 = jnp.stack(top1_terms, axis=3)
        weight = jnp.stack(weight_terms, axis=3)
        q0_nll = jnp.stack(q0_nll_terms, axis=3)
        repeat_nll = jnp.stack(repeat_nll_terms, axis=3)
        repeat_weight = jnp.stack(repeat_weight_terms, axis=3)
        event_count = jnp.maximum(weight.sum(), 1.0)
        row_count = jnp.maximum(weight.sum(axis=(-1, -2)), 1.0)
        row_loss = (nll * weight).sum(axis=(-1, -2)) / row_count
        row_valid = (weight.sum(axis=(-1, -2)) > 0).astype(jnp.float32)
        row_loss *= row_valid / jnp.maximum(row_valid.mean(), 1e-8)
        padded = jnp.pad(
            row_loss,
            ((0, 0), (0, self.ctde_multistep_jepa_max_horizon), (0, 0)),
        )
        padded *= length / roots
        metrics.update(
            {
                "ctde/teammate_plan_recent_nll": (nll * weight).sum() / event_count,
                "ctde/teammate_plan_recent_top1": (
                    top1.astype(jnp.float32) * weight
                ).sum()
                / event_count,
                "ctde/teammate_plan_recent_count": weight.sum(),
                "ctde/teammate_plan_recent_nll_gain_vs_q0": (
                    ((q0_nll - nll) * weight).sum() / event_count
                ),
                "ctde/teammate_plan_recent_nll_gain_vs_root_repeat": (
                    ((repeat_nll - nll) * repeat_weight).sum()
                    / jnp.maximum(repeat_weight.sum(), 1.0)
                ),
            }
        )
        return self.team.fold_sequence(padded), metrics

    def _ctde_mask_calibration_losses(
        self,
        dyn_entries,
        present,
        controllable_alive,
        is_first,
        action_mask,
        actions,
    ):
        """Calibrate availability and liveness on closed-loop CTDE states.

        The replay suffix root is already burn-in conditioned and therefore gives
        one unbiased synchronized root per sampled sequence. The rollout feeds its
        own predicted local posterior back for up to 15 steps under replay actions.
        Every model state is stopped before either calibrated head sees it, so the
        added gradients reach only ``actmask`` and ``ctde_alive``.
        """

        batch, length, agents = present.shape
        max_horizon = max(self.ctde_mask_horizons)
        if length <= max_horizon:
            raise ValueError(
                "CTDE mask calibration needs replay length greater than "
                f"{max_horizon}, got {length}"
            )

        root_present = present[:, 0]
        root_alive = controllable_alive[:, 0]
        rollout_alive = (
            root_alive.astype(jnp.float32) if self.ctde_soft_liveness else root_alive
        )
        initial = (
            nn.cast(self.dyn.start_at(dyn_entries, 0)),
            dyn_entries["ctde_joint_carry"],
            root_present,
            rollout_alive,
            is_first[:, 0],
            jnp.ones((batch,), bool),
        )

        def transition(state, inputs):
            (
                local_carry,
                joint_carry,
                current_present,
                current_alive,
                current_reset,
                within_episode,
            ) = state
            grouped_action, next_first = inputs
            folded_action = {self.ctde_action_key: self.team.fold_batch(grouped_action)}
            folded_present = self.team.fold_batch(current_present)
            local_features = {
                "deter": local_carry["deter"],
                "stoch": local_carry["stoch"],
            }
            grouped_state = self.team.unfold_batch(self.feat2tensor(local_features))
            local_cache, deter = self.dyn.advance(
                local_carry,
                folded_action,
                training=False,
                active=folded_present,
            )
            joint_carry, prediction = self.ctde_joint.step(
                joint_carry,
                grouped_state,
                grouped_action,
                current_present,
                current_alive,
                current_reset,
                training=False,
            )
            next_carry, _ = self.dyn.complete_from_observation(
                local_cache,
                deter,
                self.team.fold_batch(prediction["embedding"]),
                sample=False,
            )

            def preserve_absent(next_value, current_value):
                mask = folded_present.reshape(
                    (folded_present.shape[0],) + (1,) * (next_value.ndim - 1)
                )
                return jnp.where(mask, next_value, current_value)

            next_carry = jax.tree.map(preserve_absent, next_carry, local_carry)
            stopped_local = jax.lax.stop_gradient(
                self.feat2tensor(
                    {
                        "deter": next_carry["deter"],
                        "stoch": next_carry["stoch"],
                    }
                )
            )
            availability_output = self.actmask(stopped_local, 1)
            availability_binary = (
                availability_output.output
                if hasattr(availability_output, "output")
                else availability_output
            )
            availability_logits = availability_binary.logit
            availability_logits = self.team.unfold_batch(availability_logits)
            hidden = jax.lax.stop_gradient(prediction["hidden"])
            alive_output = self.ctde_alive(hidden, 2)
            alive_binary = (
                alive_output.output if hasattr(alive_output, "output") else alive_output
            )
            alive_logits = alive_binary.logit
            alive_probability = jax.nn.sigmoid(alive_logits)
            next_present = current_present
            if self.ctde_soft_liveness:
                next_alive = jax.lax.stop_gradient(
                    current_alive * next_present.astype(jnp.float32) * alive_probability
                )
            else:
                next_alive = jax.lax.stop_gradient(
                    current_alive & next_present & (alive_probability >= 0.5)
                )
            within_episode &= ~next_first
            next_state = (
                jax.lax.stop_gradient(next_carry),
                jax.lax.stop_gradient(joint_carry),
                next_present,
                next_alive,
                jnp.zeros_like(current_reset),
                within_episode,
            )
            return next_state, (
                availability_logits,
                alive_logits,
                alive_probability,
                next_alive,
                within_episode,
            )

        _, rollout = nj.scan(
            transition,
            initial,
            (
                actions[:, 1 : max_horizon + 1],
                is_first[:, 1 : max_horizon + 1],
            ),
            axis=1,
        )
        (
            availability_logits,
            alive_logits,
            alive_probability,
            rollout_alive,
            within_episode,
        ) = rollout
        predicted_alive = (
            rollout_alive if self.ctde_soft_liveness else alive_probability
        )
        mask_terms = []
        alive_terms = []
        metrics = {}

        def normalized(value, valid):
            valid = valid.astype(jnp.float32)
            return value.astype(jnp.float32) * valid / jnp.maximum(valid.mean(), 1e-8)

        for horizon in self.ctde_mask_horizons:
            index = horizon - 1
            target_mask = action_mask[:, horizon].astype(jnp.float32)
            target_alive = controllable_alive[:, horizon].astype(jnp.float32)
            mask_logit = availability_logits[:, index].astype(jnp.float32)
            alive_logit = alive_logits[:, index].astype(jnp.float32)
            mask_loss = jax.nn.softplus(mask_logit) - target_mask * mask_logit
            if self.action_mask_reduction == "mean":
                mask_loss = mask_loss.mean(axis=-1)
            elif self.action_mask_reduction == "balanced":
                mask_loss = balanced_binary_event_loss(mask_loss, target_mask)
            else:
                mask_loss = mask_loss.sum(axis=-1)
            alive_loss = jax.nn.softplus(alive_logit) - target_alive * alive_logit
            mask_valid = (
                root_alive & present[:, horizon] & within_episode[:, index, None]
            )
            alive_source = (
                controllable_alive[:, horizon - 1]
                if self.ctde_soft_liveness
                else root_present
            )
            alive_valid = (
                alive_source & present[:, horizon] & within_episode[:, index, None]
            )
            mask_terms.append(normalized(mask_loss, mask_valid))
            alive_terms.append(normalized(alive_loss, alive_valid))

            mask_probability = jax.nn.sigmoid(mask_logit)
            mask_brier = jnp.square(mask_probability - target_mask).mean(axis=-1)
            alive_brier = jnp.square(predicted_alive[:, index] - target_alive)
            metrics[f"ctde/mask_calibration_h{horizon}_brier"] = normalized(
                mask_brier, mask_valid
            ).mean()
            metrics[f"ctde/alive_calibration_h{horizon}_brier"] = normalized(
                alive_brier, alive_valid
            ).mean()
            attack_start = min(6, mask_probability.shape[-1])
            attack_probability = mask_probability[..., attack_start:]
            attack_target = target_mask[..., attack_start:].astype(bool)
            attack_valid = mask_valid[..., None]
            attack_prediction = attack_probability >= 0.5
            positive = attack_valid & attack_target
            negative = attack_valid & ~attack_target
            metrics[f"ctde/mask_calibration_h{horizon}_attack_recall"] = (
                attack_prediction & positive
            ).sum() / jnp.maximum(positive.sum(), 1)
            metrics[f"ctde/mask_calibration_h{horizon}_attack_fpr"] = (
                attack_prediction & negative
            ).sum() / jnp.maximum(negative.sum(), 1)
            valid_probability = mask_probability * mask_valid[..., None]
            metrics[f"ctde/mask_calibration_h{horizon}_illegal_mass"] = (
                valid_probability * (1.0 - target_mask)
            ).sum() / jnp.maximum(valid_probability.sum(), 1e-8)
            metrics[f"ctde/alive_calibration_h{horizon}_predicted_count"] = (
                (predicted_alive[:, index] * present[:, horizon]).sum(axis=-1).mean()
            )
            metrics[f"ctde/alive_calibration_h{horizon}_target_count"] = (
                (target_alive * present[:, horizon]).sum(axis=-1).mean()
            )

        def source_grid(terms):
            value = jnp.stack(terms).mean(axis=0)
            grid = jnp.zeros((batch, length, agents), jnp.float32)
            grid = grid.at[:, 0].set(value * length)
            return self.team.fold_sequence(grid)

        metrics["ctde/mask_calibration_horizons"] = jnp.asarray(
            len(self.ctde_mask_horizons), jnp.float32
        )
        return {
            "ctde_mask_calibration": source_grid(mask_terms),
            "ctde_alive_calibration": source_grid(alive_terms),
        }, metrics

    def _ctde_multistep_losses(
        self,
        first_prediction,
        joint_snapshots,
        repfeat,
        dyn_entries,
        online_embedding,
        ema_embedding,
        present,
        controllable_alive,
        is_first,
        reward,
        is_terminal,
        action_mask,
        actions,
        learner_valid,
    ):
        """Train the last step of a bounded two-step self-fed rollout."""

        batch, length, agents = present.shape
        if length < 3:
            raise ValueError("two-step CTDE training requires replay length at least 3")
        anchors = sample_two_step_anchors(
            nj.seed(),
            two_step_anchor_mask(is_first, controllable_alive),
            min(int(self.config.marl.ctde.multistep.anchors), batch * (length - 2)),
        )

        source_present = gather_anchors(present, anchors, offset=0)
        source_alive = gather_anchors(controllable_alive, anchors, offset=0)
        first_embedding = gather_anchors(
            first_prediction["embedding"], anchors, offset=0
        )
        first_hidden = gather_anchors(first_prediction["hidden"], anchors, offset=0)
        first_alive_probability = self.ctde_alive(first_hidden, 2).prob(1)
        next_present = gather_anchors(present, anchors, offset=1)
        predicted_alive = predicted_controllable_alive(
            source_alive, next_present, first_alive_probability
        )

        local_carry = {
            "deter": self.team.unfold_sequence(repfeat["deter"]),
            "stoch": self.team.unfold_sequence(repfeat["stoch"]),
            **{
                key: self.team.unfold_sequence(dyn_entries[key])
                for key in ("keys", "values", "valid", "position")
            },
        }
        local_carry = self.team.fold_tree_batch(
            gather_anchors(local_carry, anchors, offset=0)
        )
        first_actions = gather_anchors(actions, anchors, offset=1)
        folded_actions = {self.ctde_action_key: self.team.fold_batch(first_actions)}
        folded_present = self.team.fold_batch(source_present)
        local_cache, local_deter = self.dyn.advance(
            nn.cast(local_carry),
            folded_actions,
            training=False,
            active=folded_present,
        )
        next_local_carry, _ = self.dyn.complete_from_observation(
            local_cache,
            local_deter,
            self.team.fold_batch(first_embedding),
            sample=False,
        )

        def preserve_absent(next_value, source_value):
            mask = folded_present.reshape(
                (folded_present.shape[0],) + (1,) * (next_value.ndim - 1)
            )
            return jnp.where(mask, next_value, source_value)

        next_local_carry = jax.tree.map(preserve_absent, next_local_carry, local_carry)
        grouped_snapshots = jax.tree.map(self.team.unfold_sequence, joint_snapshots)
        next_joint_carry = self.team.fold_tree_batch(
            gather_anchors(grouped_snapshots, anchors, offset=0)
        )
        next_local_carry, next_joint_carry = detach_self_feed(
            (next_local_carry, next_joint_carry)
        )
        next_local_state = self.team.unfold_batch(
            self.feat2tensor(
                {
                    "deter": next_local_carry["deter"],
                    "stoch": next_local_carry["stoch"],
                }
            )
        )

        second_actions = gather_anchors(actions, anchors, offset=2)
        _, second_prediction = self.ctde_joint.step(
            next_joint_carry,
            next_local_state,
            second_actions,
            next_present,
            predicted_alive,
            jnp.zeros((anchors.batch.shape[0],), bool),
            training=False,
        )

        target_embedding = gather_anchors(ema_embedding, anchors, offset=2)
        target_online = jax.lax.stop_gradient(
            gather_anchors(online_embedding, anchors, offset=2)
        )
        interface_error = jnp.abs(
            second_prediction["embedding"].astype(jnp.float32)
            - target_online.astype(jnp.float32)
        )
        interface_loss = jnp.where(
            interface_error < 1.0,
            0.5 * jnp.square(interface_error),
            interface_error - 0.5,
        ).mean(axis=-1)

        hidden = second_prediction["hidden"]
        reward_loss = self.ctde_rew(hidden, 2).loss(
            gather_anchors(reward, anchors, offset=2)
        )
        continuation = (~gather_anchors(is_terminal, anchors, offset=2)).astype(
            jnp.float32
        )
        if self.config.contdisc:
            continuation *= 1.0 - 1.0 / float(self.config.horizon)
        continuation_loss = self.ctde_con(hidden, 2).loss(continuation)
        mask_loss = binary_vector_loss(
            self.ctde_mask(hidden, 2),
            gather_anchors(action_mask, anchors, offset=2),
            self.action_mask_reduction,
        )
        alive_loss = self.ctde_alive(hidden, 2).loss(
            gather_anchors(controllable_alive, anchors, offset=2)
        )

        target_present = gather_anchors(present, anchors, offset=2)
        standard_valid = source_alive & target_present
        alive_valid = source_present & target_present
        sampled_losses, sampled_metrics = two_step_objective(
            second_prediction["embedding"],
            target_embedding,
            {
                "interface": interface_loss,
                "reward": reward_loss,
                "continuation": continuation_loss,
                "action_mask": mask_loss,
                "alive": alive_loss,
            },
            anchors,
            standard_valid,
            learner_valid,
            auxiliary_valid={"alive": alive_valid},
        )

        target_deter = jax.lax.stop_gradient(
            gather_anchors(
                self.team.unfold_sequence(repfeat["deter"]), anchors, offset=2
            )
        )
        predicted_logits = jax.lax.stop_gradient(
            self.dyn.posterior(
                self.team.fold_batch(second_prediction["embedding"]),
                self.team.fold_batch(target_deter),
            )
        )
        factual_logits = jax.lax.stop_gradient(
            self.dyn.posterior(
                self.team.fold_batch(target_online),
                self.team.fold_batch(target_deter),
            )
        )
        predicted_logprob = jax.nn.log_softmax(
            predicted_logits.astype(jnp.float32), axis=-1
        )
        factual_logprob = jax.nn.log_softmax(
            factual_logits.astype(jnp.float32), axis=-1
        )
        posterior_kl = jnp.sum(
            jnp.exp(factual_logprob) * (factual_logprob - predicted_logprob),
            axis=(-1, -2),
        )
        posterior_kl = self.team.unfold_batch(posterior_kl)
        posterior_valid = (
            anchors.valid[:, None]
            & standard_valid
            & learner_valid[anchors.batch, anchors.time]
        )
        posterior_weight = posterior_valid.astype(jnp.float32)
        posterior_metric = (
            posterior_kl.astype(jnp.float32) * posterior_weight
        ).sum() / jnp.maximum(posterior_weight.sum(), 1.0)

        losses = {
            f"ctde_multistep_{name}": self.team.fold_sequence(value)
            for name, value in sampled_losses.items()
        }
        metrics = {
            f"ctde/multistep_{name}": value for name, value in sampled_metrics.items()
        }
        metrics["ctde/multistep_posterior_kl"] = posterior_metric
        metrics["ctde/multistep_anchors"] = anchors.valid.astype(jnp.float32).sum()
        return losses, metrics

    def _apply_replay_context(self, carry, data):
        if not self.ctde_enabled or not self.config.replay_context:
            return super()._apply_replay_context(carry, data)

        enc_carry, dyn_carry, dec_carry, prevact = carry
        normal_carry = (enc_carry, dyn_carry, dec_carry)
        stepid = data["stepid"]
        obs = {key: data[key] for key in self.obs_space}

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
        replay_obs = {key: rhs(data[key]) for key in self.obs_space}
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
        if not self.ctde_enabled:
            return suffix_obs
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
        if not self.ctde_enabled:
            return entries
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
        if not self.ctde_enabled:
            return result
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
        if self.ctde_enabled:
            present = self._present(obs)[:, -starts_count:].reshape((-1,))
            grouped_present = self.team.group_starts(present, starts_count).astype(bool)
            alive = self._controllable(obs)[:, -starts_count:].reshape((-1,))
            grouped_alive = self.team.group_starts(alive, starts_count).astype(bool)
            grouped_present &= ~grouped_last
            grouped_alive &= ~grouped_last
            action_mask = obs["action_mask"][:, -starts_count:]
            action_mask = action_mask.reshape((-1, *action_mask.shape[2:]))
            grouped_mask = self.team.group_starts(action_mask, starts_count).astype(
                bool
            )
            noop = jnp.zeros_like(grouped_mask).at[..., 0].set(True)
            grouped_mask = jnp.where(grouped_alive[..., None], grouped_mask, noop)
            grouped_state = self.team.unfold_sequence(self.feat2tensor(repfeat))
            grouped_action = self.team.unfold_sequence(
                prevact[self.ctde_action_key]
            ).astype(jnp.int32)
            grouped_history_present = self.team.unfold_sequence(
                self._present(obs)
            ).astype(bool)
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
                    folded_first.reshape(
                        (*folded_first.shape, *((1,) * (value.ndim - 2)))
                    ),
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

    def imagine_with_aux(self, starts, policy, horizon, training, context=None):
        if not self.ctde_enabled:
            return super().imagine_with_aux(starts, policy, horizon, training, context)
        if self.ctde_probabilistic_availability:
            return self._imagine_with_calibrated_mask(
                starts, horizon, training, context
            )
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
            folded_action = sample(distribution)
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
            predicted_embedding = self.team.fold_batch(prediction["embedding"])
            folded_next, next_features = self.dyn.complete_from_observation(
                local_cache,
                deter,
                predicted_embedding,
                sample=True,
            )
            next_carry = self.team.unfold_tree_batch(folded_next)
            next_features = self.team.unfold_tree_batch(next_features)

            hidden = prediction["hidden"]
            reward = self.ctde_rew(hidden, 2).pred()
            continuation = self.ctde_con(hidden, 2).prob(1)
            alive_probability = self.ctde_alive(hidden, 2).prob(1)
            next_present = current_present
            next_alive = jax.lax.stop_gradient(
                current_alive & next_present & (alive_probability >= 0.5)
            )
            mask_output = self.ctde_mask(hidden, 2)
            mask_probability = jax.nn.sigmoid(mask_output.output.logit)
            next_mask = jax.lax.stop_gradient(mask_probability >= 0.5)
            noop = jnp.zeros_like(next_mask)
            noop = noop.at[..., 0].set(True)
            next_mask = jnp.where(
                next_mask.any(axis=-1, keepdims=True), next_mask, noop
            )
            next_mask = jnp.where(next_alive[..., None], next_mask, noop)

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

    def _ctde_probabilistic_policy(
        self,
        tensor,
        bdims,
        alive,
        *,
        availability_logits=None,
        exact_mask=None,
        exact_rows=None,
    ):
        """Apply local probabilistic availability, retaining exact root support."""

        base_distribution, _ = self._teammate_policy_before_mask(tensor, bdims)
        if availability_logits is None:
            availability_output = self.actmask(tensor, bdims=bdims)
            availability_binary = (
                availability_output.output
                if hasattr(availability_output, "output")
                else availability_output
            )
            availability_logits = availability_binary.logit
        availability_logits = jax.lax.stop_gradient(availability_logits)
        if self.ctde_support_preserving:
            distribution = apply_support_preserving_availability(
                base_distribution,
                availability_logits,
                alive,
                self.ctde_action_key,
                probability_floor=self.ctde_support_probability_floor,
            )
        elif self.ctde_soft_liveness:
            alive_probability = jnp.clip(alive.astype(jnp.float32), 0.0, 1.0)
            availability = jax.nn.sigmoid(availability_logits)
            availability *= alive_probability[..., None]
            noop_availability = (
                1.0
                - alive_probability
                + alive_probability * jax.nn.sigmoid(availability_logits[..., 0])
            )
            availability = availability.at[..., 0].set(noop_availability)
            base_action = base_distribution[self.ctde_action_key]
            soft_action = jaxouts.Categorical(
                base_action.logits + jnp.log(jnp.clip(availability, 1e-6, 1.0))
            )
            for name in ("minent", "maxent"):
                if hasattr(base_action, name):
                    setattr(soft_action, name, getattr(base_action, name))
            distribution = dict(
                base_distribution, **{self.ctde_action_key: soft_action}
            )
        else:
            distribution = apply_predicted_action_mask(
                base_distribution,
                availability_logits,
                self.ctde_action_key,
            )
            noop = jnp.zeros((*alive.shape, self.ctde_action_count), bool)
            noop = noop.at[..., 0].set(True)
            alive_support = jnp.where(alive[..., None], jnp.ones_like(noop), noop)
            distribution = apply_action_mask(
                distribution,
                alive_support,
                self.ctde_action_key,
            )
        if exact_mask is None:
            return distribution, availability_logits
        if exact_rows is None or exact_rows.shape != alive.shape:
            raise ValueError("exact CTDE root rows must match folded liveness")
        exact = apply_action_mask(
            base_distribution,
            exact_mask,
            self.ctde_action_key,
        )
        soft_action = distribution[self.ctde_action_key]
        exact_action = exact[self.ctde_action_key]
        logits = jnp.where(
            exact_rows[..., None], exact_action.logits, soft_action.logits
        )
        selected = jaxouts.Categorical(logits)
        for name in ("minent", "maxent"):
            if hasattr(soft_action, name):
                setattr(selected, name, getattr(soft_action, name))
        return (
            dict(distribution, **{self.ctde_action_key: selected}),
            availability_logits,
        )

    def _imagine_with_calibrated_mask(self, starts, horizon, training, context):
        """CTDE imagination with exact roots and local probabilistic availability."""

        grouped_carry = nn.cast(self.team.unfold_tree_batch(starts))
        present = context["present"].astype(bool)
        alive = context["controllable_alive"].astype(bool)
        if self.ctde_soft_liveness:
            alive = alive.astype(jnp.float32)
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
                exact_root,
            ) = state
            local_features = {
                "deter": local_carry["deter"],
                "stoch": local_carry["stoch"],
            }
            folded_features = self.team.fold_tree_batch(local_features)
            folded_mask = self.team.fold_batch(current_mask)
            folded_alive = self.team.fold_batch(current_alive)
            exact_rows = self.team.fold_batch(
                jnp.broadcast_to(exact_root[:, None], current_alive.shape)
            )
            distribution, availability_logits = self._ctde_probabilistic_policy(
                self.feat2tensor(folded_features),
                1,
                folded_alive,
                exact_mask=folded_mask,
                exact_rows=exact_rows,
            )
            folded_action = sample(distribution)
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
            predicted_embedding = self.team.fold_batch(prediction["embedding"])
            folded_next, folded_next_features = self.dyn.complete_from_observation(
                local_cache,
                deter,
                predicted_embedding,
                sample=True,
            )
            next_carry = self.team.unfold_tree_batch(folded_next)
            next_features = self.team.unfold_tree_batch(folded_next_features)

            hidden = prediction["hidden"]
            reward = self.ctde_rew(hidden, 2).pred()
            continuation = self.ctde_con(hidden, 2).prob(1)
            alive_probability = self.ctde_alive(hidden, 2).prob(1)
            next_present = current_present
            if self.ctde_soft_liveness:
                next_alive = jax.lax.stop_gradient(
                    current_alive * next_present.astype(jnp.float32) * alive_probability
                )
            else:
                next_alive = jax.lax.stop_gradient(
                    current_alive & next_present & (alive_probability >= 0.5)
                )
            noop = jnp.zeros_like(current_mask).at[..., 0].set(True)
            next_mask = jnp.ones_like(current_mask)
            if not self.ctde_soft_liveness:
                next_mask = jnp.where(next_alive[..., None], next_mask, noop)

            next_state = (
                next_carry,
                joint_carry,
                next_present,
                next_alive,
                next_mask,
                jnp.zeros_like(current_reset),
                jnp.zeros_like(exact_root),
            )
            outputs = (
                next_features,
                self.team.unfold_tree_batch(folded_action),
                reward,
                continuation,
                next_mask,
                next_present,
                next_alive,
                self.team.unfold_batch(availability_logits),
            )
            return next_state, outputs

        state = (
            grouped_carry,
            central_carry,
            present,
            alive,
            action_mask,
            reset,
            jnp.ones((teams,), bool),
        )
        state, outputs = nj.scan(transition, state, (), horizon, axis=1)
        (
            next_features,
            actions,
            rewards,
            continuations,
            masks,
            present_sequence,
            alive_sequence,
            availability_sequence,
        ) = outputs
        local_carry = self.team.fold_tree_batch(state[0])
        features = jax.tree.map(self.team.fold_sequence, next_features)
        actions = jax.tree.map(self.team.fold_sequence, actions)
        final_features = {
            "deter": state[0]["deter"],
            "stoch": state[0]["stoch"],
        }
        final_availability_output = self.actmask(
            self.feat2tensor(self.team.fold_tree_batch(final_features)), 1
        )
        final_availability_binary = (
            final_availability_output.output
            if hasattr(final_availability_output, "output")
            else final_availability_output
        )
        final_availability = final_availability_binary.logit
        final_availability = self.team.unfold_batch(
            jax.lax.stop_gradient(final_availability)
        )
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
            "availability_logits": jnp.concatenate(
                [availability_sequence, final_availability[:, None]], axis=1
            ),
            "present": jnp.concatenate([present[:, None], present_sequence], axis=1),
            "controllable_alive": jnp.concatenate(
                [alive[:, None], alive_sequence], axis=1
            ),
        }
        return local_carry, features, actions, auxiliary

    def imagination_last_action(self, policy_features, auxiliary, policyfn):
        if not self.ctde_enabled:
            return super().imagination_last_action(policy_features, auxiliary, policyfn)
        del policyfn
        last_features = jax.tree.map(lambda value: value[:, -1], policy_features)
        if self.ctde_probabilistic_availability:
            last_alive = self.team.fold_batch(auxiliary["controllable_alive"][:, -1])
            distribution, _ = self._ctde_probabilistic_policy(
                self.feat2tensor(last_features),
                1,
                last_alive,
                availability_logits=self.team.fold_batch(
                    auxiliary["availability_logits"][:, -1]
                ),
            )
            return sample(distribution)
        last_mask = self.team.fold_batch(auxiliary["action_mask"][:, -1])
        distribution = self.policy_distribution(
            self.feat2tensor(last_features),
            1,
            action_mask=last_mask,
        )
        return sample(distribution)

    def imagination_policy_distribution(self, policy_inputs, auxiliary):
        if not self.ctde_enabled:
            return super().imagination_policy_distribution(policy_inputs, auxiliary)
        action_mask = self.team.fold_sequence(auxiliary["action_mask"])
        if self.ctde_probabilistic_availability:
            grouped_alive = auxiliary["controllable_alive"]
            folded_alive = self.team.fold_sequence(grouped_alive)
            root_rows = jnp.zeros_like(grouped_alive)
            root_rows = root_rows.at[:, 0].set(True)
            distribution, _ = self._ctde_probabilistic_policy(
                policy_inputs,
                2,
                folded_alive,
                availability_logits=self.team.fold_sequence(
                    auxiliary["availability_logits"]
                ),
                exact_mask=action_mask,
                exact_rows=self.team.fold_sequence(root_rows),
            )
            return distribution
        return self.policy_distribution(policy_inputs, 2, action_mask=action_mask)

    def imagination_reward_continuation(self, local_inputs, auxiliary):
        if not self.ctde_enabled:
            return super().imagination_reward_continuation(local_inputs, auxiliary)
        del local_inputs
        return (
            self.team.fold_sequence(auxiliary["reward"]),
            self.team.fold_sequence(auxiliary["continuation"]),
        )

    def restore_imagination_results(self, losses, outputs, context=None):
        starts_count = context["starts_count"] if self.ctde_enabled else context[0]

        def restore(value):
            grouped = self.team.unfold_batch(value)
            return self.team.ungroup_starts(grouped, starts_count)

        return jax.tree.map(restore, (losses, outputs))

    def imagination_validity(self, context, horizon, auxiliary=None):
        if self.ctde_enabled:
            if auxiliary is None:
                raise ValueError("CTDE imagination requires predicted activity")
            present = auxiliary["present"]
            validity = present
            if self.ctde_death_masking or self.ctde_soft_liveness:
                validity = validity.astype(jnp.float32) * auxiliary[
                    "controllable_alive"
                ].astype(jnp.float32)
            folded = self.team.fold_sequence(validity)
            return folded[:, :horizon]
        _, active = context
        return jnp.broadcast_to(active[:, None], (active.shape[0], horizon))

    def replay_value_validity(self, obs):
        validity = super().replay_value_validity(obs)
        if self.ctde_enabled and self.ctde_death_masking:
            validity *= self._controllable(obs).astype(jnp.float32)
        return validity

    def imagination_behavior_metrics(self, actions, validity, auxiliary=None):
        """Summarize the actions that actually drive CTDE imagination."""

        if not self.ctde_enabled:
            return super().imagination_behavior_metrics(actions, validity, auxiliary)
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
