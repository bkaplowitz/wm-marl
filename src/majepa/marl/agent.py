"""Multi-agent model construction and decentralized execution boundary.

Team identity is retained for the joint predictor and critic. Executable actors
share parameters but consume only their own observation and local history.
"""

from __future__ import annotations

import math

import elements
import embodied.jax
import embodied.jax.outs as jaxouts
import jax.numpy as jnp

from ..agent import Agent as LocalAgent
from ..models.ctde import (
    CentralAttentionCritic,
    JointObservationJEPA,
)
from ..models.multistep_jepa import (
    ActionConditionedMultiStepJEPA,
)
from ..models.target import CriticTarget
from ..training.joint import joint_world_model_losses
from .axes import (
    BEHAVIOR_REPLAY_PREFIX,
    TeamAxis,
    is_environment_field,
    split_prefixed_data,
)
from .imagination import JointImagination
from .replay import TeamReplay
from .spaces import (
    add_agent_axis,
    local_action_spaces,
    local_observation_spaces,
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


class MARLCore(
    TeamReplay,
    JointImagination,
    TeamAxisAdapter,
    LocalAgent,
):
    """Agent-axis learner with a permanently decentralized actor path."""

    def __init__(self, obs_space, act_space, config):
        marl = config.marl
        if str(marl.stage) != "ctde":
            raise ValueError(f"unsupported MARL stage: {marl.stage!r}")
        if str(marl.execution) != "strict_decentralized":
            raise ValueError(f"unsupported execution contract: {marl.execution!r}")
        self.team = TeamAxis(int(config.num_agents))
        self.public_obs_space = dict(obs_space)
        self.public_act_space = dict(act_space)
        self.marl_stage = str(marl.stage)
        self.ctde_enabled = self.marl_stage == "ctde" and self.team.size > 1
        self.ctde_self_fed_enabled = bool(
            marl.ctde.get("self_fed", {}).get("enabled", False)
        )
        if self.ctde_self_fed_enabled:
            self_fed = marl.ctde.self_fed
            if int(self_fed.get("bptt_steps", 1)) not in (1, 2):
                raise ValueError("Self-fed bptt_steps must be 1 or 2")
            horizons = tuple(int(value) for value in self_fed.horizons)
            if (
                not horizons
                or horizons != tuple(sorted(set(horizons)))
                or min(horizons) < 2
                or max(horizons) > 15
                or int(self_fed.anchors) < 1
                or any(
                    not math.isfinite(value) or value < 0
                    for value in (
                        float(self_fed.scale),
                        float(self_fed.get("trajectory_kl_scale", 0.0)),
                    )
                )
            ):
                raise ValueError(
                    "Self-fed training needs sorted unique H2..15, positive anchors and finite nonnegative scales"
                )
            if int(config.batch_length) < 3:
                raise ValueError(
                    "Self-fed training requires at least three replay states"
                )
            # A zero-scale control must preserve the complete baseline graph,
            # including loss keys, parameter creation and random-key ordering.
            self.ctde_self_fed_enabled = float(self_fed.scale) > 0
        self.ctde_imagination_mask_sampling = marl.ctde.get(
            "imagination_mask_sampling", "threshold"
        )
        self.ctde_imagination_mask_source = marl.ctde.get(
            "imagination_mask_source", "joint"
        )
        self.joint_mask_enabled = True

        if self.ctde_imagination_mask_source not in {"joint", "local"}:
            raise ValueError("imagination_mask_source must be joint or local")
        if self.ctde_imagination_mask_sampling not in {"threshold", "bernoulli"}:
            raise ValueError("imagination_mask_sampling must be threshold or bernoulli")
        self.ctde_multistep_jepa_enabled = bool(marl.ctde.multistep_jepa.enabled)
        self.ctde_multistep_jepa_action_scale = (
            float(config.loss_scales.ctde_multistep_jepa_action)
            if self.ctde_multistep_jepa_enabled
            else 0.0
        )
        if self.ctde_multistep_jepa_action_scale < 0.0:
            raise ValueError("multi-step JEPA action loss scale must be nonnegative")
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
        self.action_mask_reduction = str(
            getattr(config, "action_mask_reduction", "sum")
        )
        if self.action_mask_reduction not in {"sum", "mean", "balanced"}:
            raise ValueError(
                "action_mask_reduction must be 'sum', 'mean', or 'balanced'"
            )
        self.ctde_posterior_alignment_scale = float(
            config.loss_scales.get("ctde_posterior_alignment", 0.0)
        )
        if (
            not math.isfinite(self.ctde_posterior_alignment_scale)
            or self.ctde_posterior_alignment_scale < 0
        ):
            raise ValueError("Latent alignment scale must be finite and nonnegative")
        local_obs_space = local_observation_spaces(obs_space, self.team.size)
        local_act_space = local_action_spaces(act_space, self.team.size)
        super().__init__(
            local_obs_space,
            local_act_space,
            config,
        )
        if self.ctde_self_fed_enabled:
            scale = float(marl.ctde.self_fed.scale)
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
            if float(marl.ctde.self_fed.get("trajectory_kl_scale", 0.0)):
                self.scales["ctde_self_fed_trajectory_kl"] = scale * float(
                    marl.ctde.self_fed.trajectory_kl_scale
                )
        if self.ctde_multistep_jepa_enabled:
            if int(config.batch_length) <= self.ctde_multistep_jepa_max_horizon:
                raise ValueError(
                    "multi-step JEPA batch_length must exceed max_horizon, got "
                    f"{config.batch_length} and "
                    f"{self.ctde_multistep_jepa_max_horizon}"
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
        ctde_modules = [module for module in ctde_modules if module is not None]
        actor_modules = []
        if self.ctde_multistep_jepa_enabled:
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

    def additional_world_model_losses(
        self, tokens, repfeat, dyn_entries, target_tokens, obs, prevact, training
    ):
        return joint_world_model_losses(
            self, tokens, repfeat, dyn_entries, target_tokens, obs, prevact, training
        )
