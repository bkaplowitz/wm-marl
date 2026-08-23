"""Agent-axis runtime for the maintained DreaMARL learner.

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
from ..models.ctde import CentralAttentionCritic, JointObservationJEPA
from ..training.ctde import (
    detach_self_feed,
    gather_anchors,
    predicted_controllable_alive,
    sample_two_step_anchors,
    two_step_anchor_mask,
    two_step_objective,
)
from ..training.common import sample
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
    """Agent-axis learner with a permanently decentralized actor path."""

    def __init__(self, obs_space, act_space, config):
        marl = config.marl
        if str(marl.stage) not in {"local", "ctde"}:
            raise ValueError(f"unsupported MARL stage: {marl.stage!r}")
        if str(marl.execution) != "strict_decentralized":
            raise ValueError(f"unsupported execution contract: {marl.execution!r}")
        self.team = TeamAxis(int(config.num_agents))
        self.marl_stage = str(marl.stage)
        self.ctde_enabled = self.marl_stage == "ctde" and self.team.size > 1
        self.ctde_rollout_steps = (
            int(marl.ctde.rollout_steps) if self.ctde_enabled else 1
        )
        if self.ctde_rollout_steps not in {1, 2}:
            raise ValueError("CTDE rollout_steps must be 1 or 2")
        local_obs_space = local_observation_spaces(obs_space, self.team.size)
        local_act_space = local_action_spaces(act_space, self.team.size)
        super().__init__(
            local_obs_space,
            local_act_space,
            config,
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
            grouped_alive = context["controllable_alive"].astype(bool)
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
            self.ctde_modules = (
                self.ctde_joint,
                self.ctde_rew,
                self.ctde_con,
                self.ctde_mask,
                self.ctde_alive,
            )
            modules.extend(self.ctde_modules)
        return modules

    def imagination_critic_context(self, features, context, auxiliary=None):
        if not self.ctde_enabled:
            return super().imagination_critic_context(features, context, auxiliary)
        if auxiliary is None:
            raise ValueError("CTDE critic requires imagined activity")
        return {
            "present": auxiliary["present"],
            "controllable_alive": auxiliary["controllable_alive"],
        }, {}

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
        mask_loss = self.ctde_mask(hidden, 3).loss(grouped_mask[:, 1:])
        alive_loss = self.ctde_alive(hidden, 3).loss(grouped_alive[:, 1:])
        alive_valid = source_present & ~next_first[..., None]
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

        losses = {
            "ctde_embedding": folded(embedding_loss),
            "ctde_interface": folded(interface_loss),
            "ctde_reward": folded(reward_loss),
            "ctde_continuation": folded(continuation_loss),
            "ctde_action_mask": folded(mask_loss),
            "ctde_alive": folded_alive(alive_loss),
        }
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
        mask_loss = self.ctde_mask(hidden, 2).loss(
            gather_anchors(action_mask, anchors, offset=2)
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
        if self.ctde_enabled:
            present = self._present(obs)[:, -starts_count:].reshape((-1,))
            grouped_present = self.team.group_starts(present, starts_count).astype(bool)
            alive = self._controllable(obs)[:, -starts_count:].reshape((-1,))
            grouped_alive = self.team.group_starts(alive, starts_count).astype(bool)
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

    def imagination_last_action(self, policy_features, auxiliary, policyfn):
        if not self.ctde_enabled:
            return super().imagination_last_action(policy_features, auxiliary, policyfn)
        del policyfn
        last_features = jax.tree.map(lambda value: value[:, -1], policy_features)
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
            folded = self.team.fold_sequence(present)
            return folded[:, :horizon]
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
