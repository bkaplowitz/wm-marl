"""World-model and imagined PPO training orchestration."""

import elements
import jax.numpy as jnp
import ninjax as nj

from ..models.heads import binary_vector_loss
from .behavior import BehaviorLearning
from .common import masked_mean, sg
from .representation import (
    embedding_prediction_loss,
    embedding_std,
    sigreg_loss,
)


class LearnerMixin(BehaviorLearning):
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
        entropy_coefficient = jnp.asarray(
            self.config.ppo.entropy_coefficient, jnp.float32
        )
        ppo_batch, batch_metrics = self._prepare_ppo_batch(
            behavior_data,
            entropy_coefficient,
        )
        metrics.update(batch_metrics)
        actor_epochs = []
        critic_epochs = []
        for epoch in range(max(self.ppo_actor_epochs, self.ppo_critic_epochs)):
            if epoch < self.ppo_actor_epochs:
                actor_optimizer, actor_metrics = self.opt.step_group(
                    "actor",
                    self._ppo_actor_loss,
                    ppo_batch,
                    has_aux=True,
                    active=ppo_active,
                )
                actor_epochs.append(actor_metrics)
            if epoch < self.ppo_critic_epochs:
                critic_optimizer, critic_metrics = self.opt.step_group(
                    "critic",
                    self._ppo_critic_loss,
                    ppo_batch,
                    has_aux=True,
                    active=ppo_active,
                )
                critic_epochs.append(critic_metrics)
        metrics.update(actor_optimizer)
        metrics.update(critic_optimizer)
        metrics.update(self._ppo_epoch_metrics("actor", actor_epochs))
        metrics.update(self._ppo_epoch_metrics("critic", critic_epochs))
        metrics["ppo/epochs"] = jnp.asarray(self.config.ppo.epochs, jnp.float32)
        metrics["ppo/actor_epochs"] = jnp.asarray(self.ppo_actor_epochs, jnp.float32)
        metrics["ppo/critic_epochs"] = jnp.asarray(self.ppo_critic_epochs, jnp.float32)
        metrics["ppo/active"] = ppo_active.astype(jnp.float32)
        metrics["ppo/entropy_coefficient"] = entropy_coefficient

        metrics.update(self._ppo_post_update_metrics(ppo_batch))

        self._update_slow_models(ppo_active)
        if self.slowenc is not None:
            self._gated_slow_update(self.slowenc, metrics["opt/finite"])
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

    def _ppo_post_update_metrics(self, batch):
        """Measure final parameters without consuming training RNG or writing state.

        Historical final_* auxiliaries precede the last optimizer update. This
        separate series reuses the immutable batch, including its realized masks,
        before slow-target copying. The nested context disallows state mutation.
        """

        def evaluate(batch):
            return self._ppo_actor_loss(batch)[1], self._ppo_critic_loss(batch)[1]

        _, (actor, critic) = nj.pure(evaluate, nested=True)(
            dict(nj.context()),
            batch,
            seed=719_243,
            create=False,
            modify=False,
        )
        return {
            f"ppo/{group}/post_update_{key}": sg(value)
            for group, values in (("actor", actor), ("critic", critic))
            for key, value in values.items()
        }

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

    def representation_prediction_branches(self, repfeat, dynamics_aux):
        """Return predictive states that share the maintained JEPA targets."""

        del dynamics_aux
        return {"model": repfeat}

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
        dec_entries = {}
        if self.slowenc is not None:
            _, _, target_tokens = self.slowenc(
                self.target_enc.initial(batch),
                obs,
                reset,
                training=False,
            )
        else:
            target_tokens = tokens
        stop_target = True

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

        add_embedding_loss("posterior_jepa", self.feat2tensor)
        add_embedding_loss(
            "dynamics_jepa", lambda features: features["deter"], "dynpred"
        )
        return (
            (enc_carry, dyn_carry, dec_carry),
            (enc_entries, dyn_entries, dec_entries),
            tokens,
            repfeat,
            losses,
            metrics,
            target_tokens,
        )

    def report(self, carry, data):
        if not self.config.report:
            return carry, {}
        carry, obs, prevact, _ = self._apply_replay_context(carry, data)
        _, (new_carry, _, _, metrics) = self.loss(carry, obs, prevact, training=False)
        carry = (*new_carry, {key: data[key][:, -1] for key in self.act_space})
        return carry, metrics
