"""Training summaries and open-loop latent prediction reports."""

import jax
import ninjax as nj
import embodied.jax.nets as nn
from .ctde import shared_team_outcomes
from .self_fed import mixed_posterior_kl
from .availability import availability_metrics
import jax.numpy as jnp

from .representation import embedding_prediction_loss


class ReportingMixin:
    """Report training losses and open-loop latent/interface accuracy."""

    def report(self, carry, data):
        if not self.config.report:
            return carry, {}

        carry, obs, prevact, _ = self._apply_replay_context(carry, data)
        _, dyn_carry, dec_carry = carry
        batch, length = obs["is_first"].shape
        rows = self.report_rows(batch)
        metrics = {}

        _, (new_carry, entries, outs, loss_metrics) = self.loss(
            carry, obs, prevact, training=False
        )
        metrics.update(loss_metrics)

        def firsthalf(tree):
            return jax.tree.map(lambda value: value[:rows, : length // 2], tree)

        def secondhalf(tree):
            return jax.tree.map(lambda value: value[:rows, length // 2 :], tree)

        midpoint = length // 2
        dyn_carry = self.report_imagination_start(entries[1], midpoint - 1)
        dyn_carry = jax.tree.map(lambda value: value[:rows], dyn_carry)
        dec_carry = jax.tree.map(lambda value: value[:rows], dec_carry)
        _, imagined, _ = self.report_imagination(
            dyn_carry,
            secondhalf(prevact),
            length=length - length // 2,
            training=False,
        )
        if "target_tokens" in outs:
            predictor_name = "dynpred"
            metric_name = "dynamics_jepa"
            target = secondhalf(outs["target_tokens"])
            branch_cosines = []
            branch_mses = []
            branches = self.openloop_prediction_branches(imagined)
            for branch, features in branches.items():
                predicted = self.dyn.predictor(features["deter"], name=predictor_name)
                _, cosine, mse = embedding_prediction_loss(
                    predicted,
                    target,
                    distance="cosine",
                    stop_target=True,
                )
                branch_cosines.append(cosine)
                branch_mses.append(mse)
                if len(branches) > 1:
                    metrics[f"openloop/{metric_name}_{branch}_cosine"] = cosine.mean()
                    metrics[f"openloop/{metric_name}_{branch}_mse"] = mse.mean()
                    for horizon in (1, 2, 4, 8):
                        if horizon <= cosine.shape[1]:
                            metrics[
                                f"openloop/{metric_name}_{branch}_cosine_h{horizon}"
                            ] = cosine[:, horizon - 1].mean()
                            metrics[
                                f"openloop/{metric_name}_{branch}_mse_h{horizon}"
                            ] = mse[:, horizon - 1].mean()
            cosine = jnp.stack(branch_cosines).mean(0)
            mse = jnp.stack(branch_mses).mean(0)
            metrics[f"openloop/{metric_name}_cosine"] = cosine.mean()
            metrics[f"openloop/{metric_name}_mse"] = mse.mean()
            for horizon in (1, 2, 4, 8):
                if horizon <= cosine.shape[1]:
                    metrics[f"openloop/{metric_name}_cosine_h{horizon}"] = cosine[
                        :, horizon - 1
                    ].mean()
                    metrics[f"openloop/{metric_name}_mse_h{horizon}"] = mse[
                        :, horizon - 1
                    ].mean()

        carry = (*new_carry, {key: data[key][:, -1] for key in self.act_space})
        return carry, metrics

    def report_imagination(self, carry, actions, length, training):
        return self.dyn.imagine(carry, actions, length, training)

    def report_imagination_start(self, entries, index):
        return self.dyn.start_at(entries, index)

    def openloop_prediction_branches(self, imagined):
        return {"model": imagined}

    def _ctde_self_fed_report(
        self,
        repfeat,
        dyn_entries,
        joint_snapshots,
        online_tokens,
        target_tokens,
        obs,
        prevact,
    ):
        """Check PPO's actual joint/posterior simulator along factual actions.

        A single current-weight factual root per team is rolled forward without
        observation feedback. This differs from the local-prior open-loop and
        direct multi-horizon JEPA reports, neither of which runs this simulator.
        Episode-reset crossings are excluded, while terminal-transition rewards
        and after-death team outcomes remain in the diagnostics.
        """

        length = online_tokens.shape[1]
        horizon = min(8, length - 1)
        if horizon < 1:
            return {}
        teams = min(4, online_tokens.shape[0] // self.team.size)
        rows = teams * self.team.size
        root = min(length // 2, length - horizon - 1)
        selected = (1, 2, 4, 5, 8)

        def grouped(value):
            return self.team.unfold_sequence(value[:rows])

        def targets(value):
            return grouped(value)[:, root + 1 : root + horizon + 1]

        present_history = grouped(self._present(obs)).astype(bool)
        alive_history = grouped(self._controllable(obs)).astype(bool)
        first_history = grouped(obs["is_first"]).any(axis=-1)
        local_carry = {
            key: value[:rows, root]
            for key, value in dyn_entries.items()
            if key in {"deter", "stoch", "keys", "values", "valid", "position"}
        }
        joint_carry = {
            key: (joint_snapshots[key][:rows, root - 1] if root else value[:rows])
            for key, value in dyn_entries["ctde_joint_carry"].items()
        }
        source_present = present_history[:, root]
        source_alive = alive_history[:, root]
        actions = grouped(prevact[self.ctde_action_key])[
            :, root + 1 : root + horizon + 1
        ]

        def transition(state, action):
            local, joint, present, alive, reset = state
            local_features = {"deter": local["deter"], "stoch": local["stoch"]}
            local_state = self.team.unfold_batch(self.feat2tensor(local_features))
            joint, prediction = self.ctde_joint.step(
                joint, local_state, action, present, alive, reset, training=False
            )
            cache, deter = self.dyn.advance(
                local,
                {self.ctde_action_key: self.team.fold_batch(action)},
                training=False,
                active=self.team.fold_batch(present),
            )
            local, next_features = self._ctde_complete(cache, deter, prediction)
            hidden = prediction["hidden"]
            alive_probability = self.ctde_alive(hidden, 2).prob(1)
            next_alive = alive & present & (alive_probability >= 0.5)
            reward, continuation = shared_team_outcomes(
                self.ctde_rew(hidden, 2).pred(),
                self.ctde_con(hidden, 2).prob(1),
                present,
                alive,
                next_alive,
            )
            mask_output = self.ctde_mask(hidden, 2)
            binary = (
                mask_output.output if hasattr(mask_output, "output") else mask_output
            )
            mask_probability = jax.nn.sigmoid(binary.logit)
            mask_logits = binary.logit
            mask_prediction = mask_probability >= 0.5
            noop = jnp.zeros_like(mask_prediction).at[..., 0].set(True)
            mask_prediction = jnp.where(
                mask_prediction.any(axis=-1, keepdims=True), mask_prediction, noop
            )
            mask_prediction = jnp.where(
                (next_alive >= 0.5)[..., None], mask_prediction, noop
            )
            outputs = {
                "embedding": prediction["embedding"],
                "posterior": self.team.unfold_batch(next_features["logit"]),
                "reward": reward,
                "continuation": continuation,
                "action_mask": mask_prediction,
                "mask_logits": mask_logits,
                "predicted_alive": next_alive,
                "alive_probability": alive.astype(jnp.float32) * alive_probability,
            }
            return (local, joint, present, next_alive, jnp.zeros_like(reset)), outputs

        initial = (
            nn.cast(jax.lax.stop_gradient(local_carry)),
            nn.cast(jax.lax.stop_gradient(joint_carry)),
            source_present,
            source_alive,
            first_history[:, root],
        )
        _, prediction = nj.scan(transition, initial, actions, axis=1)
        prediction = jax.lax.stop_gradient(prediction)
        path_valid = jnp.cumprod(
            (~first_history[:, root + 1 : root + horizon + 1]).astype(jnp.int32),
            axis=1,
        ).astype(bool)
        # Shared reward/continuation remain valid for a dead focal roster slot.
        valid = source_present[:, None] & targets(self._present(obs)).astype(bool)
        valid &= path_valid[..., None]
        local_valid = valid & alive_history[:, root : root + horizon]
        target_reward = targets(obs["reward"])
        target_continuation = (~targets(obs["is_terminal"])).astype(jnp.float32)
        if self.config.contdisc:
            target_continuation *= 1.0 - 1.0 / float(self.config.horizon)
        target_mask = targets(obs["action_mask"]).astype(bool)
        target_alive = targets(self._controllable(obs)).astype(jnp.float32)
        ema = targets(target_tokens).astype(jnp.float32)
        predicted_embedding = prediction["embedding"].astype(jnp.float32)
        ema_unit = ema / jnp.maximum(jnp.linalg.norm(ema, axis=-1, keepdims=True), 1e-8)
        pred_unit = predicted_embedding / jnp.maximum(
            jnp.linalg.norm(predicted_embedding, axis=-1, keepdims=True), 1e-8
        )
        cosine = (ema_unit * pred_unit).sum(axis=-1)
        factual_logits = self.dyn.posterior(
            online_tokens[:rows, root + 1 : root + horizon + 1],
            repfeat["deter"][:rows, root + 1 : root + horizon + 1],
        )
        factual_logprob = jax.nn.log_softmax(
            grouped(factual_logits).astype(jnp.float32), axis=-1
        )
        predicted_logprob = jax.nn.log_softmax(
            prediction["posterior"].astype(jnp.float32), axis=-1
        )
        posterior_kl = (
            jnp.exp(factual_logprob) * (factual_logprob - predicted_logprob)
        ).sum(axis=(-1, -2))
        executable_posterior_kl = mixed_posterior_kl(
            prediction["posterior"], grouped(factual_logits), self.dyn.unimix
        )
        reward_error = prediction["reward"] - target_reward
        metrics = {}

        for step in selected:
            if step > horizon:
                continue
            index = step - 1
            weight = valid[:, index].astype(jnp.float32)
            local_weight = local_valid[:, index].astype(jnp.float32)

            def mean(value, selected_weight=weight):
                return (
                    value.astype(jnp.float32) * selected_weight
                ).sum() / jnp.maximum(selected_weight.sum(), 1.0)

            mask_predicted = prediction["action_mask"][:, index]
            mask_actual = target_mask[:, index]
            positive = local_weight[..., None] * mask_actual.astype(jnp.float32)
            negative = local_weight[..., None] * (~mask_actual).astype(jnp.float32)
            prefix = f"ctde/self_fed_h{step}"
            support_metrics = availability_metrics(
                prediction["mask_logits"][:, index],
                mask_actual,
                mask_predicted,
                prediction["predicted_alive"][:, index],
                target_alive[:, index],
                local_weight,
            )
            metrics.update(
                {f"{prefix}/{key}": value for key, value in support_metrics.items()}
            )
            metrics.update(
                {
                    f"{prefix}/valid_count": weight.sum(),
                    f"{prefix}/local_valid_count": local_weight.sum(),
                    f"{prefix}/reward_rmse": jnp.sqrt(
                        mean(jnp.square(reward_error[:, index]))
                    ),
                    f"{prefix}/reward_bias": mean(reward_error[:, index]),
                    f"{prefix}/continuation_brier": mean(
                        jnp.square(
                            prediction["continuation"][:, index]
                            - target_continuation[:, index]
                        )
                    ),
                    f"{prefix}/alive_brier": mean(
                        jnp.square(
                            prediction["alive_probability"][:, index]
                            - target_alive[:, index]
                        )
                    ),
                    f"{prefix}/action_mask_false_positive": mean(
                        mask_predicted, negative
                    ),
                    f"{prefix}/action_mask_false_negative": mean(
                        ~mask_predicted, positive
                    ),
                    f"{prefix}/embedding_cosine": mean(cosine[:, index], local_weight),
                    f"{prefix}/posterior_kl": mean(
                        posterior_kl[:, index], local_weight
                    ),
                    f"{prefix}/mixed_posterior_kl": mean(
                        executable_posterior_kl[:, index], local_weight
                    ),
                }
            )
        return jax.lax.stop_gradient(metrics)
