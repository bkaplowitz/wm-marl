"""Training summaries and open-loop latent prediction reports."""

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from .common import i32
from .representation import embedding_prediction_loss


class ReportingMixin:
    """Report loss summaries and a small open-loop visual prediction grid."""

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

        if self.config.report_gradnorms:
            for key in self.scales:
                try:

                    def lossfn(data, carry):
                        del data
                        return self.loss(carry, obs, prevact, training=False)[1][2][
                            "losses"
                        ][key].mean()

                    grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
                    metrics[f"gradnorm/{key}"] = optax.global_norm(grad)
                except KeyError:
                    print(f"Skipping gradnorm summary for missing loss: {key}")

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
        if self.dec is not None:
            observed = firsthalf(outs["repfeat"])
            dec_carry, _, observed_reconstructions = self.dec(
                dec_carry,
                observed,
                firsthalf(obs["is_first"]),
                training=False,
            )
            _, _, imagined_reconstructions = self.dec(
                dec_carry,
                imagined,
                jnp.zeros_like(secondhalf(obs["is_first"])),
                training=False,
            )
            self._reconstruction_report(
                metrics,
                obs,
                observed_reconstructions,
                imagined_reconstructions,
                rows,
                length,
            )
        if self.dynamics_jepa and "target_tokens" in outs:
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
                    distance=self.embedding_loss,
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

    def _reconstruction_report(
        self,
        metrics,
        obs,
        observed_reconstructions,
        imagined_reconstructions,
        rows,
        length,
    ):
        for key in self.dec.imgkeys:
            true = obs[key][:rows]
            assert true.dtype == jnp.uint8
            predicted = jnp.concatenate(
                [
                    observed_reconstructions[key].pred(),
                    imagined_reconstructions[key].pred(),
                ],
                1,
            )
            predicted = jnp.clip(predicted * 255, 0, 255).astype(jnp.uint8)
            error = ((i32(predicted) - i32(true) + 255) / 2).astype(np.uint8)
            video = jnp.concatenate([true, predicted, error], 2)
            video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
            mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
            border = jnp.full((length, 3), jnp.array([0, 255, 0]), jnp.uint8)
            border = border.at[length // 2 :].set(jnp.array([255, 0, 0], jnp.uint8))
            video = jnp.where(mask, video, border[None, :, None, None, :])
            video = jnp.concatenate([video, 0 * video[:, :10]], 1)
            _, time, height, width, channels = video.shape
            grid = video.transpose((1, 2, 0, 3, 4)).reshape(
                (time, height, rows * width, channels)
            )
            metrics[f"openloop/{key}"] = grid
