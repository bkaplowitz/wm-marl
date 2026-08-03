"""First-party agent-axis-native DreaMARL learner."""

import re

import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import rssm, transformer_rssm
from .axes import (
    GLOBAL_OBSERVATION_KEYS,
    GLOBAL_REPLAY_KEYS,
    broadcast_global_batch,
    broadcast_global_sequence,
    fold_agent_batch,
    fold_agent_sequence,
    fold_tree_batch,
    restore_folded_start_order,
    select_joint_starts,
    unfold_agent_sequence,
    unfold_tree_batch,
)

f32 = jnp.float32
i32 = jnp.int32


def sg(xs, skip=False):
    return xs if skip else jax.lax.stop_gradient(xs)


def sample(xs):
    return jax.tree.map(lambda x: x.sample(nj.seed()), xs)


def deterministic(xs):
    return jax.tree.map(lambda x: x.pred(), xs)


def prefix(xs, value):
    return {f"{value}/{key}": item for key, item in xs.items()}


def concat(xs, axis):
    return jax.tree.map(lambda *x: jnp.concatenate(x, axis), *xs)


def isimage(space):
    return space.dtype == np.uint8 and len(space.shape) == 3


class Agent(embodied.jax.Agent):
    banner = [
        r"---  ____                 __  ___    _    ____  _     ---",
        r"--- |  _ \ _ __ ___  __ _|  \/  |  / \  |  _ \| |    ---",
        r"--- | | | | '__/ _ \/ _` | |\/| | / _ \ | |_) | |    ---",
        r"--- | |_| | | |  __/ (_| | |  | |/ ___ \|  _ <| |___ ---",
        r"--- |____/|_|  \___|\__,_|_|  |_/_/   \_\_| \_\_____|---",
    ]

    def __init__(self, obs_space, act_space, config):
        self.num_agents = int(config.num_agents)
        if self.num_agents < 1:
            raise ValueError("num_agents must be positive")
        self.joint_obs_space = obs_space
        self.joint_act_space = act_space
        obs_space = {
            key: (
                space
                if key in GLOBAL_OBSERVATION_KEYS
                else _remove_agent_axis(key, space, self.num_agents)
            )
            for key, space in obs_space.items()
        }
        act_space = {
            key: _remove_agent_axis(key, space, self.num_agents)
            for key, space in act_space.items()
        }
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config

        exclude = ("is_first", "is_last", "is_terminal", "reward")
        enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
        dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
        self.enc = {
            "simple": rssm.Encoder,
        }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name="enc")

        self.enc_output_dim = self.enc.calculate_encoder_output_dim(
            obs_space, config.enc
        )

        if hasattr(config, "slowenc") and config.slowenc.enable:
            self.slowenc = embodied.jax.SlowModel(
                self.enc, source=self.enc, **config.slowenc, name="slowenc"
            )
        else:
            self.slowenc = None

        dynamics_type = config.dyn.typ
        dynamics_config = config.dyn[dynamics_type]
        if dynamics_type == "jepa_transformer":
            dynamics_config = dynamics_config.update(
                num_agents=self.num_agents,
                memory_seed=int(config.seed) + 10_000,
                joint_seed=int(config.seed) + 20_000,
            )
        self.dyn = {
            "rssm": rssm.RSSM,
            "jepa_transformer": transformer_rssm.TransformerRSSM,
        }[dynamics_type](act_space, self.enc_output_dim, **dynamics_config, name="dyn")

        def feat2tensor(feat):
            if hasattr(self.dyn, "control_feature"):
                return self.dyn.control_feature(feat)
            return jnp.concatenate(
                [
                    nn.cast(feat["deter"]),
                    nn.cast(feat["stoch"].reshape((*feat["stoch"].shape[:-2], -1))),
                ],
                -1,
            )

        self.feat2tensor = feat2tensor

        self.world_feat2tensor = self.feat2tensor
        self.dec = {
            "simple": rssm.Decoder,
        }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name="dec")

        scalar = elements.Space(np.float32, ())
        binary = elements.Space(bool, (), 0, 2)
        self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name="rew")
        self.con = embodied.jax.MLPHead(binary, **config.conhead, name="con")

        d1, d2 = config.policy_dist_disc, config.policy_dist_cont
        outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
        self.pol = embodied.jax.MLPHead(act_space, outs, **config.policy, name="pol")

        self.val = embodied.jax.MLPHead(scalar, **config.value, name="val")
        self.slowval = embodied.jax.SlowModel(
            embodied.jax.MLPHead(scalar, **config.value, name="slowval"),
            source=self.val,
            **config.slowvalue,
        )

        self.retnorm = embodied.jax.Normalize(**config.retnorm, name="retnorm")
        self.valnorm = embodied.jax.Normalize(**config.valnorm, name="valnorm")
        self.advnorm = embodied.jax.Normalize(**config.advnorm, name="advnorm")

        self.modules = [
            self.dyn,
            self.enc,
            self.dec,
            self.rew,
            self.con,
            self.pol,
            self.val,
        ]

        if hasattr(config, "enc_lr") or hasattr(config, "dyn_lr"):
            enc_lr = getattr(config, "enc_lr", config.opt.lr)
            dyn_lr = getattr(config, "dyn_lr", config.opt.lr)

            def label_fn(params):
                labels = {}
                for name in params.keys():
                    if name.startswith("enc/"):
                        labels[name] = "enc"
                    elif name.startswith("dyn/"):
                        labels[name] = "dyn"
                    else:
                        labels[name] = "other"
                return labels

            enc_opt_config = dict(config.opt)
            enc_opt_config["lr"] = enc_lr

            dyn_opt_config = dict(config.opt)
            dyn_opt_config["lr"] = dyn_lr
            dyn_opt_config["schedule"] = "const"

            optimizer = optax.multi_transform(
                {
                    "enc": self._make_opt(**enc_opt_config),
                    "dyn": self._make_opt(**dyn_opt_config),
                    "other": self._make_opt(**config.opt),
                },
                label_fn,
            )

        else:
            optimizer = self._make_opt(**config.opt)

        self.opt = embodied.jax.Optimizer(
            self.modules, optimizer, summary_depth=1, name="opt"
        )

        scales = self.config.loss_scales.copy()
        rec = scales.pop("rec")
        if scales.get("memory_dyn") == 0:
            scales.pop("memory_dyn")
        scales.update({k: rec for k in dec_space})
        self.scales = scales

    @property
    def policy_keys(self):
        return "^(enc|dyn|dec|pol|rew|val|con)/"

    @property
    def ext_space(self):
        spaces = {}
        spaces["consec"] = elements.Space(np.int32)
        spaces["stepid"] = elements.Space(np.uint8, 20)
        if self.config.replay_context:
            spaces.update(
                elements.tree.flatdict(
                    dict(
                        enc=self.enc.entry_space,
                        dyn=self.dyn.entry_space,
                        dec=self.dec.entry_space,
                    )
                )
            )
        return {
            key: (
                space
                if key in GLOBAL_REPLAY_KEYS
                else _add_agent_axis(space, self.num_agents)
            )
            for key, space in spaces.items()
        }

    def init_policy(self, batch_size):
        batch_size *= self.num_agents

        def zeros(value):
            return jnp.zeros((batch_size, *value.shape), value.dtype)

        carry = (
            self.enc.initial(batch_size),
            self.dyn.initial(batch_size),
            self.dec.initial(batch_size),
            jax.tree.map(zeros, self.act_space),
        )
        return unfold_tree_batch(carry, self.num_agents)

    def init_train(self, batch_size):
        return self.init_policy(batch_size)

    def init_report(self, batch_size):
        return self.init_policy(batch_size)

    def policy(self, carry, obs, mode="train"):
        flat_carry = fold_tree_batch(carry, self.num_agents)
        flat_obs = {
            key: (
                broadcast_global_batch(value, self.num_agents)
                if key in GLOBAL_OBSERVATION_KEYS
                else fold_agent_batch(value, self.num_agents)
            )
            for key, value in obs.items()
        }
        flat_carry, flat_actions, flat_outputs = self._policy_local(
            flat_carry, flat_obs, mode
        )
        return (
            unfold_tree_batch(flat_carry, self.num_agents),
            unfold_tree_batch(flat_actions, self.num_agents),
            unfold_tree_batch(flat_outputs, self.num_agents),
        )

    def _policy_local(self, carry, obs, mode="train"):
        (enc_carry, dyn_carry, dec_carry, prevact) = carry
        kw = dict(training=False, single=True)
        reset = obs["is_first"]
        enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
        dyn_carry, dyn_entry, feat, _ = self.dyn.observe(
            dyn_carry, tokens, prevact, reset, **kw
        )
        dec_entry = {}
        dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
        policy = self.pol(self.feat2tensor(feat), bdims=1)
        rew = self.rew(self.feat2tensor(feat), bdims=1).pred()
        val = self.val(self.feat2tensor(feat), bdims=1).pred()
        con = self.con(self.feat2tensor(feat), bdims=1).prob(1)
        if mode == "train":
            act = sample(policy)
        elif mode == "eval":
            act = deterministic(policy)
        else:
            raise ValueError(f"unknown policy mode: {mode}")
        out = {}
        out["finite"] = elements.tree.flatdict(
            jax.tree.map(
                lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
                dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act),
            )
        )
        carry = (enc_carry, dyn_carry, dec_carry, act)
        if self.config.replay_context:
            out.update(
                elements.tree.flatdict(
                    dict(enc=enc_entry, dyn=dyn_entry, dec=dec_entry)
                )
            )
        else:
            recons_pred = {}
            for key, recon in recons.items():
                if hasattr(recon, "pred"):
                    recons_pred[key] = recon.pred()
                else:
                    recons_pred[key] = recon
            out.update(
                elements.tree.flatdict(
                    dict(
                        enc=enc_entry,
                        dyn=dyn_entry,
                        dec=dec_entry,
                        rew=rew,
                        val=val,
                        con=con,
                        recons=recons_pred,
                    )
                )
            )
        return carry, act, out

    def train(self, carry, data):
        flat_carry = fold_tree_batch(carry, self.num_agents)
        flat_carry, outputs, metrics = self._train_local(
            flat_carry, self._fold_replay(data)
        )
        if "replay" in outputs:
            outputs = {
                **outputs,
                "replay": self._unfold_replay_updates(outputs["replay"]),
            }
        return unfold_tree_batch(flat_carry, self.num_agents), outputs, metrics

    def _train_local(self, carry, data):
        carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
        metrics, (carry, entries, outs, mets, replay_loss) = self.opt(
            self.loss, carry, obs, prevact, training=True, has_aux=True
        )
        metrics.update(mets)
        self.slowval.update()
        if self.slowenc is not None:
            self.slowenc.update()
        outs = {}
        if self.config.replay_context:
            updates = elements.tree.flatdict(
                dict(
                    stepid=stepid,
                    enc=entries[0],
                    dyn={key: entries[1][key] for key in self.dyn.entry_space},
                    dec=entries[2],
                )
            )
            B, T = obs["is_first"].shape
            assert all(x.shape[:2] == (B, T) for x in updates.values()), (
                (B, T),
                {k: v.shape for k, v in updates.items()},
            )
            outs["replay"] = updates
        carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
        return carry, outs, metrics

    def loss(self, carry, obs, prevact, training):
        enc_carry, dyn_carry, dec_carry = carry
        reset = obs["is_first"]
        B, T = reset.shape
        losses = {}
        metrics = {}

        # World model
        enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training)
        if self.slowenc is not None:
            _, _, slow_tokens = self.slowenc(enc_carry, obs, reset, training)
        dyn_carry, dyn_entries, los, repfeat, mets, _ = self.dyn.loss(
            dyn_carry,
            tokens,
            prevact,
            reset,
            training,
            slow_tokens=(slow_tokens if self.slowenc is not None else tokens),
        )

        losses.update(los)
        metrics.update(mets)
        dec_carry, dec_entries, recons = self.dec(
            dec_carry,
            jax.tree.map(lambda x: sg(x, skip=self.config.dec_grad), repfeat),
            reset,
            training,
        )
        inp = sg(self.world_feat2tensor(repfeat), skip=self.config.reward_grad)
        losses["rew"] = self.rew(inp, 2).loss(obs["reward"])
        con = f32(~obs["is_terminal"])
        if self.config.contdisc:
            con *= 1 - 1 / self.config.horizon
        losses["con"] = self.con(self.world_feat2tensor(repfeat), 2).loss(con)
        for key, recon in recons.items():
            space, value = self.obs_space[key], obs[key]
            assert value.dtype == space.dtype, (key, space, value.dtype)
            target = f32(value) / 255 if isimage(space) else value
            losses[key] = recon.loss(sg(target))

        B, T = reset.shape
        shapes = {k: v.shape for k, v in losses.items()}
        assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

        # Imagination
        K = min(self.config.imag_last or T, T)
        H = self.config.imag_length
        starts = self.dyn.starts(dyn_entries, dyn_carry, K)

        def policyfn(feat):
            return sample(self.pol(self.feat2tensor(feat), 1))

        _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
        first = jax.tree.map(
            lambda x: select_joint_starts(x, self.num_agents, K)[:, None],
            repfeat,
        )
        imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
        lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
        lastact = jax.tree.map(lambda x: x[:, None], lastact)
        imgact = concat([imgprevact, lastact], 1)
        assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
        assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
        inp = self.feat2tensor(imgfeat)
        world_inp = self.world_feat2tensor(imgfeat)
        los, imgloss_out, mets = imag_loss(
            imgact,
            self.rew(world_inp, 2).pred(),
            self.con(world_inp, 2).prob(1),
            self.pol(inp, 2),
            self.val(inp, 2),
            self.slowval(inp, 2),
            self.retnorm,
            self.valnorm,
            self.advnorm,
            update=training,
            contdisc=self.config.contdisc,
            horizon=self.config.horizon,
            **self.config.imag_loss,
        )
        losses.update(
            {
                key: restore_folded_start_order(value.mean(1), self.num_agents, K)
                for key, value in los.items()
            }
        )
        metrics.update(mets)

        # Replay
        if self.config.repval_loss:
            feat = sg(repfeat, skip=self.config.repval_grad)
            last, term, rew = [obs[k] for k in ("is_last", "is_terminal", "reward")]
            boot = restore_folded_start_order(
                imgloss_out["ret"][:, 0], self.num_agents, K
            )
            feat, last, term, rew, boot = jax.tree.map(
                lambda x: x[:, -K:], (feat, last, term, rew, boot)
            )
            inp = self.feat2tensor(feat)
            los, reploss_out, mets = repl_loss(
                last,
                term,
                rew,
                boot,
                self.val(inp, 2),
                self.slowval(inp, 2),
                self.valnorm,
                update=training,
                horizon=self.config.horizon,
                **self.config.repl_loss,
            )
            losses.update(los)
            metrics.update(prefix(mets, "reploss"))

        assert set(losses.keys()) == set(self.scales.keys()), (
            sorted(losses.keys()),
            sorted(self.scales.keys()),
        )
        metrics.update({f"loss/{k}": v.mean() for k, v in losses.items()})
        loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

        carry = (enc_carry, dyn_carry, dec_carry)
        entries = (enc_entries, dyn_entries, dec_entries)
        outs = {"tokens": tokens, "repfeat": repfeat, "losses": losses}
        return loss, (carry, entries, outs, metrics, losses["dyn_deter"])

    def report(self, carry, data):
        flat_carry = fold_tree_batch(carry, self.num_agents)
        flat_carry, metrics = self._report_local(flat_carry, self._fold_replay(data))
        return unfold_tree_batch(flat_carry, self.num_agents), metrics

    def _report_local(self, carry, data):
        if not self.config.report:
            return carry, {}

        carry, obs, prevact, _ = self._apply_replay_context(carry, data)
        (enc_carry, dyn_carry, dec_carry) = carry
        B, T = obs["is_first"].shape
        complete_groups = B // self.num_agents
        report_groups = min(complete_groups, max(1, 6 // self.num_agents))
        RB = report_groups * self.num_agents
        metrics = {}

        # Train metrics
        _, (new_carry, entries, outs, mets, _) = self.loss(
            carry, obs, prevact, training=False
        )
        metrics.update(mets)

        # Grad norms
        if self.config.report_gradnorms:
            for key in self.scales:
                try:

                    def lossfn(data, carry):
                        return self.loss(carry, obs, prevact, training=False)[1][2][
                            "losses"
                        ][key].mean()

                    grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
                    metrics[f"gradnorm/{key}"] = optax.global_norm(grad)
                except KeyError:
                    print(f"Skipping gradnorm summary for missing loss: {key}")

        # Open loop
        def firsthalf(xs):
            return jax.tree.map(lambda x: x[:RB, : T // 2], xs)

        def secondhalf(xs):
            return jax.tree.map(lambda x: x[:RB, T // 2 :], xs)

        dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
        dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
        dyn_carry, _, obsfeat, _ = self.dyn.observe(
            dyn_carry,
            firsthalf(outs["tokens"]),
            firsthalf(prevact),
            firsthalf(obs["is_first"]),
            training=False,
        )
        _, imgfeat, _ = self.dyn.imagine(
            dyn_carry, secondhalf(prevact), length=T - T // 2, training=False
        )
        metrics.update(
            self._recursive_world_model_metrics(
                imgfeat,
                secondhalf(outs["tokens"]),
                secondhalf(obs["reward"]),
                secondhalf(obs["is_terminal"]),
            )
        )
        dec_carry, _, obsrecons = self.dec(
            dec_carry, obsfeat, firsthalf(obs["is_first"]), training=False
        )
        dec_carry, _, imgrecons = self.dec(
            dec_carry,
            imgfeat,
            jnp.zeros_like(secondhalf(obs["is_first"])),
            training=False,
        )
        metrics.update(
            self._recursive_decoder_metrics(
                imgrecons,
                {key: secondhalf(obs[key]) for key in self.dec.imgkeys},
            )
        )

        # Video preds
        for key in self.dec.imgkeys:
            if not getattr(self.config, "report_video", True):
                break
            assert obs[key].dtype == jnp.uint8
            true = obs[key][:RB]
            pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
            pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
            error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
            video = jnp.concatenate([true, pred, error], 2)

            video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
            mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
            border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
            border = border.at[T // 2 :].set(jnp.array([255, 0, 0], jnp.uint8))
            video = jnp.where(mask, video, border[None, :, None, None, :])
            video = jnp.concatenate([video, 0 * video[:, :10]], 1)

            B, T, H, W, C = video.shape
            grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
            metrics[f"openloop/{key}"] = grid

        carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
        return carry, metrics

    def _recursive_world_model_metrics(
        self, imagined, target_tokens, target_rewards, target_terminals
    ):
        predicted_tokens = self.dyn.predictor(imagined["deter"])
        world_state = self.world_feat2tensor(imagined)
        predicted_rewards = self.rew(world_state, 2).pred()
        predicted_continuation = self.con(world_state, 2).prob(1)
        target_continuation = f32(~target_terminals)
        metrics = {}
        length = predicted_tokens.shape[1]
        for horizon in (1, 2, 4, 8):
            if horizon > length:
                continue
            index = horizon - 1
            latent_error = optax.losses.cosine_distance(
                f32(predicted_tokens[:, index]),
                sg(f32(target_tokens[:, index])),
                axis=-1,
                epsilon=1e-8,
            )
            reward_error = jnp.abs(
                f32(predicted_rewards[:, index]) - f32(target_rewards[:, index])
            )
            continuation_error = jnp.square(
                f32(predicted_continuation[:, index])
                - f32(target_continuation[:, index])
            )
            metrics[f"world_model/h{horizon}_latent_cosine"] = latent_error.mean()
            metrics[f"world_model/h{horizon}_reward_mae"] = reward_error.mean()
            metrics[f"world_model/h{horizon}_continuation_brier"] = (
                continuation_error.mean()
            )
            for name in ("deter", "memory"):
                key = f"interaction_{name}"
                if key not in imagined:
                    continue
                residual = f32(imagined[key][:, index])
                axes = tuple(range(1, residual.ndim))
                residual_rms = jnp.sqrt(jnp.mean(residual**2, axis=axes))
                metrics[f"world_model/h{horizon}_interaction_{name}_rms"] = (
                    residual_rms.mean()
                )
                metrics[f"world_model/h{horizon}_interaction_{name}_latent_corr"] = (
                    _pearson_correlation(residual_rms, latent_error)
                )
                metrics[f"world_model/h{horizon}_interaction_{name}_reward_corr"] = (
                    _pearson_correlation(residual_rms, reward_error)
                )
        return metrics

    def _recursive_decoder_metrics(self, predictions, targets):
        metrics = {}
        for key in self.dec.imgkeys:
            prediction = f32(predictions[key].pred())
            target = f32(targets[key]) / 255.0
            length = prediction.shape[1]
            for horizon in (1, 2, 4, 8):
                if horizon > length:
                    continue
                index = horizon - 1
                error = jnp.abs(prediction[:, index] - target[:, index])
                metrics[f"world_model/h{horizon}_{key}_mae"] = error.mean()
        return metrics

    def _fold_replay(self, data):
        per_agent = set(self.joint_obs_space) - GLOBAL_OBSERVATION_KEYS
        per_agent.update(self.joint_act_space)
        per_agent.update(
            set(data)
            - set(self.joint_obs_space)
            - set(self.joint_act_space)
            - GLOBAL_REPLAY_KEYS
        )
        return {
            key: (
                fold_agent_sequence(value, self.num_agents)
                if key in per_agent
                else broadcast_global_sequence(value, self.num_agents)
            )
            for key, value in data.items()
        }

    def _unfold_replay_updates(self, updates):
        result = {}
        for key, value in updates.items():
            restored = unfold_agent_sequence(value, self.num_agents)
            # Step IDs address joint replay rows. They are broadcast only so each
            # shared trajectory can execute the unchanged replay-context logic.
            result[key] = restored[:, :, 0] if key == "stepid" else restored
        return result

    def _apply_replay_context(self, carry, data):
        (enc_carry, dyn_carry, dec_carry, prevact) = carry
        carry = (enc_carry, dyn_carry, dec_carry)
        stepid = data["stepid"]
        obs = {k: data[k] for k in self.obs_space}

        def prepend(x, y):
            return jnp.concatenate([x[:, None], y[:, :-1]], 1)

        prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
        if not self.config.replay_context:
            return carry, obs, prevact, stepid

        K = self.config.replay_context
        nested = elements.tree.nestdict(data)
        entries = [nested.get(k, {}) for k in ("enc", "dyn", "dec")]

        def lhs(xs):
            return jax.tree.map(lambda x: x[:, :K], xs)

        def rhs(xs):
            return jax.tree.map(lambda x: x[:, K:], xs)

        rep_carry = (
            self.enc.truncate(lhs(entries[0]), enc_carry),
            self.dyn.truncate(lhs(entries[1]), dyn_carry),
            self.dec.truncate(lhs(entries[2]), dec_carry),
        )
        rep_obs = {k: rhs(data[k]) for k in self.obs_space}
        rep_prevact = {k: data[k][:, K - 1 : -1] for k in self.act_space}
        rep_stepid = rhs(stepid)

        first_chunk = data["consec"][:, 0] == 0
        carry, obs, prevact, stepid = jax.tree.map(
            lambda normal, replay: nn.where(first_chunk, replay, normal),
            (carry, rhs(obs), rhs(prevact), rhs(stepid)),
            (rep_carry, rep_obs, rep_prevact, rep_stepid),
        )
        return carry, obs, prevact, stepid

    def _make_opt(
        self,
        lr: float = 4e-5,
        agc: float = 0.3,
        eps: float = 1e-20,
        beta1: float = 0.9,
        beta2: float = 0.999,
        momentum: bool = True,
        nesterov: bool = False,
        wd: float = 0.0,
        wdregex: str = r"/kernel$",
        schedule: str = "const",
        warmup: int = 1000,
        anneal: int = 0,
    ):
        chain = []
        chain.append(embodied.jax.opt.clip_by_agc(agc))
        chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
        chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
        if wd:
            assert not wdregex[0].isnumeric(), wdregex
            pattern = re.compile(wdregex)

            def wdmask(params):
                return {key: bool(pattern.search(key)) for key in params}

            chain.append(optax.add_decayed_weights(wd, wdmask))
        assert anneal > 0 or schedule == "const"
        if schedule == "const":
            sched = optax.constant_schedule(lr)
        elif schedule == "linear":
            sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
        elif schedule == "cosine":
            sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
        else:
            raise NotImplementedError(schedule)
        if warmup:
            ramp = optax.linear_schedule(0.0, lr, warmup)
            sched = optax.join_schedules([ramp, sched], [warmup])
        chain.append(optax.scale_by_learning_rate(sched))
        return optax.chain(*chain)


def _remove_agent_axis(name, space, num_agents):
    if not space.shape or space.shape[0] != num_agents:
        raise ValueError(
            f"{name!r} must expose leading agent axis {num_agents}, "
            f"got shape {space.shape}"
        )
    return elements.Space(
        space.dtype,
        space.shape[1:],
        _remove_bound_axis(space.low),
        _remove_bound_axis(space.high),
    )


def _add_agent_axis(space, num_agents):
    shape = (num_agents, *space.shape)
    low = None if space.low is None else np.broadcast_to(space.low, shape)
    high = None if space.high is None else np.broadcast_to(space.high, shape)
    return elements.Space(space.dtype, shape, low, high)


def _remove_bound_axis(bound):
    if bound is None:
        return None
    return np.asarray(bound)[0]


def _pearson_correlation(left, right):
    left = f32(left).reshape((-1,))
    right = f32(right).reshape((-1,))
    left = left - left.mean()
    right = right - right.mean()
    denominator = jnp.sqrt(jnp.sum(left**2) * jnp.sum(right**2))
    return jnp.where(denominator > 1e-8, jnp.sum(left * right) / denominator, 0.0)


def imag_loss(
    act,
    rew,
    con,
    policy,
    value,
    slowvalue,
    retnorm,
    valnorm,
    advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
    losses = {}
    metrics = {}

    voffset, vscale = valnorm.stats()
    val = value.pred() * vscale + voffset
    slowval = slowvalue.pred() * vscale + voffset
    tarval = slowval if slowtar else val
    disc = 1 if contdisc else 1 - 1 / horizon
    weight = jnp.cumprod(disc * con, 1) / disc
    last = jnp.zeros_like(con)
    term = 1 - con
    ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

    roffset, rscale = retnorm(ret, update)
    adv = (ret - tarval[:, :-1]) / rscale
    aoffset, ascale = advnorm(adv, update)
    adv_normed = (adv - aoffset) / ascale
    logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
    ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
    policy_loss = sg(weight[:, :-1]) * -(
        logpi * sg(adv_normed) + actent * sum(ents.values())
    )
    losses["policy"] = policy_loss

    voffset, vscale = valnorm(ret, update)
    tar_normed = (ret - voffset) / vscale
    tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
    losses["value"] = (
        sg(weight[:, :-1])
        * (value.loss(sg(tar_padded)) + slowreg * value.loss(sg(slowvalue.pred())))[
            :, :-1
        ]
    )

    ret_normed = (ret - roffset) / rscale
    metrics["adv"] = adv.mean()
    metrics["adv_std"] = adv.std()
    metrics["adv_mag"] = jnp.abs(adv).mean()
    metrics["rew"] = rew.mean()
    metrics["con"] = con.mean()
    metrics["ret"] = ret_normed.mean()
    metrics["val"] = val.mean()
    metrics["tar"] = tar_normed.mean()
    metrics["weight"] = weight.mean()
    metrics["slowval"] = slowval.mean()
    metrics["ret_min"] = ret_normed.min()
    metrics["ret_max"] = ret_normed.max()
    metrics["ret_rate"] = (jnp.abs(ret_normed) >= 1.0).mean()
    for k in act:
        metrics[f"ent/{k}"] = ents[k].mean()
        if hasattr(policy[k], "minent"):
            lo, hi = policy[k].minent, policy[k].maxent
            metrics[f"rand/{k}"] = (ents[k].mean() - lo) / (hi - lo)

    outs = {}
    outs["ret"] = ret
    return losses, outs, metrics


def repl_loss(
    last,
    term,
    rew,
    boot,
    value,
    slowvalue,
    valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
    losses = {}

    voffset, vscale = valnorm.stats()
    val = value.pred() * vscale + voffset
    slowval = slowvalue.pred() * vscale + voffset
    tarval = slowval if slowtar else val
    disc = 1 - 1 / horizon
    weight = f32(~last)
    ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

    voffset, vscale = valnorm(ret, update)
    ret_normed = (ret - voffset) / vscale
    ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
    losses["repval"] = (
        weight[:, :-1]
        * (value.loss(sg(ret_padded)) + slowreg * value.loss(sg(slowvalue.pred())))[
            :, :-1
        ]
    )

    outs = {}
    outs["ret"] = ret
    metrics = {}

    return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
    chex.assert_equal_shape((last, term, rew, val, boot))
    rets = [boot[:, -1]]
    live = (1 - f32(term))[:, 1:] * disc
    cont = (1 - f32(last))[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return jnp.stack(list(reversed(rets))[:-1], 1)
