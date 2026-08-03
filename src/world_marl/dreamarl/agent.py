"""MARL-first DreaMARL agent with local execution and joint imagination."""

from __future__ import annotations

import re

import chex
import elements
import embodied
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import perception
from .axes import (
    GLOBAL_OBSERVATION_KEYS,
    broadcast_global_batch,
    broadcast_global_sequence,
    fold_agent_batch,
    fold_agent_sequence,
    fold_tree_batch,
    unfold_agent_sequence,
    unfold_tree_batch,
)
from .joint_model import JointWorldModel
from .local_belief import LocalBelief


f32 = jnp.float32


def sg(value, skip=False):
    return value if skip else jax.lax.stop_gradient(value)


def sample(distributions):
    return {key: value.sample(seed=nj.seed()) for key, value in distributions.items()}


def deterministic(distributions):
    return {key: value.pred() for key, value in distributions.items()}


class Agent(embodied.jax.Agent):
    """Joint-world model with strictly decentralized shared actors."""

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
        self.config = config
        self.joint_obs_space = obs_space
        self.joint_act_space = act_space
        self.obs_space = {
            key: (
                space
                if key in GLOBAL_OBSERVATION_KEYS
                else _remove_agent_axis(key, space, self.num_agents)
            )
            for key, space in obs_space.items()
        }
        self.act_space = {
            key: _remove_agent_axis(key, space, self.num_agents)
            for key, space in act_space.items()
        }

        excluded = {"is_first", "is_last", "is_terminal", "reward"}
        enc_space = {key: value for key, value in self.obs_space.items() if key not in excluded}
        self.enc = perception.Encoder(enc_space, **config.enc.simple, name="enc")
        self.embedding_dim = self.enc.calculate_output_dim()
        self.belief = LocalBelief(
            self.act_space,
            self.embedding_dim,
            **config.local_belief,
            name="belief",
        )
        self.world = JointWorldModel(
            self.act_space,
            self.embedding_dim,
            int(config.local_belief.units),
            self.num_agents,
            **config.joint,
            name="world",
        )
        scalar = elements.Space(np.float32, ())
        binary = elements.Space(bool, (), 0, 2)
        self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name="rew")
        self.con = embodied.jax.MLPHead(binary, **config.conhead, name="con")
        distributions = {
            key: config.policy_dist_disc if value.discrete else config.policy_dist_cont
            for key, value in self.act_space.items()
        }
        self.pol = embodied.jax.MLPHead(
            self.act_space, distributions, **config.policy, name="pol"
        )
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
            self.enc,
            self.belief,
            self.world,
            self.rew,
            self.con,
            self.pol,
            self.val,
        ]
        optimizer = self._optimizer(config)
        self.opt = embodied.jax.Optimizer(
            self.modules, optimizer, summary_depth=1, name="opt"
        )

        self.scales = dict(config.loss_scales)

    @property
    def policy_keys(self):
        return "^(enc|belief|pol)/"

    @property
    def ext_space(self):
        return {
            "consec": elements.Space(np.int32),
            "stepid": elements.Space(np.uint8, 20),
        }

    def init_policy(self, batch_size):
        local_batch = batch_size * self.num_agents

        def zeros(space):
            return jnp.zeros((local_batch, *space.shape), space.dtype)

        carry = (
            self.enc.initial(local_batch),
            self.belief.initial(local_batch),
            {key: zeros(space) for key, space in self.act_space.items()},
        )
        return unfold_tree_batch(carry, self.num_agents)

    def init_train(self, batch_size):
        return jnp.zeros((batch_size,), jnp.int32)

    def init_report(self, batch_size):
        return jnp.zeros((batch_size,), jnp.int32)

    def policy(self, carry, obs, mode="train"):
        enc_carry, belief_carry, previous_action = fold_tree_batch(
            carry, self.num_agents
        )
        local_obs = {
            key: (
                broadcast_global_batch(value, self.num_agents)
                if key in GLOBAL_OBSERVATION_KEYS
                else fold_agent_batch(value, self.num_agents)
            )
            for key, value in obs.items()
        }
        reset = local_obs["is_first"]
        enc_carry, _, embedding = self.enc(
            enc_carry, local_obs, reset, training=False, single=True
        )
        belief_carry, belief, _ = self.belief.observe(
            belief_carry,
            embedding,
            previous_action,
            reset,
            training=False,
            single=True,
        )
        policy = self.pol(belief, bdims=1)
        if mode == "train":
            action = sample(policy)
        elif mode == "eval":
            action = deterministic(policy)
        else:
            raise ValueError(f"unknown policy mode: {mode}")
        output = {
            "finite": elements.tree.flatdict(
                jax.tree.map(
                    lambda value: jnp.isfinite(value).all(range(1, value.ndim)),
                    {"embedding": embedding, "belief": belief, "action": action},
                )
            )
        }
        carry = (enc_carry, belief_carry, action)
        return (
            unfold_tree_batch(carry, self.num_agents),
            unfold_tree_batch(action, self.num_agents),
            unfold_tree_batch(output, self.num_agents),
        )

    def train(self, carry, data):
        metrics, (_, model_metrics) = self.opt(
            self.loss, data, training=True, has_aux=True
        )
        metrics.update(model_metrics)
        self.slowval.update()
        return carry, {}, metrics

    def loss(self, data, training):
        artifacts = self._model_pass(data, training)
        prefix = int(self.config.replay_context)
        target = slice(prefix, None)
        losses = {
            key: value[:, target]
            for key, value in artifacts["world_losses"].items()
        }
        metrics = dict(artifacts["metrics"])

        agent_feature = self.world.agent_feature(artifacts["world_features"])
        team_feature = self.world.team_feature(artifacts["world_features"])
        reward_output = self.rew(agent_feature, bdims=3)
        continuation_output = self.con(team_feature, bdims=2)
        losses["rew"] = reward_output.loss(data["reward"])[:, target]
        continuation_target = f32(~data["is_terminal"])
        if self.config.contdisc:
            continuation_target *= 1 - 1 / self.config.horizon
        losses["con"] = continuation_output.loss(continuation_target)[:, target]
        reward_prediction = reward_output.pred()
        metrics["world_model/reward_vector_mae"] = jnp.abs(
            reward_prediction[:, target] - data["reward"][:, target]
        ).mean()

        sliced_features = {
            key: value[:, target]
            for key, value in artifacts["world_features"].items()
        }
        sliced_actions = {
            key: value[:, target] for key, value in self._joint_actions(data).items()
        }
        sliced_targets = artifacts["target_embeddings"][:, target]
        sliced_resets = data["is_first"][:, target]
        overshoot, overshoot_metrics = self.world.overshoot_loss(
            sliced_features,
            sliced_actions,
            sliced_targets,
            sliced_resets,
            training=training,
        )
        losses["overshoot"] = overshoot
        metrics.update(overshoot_metrics)

        imagination_losses, imagination_metrics, imagination_output = self._control_loss(
            artifacts, data, target, training
        )
        losses.update(imagination_losses)
        metrics.update(imagination_metrics)

        replay_losses, replay_metrics = self._replay_value_loss(
            sliced_features,
            data,
            target,
            imagination_output,
            training,
        )
        losses.update(replay_losses)
        metrics.update(replay_metrics)

        missing = set(self.scales) - set(losses)
        extra = set(losses) - set(self.scales)
        if missing or extra:
            raise ValueError(
                f"loss scale mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        means = {key: f32(value).mean() for key, value in losses.items()}
        metrics.update({f"loss/{key}": value for key, value in means.items()})
        total = sum(means[key] * self.scales[key] for key in means)
        return f32(total), ((), metrics)

    def report(self, carry, data):
        _, (_, metrics) = self.loss(data, training=False)
        return carry, metrics

    def _model_pass(self, data, training):
        reset = data["is_first"]
        batch, length = reset.shape
        if length <= self.config.replay_context:
            raise ValueError(
                f"sequence length {length} must exceed replay context "
                f"{self.config.replay_context}"
            )
        local_obs = self._local_observations(data)
        local_reset = broadcast_global_sequence(reset, self.num_agents)
        local_batch = batch * self.num_agents
        _, _, embeddings = self.enc(
            self.enc.initial(local_batch),
            local_obs,
            local_reset,
            training,
        )
        target_embeddings = sg(embeddings)
        previous_actions = self._previous_actions(data)
        flat_previous_actions = {
            key: fold_agent_sequence(value, self.num_agents)
            for key, value in previous_actions.items()
        }
        _, beliefs, belief_entries = self.belief.observe(
            self.belief.initial(local_batch),
            embeddings,
            flat_previous_actions,
            local_reset,
            training,
        )
        joint_embeddings = unfold_agent_sequence(embeddings, self.num_agents)
        joint_targets = unfold_agent_sequence(target_embeddings, self.num_agents)
        joint_beliefs = unfold_agent_sequence(beliefs, self.num_agents)
        _, world_features, world_losses, metrics = self.world.loss(
            self.world.initial(batch),
            joint_embeddings,
            joint_beliefs,
            previous_actions,
            reset,
            training,
            target_embeddings=joint_targets,
        )
        joint_belief_entries = jax.tree.map(
            lambda value: unfold_agent_sequence(value, self.num_agents),
            belief_entries,
        )
        return {
            "world_features": world_features,
            "world_losses": world_losses,
            "beliefs": joint_beliefs,
            "belief_entries": joint_belief_entries,
            "target_embeddings": joint_targets,
            "metrics": metrics,
        }

    def _control_loss(self, artifacts, data, target, training):
        features = {
            key: value[:, target] for key, value in artifacts["world_features"].items()
        }
        belief_entries = {
            key: value[:, target] for key, value in artifacts["belief_entries"].items()
        }
        batch, length = features["global"].shape[:2]
        starts = min(self.config.imag_last or length, length)
        world_start = {
            key: features[key][:, -starts:].reshape(
                (batch * starts, *features[key].shape[2:])
            )
            for key in ("global", "deter", "stoch", "logit")
        }
        belief_start = {
            key: value[:, -starts:].reshape(
                (batch * starts * self.num_agents, *value.shape[3:])
            )
            for key, value in belief_entries.items()
        }
        world_sequence, belief_sequence, action_sequence = self._imagine(
            world_start,
            belief_start,
            self.config.imag_length,
            training,
        )
        ac_world = jax.tree.map(
            lambda value: sg(value, skip=self.config.ac_grads), world_sequence
        )
        ac_belief = sg(belief_sequence, skip=self.config.ac_grads)
        team_feature = self.world.team_feature(ac_world)
        agent_feature = self.world.agent_feature(ac_world)
        del agent_feature
        reward_vector = self.rew(self.world.agent_feature(ac_world), 3).pred()
        team_reward = reward_vector.mean(-1)
        continuation = self.con(team_feature, 2).prob(1)
        policy = self.pol(ac_belief, 3)
        value = self.val(team_feature, 2)
        slowvalue = self.slowval(team_feature, 2)
        losses, output, metrics = joint_imag_loss(
            action_sequence,
            team_reward,
            continuation,
            policy,
            value,
            slowvalue,
            self.retnorm,
            self.valnorm,
            self.advnorm,
            update=training,
            contdisc=self.config.contdisc,
            horizon=self.config.horizon,
            **self.config.imag_loss,
        )
        output["starts"] = starts
        output["batch"] = batch
        return losses, metrics, output

    def _imagine(self, world_start, belief_start, length, training):
        environments = world_start["global"].shape[0]

        def step(carry, _):
            world_state, belief_state = carry
            belief = belief_state["belief"].reshape(
                (environments, self.num_agents, -1)
            )
            policy = self.pol(belief, bdims=2)
            action = sample(policy)
            reset = jnp.zeros((environments,), bool)
            next_world = self.world.imagine_step(
                world_state, action, reset, training
            )
            predicted_embedding = self.world.predict_embedding(next_world)
            flat_embedding = predicted_embedding.reshape(
                (environments * self.num_agents, self.embedding_dim)
            )
            flat_action = {
                key: value.reshape(
                    (environments * self.num_agents, *value.shape[2:])
                )
                for key, value in action.items()
            }
            local_reset = jnp.zeros((environments * self.num_agents,), bool)
            next_belief_state, next_belief, _ = self.belief.observe(
                belief_state,
                flat_embedding,
                flat_action,
                local_reset,
                training,
                single=True,
            )
            output_world = {
                key: next_world[key] for key in ("global", "deter", "stoch", "logit")
            }
            output_belief = next_belief.reshape(
                (environments, self.num_agents, -1)
            )
            return (next_world, next_belief_state), (
                output_world,
                output_belief,
                action,
            )

        _, (imagined_world, imagined_belief, actions) = nj.scan(
            step,
            (world_start, belief_start),
            (),
            length,
            axis=1,
        )
        start_world = {
            key: value[:, None]
            for key, value in world_start.items()
            if key in ("global", "deter", "stoch", "logit")
        }
        world_sequence = {
            key: jnp.concatenate([start_world[key], imagined_world[key]], 1)
            for key in start_world
        }
        start_belief = belief_start["belief"].reshape(
            (environments, self.num_agents, -1)
        )
        belief_sequence = jnp.concatenate(
            [start_belief[:, None], imagined_belief], 1
        )
        last_policy = self.pol(belief_sequence[:, -1], bdims=2)
        last_action = {key: value[:, None] for key, value in sample(last_policy).items()}
        action_sequence = {
            key: jnp.concatenate([value, last_action[key]], 1)
            for key, value in actions.items()
        }
        return world_sequence, belief_sequence, action_sequence

    def _replay_value_loss(
        self, features, data, target, imagination_output, training
    ):
        starts = imagination_output["starts"]
        batch = imagination_output["batch"]
        boot = imagination_output["ret"][:, 0].reshape((batch, starts))
        replay_features = {
            key: sg(value[:, -starts:], skip=self.config.repval_grad)
            for key, value in features.items()
        }
        team_feature = self.world.team_feature(replay_features)
        team_reward = data["reward"][:, target].mean(-1)[:, -starts:]
        last = data["is_last"][:, target][:, -starts:]
        terminal = data["is_terminal"][:, target][:, -starts:]
        losses, _, metrics = replay_value_loss(
            last,
            terminal,
            team_reward,
            boot,
            self.val(team_feature, 2),
            self.slowval(team_feature, 2),
            self.valnorm,
            update=training,
            horizon=self.config.horizon,
            **self.config.repl_loss,
        )
        return losses, {f"replay_value/{key}": value for key, value in metrics.items()}

    def _local_observations(self, data):
        result = {}
        for key in self.obs_space:
            if key in GLOBAL_OBSERVATION_KEYS:
                result[key] = broadcast_global_sequence(data[key], self.num_agents)
            else:
                result[key] = fold_agent_sequence(data[key], self.num_agents)
        return result

    def _joint_actions(self, data):
        return {key: data[key] for key in self.joint_act_space}

    def _previous_actions(self, data):
        result = {}
        for key, value in self._joint_actions(data).items():
            zeros = jnp.zeros_like(value[:, :1])
            result[key] = jnp.concatenate([zeros, value[:, :-1]], 1)
        return result

    def _optimizer(self, config):
        if not (
            hasattr(config, "enc_lr")
            or hasattr(config, "world_lr")
            or hasattr(config, "belief_lr")
        ):
            return self._make_opt(**config.opt)
        enc_lr = getattr(config, "enc_lr", config.opt.lr)
        world_lr = getattr(config, "world_lr", config.opt.lr)
        belief_lr = getattr(config, "belief_lr", world_lr)

        def labels(params):
            result = {}
            for name in params:
                if name.startswith("enc/"):
                    result[name] = "enc"
                elif name.startswith("belief/"):
                    result[name] = "belief"
                elif name.startswith("world/"):
                    result[name] = "world"
                else:
                    result[name] = "other"
            return result

        def configured(lr):
            values = dict(config.opt)
            values["lr"] = lr
            values["schedule"] = "const"
            return self._make_opt(**values)

        return optax.multi_transform(
            {
                "enc": configured(enc_lr),
                "belief": configured(belief_lr),
                "world": configured(world_lr),
                "other": self._make_opt(**config.opt),
            },
            labels,
        )

    def _make_opt(
        self,
        lr=4e-5,
        agc=0.3,
        eps=1e-20,
        beta1=0.9,
        beta2=0.999,
        momentum=True,
        nesterov=False,
        wd=0.0,
        wdregex=r"/kernel$",
        schedule="const",
        warmup=1000,
        anneal=0,
    ):
        chain = [embodied.jax.opt.clip_by_agc(agc)]
        chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
        if momentum:
            chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
        if wd:
            pattern = re.compile(wdregex)
            chain.append(
                optax.add_decayed_weights(
                    wd, lambda params: {key: bool(pattern.search(key)) for key in params}
                )
            )
        if schedule == "const":
            learning_rate = optax.constant_schedule(lr)
        elif schedule == "linear":
            learning_rate = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
        elif schedule == "cosine":
            learning_rate = optax.cosine_decay_schedule(
                lr, anneal - warmup, alpha=0.1
            )
        else:
            raise NotImplementedError(schedule)
        if warmup:
            learning_rate = optax.join_schedules(
                [optax.linear_schedule(0.0, lr, warmup), learning_rate], [warmup]
            )
        chain.append(optax.scale_by_learning_rate(learning_rate))
        return optax.chain(*chain)


def joint_imag_loss(
    action,
    reward,
    continuation,
    policy,
    value,
    slowvalue,
    return_norm,
    value_norm,
    advantage_norm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
    value_offset, value_scale = value_norm.stats()
    values = value.pred() * value_scale + value_offset
    slow_values = slowvalue.pred() * value_scale + value_offset
    target_values = slow_values if slowtar else values
    discount = 1 if contdisc else 1 - 1 / horizon
    weight = jnp.cumprod(discount * continuation, 1) / discount
    returns = lambda_return(
        jnp.zeros_like(continuation),
        1 - continuation,
        reward,
        target_values,
        target_values,
        discount,
        lam,
    )
    return_offset, return_scale = return_norm(returns, update)
    advantage = (returns - target_values[:, :-1]) / return_scale
    advantage_offset, advantage_scale = advantage_norm(advantage, update)
    normalized_advantage = (advantage - advantage_offset) / advantage_scale
    log_probability = sum(
        distribution.logp(sg(action[key]))[:, :-1]
        for key, distribution in policy.items()
    )
    entropies = {
        key: distribution.entropy()[:, :-1]
        for key, distribution in policy.items()
    }
    policy_loss = sg(weight[:, :-1, None]) * -(
        log_probability * sg(normalized_advantage[:, :, None])
        + actent * sum(entropies.values())
    )

    value_offset, value_scale = value_norm(returns, update)
    normalized_target = (returns - value_offset) / value_scale
    padded_target = jnp.concatenate(
        [normalized_target, 0 * normalized_target[:, -1:]], 1
    )
    value_loss = sg(weight[:, :-1]) * (
        value.loss(sg(padded_target))
        + slowreg * value.loss(sg(slowvalue.pred()))
    )[:, :-1]
    metrics = {
        "adv": advantage.mean(),
        "adv_std": advantage.std(),
        "adv_mag": jnp.abs(advantage).mean(),
        "rew": reward.mean(),
        "con": continuation.mean(),
        "ret": ((returns - return_offset) / return_scale).mean(),
        "val": values.mean(),
        "slowval": slow_values.mean(),
        "weight": weight.mean(),
    }
    for key, entropy in entropies.items():
        metrics[f"ent/{key}"] = entropy.mean()
        if hasattr(policy[key], "minent"):
            low, high = policy[key].minent, policy[key].maxent
            metrics[f"rand/{key}"] = (entropy.mean() - low) / (high - low)
    return {"policy": policy_loss, "value": value_loss}, {"ret": returns}, metrics


def replay_value_loss(
    last,
    terminal,
    reward,
    bootstrap,
    value,
    slowvalue,
    value_norm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
    offset, scale = value_norm.stats()
    values = value.pred() * scale + offset
    slow_values = slowvalue.pred() * scale + offset
    target_values = slow_values if slowtar else values
    returns = lambda_return(
        last,
        terminal,
        reward,
        target_values,
        bootstrap,
        1 - 1 / horizon,
        lam,
    )
    offset, scale = value_norm(returns, update)
    normalized = (returns - offset) / scale
    padded = jnp.concatenate([normalized, 0 * normalized[:, -1:]], 1)
    loss = f32(~last)[:, :-1] * (
        value.loss(sg(padded)) + slowreg * value.loss(sg(slowvalue.pred()))
    )[:, :-1]
    return {"repval": loss}, {"ret": returns}, {}


def lambda_return(last, terminal, reward, value, bootstrap, discount, lam):
    chex.assert_equal_shape((last, terminal, reward, value, bootstrap))
    returns = [bootstrap[:, -1]]
    live = (1 - f32(terminal))[:, 1:] * discount
    trace = (1 - f32(last))[:, 1:] * lam
    interm = reward[:, 1:] + (1 - trace) * live * bootstrap[:, 1:]
    for index in reversed(range(live.shape[1])):
        returns.append(interm[:, index] + live[:, index] * trace[:, index] * returns[-1])
    return jnp.stack(list(reversed(returns))[:-1], 1)


def _remove_agent_axis(name, space, num_agents):
    if not space.shape or space.shape[0] != num_agents:
        raise ValueError(
            f"{name!r} must expose leading agent axis {num_agents}, got {space.shape}"
        )
    return elements.Space(
        space.dtype,
        space.shape[1:],
        _remove_bound_axis(space.low),
        _remove_bound_axis(space.high),
    )


def _remove_bound_axis(bound):
    return None if bound is None else np.asarray(bound)[0]
