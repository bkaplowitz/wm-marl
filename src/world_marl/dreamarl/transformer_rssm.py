"""Causal Transformer dynamics overlay for the pinned Dreamer-CDP source.

This file is copied into a clean runtime snapshot of Dreamer-CDP. It subclasses
the official RSSM so the categorical latent, JEPA predictor, prior/posterior,
and losses remain unchanged; only the deterministic recurrent transition is
replaced by a bounded causal Transformer with a recurrent KV cache.
"""

import math

import elements
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from . import rssm
from .axes import (
    fold_agent_batch,
    fold_agent_sequence,
    select_joint_starts,
    unfold_agent_batch,
    unfold_agent_sequence,
)
from .interaction import AgentInteraction, InteractionResidual
from .local_memory import LocalMemorySidecar


f32 = jnp.float32
sg = jax.lax.stop_gradient


def _encoded_action_dim(act_space):
    """Return the width produced by ``nn.DictConcat`` for an action tree."""

    total = 0
    for space in act_space.values():
        elements_count = math.prod(space.shape) if space.shape else 1
        if space.discrete:
            classes = np.asarray(space.classes).reshape(-1)
            if not (classes == classes[0]).all():
                raise ValueError(
                    "each discrete action tensor must share one cardinality"
                )
            elements_count *= int(classes[0])
        total += elements_count
    return total


class CausalKVTransformer(nj.Module):
    units: int = 512
    output: int = 8192
    layers: int = 4
    heads: int = 8
    context: int = 64
    ffup: int = 4
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"

    def __init__(self, pair_dim, **kw):
        assert self.units % self.heads == 0
        self.pair_dim = pair_dim
        self.kw = kw

    def initial(self, batch_size):
        head_dim = self.units // self.heads
        dtype = nn.COMPUTE_DTYPE
        return dict(
            keys=jnp.zeros(
                (batch_size, self.layers, self.context, self.heads, head_dim), dtype
            ),
            values=jnp.zeros(
                (batch_size, self.layers, self.context, self.heads, head_dim), dtype
            ),
            valid=jnp.zeros((batch_size, self.context), bool),
            position=-jnp.ones((batch_size,), jnp.int32),
        )

    def step(self, cache, pair, reset):
        assert pair.shape == (pair.shape[0], self.pair_dim), pair.shape
        assert reset.shape == (pair.shape[0],), reset.shape
        pair = nn.cast(pair)
        cache = jax.tree.map(lambda x: nn.where(reset, jnp.zeros_like(x), x), cache)
        position = jnp.where(reset, 0, cache["position"] + 1)
        projected = self.sub(
            "pair_projection", nn.Linear, self.units, winit=self.winit
        )(pair)
        start = self.value("start_token", nn.init("trunc_normal"), (self.units,), f32)
        start = nn.cast(jnp.broadcast_to(start, projected.shape))
        x = nn.where(reset, start, projected)

        valid = jnp.concatenate(
            [
                cache["valid"][:, 1:],
                jnp.ones((pair.shape[0], 1), bool),
            ],
            1,
        )
        next_keys = []
        next_values = []
        for index in range(self.layers):
            with nj.scope(f"layer{index}"):
                residual = x
                normed = self.sub("attention_norm", nn.Norm, self.norm)(x)
                qkv = self.sub("qkv", nn.Linear, 3 * self.units, winit=self.winit)(
                    normed
                )
                query, key, value = jnp.split(qkv, 3, -1)
                shape = (pair.shape[0], self.heads, self.units // self.heads)
                query, key, value = [
                    item.reshape(shape) for item in (query, key, value)
                ]
                timestamp = position[:, None]
                query = nn.rope(query[:, None], timestamp)[:, 0]
                key = nn.rope(key[:, None], timestamp)[:, 0]
                keys = jnp.concatenate([cache["keys"][:, index, 1:], key[:, None]], 1)
                values = jnp.concatenate(
                    [cache["values"][:, index, 1:], value[:, None]], 1
                )
                logits = jnp.einsum("bhd,bthd->bht", query, keys)
                logits = f32(logits) / math.sqrt(key.shape[-1])
                logits = jnp.where(valid[:, None], logits, -1e30)
                weights = jax.nn.softmax(logits).astype(x.dtype)
                attended = jnp.einsum("bht,bthd->bhd", weights, values)
                attended = attended.reshape((pair.shape[0], self.units))
                attended = self.sub(
                    "attention_out", nn.Linear, self.units, winit=self.winit
                )(attended)
                x = residual + attended

                residual = x
                x = self.sub("ffn_norm", nn.Norm, self.norm)(x)
                x = self.sub(
                    "ffn_in", nn.Linear, self.units * self.ffup, winit=self.winit
                )(x)
                x = nn.act(self.act)(x)
                x = self.sub("ffn_out", nn.Linear, self.units, winit=self.winit)(x)
                x = residual + x
                next_keys.append(keys)
                next_values.append(values)

        x = self.sub("output_norm", nn.Norm, self.norm)(x)
        deter = self.sub("state_projection", nn.Linear, self.output, winit=self.winit)(
            x
        )
        next_cache = dict(
            keys=jnp.stack(next_keys, 1),
            values=jnp.stack(next_values, 1),
            valid=valid,
            position=position,
        )
        return nn.cast(next_cache), nn.cast(deter)


class TransformerRSSM(rssm.RSSM):
    deter: int = 8192
    hidden: int = 1024
    stoch: int = 32
    classes: int = 64
    norm: str = "rms"
    act: str = "silu"
    unroll: bool = False
    unimix: float = 0.01
    outscale: float = 1.0
    imglayers: int = 2
    obslayers: int = 1
    dynlayers: int = 1
    absolute: bool = False
    blocks: int = 8
    free_nats: float = 1.0
    model: int = 512
    layers: int = 4
    heads: int = 8
    context: int = 64
    ffup: int = 4
    num_agents: int = 1
    interaction: str = "none"
    interaction_units: int = 128
    interaction_heads: int = 4
    interaction_seed: int = 0
    memory_tokens: int = 0
    memory_units: int = 256
    memory_heads: int = 4
    memory_ffup: int = 2
    memory_seed: int = 0

    def __init__(self, act_space, enc_output, **kw):
        super().__init__(act_space, enc_output, **kw)
        if self.interaction not in {"none", "aligned", "shuffled"}:
            raise ValueError(f"unknown interaction context: {self.interaction!r}")
        if self.num_agents < 1:
            raise ValueError("num_agents must be positive")
        self.action_dim = _encoded_action_dim(act_space)
        self.pair_dim = self.stoch * self.classes + self.action_dim
        self.local_feature_dim = self.deter + self.stoch * self.classes
        self.memory_enabled = self.memory_tokens > 0

    @property
    def entry_space(self):
        spaces = dict(
            deter=elements.Space(np.float32, self.deter),
            stoch=elements.Space(np.float32, (self.stoch, self.classes)),
            pair=elements.Space(np.float32, self.pair_dim),
            reset=elements.Space(bool),
        )
        if self.memory_enabled:
            spaces["memory"] = elements.Space(
                np.float32, (self.memory_tokens, self.memory_units)
            )
        return spaces

    def initial(self, batch_size):
        temporal = self._temporal()
        cache = temporal.initial(batch_size)
        carry = dict(
            deter=jnp.zeros((batch_size, self.deter), f32),
            stoch=jnp.zeros((batch_size, self.stoch, self.classes), f32),
            **cache,
        )
        if self.memory_enabled:
            carry["memory"] = self._memory().initial(batch_size)
        return nn.cast(carry)

    def truncate(self, entries, carry=None):
        assert entries["pair"].ndim == 3, entries["pair"].shape
        batch_size = entries["pair"].shape[0]
        carry = self.initial(batch_size)

        def advance(current, inputs):
            pair, reset, stoch, memory = inputs
            cache, deter = self._temporal().step(self._cache(current), pair, reset)
            current = dict(deter=deter, stoch=nn.cast(stoch), **cache)
            if self.memory_enabled:
                current["memory"] = nn.cast(memory)
            return current, ()

        memory = (
            entries["memory"]
            if self.memory_enabled
            else jnp.zeros((*entries["reset"].shape, 0, 0), f32)
        )
        carry, _ = nj.scan(
            advance,
            carry,
            (entries["pair"], entries["reset"], entries["stoch"], memory),
            axis=1,
        )
        return carry

    def starts(self, entries, carry, nlast):
        del carry
        keys = ["deter", "stoch", "keys", "values", "valid", "position"]
        if self.memory_enabled:
            keys.append("memory")
        return {
            key: select_joint_starts(entries[key], self.num_agents, nlast)
            for key in keys
        }

    def observe(self, carry, tokens, action, reset, training, single=False):
        carry, tokens, action = nn.cast((carry, tokens, action))
        if single:
            carry, (entry, feat, posterior_input) = self._observe(
                carry, tokens, action, reset, training, include_cache=False
            )
            return carry, entry, feat, posterior_input
        unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
        carry, (entries, feat, posterior_input) = nj.scan(
            lambda state, inputs: self._observe(
                state, *inputs, training, include_cache=True
            ),
            carry,
            (tokens, action, reset),
            unroll=unroll,
            axis=1,
        )
        return carry, entries, feat, posterior_input

    def _observe(self, carry, tokens, action, reset, training, include_cache=False):
        del training
        action = nn.DictConcat(self.act_space, 1)(action)
        action = nn.mask(action, ~reset)
        action /= sg(jnp.maximum(1, jnp.abs(action)))
        previous_stoch = carry["stoch"].reshape((carry["stoch"].shape[0], -1))
        pair = jnp.concatenate([previous_stoch, action], -1)
        cache, deter = self._temporal().step(self._cache(carry), pair, reset)
        tokens = tokens.reshape((*deter.shape[:-1], -1))
        memory_values = {}
        if self.memory_enabled:
            previous_belief = jnp.concatenate([carry["deter"], previous_stoch], -1)
            memory_prior = self._memory().imagine(
                carry["memory"], previous_belief, action, reset
            )
            memory, memory_target = self._memory().observe(
                carry["memory"], tokens, action, reset
            )
            memory_values = dict(
                memory=memory,
                memory_prior=memory_prior,
                memory_target=memory_target,
            )
        x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
        for index in range(self.obslayers):
            x = self.sub(f"obs{index}", nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f"obs{index}norm", nn.Norm, self.norm)(x))
        logit = self._logit("obslogit", x)
        stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
        carry = nn.cast(
            dict(
                deter=deter,
                stoch=stoch,
                **cache,
                **({"memory": memory_values["memory"]} if self.memory_enabled else {}),
            )
        )
        feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit, **memory_values))
        entry = dict(
            deter=f32(deter),
            stoch=f32(stoch),
            pair=f32(pair),
            reset=reset,
        )
        if self.memory_enabled:
            entry["memory"] = f32(memory_values["memory"])
        if include_cache:
            entry.update(cache)
        return carry, (entry, feat, x)

    def imagine(self, carry, policy, length, training, single=False):
        if single:
            action = policy(sg(carry)) if callable(policy) else policy
            action_embedding = nn.DictConcat(self.act_space, 1)(action)
            action_embedding /= sg(jnp.maximum(1, jnp.abs(action_embedding)))
            message, has_other = self._interaction_step(carry, action_embedding)
            stoch = carry["stoch"].reshape((carry["stoch"].shape[0], -1))
            memory = None
            if self.memory_enabled:
                belief = jnp.concatenate([carry["deter"], stoch], -1)
                memory = self._memory().imagine(
                    carry["memory"],
                    belief,
                    action_embedding,
                    jnp.zeros((stoch.shape[0],), bool),
                )
            pair = jnp.concatenate([stoch, action_embedding], -1)
            reset = jnp.zeros((pair.shape[0],), bool)
            cache, deter = self._temporal().step(self._cache(carry), pair, reset)
            base_prior = self._prior(deter)
            logit, _, world_delta = self._interaction_outputs(
                deter, message, has_other, base_prior=base_prior
            )
            stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
            carry = nn.cast(
                dict(
                    deter=deter,
                    stoch=stoch,
                    **cache,
                    **({"memory": memory} if self.memory_enabled else {}),
                )
            )
            feat = nn.cast(
                dict(
                    deter=deter,
                    stoch=stoch,
                    logit=logit,
                    world_delta=world_delta,
                    **({"memory": memory} if self.memory_enabled else {}),
                )
            )
            return carry, (feat, action)
        unroll = length if self.unroll else 1
        if callable(policy):
            carry, (feat, action) = nj.scan(
                lambda state, _: self.imagine(state, policy, 1, training, single=True),
                nn.cast(carry),
                (),
                length,
                unroll=unroll,
                axis=1,
            )
        else:
            carry, (feat, action) = nj.scan(
                lambda state, act: self.imagine(state, act, 1, training, single=True),
                nn.cast(carry),
                nn.cast(policy),
                length,
                unroll=unroll,
                axis=1,
            )
        return carry, feat, action

    def loss(self, carry, tokens, acts, reset, training, slow_tokens=None):
        metrics = {}
        initial_carry = carry
        carry, entries, feat, _ = self.observe(carry, tokens, acts, reset, training)
        base_prior = self._prior(feat["deter"])
        message, has_other = self._interaction_sequence(
            initial_carry, feat, acts, reset
        )
        prior, pred_delta, world_delta = self._interaction_outputs(
            feat["deter"], message, has_other, base_prior=base_prior
        )
        post = feat["logit"]
        dyn = self._dist(sg(post)).kl(self._dist(prior))
        rep = self._dist(post).kl(self._dist(sg(base_prior)))
        if self.free_nats:
            dyn = jnp.maximum(dyn, self.free_nats)
            rep = jnp.maximum(rep, self.free_nats)
        pred_enc = self.predictor(feat["deter"]) + pred_delta
        dyn_deter = optax.losses.cosine_distance(
            sg(slow_tokens), pred_enc, axis=-1, epsilon=1e-8
        )
        losses = dict(dyn=dyn, rep=rep, dyn_deter=dyn_deter)
        if self.memory_enabled:
            memory_target = sg(feat["memory_target"])
            memory_prior = feat["memory_prior"]
            memory_dyn = optax.losses.cosine_distance(
                memory_target,
                memory_prior,
                axis=-1,
                epsilon=1e-8,
            ).mean(-1)
            losses["memory_dyn"] = nn.mask(memory_dyn, ~reset)
            metrics["memory/dyn_error"] = losses["memory_dyn"].mean()
            metrics["memory/target_rms"] = jnp.sqrt(
                jnp.mean(f32(memory_target) ** 2)
            )
            metrics["memory/target_std"] = jnp.std(f32(memory_target))
            metrics["memory/prior_rms"] = jnp.sqrt(
                jnp.mean(f32(memory_prior) ** 2)
            )
            metrics["memory/prior_std"] = jnp.std(f32(memory_prior))
            metrics["memory/posterior_gate"] = self._memory().gate("posterior_gate")
            metrics["memory/control_gate"] = self._memory().gate("control_gate")
        else:
            losses["memory_dyn"] = jnp.zeros_like(dyn_deter)
        metrics["dyn_ent"] = self._dist(prior).entropy().mean()
        metrics["local_dyn_ent"] = self._dist(base_prior).entropy().mean()
        metrics["rep_ent"] = self._dist(post).entropy().mean()
        metrics["interaction/message_rms"] = self._masked_rms(message, has_other)
        metrics["interaction/prior_delta_rms"] = self._masked_rms(
            prior - base_prior, has_other
        )
        metrics["interaction/predictor_delta_rms"] = self._masked_rms(
            pred_delta, has_other
        )
        metrics["interaction/active_fraction"] = has_other.mean()
        if self.memory_enabled:
            feat = {
                key: value
                for key, value in feat.items()
                if key not in {"memory_prior", "memory_target"}
            }
        feat = {**feat, "world_delta": world_delta}
        return carry, entries, losses, feat, metrics, None

    def representation_diagnostics(
        self, carry, tokens, acts, reset, training=False, slow_tokens=None
    ):
        """Return transition-aligned frozen-model tensors for offline studies."""

        initial_carry = carry
        carry, entries, feat, _ = self.observe(carry, tokens, acts, reset, training)
        base_prior = self._prior(feat["deter"])
        message, has_other = self._interaction_sequence(
            initial_carry, feat, acts, reset
        )
        prior, pred_delta, _ = self._interaction_outputs(
            feat["deter"], message, has_other, base_prior=base_prior
        )
        pred_token = self.predictor(feat["deter"]) + pred_delta
        target_token = tokens if slow_tokens is None else slow_tokens
        return carry, {
            "pair": f32(entries["pair"]),
            "deter": f32(feat["deter"]),
            "stoch": f32(feat["stoch"]),
            "post_logit": f32(feat["logit"]),
            "prior_logit": f32(prior),
            "base_prior_logit": f32(base_prior),
            "pred_token": f32(pred_token),
            "target_token": f32(target_token),
            "reset": reset,
            **(
                {
                    "memory": f32(feat["memory"]),
                    "memory_prior": f32(feat["memory_prior"]),
                    "memory_target": f32(feat["memory_target"]),
                }
                if self.memory_enabled
                else {}
            ),
        }

    def control_feature(self, feat):
        """Return the unchanged belief plus a zero-gated memory residual."""

        stoch = feat["stoch"].reshape((*feat["stoch"].shape[:-2], -1))
        belief = jnp.concatenate([nn.cast(feat["deter"]), nn.cast(stoch)], -1)
        if not self.memory_enabled:
            return belief
        residual = self._memory().control_residual(
            feat["memory"], self.local_feature_dim
        )
        return belief + residual

    def _interaction_sequence(self, initial_carry, feat, actions, reset):
        source_deter = jnp.concatenate(
            [initial_carry["deter"][:, None], feat["deter"][:, :-1]], 1
        )
        source_stoch = jnp.concatenate(
            [initial_carry["stoch"][:, None], feat["stoch"][:, :-1]], 1
        )
        action = nn.DictConcat(self.act_space, 1)(actions)
        action = nn.mask(action, ~reset)
        action /= sg(jnp.maximum(1, jnp.abs(action)))
        belief = jnp.concatenate(
            [source_deter, source_stoch.reshape((*source_stoch.shape[:-2], -1))],
            -1,
        )
        token = jnp.concatenate([belief, action], -1)
        grouped_belief = unfold_agent_sequence(belief, self.num_agents)
        grouped_token = unfold_agent_sequence(token, self.num_agents)
        if self.interaction == "none":
            shape = (*grouped_belief.shape[:-1], self.interaction_units)
            message = jnp.zeros(shape, grouped_belief.dtype)
            has_other = jnp.zeros((*shape[:-1], 1), bool)
            return (
                fold_agent_sequence(message, self.num_agents),
                fold_agent_sequence(has_other, self.num_agents),
            )
        valid = jnp.ones(grouped_belief.shape[:-1], bool)
        message, has_other = self._interaction()(
            grouped_belief,
            grouped_token,
            valid,
            shuffled=self.interaction == "shuffled",
        )
        message = fold_agent_sequence(message, self.num_agents)
        has_other = fold_agent_sequence(has_other, self.num_agents)
        message = nn.mask(message, ~reset)
        has_other = has_other & (~reset[..., None])
        return message, has_other

    def _interaction_step(self, carry, action):
        stoch = carry["stoch"].reshape((carry["stoch"].shape[0], -1))
        belief = jnp.concatenate([carry["deter"], stoch], -1)
        token = jnp.concatenate([belief, action], -1)
        grouped_belief = unfold_agent_batch(belief, self.num_agents)
        grouped_token = unfold_agent_batch(token, self.num_agents)
        if self.interaction == "none":
            shape = (*grouped_belief.shape[:-1], self.interaction_units)
            message = jnp.zeros(shape, grouped_belief.dtype)
            has_other = jnp.zeros((*shape[:-1], 1), bool)
            return (
                fold_agent_batch(message, self.num_agents),
                fold_agent_batch(has_other, self.num_agents),
            )
        valid = jnp.ones(grouped_belief.shape[:-1], bool)
        message, has_other = self._interaction()(
            grouped_belief,
            grouped_token,
            valid,
            shuffled=self.interaction == "shuffled",
        )
        return (
            fold_agent_batch(message, self.num_agents),
            fold_agent_batch(has_other, self.num_agents),
        )

    def _interaction_outputs(self, deter, message, has_other, *, base_prior):
        if self.interaction == "none":
            return (
                base_prior,
                jnp.zeros((*deter.shape[:-1], self.enc_output), deter.dtype),
                jnp.zeros((*deter.shape[:-1], self.local_feature_dim), deter.dtype),
            )
        prior_delta = self.sub(
            "interaction_prior",
            InteractionResidual,
            self.stoch * self.classes,
            hidden=self.interaction_units,
            act=self.act,
            norm=self.norm,
            seed=self.interaction_seed + 100,
        )(deter, message, has_other)
        prior_delta = prior_delta.reshape(base_prior.shape)
        pred_delta = self.sub(
            "interaction_predictor",
            InteractionResidual,
            self.enc_output,
            hidden=self.interaction_units,
            act=self.act,
            norm=self.norm,
            seed=self.interaction_seed + 200,
        )(deter, message, has_other)
        world_delta = self.sub(
            "interaction_world",
            InteractionResidual,
            self.local_feature_dim,
            hidden=self.interaction_units,
            act=self.act,
            norm=self.norm,
            seed=self.interaction_seed + 300,
        )(deter, message, has_other)
        return base_prior + prior_delta, pred_delta, world_delta

    def _interaction(self):
        return self.sub(
            "interaction",
            AgentInteraction,
            self.local_feature_dim,
            self.local_feature_dim + self.action_dim,
            units=self.interaction_units,
            heads=self.interaction_heads,
            norm=self.norm,
            seed=self.interaction_seed,
        )

    @staticmethod
    def _masked_rms(value, mask):
        while mask.ndim < value.ndim:
            mask = mask[..., None]
        expanded = jnp.broadcast_to(mask, value.shape)
        count = jnp.maximum(expanded.sum(), 1)
        return jnp.sqrt((jnp.square(f32(value)) * expanded).sum() / count)

    def _cache(self, carry):
        return {key: carry[key] for key in ("keys", "values", "valid", "position")}

    def _memory(self):
        if not self.memory_enabled:
            raise RuntimeError("local memory sidecar is disabled")
        return self.sub(
            "local_memory",
            LocalMemorySidecar,
            self.enc_output,
            self.local_feature_dim,
            self.action_dim,
            tokens=self.memory_tokens,
            units=self.memory_units,
            heads=self.memory_heads,
            ffup=self.memory_ffup,
            act=self.act,
            norm=self.norm,
            seed=self.memory_seed,
        )

    def _temporal(self):
        return self.sub(
            "temporal",
            CausalKVTransformer,
            self.pair_dim,
            units=self.model,
            output=self.deter,
            layers=self.layers,
            heads=self.heads,
            context=self.context,
            ffup=self.ffup,
            act=self.act,
            norm=self.norm,
            winit=self.kw.get("winit", "trunc_normal_in"),
        )
