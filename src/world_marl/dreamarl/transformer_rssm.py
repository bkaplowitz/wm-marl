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
from .axes import select_joint_starts
from .joint_transition import JointInteractionResidual
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
    memory_tokens: int = 0
    memory_units: int = 256
    memory_heads: int = 4
    memory_ffup: int = 2
    memory_seed: int = 0
    memory_mode: str = "residual"
    joint_enabled: bool = False
    joint_units: int = 256
    joint_heads: int = 4
    joint_ffup: int = 2
    joint_seed: int = 0

    def __init__(self, act_space, enc_output, **kw):
        super().__init__(act_space, enc_output, **kw)
        if self.num_agents < 1:
            raise ValueError("num_agents must be positive")
        self.action_dim = _encoded_action_dim(act_space)
        self.pair_dim = self.stoch * self.classes + self.action_dim
        self.local_feature_dim = self.deter + self.stoch * self.classes
        self.memory_enabled = self.memory_tokens > 0
        if self.memory_mode not in {"residual", "unified"}:
            raise ValueError(f"unknown memory mode: {self.memory_mode}")
        if self.memory_mode == "unified" and not self.memory_enabled:
            raise ValueError("unified memory mode requires memory tokens")

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
        memory_prior = None
        if self.memory_enabled:
            previous_belief = jnp.concatenate([carry["deter"], previous_stoch], -1)
            memory_prior = self._memory().imagine(
                carry["memory"],
                previous_belief,
                action,
                reset,
                use_belief=self.memory_mode == "residual",
            )
        deter_residual, memory_residual = self._interaction_residual(
            carry, action, reset
        )
        deter = deter + deter_residual
        if memory_prior is not None:
            memory_prior = memory_prior + memory_residual

        if self.memory_enabled:
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
        feat = nn.cast(
            dict(
                deter=deter,
                stoch=stoch,
                logit=logit,
                **memory_values,
                **(
                    {
                        "interaction_deter": deter_residual,
                        **(
                            {"interaction_memory": memory_residual}
                            if memory_residual is not None
                            else {}
                        ),
                    }
                    if self.joint_enabled
                    else {}
                ),
            )
        )
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
            stoch = carry["stoch"].reshape((carry["stoch"].shape[0], -1))
            memory = None
            if self.memory_enabled:
                belief = jnp.concatenate([carry["deter"], stoch], -1)
                memory = self._memory().imagine(
                    carry["memory"],
                    belief,
                    action_embedding,
                    jnp.zeros((stoch.shape[0],), bool),
                    use_belief=self.memory_mode == "residual",
                )
            pair = jnp.concatenate([stoch, action_embedding], -1)
            reset = jnp.zeros((pair.shape[0],), bool)
            cache, deter = self._temporal().step(self._cache(carry), pair, reset)
            deter_residual, memory_residual = self._interaction_residual(
                carry, action_embedding, reset
            )
            deter = deter + deter_residual
            if memory is not None:
                memory = memory + memory_residual
            logit = self._prior(deter)
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
                    **({"memory": memory} if self.memory_enabled else {}),
                    **(
                        {
                            "interaction_deter": deter_residual,
                            **(
                                {"interaction_memory": memory_residual}
                                if memory_residual is not None
                                else {}
                            ),
                        }
                        if self.joint_enabled
                        else {}
                    ),
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
        carry, entries, feat, _ = self.observe(carry, tokens, acts, reset, training)
        prior = self._prior(feat["deter"])
        post = feat["logit"]
        dyn = self._dist(sg(post)).kl(self._dist(prior))
        rep = self._dist(post).kl(self._dist(sg(prior)))
        if self.free_nats:
            dyn = jnp.maximum(dyn, self.free_nats)
            rep = jnp.maximum(rep, self.free_nats)
        pred_enc = self.predictor(feat["deter"])
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
        if self.joint_enabled:
            metrics["interaction/deter_rms"] = jnp.sqrt(
                jnp.mean(f32(feat["interaction_deter"]) ** 2)
            )
            if self.memory_enabled:
                metrics["interaction/memory_rms"] = jnp.sqrt(
                    jnp.mean(f32(feat["interaction_memory"]) ** 2)
                )
        metrics["dyn_ent"] = self._dist(prior).entropy().mean()
        metrics["rep_ent"] = self._dist(post).entropy().mean()
        if self.memory_enabled:
            feat = {
                key: value
                for key, value in feat.items()
                if key not in {"memory_prior", "memory_target"}
            }
        return carry, entries, losses, feat, metrics, None

    def control_feature(self, feat):
        """Return the unchanged belief plus a zero-gated memory residual."""

        stoch = feat["stoch"].reshape((*feat["stoch"].shape[:-2], -1))
        belief = jnp.concatenate([nn.cast(feat["deter"]), nn.cast(stoch)], -1)
        if not self.memory_enabled:
            return belief
        if self.memory_mode == "unified":
            return self._memory().control_state(
                feat["memory"], self.local_feature_dim
            )
        return belief + self._memory().control_residual(
            feat["memory"], self.local_feature_dim
        )

    def _cache(self, carry):
        return {key: carry[key] for key in ("keys", "values", "valid", "position")}

    def _interaction_residual(self, carry, action, reset):
        if not self.joint_enabled:
            deter = jnp.zeros_like(carry["deter"])
            memory = (
                jnp.zeros_like(carry["memory"]) if self.memory_enabled else None
            )
            return deter, memory
        stoch = carry["stoch"].reshape((carry["stoch"].shape[0], -1))
        state = jnp.concatenate([carry["deter"], stoch], -1)
        memory = carry["memory"] if self.memory_enabled else None
        return self._interaction()(state, memory, action, reset)

    def _interaction(self):
        memory_shape = (
            (self.memory_tokens, self.memory_units) if self.memory_enabled else None
        )
        return self.sub(
            "joint_interaction",
            JointInteractionResidual,
            self.local_feature_dim,
            self.action_dim,
            self.deter,
            memory_shape,
            self.num_agents,
            units=self.joint_units,
            heads=self.joint_heads,
            ffup=self.joint_ffup,
            act=self.act,
            norm=self.norm,
            seed=self.joint_seed,
        )

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
