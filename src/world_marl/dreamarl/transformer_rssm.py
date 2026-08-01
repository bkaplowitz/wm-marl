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
                raise ValueError("each discrete action tensor must share one cardinality")
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

    def __init__(self, act_space, enc_output, **kw):
        super().__init__(act_space, enc_output, **kw)
        self.action_dim = _encoded_action_dim(act_space)
        self.pair_dim = self.stoch * self.classes + self.action_dim

    @property
    def entry_space(self):
        return dict(
            deter=elements.Space(np.float32, self.deter),
            stoch=elements.Space(np.float32, (self.stoch, self.classes)),
            pair=elements.Space(np.float32, self.pair_dim),
            reset=elements.Space(bool),
        )

    def initial(self, batch_size):
        temporal = self._temporal()
        cache = temporal.initial(batch_size)
        return nn.cast(
            dict(
                deter=jnp.zeros((batch_size, self.deter), f32),
                stoch=jnp.zeros((batch_size, self.stoch, self.classes), f32),
                **cache,
            )
        )

    def truncate(self, entries, carry=None):
        assert entries["pair"].ndim == 3, entries["pair"].shape
        batch_size = entries["pair"].shape[0]
        carry = self.initial(batch_size)

        def advance(current, inputs):
            pair, reset, stoch = inputs
            cache, deter = self._temporal().step(self._cache(current), pair, reset)
            current = dict(deter=deter, stoch=nn.cast(stoch), **cache)
            return current, ()

        carry, _ = nj.scan(
            advance,
            carry,
            (entries["pair"], entries["reset"], entries["stoch"]),
            axis=1,
        )
        return carry

    def starts(self, entries, carry, nlast):
        del carry
        batch = entries["deter"].shape[0]
        keys = ("deter", "stoch", "keys", "values", "valid", "position")
        return {
            key: entries[key][:, -nlast:].reshape(
                (batch * nlast, *entries[key].shape[2:])
            )
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
        x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
        for index in range(self.obslayers):
            x = self.sub(f"obs{index}", nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f"obs{index}norm", nn.Norm, self.norm)(x))
        logit = self._logit("obslogit", x)
        stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
        carry = nn.cast(dict(deter=deter, stoch=stoch, **cache))
        feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))
        entry = dict(
            deter=f32(deter),
            stoch=f32(stoch),
            pair=f32(pair),
            reset=reset,
        )
        if include_cache:
            entry.update(cache)
        return carry, (entry, feat, x)

    def imagine(self, carry, policy, length, training, single=False):
        if single:
            action = policy(sg(carry)) if callable(policy) else policy
            action_embedding = nn.DictConcat(self.act_space, 1)(action)
            action_embedding /= sg(jnp.maximum(1, jnp.abs(action_embedding)))
            stoch = carry["stoch"].reshape((carry["stoch"].shape[0], -1))
            pair = jnp.concatenate([stoch, action_embedding], -1)
            reset = jnp.zeros((pair.shape[0],), bool)
            cache, deter = self._temporal().step(self._cache(carry), pair, reset)
            logit = self._prior(deter)
            stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
            carry = nn.cast(dict(deter=deter, stoch=stoch, **cache))
            feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))
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
        metrics["dyn_ent"] = self._dist(prior).entropy().mean()
        metrics["rep_ent"] = self._dist(post).entropy().mean()
        return carry, entries, losses, feat, metrics, None

    def representation_diagnostics(
        self, carry, tokens, acts, reset, training=False, slow_tokens=None
    ):
        """Return transition-aligned frozen-model tensors for offline studies."""

        carry, entries, feat, _ = self.observe(
            carry, tokens, acts, reset, training
        )
        prior = self._prior(feat["deter"])
        pred_token = self.predictor(feat["deter"])
        target_token = tokens if slow_tokens is None else slow_tokens
        return carry, {
            "pair": f32(entries["pair"]),
            "deter": f32(feat["deter"]),
            "stoch": f32(feat["stoch"]),
            "post_logit": f32(feat["logit"]),
            "prior_logit": f32(prior),
            "pred_token": f32(pred_token),
            "target_token": f32(target_token),
            "reset": reset,
        }

    def _cache(self, carry):
        return {key: carry[key] for key in ("keys", "values", "valid", "position")}

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
