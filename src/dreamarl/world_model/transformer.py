"""Stochastic local world model with strict-causal Transformer dynamics.

Replay processes shifted latent-action pairs in parallel. Collection and
imagination use the same Transformer parameters through a bounded KV cache.
"""

import math
from types import MappingProxyType

import elements
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from ..models import visual
from ..models.latent import CategoricalLatent
from .backend import WorldModelBackend


f32 = jnp.float32
sg = jax.lax.stop_gradient


def _feature_tensor(features):
    return jnp.concatenate(
        [
            nn.cast(features["deter"]),
            nn.cast(features["stoch"].reshape((*features["stoch"].shape[:-2], -1))),
        ],
        -1,
    )


class CausalTransformer(nj.Module):
    """Causal sequence model with equivalent parallel and recurrent paths."""

    units: int = 512
    output: int = 2048
    layers: int = 2
    heads: int = 8
    context: int = 64
    ffup: int = 4
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"
    team_size: int = 1

    def __init__(self, pair_dim, **kw):
        self.pair_dim = pair_dim
        self.kw = kw

    def initial(self, batch_size):
        head_dim = self.units // self.heads
        dtype = nn.COMPUTE_DTYPE
        return {
            "keys": jnp.zeros(
                (batch_size, self.layers, self.context, self.heads, head_dim),
                dtype,
            ),
            "values": jnp.zeros(
                (batch_size, self.layers, self.context, self.heads, head_dim),
                dtype,
            ),
            "valid": jnp.zeros((batch_size, self.context), bool),
            "position": -jnp.ones((batch_size,), jnp.int32),
        }

    def sequence(self, cache, previous_pairs, resets, active=None):
        """Process replay in parallel and retain every imagination-start cache."""

        batch, length = previous_pairs.shape[:2]
        projected = self.sub(
            "pair_projection", nn.Linear, self.units, winit=self.winit
        )(nn.cast(previous_pairs))
        projected += self._peer_residual(previous_pairs, active)
        start = self.value("start_token", nn.init("trunc_normal"), (self.units,), f32)
        start = nn.cast(jnp.broadcast_to(start, projected.shape))
        x = jnp.where(resets[..., None], start, projected)
        positions = _episode_positions(resets, cache["position"])
        segments = jnp.cumsum(resets.astype(jnp.int32), axis=1)

        old_segments = jnp.zeros((batch, self.context), jnp.int32)
        key_segments = jnp.concatenate([old_segments, segments], axis=1)
        key_valid = jnp.concatenate(
            [cache["valid"], jnp.ones((batch, length), bool)], axis=1
        )
        old_mask = cache["valid"][:, None, :] & (segments[:, :, None] == 0)
        causal = jnp.arange(length)[None, None, :] <= jnp.arange(length)[None, :, None]
        new_mask = causal & (segments[:, :, None] == segments[:, None, :])
        attention_mask = jnp.concatenate([old_mask, new_mask], axis=-1)

        snapshot_indices = (
            jnp.arange(length)[:, None] + jnp.arange(self.context)[None, :] + 1
        )
        gathered_valid = key_valid[:, snapshot_indices]
        gathered_segments = key_segments[:, snapshot_indices]
        snapshot_valid = gathered_valid & (gathered_segments == segments[:, :, None])

        key_snapshots = []
        value_snapshots = []
        for index in range(self.layers):
            with nj.scope(f"layer{index}"):
                residual = x
                normed = self.sub("attention_norm", nn.Norm, self.norm)(x)
                qkv = self.sub("qkv", nn.Linear, 3 * self.units, winit=self.winit)(
                    normed
                )
                query, key, value = jnp.split(qkv, 3, axis=-1)
                shape = (batch, length, self.heads, self.units // self.heads)
                query, key, value = [
                    item.reshape(shape) for item in (query, key, value)
                ]
                query = nn.rope(query, positions)
                key = nn.rope(key, positions)
                key_bank = jnp.concatenate([cache["keys"][:, index], key], axis=1)
                value_bank = jnp.concatenate([cache["values"][:, index], value], axis=1)
                logits = jnp.einsum("bthd,bshd->bhts", query, key_bank)
                logits = f32(logits) / math.sqrt(key.shape[-1])
                logits = jnp.where(attention_mask[:, None], logits, -1e30)
                weights = jax.nn.softmax(logits, axis=-1).astype(x.dtype)
                attended = jnp.einsum("bhts,bshd->bthd", weights, value_bank)
                attended = attended.reshape((batch, length, self.units))
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

                key_window = key_bank[:, snapshot_indices]
                value_window = value_bank[:, snapshot_indices]
                mask = snapshot_valid[..., None, None]
                key_snapshots.append(jnp.where(mask, key_window, 0))
                value_snapshots.append(jnp.where(mask, value_window, 0))

        x = self.sub("output_norm", nn.Norm, self.norm)(x)
        states = self.sub("state_projection", nn.Linear, self.output, winit=self.winit)(
            x
        )
        key_snapshots = jnp.stack(key_snapshots, axis=2)
        value_snapshots = jnp.stack(value_snapshots, axis=2)
        snapshots = {
            "keys": key_snapshots,
            "values": value_snapshots,
            "valid": snapshot_valid,
            "position": positions,
        }
        final = {
            "keys": key_snapshots[:, -1],
            "values": value_snapshots[:, -1],
            "valid": snapshot_valid[:, -1],
            "position": positions[:, -1],
        }
        return nn.cast(final), nn.cast(states), nn.cast(snapshots)

    def step(self, cache, previous_pair, reset, active=None):
        """Advance one timestep using the same weights as ``sequence``."""

        batch = previous_pair.shape[0]
        cache = {
            key: _reset_array(value, reset)
            for key, value in cache.items()
            if key != "position"
        } | {"position": jnp.where(reset, -1, cache["position"])}
        position = cache["position"] + 1
        projected = self.sub(
            "pair_projection", nn.Linear, self.units, winit=self.winit
        )(nn.cast(previous_pair))
        projected += self._peer_residual(previous_pair, active)
        start = self.value("start_token", nn.init("trunc_normal"), (self.units,), f32)
        x = jnp.where(reset[:, None], nn.cast(start), projected)
        valid = jnp.concatenate(
            [cache["valid"][:, 1:], jnp.ones((batch, 1), bool)], axis=1
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
                query, key, value = jnp.split(qkv, 3, axis=-1)
                shape = (batch, self.heads, self.units // self.heads)
                query, key, value = [
                    item.reshape(shape) for item in (query, key, value)
                ]
                timestamp = position[:, None]
                query = nn.rope(query[:, None], timestamp)[:, 0]
                key = nn.rope(key[:, None], timestamp)[:, 0]
                keys = jnp.concatenate(
                    [cache["keys"][:, index, 1:], key[:, None]], axis=1
                )
                values = jnp.concatenate(
                    [cache["values"][:, index, 1:], value[:, None]], axis=1
                )
                logits = jnp.einsum("bhd,bthd->bht", query, keys)
                logits = f32(logits) / math.sqrt(key.shape[-1])
                logits = jnp.where(valid[:, None], logits, -1e30)
                weights = jax.nn.softmax(logits, axis=-1).astype(x.dtype)
                attended = jnp.einsum("bht,bthd->bhd", weights, values)
                attended = attended.reshape((batch, self.units))
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
        state = self.sub("state_projection", nn.Linear, self.output, winit=self.winit)(
            x
        )
        next_cache = {
            "keys": jnp.stack(next_keys, axis=1),
            "values": jnp.stack(next_values, axis=1),
            "valid": valid,
            "position": position,
        }
        return nn.cast(next_cache), nn.cast(state)

    def _peer_context(self, projected, active):
        if self.team_size == 1:
            return jnp.zeros_like(projected), jnp.zeros(projected.shape[:-1], bool)
        if active is None:
            active = jnp.ones(projected.shape[:-1], bool)
        sequence = projected.ndim == 3
        if sequence:
            batch = projected.shape[0] // self.team_size
            grouped = projected.reshape(
                (batch, self.team_size, projected.shape[1], projected.shape[2])
            ).transpose((0, 2, 1, 3))
            grouped_active = active.reshape(
                (batch, self.team_size, active.shape[1])
            ).transpose((0, 2, 1))
        else:
            batch = projected.shape[0] // self.team_size
            grouped = projected.reshape((batch, self.team_size, projected.shape[1]))
            grouped_active = active.reshape((batch, self.team_size))
        weighted = grouped * grouped_active[..., None]
        peer_sum = weighted.sum(-2, keepdims=True) - weighted
        peer_count = grouped_active.sum(-1, keepdims=True)[..., None]
        peer_count = peer_count - grouped_active[..., None]
        context = peer_sum / jnp.maximum(peer_count, 1)
        context_active = peer_count[..., 0] > 0
        if sequence:
            context = context.transpose((0, 2, 1, 3)).reshape(projected.shape)
            context_active = context_active.transpose((0, 2, 1)).reshape(active.shape)
        else:
            context = context.reshape(projected.shape)
            context_active = context_active.reshape(active.shape)
        return context, context_active

    def _peer_residual(self, pair, active):
        if self.team_size == 1:
            return jnp.zeros((*pair.shape[:-1], self.units), nn.COMPUTE_DTYPE)
        peer = self.sub(
            "peer_projection", nn.Linear, self.units, winit=self.winit
        )(sg(nn.cast(pair)))
        peer = nn.act(self.act)(self.sub("peer_norm", nn.Norm, self.norm)(peer))
        peer, peer_active = self._peer_context(peer, active)
        gate = nn.cast(self.peer_gate())
        residual = peer * gate
        return jnp.where(peer_active[..., None], residual, 0)

    def peer_gate(self):
        if self.team_size == 1:
            return jnp.zeros((self.units,), f32)
        value = self.value("peer_gate", nn.init("zeros"), (self.units,), f32)
        return jnp.tanh(value)


class ParallelTransformerDynamics(CategoricalLatent):
    """Observation-parallel posterior and causal Transformer prior dynamics."""

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
    layers: int = 2
    heads: int = 8
    context: int = 64
    ffup: int = 4
    posterior_context: str = "history"
    team_size: int = 1

    def __init__(self, act_space, enc_output, **kw):
        super().__init__(act_space, enc_output, **kw)
        if self.posterior_context not in {"observation", "history"}:
            raise ValueError(
                "posterior_context must be either 'observation' or 'history'"
            )
        self.action_dim = sum(
            _action_feature_dim(space) for space in act_space.values()
        )
        self.local_pair_dim = self.stoch * self.classes + self.action_dim
        self.pair_dim = self.local_pair_dim

    @property
    def entry_space(self):
        return {
            "stoch": elements.Space(np.float32, (self.stoch, self.classes)),
            "pair": elements.Space(np.float32, self.pair_dim),
            "reset": elements.Space(bool),
        }

    def initial(self, batch_size):
        cache = self._temporal().initial(batch_size)
        return nn.cast(
            {
                "deter": jnp.zeros((batch_size, self.deter), f32),
                "stoch": jnp.zeros((batch_size, self.stoch, self.classes), f32),
                **cache,
            }
        )

    def truncate(self, entries, carry=None, active=None):
        del carry
        state = self.initial(entries["pair"].shape[0])

        if active is None:
            active = jnp.ones_like(entries["reset"], bool)

        def advance(current, inputs):
            pair, reset, stoch, current_active = inputs
            cache, deter = self._temporal().step(
                self._cache(current), pair, reset, active=current_active
            )
            current = nn.cast({"deter": deter, "stoch": stoch, **cache})
            return current, ()

        inputs = (
            entries["pair"],
            entries["reset"],
            entries["stoch"],
            active,
        )

        state, _ = nj.scan(advance, state, inputs, axis=1)
        return state

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

    def start_at(self, entries, index):
        """Recover one posterior/cache state from a parallel replay sequence."""
        keys = ("deter", "stoch", "keys", "values", "valid", "position")
        return {key: entries[key][:, index] for key in keys}

    def observe(
        self,
        carry,
        tokens,
        action,
        reset,
        training,
        single=False,
        active=None,
    ):
        carry, tokens, action = nn.cast((carry, tokens, action))
        if single:
            return self._observe_single(carry, tokens, action, reset, training, active)

        if self.posterior_context == "history":

            def advance(state, inputs):
                current_tokens, current_action, current_reset, current_active = inputs
                state, entry, feat, posterior = self._observe_single(
                    state,
                    current_tokens,
                    current_action,
                    current_reset,
                    training,
                    current_active,
                )
                return state, (entry, feat, posterior)

            if active is None:
                active = jnp.ones_like(reset, bool)
            carry, (entries, feat, posterior) = nj.scan(
                advance,
                carry,
                (tokens, action, reset, active),
                axis=1,
            )
            return carry, entries, feat, posterior

        posterior = self._posterior(tokens)
        stoch = nn.cast(self._dist(posterior).sample(seed=nj.seed()))
        previous_stoch = jnp.concatenate(
            [carry["stoch"][:, None], stoch[:, :-1]], axis=1
        )
        pair = self._temporal_pair(previous_stoch, action, training=training)
        if active is None:
            active = jnp.ones_like(reset, bool)
        cache, deter, snapshots = self._temporal().sequence(
            self._cache(carry), pair, reset, active=active
        )
        feat = nn.cast({"deter": deter, "stoch": stoch, "logit": posterior})
        entries = {
            "deter": f32(deter),
            "stoch": f32(stoch),
            "pair": f32(pair),
            "reset": reset,
            "active": active,
            **snapshots,
        }
        carry = nn.cast({"deter": deter[:, -1], "stoch": stoch[:, -1], **cache})
        return carry, entries, feat, posterior

    def _observe_single(self, carry, tokens, action, reset, training, active=None):
        pair = self._temporal_pair(
            carry["stoch"], action, reset=reset, training=training
        )
        if active is None:
            active = jnp.ones_like(reset, bool)
        cache, deter = self._temporal().step(
            self._cache(carry), pair, reset, active=active
        )
        posterior = self._posterior(tokens, deter)
        stoch = nn.cast(self._dist(posterior).sample(seed=nj.seed()))
        carry = nn.cast({"deter": deter, "stoch": stoch, **cache})
        feat = nn.cast({"deter": deter, "stoch": stoch, "logit": posterior})
        entry = {
            "deter": f32(deter),
            "stoch": f32(stoch),
            "pair": f32(pair),
            "reset": reset,
            "active": active,
            **cache,
        }
        return carry, entry, feat, posterior

    def imagine(
        self,
        carry,
        policy,
        length,
        training,
        single=False,
        active=None,
    ):
        if single:
            action = policy(sg(carry)) if callable(policy) else policy
            cache, deter = self.advance(carry, action, training, active=active)
            carry, feat = self.complete(cache, deter)
            return carry, (feat, action)
        unroll = length if self.unroll else 1
        if callable(policy):
            carry, (feat, action) = nj.scan(
                lambda state, _: self.imagine(
                    state,
                    policy,
                    1,
                    training,
                    single=True,
                    active=active,
                ),
                nn.cast(carry),
                (),
                length,
                unroll=unroll,
                axis=1,
            )
        else:
            carry, (feat, action) = nj.scan(
                lambda state, act: self.imagine(
                    state,
                    act,
                    1,
                    training,
                    single=True,
                    active=active,
                ),
                nn.cast(carry),
                nn.cast(policy),
                length,
                unroll=unroll,
                axis=1,
            )
        return carry, feat, action

    def advance(self, carry, action, training, active=None):
        """Compute the local temporal proposal for one imagined step."""

        pair = self._temporal_pair(carry["stoch"], action, training=training)
        reset = jnp.zeros((pair.shape[0],), bool)
        if active is None:
            active = jnp.ones_like(reset, bool)
        cache, deter = self._temporal().step(
            self._cache(carry), pair, reset, active=active
        )
        return cache, deter

    def complete(self, cache, deter, logit=None):
        """Sample the categorical state from the joint-conditioned prior."""

        logit = self._prior(deter) if logit is None else logit
        stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
        carry = nn.cast({"deter": deter, "stoch": stoch, **cache})
        feat = nn.cast({"deter": deter, "stoch": stoch, "logit": logit})
        return carry, feat

    def prior(self, deter):
        return self._prior(deter)

    def latent_losses(self, posterior, prior):
        dyn = self._dist(sg(posterior)).kl(self._dist(prior))
        rep = self._dist(posterior).kl(self._dist(sg(prior)))
        if self.free_nats:
            dyn = jnp.maximum(dyn, self.free_nats)
            rep = jnp.maximum(rep, self.free_nats)
        return {"dyn": dyn, "rep": rep}, {
            "dyn_ent": self._dist(prior).entropy().mean(),
            "rep_ent": self._dist(posterior).entropy().mean(),
        }

    def loss(
        self,
        carry,
        tokens,
        acts,
        reset,
        training,
        slow_tokens=None,
        active=None,
    ):
        del slow_tokens
        metrics = {}
        carry, entries, feat, _ = self.observe(
            carry,
            tokens,
            acts,
            reset,
            training,
            active=active,
        )
        prior = self._prior(feat["deter"])
        losses, latent_metrics = self.latent_losses(feat["logit"], prior)
        metrics.update(latent_metrics)
        if self.team_size > 1:
            gate = jnp.abs(f32(self._temporal().peer_gate()))
            metrics["interaction/gate_mean"] = gate.mean()
            metrics["interaction/gate_max"] = gate.max()
        return carry, entries, losses, feat, metrics, None

    def _posterior(self, tokens, deter=None):
        x = tokens.reshape((*tokens.shape[:-1], -1))
        if self.posterior_context == "history":
            if deter is None:
                raise ValueError("history-conditioned posterior requires deter")
            x = jnp.concatenate([deter, x], axis=-1)
        for index in range(self.obslayers):
            x = self.sub(f"obs{index}", nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(self.sub(f"obs{index}norm", nn.Norm, self.norm)(x))
        return self._logit("obslogit", x)

    def _cache(self, carry):
        return {key: carry[key] for key in ("keys", "values", "valid", "position")}

    def _temporal_pair(self, stoch, action, *, training, reset=None):
        del training
        action_embedding = nn.DictConcat(self.act_space, 1)(action)
        if reset is not None:
            action_embedding = nn.mask(action_embedding, ~reset)
        action_embedding /= sg(jnp.maximum(1, jnp.abs(action_embedding)))
        stoch = stoch.reshape((*stoch.shape[:-2], -1))
        return jnp.concatenate([stoch, action_embedding], axis=-1)

    def _temporal(self):
        return self.sub(
            "temporal",
            CausalTransformer,
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
            team_size=self.team_size,
        )


def _episode_positions(resets, initial_position):
    def advance(position, reset):
        position = jnp.where(reset, 0, position + 1)
        return position, position

    _, positions = jax.lax.scan(advance, initial_position, resets.T)
    return positions.T


def _action_feature_dim(space):
    size = math.prod(space.shape)
    if not space.discrete:
        return size
    classes = np.asarray(space.classes)
    if classes.size and not (classes == classes.flat[0]).all():
        raise ValueError("heterogeneous discrete dimensions are not supported")
    return size * int(classes.flat[0])


def _reset_array(value, reset):
    shape = (reset.shape[0],) + (1,) * (value.ndim - 1)
    return jnp.where(reset.reshape(shape), jnp.zeros_like(value), value)


def _replay_entries(entries):
    return {key: entries[key] for key in ("stoch", "pair", "reset")}


_PARALLEL_BACKEND = WorldModelBackend(
    name="parallel_transformer",
    encoders=MappingProxyType({"simple": visual.Encoder}),
    decoders=MappingProxyType({}),
    dynamics=MappingProxyType({"parallel_transformer": ParallelTransformerDynamics}),
    feature_tensor=_feature_tensor,
    replay_entries=_replay_entries,
)


def parallel_backend():
    return _PARALLEL_BACKEND
