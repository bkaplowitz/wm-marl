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
    condition_mode: str = "add"

    def __init__(self, pair_dim, **kw):
        self.pair_dim = pair_dim
        self.kw = kw
        if self.condition_mode not in {"add", "adaln"}:
            raise ValueError(
                "causal Transformer condition_mode must be 'add' or 'adaln'"
            )

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

    def sequence(self, cache, previous_pairs, resets, condition=None):
        """Process replay in parallel and retain every imagination-start cache."""

        batch, length = previous_pairs.shape[:2]
        projected = self.sub(
            "pair_projection", nn.Linear, self.units, winit=self.winit
        )(nn.cast(previous_pairs))
        if condition is not None:
            condition = nn.cast(condition)
            if condition.shape[:2] != projected.shape[:2]:
                raise ValueError(
                    "causal Transformer condition must share batch/time axes: "
                    f"{condition.shape} versus {projected.shape}"
                )
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
        old_positions = (
            cache["position"][:, None]
            - jnp.arange(self.context - 1, -1, -1, dtype=jnp.int32)[None]
        )
        key_positions = jnp.concatenate([old_positions, positions], axis=1)
        within_window = (key_positions[:, None] <= positions[:, :, None]) & (
            key_positions[:, None] > positions[:, :, None] - self.context
        )
        attention_mask &= within_window

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
                modulation = None
                if condition is not None and self.condition_mode == "add":
                    x = x + self.sub(
                        "condition", nn.Linear, self.units, winit=self.winit
                    )(condition)
                elif condition is not None:
                    modulation = self.sub(
                        "condition_adaln",
                        nn.Linear,
                        6 * self.units,
                        winit=self.winit,
                        outscale=0.0,
                    )(nn.act(self.act)(condition))
                    modulation = jnp.split(modulation, 6, axis=-1)
                residual = x
                normed = self.sub("attention_norm", nn.Norm, self.norm)(x)
                if modulation is not None:
                    shift, scale, attention_gate = modulation[:3]
                    normed = normed * (1 + scale) + shift
                qkv = self.sub("qkv", nn.Linear, 3 * self.units, winit=self.winit)(
                    normed
                )
                query, key, value = jnp.split(qkv, 3, axis=-1)
                shape = (batch, length, self.heads, self.units // self.heads)
                query, key, value = [
                    item.reshape(shape) for item in (query, key, value)
                ]
                query = _rope_f32(query, positions)
                key = _rope_f32(key, positions)
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
                if modulation is not None:
                    attended = attention_gate * attended
                x = residual + attended

                residual = x
                x = self.sub("ffn_norm", nn.Norm, self.norm)(x)
                if modulation is not None:
                    shift, scale, ffn_gate = modulation[3:]
                    x = x * (1 + scale) + shift
                x = self.sub(
                    "ffn_in", nn.Linear, self.units * self.ffup, winit=self.winit
                )(x)
                x = nn.act(self.act)(x)
                x = self.sub("ffn_out", nn.Linear, self.units, winit=self.winit)(x)
                if modulation is not None:
                    x = ffn_gate * x
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

    def step(self, cache, previous_pair, reset, condition=None):
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
        if condition is not None:
            condition = nn.cast(condition)
            if condition.shape[0] != projected.shape[0]:
                raise ValueError(
                    "causal Transformer condition must share the batch axis: "
                    f"{condition.shape} versus {projected.shape}"
                )
        start = self.value("start_token", nn.init("trunc_normal"), (self.units,), f32)
        x = jnp.where(reset[:, None], nn.cast(start), projected)
        valid = jnp.concatenate(
            [cache["valid"][:, 1:], jnp.ones((batch, 1), bool)], axis=1
        )
        next_keys = []
        next_values = []
        for index in range(self.layers):
            with nj.scope(f"layer{index}"):
                modulation = None
                if condition is not None and self.condition_mode == "add":
                    x = x + self.sub(
                        "condition", nn.Linear, self.units, winit=self.winit
                    )(condition)
                elif condition is not None:
                    modulation = self.sub(
                        "condition_adaln",
                        nn.Linear,
                        6 * self.units,
                        winit=self.winit,
                        outscale=0.0,
                    )(nn.act(self.act)(condition))
                    modulation = jnp.split(modulation, 6, axis=-1)
                residual = x
                normed = self.sub("attention_norm", nn.Norm, self.norm)(x)
                if modulation is not None:
                    shift, scale, attention_gate = modulation[:3]
                    normed = normed * (1 + scale) + shift
                qkv = self.sub("qkv", nn.Linear, 3 * self.units, winit=self.winit)(
                    normed
                )
                query, key, value = jnp.split(qkv, 3, axis=-1)
                shape = (batch, self.heads, self.units // self.heads)
                query, key, value = [
                    item.reshape(shape) for item in (query, key, value)
                ]
                timestamp = position[:, None]
                query = _rope_f32(query[:, None], timestamp)[:, 0]
                key = _rope_f32(key[:, None], timestamp)[:, 0]
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
                if modulation is not None:
                    attended = attention_gate * attended
                x = residual + attended

                residual = x
                x = self.sub("ffn_norm", nn.Norm, self.norm)(x)
                if modulation is not None:
                    shift, scale, ffn_gate = modulation[3:]
                    x = x * (1 + scale) + shift
                x = self.sub(
                    "ffn_in", nn.Linear, self.units * self.ffup, winit=self.winit
                )(x)
                x = nn.act(self.act)(x)
                x = self.sub("ffn_out", nn.Linear, self.units, winit=self.winit)(x)
                if modulation is not None:
                    x = ffn_gate * x
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
            "position": elements.Space(np.int32),
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
        state, _ = self.replay_sequence(entries, carry=carry, active=active)
        return state

    def replay_sequence(self, entries, carry=None, active=None):
        """Rebuild loss-free replay state and expose its local feature sequence."""

        del carry
        state = self.initial(entries["pair"].shape[0])
        if "position" in entries:
            initial_position = entries["position"][:, 0] - 1
            initial_position = jnp.where(
                entries["reset"][:, 0],
                -jnp.ones_like(initial_position),
                initial_position,
            )
            state = dict(state, position=initial_position)

        if active is None:
            active = jnp.ones_like(entries["reset"], bool)

        def advance(current, inputs):
            pair, reset, stoch, current_active = inputs
            cache, deter = self._temporal().step(self._cache(current), pair, reset)
            next_state = nn.cast({"deter": deter, "stoch": stoch, **cache})
            current = _where_active(current_active, next_state, current)
            return current, {
                "deter": current["deter"],
                "stoch": current["stoch"],
            }

        inputs = (
            entries["pair"],
            entries["reset"],
            entries["stoch"],
            active,
        )

        return nj.scan(advance, state, inputs, axis=1)

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
            self._cache(carry), pair, reset
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
        cache, deter = self._temporal().step(self._cache(carry), pair, reset)
        posterior = self._posterior(tokens, deter)
        stoch = nn.cast(self._dist(posterior).sample(seed=nj.seed()))
        next_carry = nn.cast({"deter": deter, "stoch": stoch, **cache})
        carry = _where_active(active, next_carry, carry)
        feat = nn.cast(
            {
                "deter": carry["deter"],
                "stoch": carry["stoch"],
                "logit": posterior,
            }
        )
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
            previous = carry
            action = policy(sg(carry)) if callable(policy) else policy
            cache, deter = self.advance(carry, action, training, active=active)
            carry, feat = self.complete(cache, deter)
            if active is not None:
                carry = _where_active(active, carry, previous)
                feat = dict(
                    feat,
                    deter=carry["deter"],
                    stoch=carry["stoch"],
                )
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
        cache, deter = self._temporal().step(self._cache(carry), pair, reset)
        if active is not None:
            cache = _where_active(active, cache, self._cache(carry))
            deter = _where_active(active, deter, carry["deter"])
        return cache, deter

    def complete(self, cache, deter, logit=None, *, sample=True):
        """Complete one local transition from its action-conditioned prior."""

        logit = self._prior(deter) if logit is None else logit
        distribution = self._dist(logit)
        stoch = distribution.sample(seed=nj.seed()) if sample else distribution.pred()
        stoch = nn.cast(stoch)
        carry = nn.cast({"deter": deter, "stoch": stoch, **cache})
        feat = nn.cast({"deter": deter, "stoch": stoch, "logit": logit})
        return carry, feat

    def posterior(self, tokens, deter):
        """Return the executable observation-conditioned posterior logits."""

        return self._posterior(tokens, deter)

    def complete_from_observation(self, cache, deter, tokens, *, sample=True):
        """Complete a local proposal from a predicted observation embedding."""

        return self.complete(
            cache,
            deter,
            logit=self.posterior(nn.cast(tokens), nn.cast(deter)),
            sample=sample,
        )

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
        )


def _episode_positions(resets, initial_position):
    def advance(position, reset):
        position = jnp.where(reset, 0, position + 1)
        return position, position

    _, positions = jax.lax.scan(advance, initial_position, resets.T)
    return positions.T


def _rope_f32(x, timestamps, maxlen=4096):
    """Apply RoPE with FP32 angles and trigonometry, then restore input dtype."""

    if x.shape[-1] % 2:
        raise ValueError(f"RoPE feature width must be even, got {x.shape[-1]}")
    timestamps = jnp.asarray(timestamps, f32)
    frequencies = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=f32)
    timescales = jnp.asarray(maxlen, f32) ** frequencies
    radians = timestamps[..., None] / timescales
    sine = jnp.sin(radians)[..., None, :]
    cosine = jnp.cos(radians)[..., None, :]
    left, right = jnp.split(f32(x), 2, axis=-1)
    rotated = jnp.concatenate(
        [left * cosine - right * sine, right * cosine + left * sine],
        axis=-1,
    )
    return rotated.astype(x.dtype)


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


def _where_active(active, current, previous):
    """Select current leaves for active rows and preserve inactive rows exactly."""

    def select(current_value, previous_value):
        shape = active.shape + (1,) * (current_value.ndim - active.ndim)
        return jnp.where(active.reshape(shape), current_value, previous_value)

    return jax.tree.map(select, current, previous)


def _replay_entries(entries):
    return {key: entries[key] for key in ("stoch", "pair", "reset", "position")}


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
