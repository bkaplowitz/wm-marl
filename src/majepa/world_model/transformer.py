"""Stochastic local world model with strict-causal Transformer dynamics.

Replay processes shifted latent-action pairs in parallel. Collection and
imagination use the same Transformer parameters through a bounded KV cache.
"""

import math

import elements
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from ..models.latent import CategoricalLatent
from .attention import CausalTransformer

f32 = jnp.float32
sg = jax.lax.stop_gradient


def feature_tensor(features):
    return jnp.concatenate(
        [
            nn.cast(features["deter"]),
            nn.cast(features["stoch"].reshape((*features["stoch"].shape[:-2], -1))),
        ],
        -1,
    )


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


def _action_feature_dim(space):
    size = math.prod(space.shape)
    if not space.discrete:
        return size
    classes = np.asarray(space.classes)
    if classes.size and not (classes == classes.flat[0]).all():
        raise ValueError("heterogeneous discrete dimensions are not supported")
    return size * int(classes.flat[0])


def _where_active(active, current, previous):
    """Select current leaves for active rows and preserve inactive rows exactly."""

    def select(current_value, previous_value):
        shape = active.shape + (1,) * (current_value.ndim - active.ndim)
        return jnp.where(active.reshape(shape), current_value, previous_value)

    return jax.tree.map(select, current, previous)


def replay_entries(entries):
    return {key: entries[key] for key in ("stoch", "pair", "reset", "position")}
