"""First-party DreamerV3 block-GRU RSSM reference dynamics."""

from types import MappingProxyType

import elements
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from . import visual
from ..models.latent import CategoricalLatent
from ..world_model.backend import WorldModelBackend


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


class GRURSSMDynamics(CategoricalLatent):
    """DreamerV3 recurrent state-space dynamics used as the parity control."""

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

    def __init__(self, act_space, enc_output, **kw):
        super().__init__(act_space, enc_output, **kw)
        if self.deter % self.blocks:
            raise ValueError("deter must be divisible by blocks")

    @property
    def entry_space(self):
        return {
            "deter": elements.Space(np.float32, self.deter),
            "stoch": elements.Space(np.float32, (self.stoch, self.classes)),
        }

    def initial(self, batch_size):
        return nn.cast(
            {
                "deter": jnp.zeros((batch_size, self.deter), f32),
                "stoch": jnp.zeros((batch_size, self.stoch, self.classes), f32),
            }
        )

    def truncate(self, entries, carry=None):
        del carry
        return jax.tree.map(lambda value: value[:, -1], entries)

    def starts(self, entries, carry, nlast):
        batch = len(jax.tree.leaves(carry)[0])
        return jax.tree.map(
            lambda value: value[:, -nlast:].reshape((batch * nlast, *value.shape[2:])),
            entries,
        )

    def start_at(self, entries, index):
        return jax.tree.map(lambda value: value[:, index], entries)

    def observe(self, carry, tokens, action, reset, training, single=False):
        carry, tokens, action = nn.cast((carry, tokens, action))
        if single:
            carry, (entry, feat, posterior) = self._observe(
                carry, tokens, action, reset, training
            )
            return carry, entry, feat, posterior
        unroll = tokens.shape[1] if self.unroll else 1
        carry, (entries, feat, posterior) = nj.scan(
            lambda state, inputs: self._observe(state, *inputs, training),
            carry,
            (tokens, action, reset),
            unroll=unroll,
            axis=1,
        )
        return carry, entries, feat, posterior

    def _observe(self, carry, tokens, action, reset, training):
        del training
        deter, stoch, action = nn.mask((carry["deter"], carry["stoch"], action), ~reset)
        action = nn.DictConcat(self.act_space, 1)(action)
        action = nn.mask(action, ~reset)
        deter = self._core(deter, stoch, action)
        posterior = self._posterior(deter, tokens)
        stoch = nn.cast(self._dist(posterior).sample(seed=nj.seed()))
        carry = nn.cast({"deter": deter, "stoch": stoch})
        feat = nn.cast({"deter": deter, "stoch": stoch, "logit": posterior})
        entry = {"deter": f32(deter), "stoch": f32(stoch)}
        return carry, (entry, feat, posterior)

    def imagine(self, carry, policy, length, training, single=False):
        if single:
            action = policy(sg(carry)) if callable(policy) else policy
            action_embedding = nn.DictConcat(self.act_space, 1)(action)
            deter = self._core(carry["deter"], carry["stoch"], action_embedding)
            logit = self._prior(deter)
            stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
            carry = nn.cast({"deter": deter, "stoch": stoch})
            feat = nn.cast({"deter": deter, "stoch": stoch, "logit": logit})
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

    def advance(self, carry, action, training):
        """Expose the backend-neutral deterministic transition interface."""

        del training
        action = nn.DictConcat(self.act_space, 1)(action)
        deter = self._core(carry["deter"], carry["stoch"], action)
        return carry, deter

    def complete(self, carry, deter, logit=None):
        """Sample the stochastic state after an optional interaction correction."""

        logit = self.prior(deter) if logit is None else logit
        stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
        carry = nn.cast({"deter": deter, "stoch": stoch})
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
        losses = {"dyn": dyn, "rep": rep}
        metrics = {
            "dyn_ent": self._dist(prior).entropy().mean(),
            "rep_ent": self._dist(posterior).entropy().mean(),
        }
        return losses, metrics

    def loss(self, carry, tokens, acts, reset, training, slow_tokens=None):
        del slow_tokens
        carry, entries, feat, _ = self.observe(carry, tokens, acts, reset, training)
        prior = self.prior(feat["deter"])
        losses, metrics = self.latent_losses(feat["logit"], prior)
        return carry, entries, losses, feat, metrics, None

    def _posterior(self, deter, tokens):
        tokens = tokens.reshape((*deter.shape[:-1], -1))
        value = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
        for index in range(self.obslayers):
            value = self.sub(f"obs{index}", nn.Linear, self.hidden, **self.kw)(value)
            value = nn.act(self.act)(
                self.sub(f"obs{index}norm", nn.Norm, self.norm)(value)
            )
        return self._logit("obslogit", value)

    def _core(self, deter, stoch, action):
        stoch = stoch.reshape((stoch.shape[0], -1))
        if isinstance(action, dict):
            action = nn.DictConcat(self.act_space, 1)(action)
        action /= sg(jnp.maximum(1, jnp.abs(action)))
        groups = self.blocks

        def grouped(value):
            return value.reshape((*value.shape[:-1], groups, -1))

        def flattened(value):
            return value.reshape((*value.shape[:-2], -1))

        deter_input = self.sub("dynin0", nn.Linear, self.hidden, **self.kw)(deter)
        deter_input = nn.act(self.act)(
            self.sub("dynin0norm", nn.Norm, self.norm)(deter_input)
        )
        stoch_input = self.sub("dynin1", nn.Linear, self.hidden, **self.kw)(stoch)
        stoch_input = nn.act(self.act)(
            self.sub("dynin1norm", nn.Norm, self.norm)(stoch_input)
        )
        action_input = self.sub("dynin2", nn.Linear, self.hidden, **self.kw)(action)
        action_input = nn.act(self.act)(
            self.sub("dynin2norm", nn.Norm, self.norm)(action_input)
        )
        inputs = jnp.concatenate([deter_input, stoch_input, action_input], -1)[
            ..., None, :
        ].repeat(groups, -2)
        value = flattened(jnp.concatenate([grouped(deter), inputs], -1))
        for index in range(self.dynlayers):
            value = self.sub(
                f"dynhid{index}", nn.BlockLinear, self.deter, groups, **self.kw
            )(value)
            value = nn.act(self.act)(
                self.sub(f"dynhid{index}norm", nn.Norm, self.norm)(value)
            )
        value = self.sub("dyngru", nn.BlockLinear, 3 * self.deter, groups, **self.kw)(
            value
        )
        reset, candidate, update = [
            flattened(part) for part in jnp.split(grouped(value), 3, -1)
        ]
        reset = jax.nn.sigmoid(reset)
        candidate = jnp.tanh(reset * candidate)
        update = jax.nn.sigmoid(update - 1)
        return update * candidate + (1 - update) * deter


def _replay_entries(entries):
    return {key: entries[key] for key in ("deter", "stoch")}


_RSSM_BACKEND = WorldModelBackend(
    name="rssm",
    encoders=MappingProxyType(
        {
            "simple": visual.Encoder,
            "vit": visual.ViTEncoder,
            "vjepa21": visual.ViTEncoder,
            "leworldmodel": visual.ViTEncoder,
        }
    ),
    decoders=MappingProxyType({"simple": visual.Decoder}),
    dynamics=MappingProxyType({"rssm": GRURSSMDynamics}),
    feature_tensor=_feature_tensor,
    replay_entries=_replay_entries,
)


def rssm_backend():
    return _RSSM_BACKEND
