"""Strictly local recurrent belief used for decentralized execution.

The module is shared across agents, but each invocation consumes only one
agent's observation embedding, previous action, and recurrent cache.  It has
no reward, value, continuation, or peer-conditioning heads.
"""

import math

import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np


f32 = jnp.float32
sg = jax.lax.stop_gradient


class LocalBelief(nj.Module):
    """Bounded causal Transformer memory for one agent trajectory."""

    units: int = 512
    layers: int = 2
    heads: int = 8
    context: int = 64
    ffup: int = 4
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"

    def __init__(self, act_space, embedding_dim: int, **kw):
        if self.units % self.heads:
            raise ValueError("local belief width must be divisible by attention heads")
        self.act_space = act_space
        self.embedding_dim = int(embedding_dim)
        self.input_dim = self.embedding_dim + _encoded_action_dim(act_space)
        self.kw = kw

    @property
    def entry_space(self):
        # Replay uses a burn-in prefix rather than serializing implementation
        # specific KV tensors into every transition.
        return {}

    def initial(self, batch_size: int):
        head_dim = self.units // self.heads
        dtype = nn.COMPUTE_DTYPE
        return {
            "belief": jnp.zeros((batch_size, self.units), dtype),
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

    def truncate(self, entries, carry=None):
        del entries
        if carry is None:
            raise ValueError("local belief burn-in requires an explicit carry")
        return carry

    def observe(self, carry, embedding, action, reset, training, single=False):
        del training
        carry, embedding, action = nn.cast((carry, embedding, action))
        if single:
            carry, (feature, entry) = self._step(carry, embedding, action, reset)
            return carry, feature, entry
        carry, (features, entries) = nj.scan(
            lambda state, inputs: self._step(state, *inputs),
            carry,
            (embedding, action, reset),
            axis=1,
        )
        return carry, features, entries

    def _step(self, carry, embedding, action, reset):
        if embedding.ndim != 2 or embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"expected local embeddings [batch, {self.embedding_dim}], "
                f"got {embedding.shape}"
            )
        encoded_action = nn.DictConcat(self.act_space, 1)(action)
        encoded_action = nn.mask(encoded_action, ~reset)
        encoded_action /= sg(jnp.maximum(1, jnp.abs(encoded_action)))
        inputs = jnp.concatenate([embedding, encoded_action], -1)
        next_cache, belief = self._core().step(self._cache(carry), inputs, reset)
        carry = nn.cast({"belief": belief, **next_cache})
        return carry, (belief, carry)

    def _cache(self, carry):
        return {key: carry[key] for key in ("keys", "values", "valid", "position")}

    def _core(self):
        return self.sub(
            "core",
            CausalBeliefTransformer,
            self.input_dim,
            units=self.units,
            layers=self.layers,
            heads=self.heads,
            context=self.context,
            ffup=self.ffup,
            act=self.act,
            norm=self.norm,
            winit=self.winit,
        )


class CausalBeliefTransformer(nj.Module):
    """Incremental Transformer whose cache is private to one local policy."""

    units: int = 512
    layers: int = 2
    heads: int = 8
    context: int = 64
    ffup: int = 4
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"

    def __init__(self, input_dim: int, **kw):
        self.input_dim = int(input_dim)
        self.kw = kw

    def step(self, cache, inputs, reset):
        if inputs.shape != (inputs.shape[0], self.input_dim):
            raise ValueError((inputs.shape, self.input_dim))
        if reset.shape != (inputs.shape[0],):
            raise ValueError((reset.shape, inputs.shape))
        cache = jax.tree.map(lambda x: nn.where(reset, jnp.zeros_like(x), x), cache)
        position = jnp.where(reset, 0, cache["position"] + 1)
        projected = self.sub(
            "input_projection", nn.Linear, self.units, winit=self.winit
        )(inputs)
        start = self.value("start_token", nn.init("trunc_normal"), (self.units,), f32)
        start = jnp.broadcast_to(nn.cast(start), projected.shape)
        x = projected + nn.where(reset, start, jnp.zeros_like(start))

        valid = jnp.concatenate(
            [cache["valid"][:, 1:], jnp.ones((inputs.shape[0], 1), bool)], 1
        )
        next_keys = []
        next_values = []
        for index in range(self.layers):
            with nj.scope(f"layer{index}"):
                residual = x
                normed = self.sub("attention_norm", nn.Norm, self.norm)(x)
                qkv = self.sub(
                    "qkv", nn.Linear, 3 * self.units, winit=self.winit
                )(normed)
                query, key, value = jnp.split(qkv, 3, -1)
                shape = (inputs.shape[0], self.heads, self.units // self.heads)
                query, key, value = [item.reshape(shape) for item in (query, key, value)]
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
                attended = attended.reshape((inputs.shape[0], self.units))
                x = residual + self.sub(
                    "attention_out", nn.Linear, self.units, winit=self.winit
                )(attended)

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

        belief = self.sub("output_norm", nn.Norm, self.norm)(x)
        next_cache = {
            "keys": jnp.stack(next_keys, 1),
            "values": jnp.stack(next_values, 1),
            "valid": valid,
            "position": position,
        }
        return nn.cast(next_cache), nn.cast(belief)


def _encoded_action_dim(act_space) -> int:
    total = 0
    for space in act_space.values():
        width = math.prod(space.shape) if space.shape else 1
        if space.discrete:
            classes = np.asarray(space.classes).reshape(-1)
            if not (classes == classes[0]).all():
                raise ValueError("discrete action elements must share cardinality")
            width *= int(classes[0])
        total += width
    return int(total)
