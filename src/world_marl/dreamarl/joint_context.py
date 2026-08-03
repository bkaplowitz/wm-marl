"""Permutation-equivariant peer context for DreaMARL transitions."""

import math

import embodied.jax.nets as nn
import jax.numpy as jnp
import ninjax as nj

from .local_memory import isolated_winit


f32 = jnp.float32


class JointTransitionContext(nj.Module):
    """Condition local transitions on peers without writing persistent state.

    Inputs use folded ``[environment * agent, ...]`` ordering. The module
    attends only to valid peers and returns gated conditioning vectors for the
    existing temporal and memory transitions. The gate is exactly zero at
    initialization, and the result remains exactly zero when no peer exists.
    """

    units: int = 256
    heads: int = 4
    ffup: int = 2
    act: str = "silu"
    norm: str = "rms"
    seed: int = 0

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        pair_dim: int,
        memory_shape: tuple[int, int] | None,
        num_agents: int,
        **kw,
    ):
        if num_agents < 1:
            raise ValueError("num_agents must be positive")
        if self.units % self.heads:
            raise ValueError((self.units, self.heads))
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.pair_dim = int(pair_dim)
        self.memory_shape = memory_shape
        self.memory_dim = 0 if memory_shape is None else math.prod(memory_shape)
        self.num_agents = int(num_agents)
        self.input_dim = self.state_dim + self.memory_dim + self.action_dim
        self.kw = kw

    def __call__(self, state, memory, action, reset):
        self._check_inputs(state, memory, action, reset)
        parts = (state, action)
        if memory is not None:
            flat_memory = memory.reshape((*memory.shape[:-2], -1))
            parts = (state, flat_memory, action)
        token = jnp.concatenate([nn.cast(value) for value in parts], -1)
        token = self.sub(
            "token_projection",
            nn.Linear,
            self.units,
            winit=self._winit("token_projection"),
        )(token)
        token = nn.act(self.act)(self.sub("token_norm", nn.Norm, self.norm)(token))

        groups = token.shape[0] // self.num_agents
        token = token.reshape((groups, self.num_agents, self.units))
        valid = (~reset).reshape((groups, self.num_agents))
        peer, has_peer = self._leave_one_out_attention(token, valid)
        context = self._feedforward(token + peer)
        context *= has_peer[..., None]
        context = context.reshape((state.shape[0], self.units))

        gate = jnp.tanh(self.value("gate", jnp.zeros, (), f32))
        active = has_peer.reshape((state.shape[0], 1))
        pair = self.sub(
            "pair_projection",
            nn.Linear,
            self.pair_dim,
            bias=False,
            winit=self._winit("pair_projection"),
        )(context)
        belief = self.sub(
            "belief_projection",
            nn.Linear,
            self.state_dim,
            bias=False,
            winit=self._winit("belief_projection"),
        )(context)
        pair = nn.cast(gate * pair) * active
        belief = nn.cast(gate * belief) * active
        return pair, belief

    def gate(self):
        return jnp.tanh(self.value("gate", jnp.zeros, (), f32))

    def _leave_one_out_attention(self, token, valid):
        normalized = self.sub("attention_norm", nn.Norm, self.norm)(token)
        query, key, value = [
            self.sub(
                f"attention_{name}",
                nn.Linear,
                self.units,
                winit=self._winit(f"attention_{name}"),
            )(normalized)
            for name in ("query", "key", "value")
        ]
        head_dim = self.units // self.heads
        shape = (*token.shape[:-1], self.heads, head_dim)
        query, key, value = [item.reshape(shape) for item in (query, key, value)]
        logits = jnp.einsum("bihd,bjhd->bhij", query, key)
        logits = f32(logits) / jnp.sqrt(f32(head_dim))

        peer_mask = ~jnp.eye(self.num_agents, dtype=bool)[None]
        peer_mask &= valid[:, :, None]
        peer_mask &= valid[:, None, :]
        mask = peer_mask[:, None]
        masked_logits = jnp.where(mask, logits, -1e30)
        maximum = jnp.max(masked_logits, axis=-1, keepdims=True)
        weights = jnp.where(mask, jnp.exp(masked_logits - maximum), 0.0)
        weights /= jnp.maximum(weights.sum(-1, keepdims=True), 1.0)
        attended = jnp.einsum("bhij,bjhd->bihd", weights.astype(value.dtype), value)
        attended = attended.reshape((*token.shape[:-1], self.units))
        attended = self.sub(
            "attention_output",
            nn.Linear,
            self.units,
            winit=self._winit("attention_output"),
        )(attended)
        return attended, peer_mask.any(-1).astype(attended.dtype)

    def _feedforward(self, value):
        residual = value
        value = self.sub("ffn_norm", nn.Norm, self.norm)(value)
        value = self.sub(
            "ffn_in",
            nn.Linear,
            self.units * self.ffup,
            winit=self._winit("ffn_in"),
        )(value)
        value = nn.act(self.act)(value)
        value = self.sub(
            "ffn_out",
            nn.Linear,
            self.units,
            winit=self._winit("ffn_out"),
        )(value)
        return residual + value

    def _winit(self, name):
        offset = sum((index + 1) * ord(char) for index, char in enumerate(name))
        return isolated_winit(self.seed + offset)

    def _check_inputs(self, state, memory, action, reset):
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError((state.shape, self.state_dim))
        if state.shape[0] % self.num_agents:
            raise ValueError((state.shape, self.num_agents))
        if action.shape != (state.shape[0], self.action_dim):
            raise ValueError((action.shape, state.shape, self.action_dim))
        if reset.shape != (state.shape[0],):
            raise ValueError((reset.shape, state.shape))
        if self.memory_shape is None:
            if memory is not None:
                raise ValueError("memory must be None when memory_shape is disabled")
        elif memory.shape != (state.shape[0], *self.memory_shape):
            raise ValueError((memory.shape, state.shape, self.memory_shape))
