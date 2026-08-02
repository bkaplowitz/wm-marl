"""Local persistent memory sidecar for DreaMARL.

The module is shared across agents and never receives another agent's state.
It exposes a posterior path for real observations and a prior path used during
imagination. The existing DreaMARL belief remains a separate, unchanged path.
"""

import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj

f32 = jnp.float32


def isolated_winit(seed):
    """Fan-in initializer that does not consume the learner RNG stream."""

    def initialize(shape, dtype=jnp.float32):
        fan_in = jnp.prod(jnp.asarray(shape[:-1]))
        value = jax.random.truncated_normal(
            jax.random.key(seed), -2.0, 2.0, shape, dtype
        )
        return value * (1.1368 / jnp.sqrt(fan_in))

    return initialize


class LocalMemorySidecar(nj.Module):
    tokens: int = 4
    units: int = 256
    heads: int = 4
    ffup: int = 2
    act: str = "silu"
    norm: str = "rms"
    seed: int = 0

    def __init__(self, observation_dim, belief_dim, action_dim, **kw):
        if self.tokens < 1:
            raise ValueError("local memory requires at least one token")
        if self.units % self.heads:
            raise ValueError((self.units, self.heads))
        if observation_dim % self.units:
            raise ValueError(
                "encoder output must divide into pre-bottleneck token channels: "
                f"{observation_dim} % {self.units}"
            )
        self.observation_dim = int(observation_dim)
        self.observation_tokens = self.observation_dim // self.units
        self.belief_dim = int(belief_dim)
        self.action_dim = int(action_dim)
        self.kw = kw

    def initial(self, batch_size):
        return jnp.zeros((batch_size, self.tokens, self.units), f32)

    def tokenize(self, observation):
        """Pool existing pre-bottleneck encoder tokens into four targets."""

        if observation.shape[-1] != self.observation_dim:
            raise ValueError((observation.shape, self.observation_dim))
        source = nn.cast(
            observation.reshape(
                (*observation.shape[:-1], self.observation_tokens, self.units)
            )
        )
        queries = self.value(
            "target_queries",
            self._small_winit("target_queries"),
            (self.tokens, self.units),
            f32,
        )
        queries = jnp.broadcast_to(
            nn.cast(queries), (*source.shape[:-2], self.tokens, self.units)
        )
        target = queries + self._attention("target_cross", queries, source)
        return nn.cast(self._feedforward("target", target))

    def observe(self, previous, observation, action, reset):
        """Update posterior memory from local observation history."""

        self._check_inputs(previous, action, reset)
        target = self.tokenize(observation)
        start = self._start(previous.shape[:-2])
        previous = nn.where(reset, start, nn.cast(previous))
        condition = self.sub(
            "posterior_action",
            nn.Linear,
            self.units,
            winit=self._winit("posterior_action"),
        )(nn.cast(action))
        update = previous + condition[..., None, :]
        update = update + self._attention("posterior_self", update, update)
        update = update + self._attention("posterior_cross", update, target)
        update = self._feedforward("posterior", update)
        gate = self.value("posterior_gate", jnp.zeros, (), f32)
        posterior = target + nn.cast(gate) * (update - target)
        return nn.cast(posterior), nn.cast(target)

    def imagine(self, previous, belief, action, reset, use_belief=True):
        """Predict the next memory without access to future observations."""

        self._check_inputs(previous, action, reset)
        if belief.shape[:-1] != previous.shape[:-2]:
            raise ValueError((belief.shape, previous.shape))
        if belief.shape[-1] != self.belief_dim:
            raise ValueError((belief.shape, self.belief_dim))
        start = self._start(previous.shape[:-2])
        previous = nn.where(reset, start, nn.cast(previous))
        belief = nn.cast(belief)
        if not use_belief:
            belief = jnp.zeros_like(belief)
        condition = jnp.concatenate([belief, nn.cast(action)], -1)
        condition = self.sub(
            "prior_condition",
            nn.Linear,
            self.units,
            winit=self._winit("prior_condition"),
        )(condition)
        predicted = previous + condition[..., None, :]
        predicted = predicted + self._attention("prior_self", predicted, predicted)
        return nn.cast(self._feedforward("prior", predicted))

    def control_residual(self, memory, output_dim):
        """Project memory through an exact zero-initialized scalar gate."""

        return self._control_projection(memory, output_dim, gated=True)

    def control_state(self, memory, output_dim):
        """Project memory directly into the shared world/control feature width."""

        return self._control_projection(memory, output_dim, gated=False)

    def _control_projection(self, memory, output_dim, gated):
        pooled = nn.cast(memory).reshape((*memory.shape[:-2], -1))
        pooled = self.sub(
            "control_bottleneck",
            nn.Linear,
            self.units,
            winit=self._winit("control_bottleneck"),
        )(pooled)
        pooled = nn.act(self.act)(
            self.sub("control_bottleneck_norm", nn.Norm, self.norm)(pooled)
        )
        residual = self.sub(
            "control_projection",
            nn.Linear,
            output_dim,
            winit=self._winit("control_projection"),
        )(pooled)
        gate = self.value("control_gate", jnp.zeros, (), f32)
        if gated:
            return nn.cast(gate) * residual
        # Keep the parameterization identical to the dual-path arm while making
        # the structured memory state the complete learned-head interface.
        return residual + nn.cast(gate) * jnp.zeros_like(residual)

    def gate(self, name):
        if name not in {"posterior_gate", "control_gate"}:
            raise ValueError(name)
        return self.value(name, jnp.zeros, (), f32)

    def _attention(self, name, query, context):
        query = self.sub(
            f"{name}_query",
            nn.Linear,
            self.units,
            winit=self._winit(f"{name}_query"),
        )(self.sub(f"{name}_query_norm", nn.Norm, self.norm)(query))
        normalized = self.sub(f"{name}_context_norm", nn.Norm, self.norm)(context)
        key = self.sub(
            f"{name}_key",
            nn.Linear,
            self.units,
            winit=self._winit(f"{name}_key"),
        )(normalized)
        value = self.sub(
            f"{name}_value",
            nn.Linear,
            self.units,
            winit=self._winit(f"{name}_value"),
        )(normalized)
        head_dim = self.units // self.heads
        query = query.reshape((*query.shape[:-1], self.heads, head_dim))
        key = key.reshape((*key.shape[:-1], self.heads, head_dim))
        value = value.reshape((*value.shape[:-1], self.heads, head_dim))
        logits = jnp.einsum("...qhd,...khd->...hqk", query, key)
        weights = jax.nn.softmax(f32(logits) / jnp.sqrt(f32(head_dim)), -1)
        attended = jnp.einsum("...hqk,...khd->...qhd", weights, value)
        attended = nn.cast(attended.reshape((*attended.shape[:-2], self.units)))
        return self.sub(
            f"{name}_output",
            nn.Linear,
            self.units,
            winit=self._winit(f"{name}_output"),
        )(attended)

    def _feedforward(self, name, value):
        value = nn.cast(value)
        residual = value
        value = self.sub(f"{name}_ff_norm", nn.Norm, self.norm)(value)
        value = self.sub(
            f"{name}_ff_in",
            nn.Linear,
            self.units * self.ffup,
            winit=self._winit(f"{name}_ff_in"),
        )(value)
        value = nn.act(self.act)(value)
        value = self.sub(
            f"{name}_ff_out",
            nn.Linear,
            self.units,
            winit=self._winit(f"{name}_ff_out"),
        )(value)
        return residual + value

    def _start(self, batch_shape):
        value = self.value(
            "start_tokens",
            self._small_winit("start_tokens"),
            (self.tokens, self.units),
            f32,
        )
        return jnp.broadcast_to(nn.cast(value), (*batch_shape, *value.shape))

    def _winit(self, name):
        offset = sum((index + 1) * ord(char) for index, char in enumerate(name))
        return isolated_winit(self.seed + offset)

    def _small_winit(self, name):
        initialize = self._winit(name)

        def small(shape, dtype=jnp.float32):
            return 0.1 * initialize(shape, dtype)

        return small

    def _check_inputs(self, previous, action, reset):
        if previous.shape[-2:] != (self.tokens, self.units):
            raise ValueError((previous.shape, self.tokens, self.units))
        if action.shape[:-1] != previous.shape[:-2]:
            raise ValueError((action.shape, previous.shape))
        if action.shape[-1] != self.action_dim:
            raise ValueError((action.shape, self.action_dim))
        if reset.shape != previous.shape[:-2]:
            raise ValueError((reset.shape, previous.shape))
