"""Training-only joint JEPA simulator and centralized attention critic.

The executable DreaMARL policy remains the unchanged local encoder, causal
Transformer, posterior, and actor.  These modules operate only on synchronized
team tensors during learning.  The simulator predicts the next observation
embedding for each local posterior; it never produces an actor feature
directly.  No fixed agent identifiers or agent-position embeddings are used.
"""

import math

import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj

from ..world_model.transformer import CausalTransformer


f32 = jnp.float32
sg = jax.lax.stop_gradient


def _masked_softmax(logits, valid):
    valid = valid.astype(bool)
    weights = jax.nn.softmax(jnp.where(valid, f32(logits), -1e30), axis=-1)
    weights = weights * valid.astype(f32)
    return weights / jnp.maximum(weights.sum(axis=-1, keepdims=True), 1e-8)


def _fold_agent_sequence(value):
    """Convert ``[B,T,A,...]`` to ``[B*A,T,...]``."""

    axes = (0, 2, 1, *range(3, value.ndim))
    value = value.transpose(axes)
    return value.reshape(
        (value.shape[0] * value.shape[1], value.shape[2], *value.shape[3:])
    )


def _unfold_agent_sequence(value, agents):
    """Convert ``[B*A,T,...]`` to ``[B,T,A,...]``."""

    batch = value.shape[0] // agents
    value = value.reshape((batch, agents, value.shape[1], *value.shape[2:]))
    axes = (0, 2, 1, *range(3, value.ndim))
    return value.transpose(axes)


class AgentInteraction(nj.Module):
    """Permutation-equivariant attention over the synchronized agent set."""

    width: int = 256
    heads: int = 4
    layers: int = 2
    ffup: int = 4
    dropout: float = 0.1
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"

    def __init__(self, **kwargs):
        if self.width < 1 or self.layers < 1 or self.heads < 1:
            raise ValueError("CTDE agent-attention dimensions must be positive")
        if self.width % self.heads:
            raise ValueError(
                f"CTDE width {self.width} must be divisible by {self.heads} heads"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("CTDE dropout must be in [0, 1)")
        del kwargs

    def __call__(self, tokens, active, training):
        if tokens.ndim < 2 or active.shape != tokens.shape[:-1]:
            raise ValueError(
                "CTDE interaction expects tokens [...,A,D] and active [...,A], "
                f"got {tokens.shape} and {active.shape}"
            )
        if tokens.shape[-1] != self.width:
            raise ValueError(
                f"expected {self.width}-wide CTDE tokens, got {tokens.shape[-1]}"
            )
        active = active.astype(bool)
        value = nn.cast(tokens) * active[..., None].astype(nn.cast(tokens).dtype)
        agents = value.shape[-2]
        head_width = self.width // self.heads

        for index in range(self.layers):
            residual = value
            update = self.sub(f"attn{index}_norm", nn.Norm, self.norm)(value)
            qkv = self.sub(
                f"attn{index}_qkv", nn.Linear, 3 * self.width, winit=self.winit
            )(update)
            query, key, item = jnp.split(qkv, 3, axis=-1)
            shape = (*value.shape[:-2], agents, self.heads, head_width)
            query, key, item = [part.reshape(shape) for part in (query, key, item)]
            logits = jnp.einsum(
                "...qhd,...khd->...hqk", f32(query), f32(key)
            ) / math.sqrt(head_width)
            weights = _masked_softmax(logits, active[..., None, None, :]).astype(
                item.dtype
            )
            weights = nn.dropout(weights, self.dropout, training)
            update = jnp.einsum("...hqk,...khd->...qhd", weights, item)
            update = update.reshape((*value.shape[:-2], agents, self.width))
            update = self.sub(
                f"attn{index}_out", nn.Linear, self.width, winit=self.winit
            )(update)
            update = nn.dropout(update, self.dropout, training)
            value = (residual + update) * active[..., None].astype(update.dtype)

            residual = value
            update = self.sub(f"ffn{index}_norm", nn.Norm, self.norm)(value)
            update = self.sub(
                f"ffn{index}_in",
                nn.Linear,
                self.width * self.ffup,
                winit=self.winit,
            )(update)
            update = nn.act(self.act)(update)
            update = self.sub(
                f"ffn{index}_out", nn.Linear, self.width, winit=self.winit
            )(update)
            update = nn.dropout(update, self.dropout, training)
            value = (residual + update) * active[..., None].astype(update.dtype)

        value = self.sub("output_norm", nn.Norm, self.norm)(value)
        return value * active[..., None].astype(value.dtype)


class JointObservationJEPA(nj.Module):
    """Joint-action simulator predicting one next local embedding per agent."""

    width: int = 256
    heads: int = 4
    agent_layers: int = 2
    temporal_layers: int = 4
    context: int = 16
    ffup: int = 4
    dropout: float = 0.1
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"
    action_conditioning: str = "add"

    def __init__(self, action_count, action_low, target_dim, **kwargs):
        self.action_count = int(action_count)
        self.action_low = int(action_low)
        self.target_dim = int(target_dim)
        if self.action_count < 2 or self.target_dim < 1:
            raise ValueError("CTDE requires a categorical action and embedding target")
        if self.width % self.heads:
            raise ValueError(
                f"CTDE width {self.width} must be divisible by {self.heads} heads"
            )
        if self.action_conditioning not in {"add", "adaln"}:
            raise ValueError("CTDE action_conditioning must be either 'add' or 'adaln'")
        del kwargs

    def initial(self, teams, agents, previous_position=None):
        teams, agents = int(teams), int(agents)
        cache = self._temporal().initial(teams * agents)
        if previous_position is None:
            return cache
        previous_position = jnp.asarray(previous_position, jnp.int32)
        if previous_position.shape == (teams,):
            previous_position = jnp.broadcast_to(
                previous_position[:, None], (teams, agents)
            )
        if previous_position.shape != (teams, agents):
            raise ValueError(
                "CTDE initial position must be shaped [B] or [B,A], got "
                f"{previous_position.shape}"
            )
        return dict(cache, position=previous_position.reshape(-1))

    def sequence(self, cache, states, actions, present, alive, reset, training):
        """Teacher-forced joint transitions for ``[B,T,A,...]`` replay."""

        if states.ndim != 4 or actions.shape != states.shape[:3]:
            raise ValueError(
                "CTDE replay expects states [B,T,A,F] and actions [B,T,A], "
                f"got {states.shape} and {actions.shape}"
            )
        if (
            present.shape != actions.shape
            or alive.shape != actions.shape
            or reset.shape != actions.shape[:2]
        ):
            raise ValueError("CTDE replay masks do not match state/action axes")
        agents = states.shape[2]
        mixed, action_condition = self._mix(states, actions, present, alive, training)
        folded = _fold_agent_sequence(mixed)
        condition = (
            folded
            if self.action_conditioning == "add"
            else _fold_agent_sequence(action_condition)
        )
        folded_reset = _fold_agent_sequence(
            jnp.broadcast_to(reset[:, :, None], reset.shape + (agents,))
        )
        cache, hidden, snapshots = self._temporal().sequence(
            cache,
            folded,
            folded_reset,
            condition=condition,
        )
        hidden = _unfold_agent_sequence(hidden, agents)
        hidden = hidden * present[..., None].astype(hidden.dtype)
        return cache, self._outputs(hidden), snapshots

    def step(self, cache, states, actions, present, alive, reset, training):
        """One synchronized imagined transition for ``[N,A,...]`` states."""

        if states.ndim != 3 or actions.shape != states.shape[:2]:
            raise ValueError(
                "CTDE step expects states [N,A,F] and actions [N,A], "
                f"got {states.shape} and {actions.shape}"
            )
        if (
            present.shape != actions.shape
            or alive.shape != actions.shape
            or reset.shape != actions.shape[:1]
        ):
            raise ValueError("CTDE step masks do not match state/action axes")
        teams, agents = actions.shape
        mixed, action_condition = self._mix(states, actions, present, alive, training)
        folded = mixed.reshape((teams * agents, self.width))
        condition = (
            folded
            if self.action_conditioning == "add"
            else action_condition.reshape((teams * agents, self.width))
        )
        folded_reset = jnp.broadcast_to(reset[:, None], (teams, agents)).reshape(-1)
        cache, hidden = self._temporal().step(
            cache,
            folded,
            folded_reset,
            condition=condition,
        )
        hidden = hidden.reshape((teams, agents, self.width))
        hidden = hidden * present[..., None].astype(hidden.dtype)
        return cache, self._outputs(hidden)

    def _mix(self, states, actions, present, alive, training):
        present = present.astype(bool)
        alive = alive.astype(bool) & present
        states = self.sub("state_projection", nn.Linear, self.width, winit=self.winit)(
            nn.cast(sg(states))
        )
        dead_state = self.value(
            "dead_state", nn.init("trunc_normal"), (self.width,), f32
        )
        states = jnp.where(alive[..., None], states, nn.cast(dead_state))
        alive_token = self.sub(
            "alive_projection", nn.Linear, self.width, winit=self.winit
        )(nn.cast(alive[..., None].astype(f32)))
        onehot = jax.nn.one_hot(
            actions.astype(jnp.int32) - self.action_low,
            self.action_count,
            dtype=f32,
        )
        action = self.sub("action_projection", nn.Linear, self.width, winit=self.winit)(
            nn.cast(onehot)
        )
        tokens = self.sub("input_norm", nn.Norm, self.norm)(
            states + action + alive_token
        )
        tokens = nn.dropout(tokens, self.dropout, training)
        tokens = self.sub(
            "agent_interaction",
            AgentInteraction,
            width=self.width,
            heads=self.heads,
            layers=self.agent_layers,
            ffup=self.ffup,
            dropout=self.dropout,
            act=self.act,
            norm=self.norm,
            winit=self.winit,
        )(tokens, present, training)
        return tokens, action

    def _outputs(self, hidden):
        prediction = self.sub(
            "embedding_prediction", nn.Linear, self.target_dim, winit=self.winit
        )(hidden)
        return {
            "hidden": hidden,
            "embedding": prediction,
        }

    def _temporal(self):
        return self.sub(
            "temporal",
            CausalTransformer,
            self.width,
            units=self.width,
            output=self.width,
            layers=self.temporal_layers,
            heads=self.heads,
            context=self.context,
            ffup=self.ffup,
            act=self.act,
            norm=self.norm,
            winit=self.winit,
            condition_mode=self.action_conditioning,
        )


class CentralAttentionCritic(nj.Module):
    """Shared per-agent value distribution with training-only team attention."""

    width: int = 256
    heads: int = 4
    layers: int = 2
    ffup: int = 4
    dropout: float = 0.0
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"
    value_layers: int = 2
    value_units: int = 256
    bins: int = 255
    outscale: float = 0.0

    def __init__(self, **kwargs):
        self.scalar = elements.Space(jnp.float32, ())
        del kwargs

    def __call__(self, local_states, present, alive, bdims):
        if (
            local_states.ndim < 3
            or present.shape != local_states.shape[:-1]
            or alive.shape != local_states.shape[:-1]
        ):
            raise ValueError(
                "central critic expects [...,A,F] states plus [...,A] roster and "
                f"liveness masks, got {local_states.shape}, {present.shape}, and "
                f"{alive.shape}"
            )
        present = present.astype(bool)
        alive = alive.astype(bool) & present
        value = self.sub("state_projection", nn.Linear, self.width, winit=self.winit)(
            nn.cast(sg(local_states))
        )
        dead_state = self.value(
            "dead_state", nn.init("trunc_normal"), (self.width,), f32
        )
        value = jnp.where(alive[..., None], value, nn.cast(dead_state))
        value += self.sub("alive_projection", nn.Linear, self.width, winit=self.winit)(
            nn.cast(alive[..., None].astype(f32))
        )
        value = self.sub(
            "agent_attention",
            AgentInteraction,
            width=self.width,
            heads=self.heads,
            layers=self.layers,
            ffup=self.ffup,
            dropout=self.dropout,
            act=self.act,
            norm=self.norm,
            winit=self.winit,
        )(value, present, training=False)
        return self.sub(
            "head",
            embodied.jax.MLPHead,
            self.scalar,
            layers=self.value_layers,
            units=self.value_units,
            act=self.act,
            norm=self.norm,
            output="symexp_twohot",
            outscale=self.outscale,
            winit=self.winit,
            bins=self.bins,
        )(value, bdims=bdims)


__all__ = ["CentralAttentionCritic", "JointObservationJEPA"]
