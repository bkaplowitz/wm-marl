"""Permutation-equivariant interaction primitives for DreaMARL.

The interaction model is centralized world-model infrastructure. Actor,
critic, posterior, and local temporal state never consume its messages.
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


def safe_masked_softmax(logits, valid, axis=-1):
    """Softmax that returns exact zeros when every entry is masked."""

    if valid.dtype != bool:
        raise ValueError(f"attention validity mask must be boolean, got {valid.dtype}")
    weights = jax.nn.softmax(jnp.where(valid, f32(logits), -1e30), axis=axis)
    weights = weights * valid
    denominator = weights.sum(axis=axis, keepdims=True)
    return weights / jnp.maximum(denominator, 1.0)


class AgentInteraction(nj.Module):
    """One shared self-excluded attention block over an explicit agent axis."""

    units: int = 128
    heads: int = 4
    norm: str = "rms"
    seed: int = 0

    def __init__(self, belief_dim, token_dim, **kw):
        if self.units % self.heads:
            raise ValueError((self.units, self.heads))
        self.belief_dim = int(belief_dim)
        self.token_dim = int(token_dim)
        self.kw = kw

    def __call__(self, belief, agent_token, valid_agents, *, shuffled=False):
        """Return messages and an exact ``has_other`` gate.

        Args:
            belief: ``[..., agent, belief_feature]`` focal queries.
            agent_token: ``[..., agent, token_feature]`` paired state/actions.
            valid_agents: ``[..., agent]`` validity mask.
            shuffled: Roll complete environment groups before building keys and
                values. The roll is constant across time for sequence inputs.
        """

        if belief.shape[:-1] != agent_token.shape[:-1]:
            raise ValueError((belief.shape, agent_token.shape))
        if belief.shape[-1] != self.belief_dim:
            raise ValueError((belief.shape, self.belief_dim))
        if agent_token.shape[-1] != self.token_dim:
            raise ValueError((agent_token.shape, self.token_dim))
        if valid_agents.shape != belief.shape[:-1]:
            raise ValueError((valid_agents.shape, belief.shape))

        context = jnp.roll(agent_token, 1, axis=0) if shuffled else agent_token
        context_valid = (
            jnp.roll(valid_agents, 1, axis=0) if shuffled else valid_agents
        )
        belief = nn.cast(belief)
        context = nn.cast(context)
        query = self.sub(
            "query", nn.Linear, self.units, winit=isolated_winit(self.seed + 1)
        )(belief)
        key = self.sub(
            "key", nn.Linear, self.units, winit=isolated_winit(self.seed + 2)
        )(context)
        value = self.sub(
            "value", nn.Linear, self.units, winit=isolated_winit(self.seed + 3)
        )(context)
        head_dim = self.units // self.heads
        shape = (*query.shape[:-1], self.heads, head_dim)
        query, key, value = [item.reshape(shape) for item in (query, key, value)]

        logits = jnp.einsum("...ihd,...jhd->...hij", query, key)
        logits = f32(logits) / jnp.sqrt(f32(head_dim))
        agents = belief.shape[-2]
        self_mask = ~jnp.eye(agents, dtype=bool)
        valid_other = (
            valid_agents[..., None]
            & context_valid[..., None, :]
            & self_mask
        )
        weights = safe_masked_softmax(logits, valid_other[..., None, :, :], -1)
        attended = jnp.einsum("...hij,...jhd->...ihd", weights, value)
        attended = nn.cast(attended.reshape((*belief.shape[:-1], self.units)))
        attended = self.sub("output_norm", nn.Norm, self.norm)(attended)
        message = self.sub(
            "output", nn.Linear, self.units, winit=isolated_winit(self.seed + 4)
        )(attended)
        has_other = valid_other.any(-1, keepdims=True)
        message = jnp.where(has_other, message, jnp.zeros_like(message))
        return nn.cast(message), has_other


class InteractionResidual(nj.Module):
    """Zero-initialized residual head with a mandatory post-projection gate."""

    hidden: int = 128
    act: str = "silu"
    norm: str = "rms"
    seed: int = 0

    def __init__(self, output_dim, **kw):
        self.output_dim = int(output_dim)
        self.kw = kw

    def __call__(self, local_state, message, has_other):
        if local_state.shape[:-1] != message.shape[:-1]:
            raise ValueError((local_state.shape, message.shape))
        if has_other.shape != (*local_state.shape[:-1], 1):
            raise ValueError((has_other.shape, local_state.shape))
        value = jnp.concatenate([nn.cast(local_state), nn.cast(message)], -1)
        value = self.sub(
            "hidden", nn.Linear, self.hidden, winit=isolated_winit(self.seed + 1)
        )(value)
        value = nn.act(self.act)(self.sub("norm", nn.Norm, self.norm)(value))
        value = self.sub(
            "output",
            nn.Linear,
            self.output_dim,
            winit=isolated_winit(self.seed + 2),
            outscale=0.0,
        )(value)
        return nn.cast(jnp.where(has_other, value, jnp.zeros_like(value)))
