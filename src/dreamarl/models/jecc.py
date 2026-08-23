"""Training-only joint-embedding models for counterfactual credit.

JECC keeps the executable B0 actor and world model local. The modules in this
file operate on grouped replay or imagination tensors with an explicit agent
axis. B0's stopped, action-conditioned next features ground each intervention;
JECC combines them with the stopped current features and factual joint action
without learning a duplicate transition model. The modules contain no agent
identities or agent-position embeddings, so their outputs are equivariant to a
simultaneous permutation of members and focal queries.
"""

import math

import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np


f32 = jnp.float32
sg = jax.lax.stop_gradient


def _masked_softmax(logits, valid):
    """FP32 softmax over valid keys, including the all-masked case."""

    valid = valid.astype(bool)
    weights = jax.nn.softmax(jnp.where(valid, f32(logits), -1e30), axis=-1)
    weights = weights * valid.astype(f32)
    return weights / jnp.maximum(weights.sum(axis=-1, keepdims=True), 1e-8)


def _unit_length(value):
    value = f32(value)
    return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-8)


def _expand_projected_agents(value, target_prefix):
    """Broadcast ``[...,A,D]`` over new aligned focal/intervention axes."""

    source_prefix = value.shape[:-2]
    target_prefix = tuple(target_prefix)
    if target_prefix[: len(source_prefix)] != source_prefix:
        raise ValueError(
            f"cannot align projected agent prefix {source_prefix} with {target_prefix}"
        )
    added = len(target_prefix) - len(source_prefix)
    if added < 0:
        raise ValueError(
            f"target prefix {target_prefix} is shorter than source {source_prefix}"
        )
    if not added:
        return value
    shape = (*source_prefix, *((1,) * added), *value.shape[-2:])
    return jnp.broadcast_to(value.reshape(shape), (*target_prefix, *value.shape[-2:]))


def _longest_compatible_prefix(*prefixes):
    """Return the longest prefix when every shorter prefix aligns with it."""

    prefixes = tuple(tuple(prefix) for prefix in prefixes)
    target = max(prefixes, key=len)
    for source in prefixes:
        if target[: len(source)] != source:
            raise ValueError(f"incompatible JECC prefixes {prefixes}")
    return target


def _expand_agent_mask(value, target_prefix, agents):
    """Broadcast [..., A] over new intervention axes after its prefix."""

    if value.shape[-1] != agents:
        raise ValueError(f"expected {agents} agent-mask entries, got {value.shape}")
    source_prefix = value.shape[:-1]
    target_prefix = tuple(target_prefix)
    if source_prefix == target_prefix:
        return value
    if target_prefix[: len(source_prefix)] != source_prefix:
        raise ValueError(
            f"cannot align agent-mask prefix {source_prefix} with {target_prefix}"
        )
    added = len(target_prefix) - len(source_prefix)
    shape = (*source_prefix, *((1,) * added), agents)
    return jnp.broadcast_to(value.reshape(shape), (*target_prefix, agents))


class _EquivariantTransformer(nj.Module):
    """Transformer over an unordered, validity-masked set of tokens."""

    width: int = 256
    heads: int = 4
    layers: int = 2
    ffup: int = 4
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"

    def __init__(self, **kwargs):
        if self.width < 1 or self.layers < 1 or self.heads < 1:
            raise ValueError("JECC Transformer dimensions must be positive")
        if self.width % self.heads:
            raise ValueError(
                f"JECC width {self.width} must be divisible by {self.heads} heads"
            )
        del kwargs

    def __call__(self, tokens, valid):
        if tokens.ndim < 2 or valid.shape != tokens.shape[:-1]:
            raise ValueError(
                "JECC Transformer expects tokens [...,N,D] and valid [...,N], "
                f"got {tokens.shape} and {valid.shape}"
            )
        if tokens.shape[-1] != self.width:
            raise ValueError(
                f"expected {self.width}-wide JECC tokens, got {tokens.shape[-1]}"
            )

        valid = valid.astype(bool)
        value = nn.cast(tokens)
        value = value * valid[..., None].astype(value.dtype)
        for index in range(self.layers):
            residual = value
            update = self.sub(f"attn{index}_norm", nn.Norm, self.norm)(value)
            query = self.sub(
                f"attn{index}_query", nn.Linear, self.width, winit=self.winit
            )(update)
            key = self.sub(f"attn{index}_key", nn.Linear, self.width, winit=self.winit)(
                update
            )
            item = self.sub(
                f"attn{index}_value", nn.Linear, self.width, winit=self.winit
            )(update)
            count = value.shape[-2]
            head_width = self.width // self.heads
            shape = (*value.shape[:-2], count, self.heads, head_width)
            query = query.reshape(shape)
            key = key.reshape(shape)
            item = item.reshape(shape)
            logits = jnp.einsum(
                "...qhd,...khd->...hqk", f32(query), f32(key)
            ) / math.sqrt(head_width)
            weights = _masked_softmax(logits, valid[..., None, None, :]).astype(
                item.dtype
            )
            update = jnp.einsum("...hqk,...khd->...qhd", weights, item)
            update = update.reshape((*value.shape[:-2], count, self.width))
            update = self.sub(
                f"attn{index}_out", nn.Linear, self.width, winit=self.winit
            )(update)
            value = (residual + update) * valid[..., None].astype(update.dtype)

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
            value = (residual + update) * valid[..., None].astype(update.dtype)

        value = self.sub("output_norm", nn.Norm, self.norm)(value)
        return value * valid[..., None].astype(value.dtype)


class OutcomeEncoder(nj.Module):
    """Encode one factual future-outcome window into a normalized embedding."""

    width: int = 128
    heads: int = 4
    layers: int = 2
    ffup: int = 4
    outcome_dim: int = 128
    horizons: tuple = (5, 15, 32)
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"

    def __init__(self, **kwargs):
        horizons = tuple(int(value) for value in self.horizons)
        if horizons != tuple(sorted(set(horizons))) or not horizons:
            raise ValueError("JECC horizons must be unique positive sorted values")
        if horizons[0] < 1 or self.outcome_dim < 1:
            raise ValueError("JECC outcome dimensions must be positive")
        del kwargs

    def __call__(self, tokens, valid, horizon):
        """Encode ``[...,H,4]`` tokens into ``[...,128]``.

        ``horizon`` is one configured Python integer. Separate calls for 5, 15,
        and 32 steps share all temporal Transformer parameters.
        """

        horizon = int(horizon)
        if horizon not in self.horizons:
            raise ValueError(f"unknown JECC horizon {horizon}: {self.horizons}")
        if tokens.ndim < 3 or tokens.shape[-2:] != (horizon, 4):
            raise ValueError(
                f"outcome tokens must be [...,{horizon},4], got {tokens.shape}"
            )
        if valid.shape != tokens.shape[:-1]:
            raise ValueError(
                f"outcome validity {valid.shape} does not match {tokens.shape}"
            )

        value = self.sub("input_projection", nn.Linear, self.width, winit=self.winit)(
            nn.cast(tokens)
        )
        offsets = self.value(
            "temporal_offsets",
            nn.init("trunc_normal"),
            (max(self.horizons), self.width),
            f32,
        )[:horizon]
        value = value + nn.cast(offsets)
        value = self.sub("input_norm", nn.Norm, self.norm)(value)

        horizon_queries = self.value(
            "horizon_queries",
            nn.init("trunc_normal"),
            (len(self.horizons), self.width),
            f32,
        )
        query = horizon_queries[self.horizons.index(horizon)]
        query = nn.cast(jnp.broadcast_to(query, (*tokens.shape[:-2], 1, self.width)))
        sequence = jnp.concatenate([query, value], axis=-2)
        sequence_valid = jnp.concatenate(
            [jnp.ones((*valid.shape[:-1], 1), bool), valid.astype(bool)], axis=-1
        )
        sequence = self.sub(
            "temporal_transformer",
            _EquivariantTransformer,
            width=self.width,
            heads=self.heads,
            layers=self.layers,
            ffup=self.ffup,
            act=self.act,
            norm=self.norm,
            winit=self.winit,
        )(sequence, sequence_valid)
        outcome = self.sub("output", nn.Linear, self.outcome_dim, winit=self.winit)(
            sequence[..., 0, :]
        )
        return _unit_length(outcome)


class OutcomePredictor(nj.Module):
    """Predict all configured outcome horizons for one focal agent.

    Focal identity is supplied as a content-aligned one-hot query, never as a
    fixed agent ID. Stopped current B0 features, the factual joint action,
    and stopped B0 features after advance/prior/complete are projected
    separately and fused with a learned normalized projection. Permuting all
    agent-aligned inputs together leaves the predicted outcome unchanged.
    """

    width: int = 256
    heads: int = 4
    layers: int = 2
    ffup: int = 4
    outcome_dim: int = 128
    horizons: tuple = (5, 15, 32)
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"

    def __init__(self, action_count, action_low=0, **kwargs):
        self.action_count = int(action_count)
        self.action_low = int(action_low)
        horizons = tuple(int(value) for value in self.horizons)
        if horizons != tuple(sorted(set(horizons))) or not horizons:
            raise ValueError("JECC horizons must be unique positive sorted values")
        if horizons[0] < 1 or self.outcome_dim < 1 or self.action_count < 2:
            raise ValueError("JECC outcome dimensions and action count must be valid")
        del kwargs

    def project_current_states(self, current_states):
        """Project stopped current B0 features ``[...,A,S]`` once."""

        if current_states.ndim < 2:
            raise ValueError(
                f"JECC current states must be [...,A,S], got {current_states.shape}"
            )
        return self.sub(
            "current_state_projection", nn.Linear, self.width, winit=self.winit
        )(nn.cast(sg(current_states)))

    def project_actions(self, actions):
        """Project the factual joint action ``[...,A]``."""

        if actions.ndim < 1:
            raise ValueError(f"JECC actions must be [...,A], got {actions.shape}")
        onehot = jax.nn.one_hot(
            actions.astype(jnp.int32) - self.action_low,
            self.action_count,
            dtype=f32,
        )
        return self.sub("action_projection", nn.Linear, self.width, winit=self.winit)(
            nn.cast(onehot)
        )

    def project_next_states(self, next_states):
        """Project stopped B0 next features ``[...,A,S]`` once."""

        if next_states.ndim < 2:
            raise ValueError(
                f"JECC next states must be [...,A,S], got {next_states.shape}"
            )
        return self.sub(
            "next_state_projection", nn.Linear, self.width, winit=self.winit
        )(nn.cast(sg(next_states)))

    def __call__(self, current_states, actions, next_states, focal, active):
        """Return normalized predictions ``[...,3,128]``."""

        return self.predict_projected(
            self.project_current_states(current_states),
            self.project_actions(actions),
            self.project_next_states(next_states),
            focal,
            active,
        )

    def predict_projected(
        self,
        projected_current_states,
        projected_actions,
        projected_next_states,
        focal,
        active,
    ):
        """Predict from reusable projected states, actions, and next states.

        Base current features and factual actions may omit focal and
        alternative-action axes, while candidate next features may include
        both. Every shorter compatible prefix is broadcast to the longest
        prefix here.
        """

        for name, value in (
            ("current states", projected_current_states),
            ("actions", projected_actions),
            ("next states", projected_next_states),
        ):
            if value.ndim < 2 or value.shape[-1] != self.width:
                raise ValueError(
                    f"projected JECC {name} must be [...,A,{self.width}], got "
                    f"{value.shape}"
                )
        if (
            projected_current_states.shape[-2] != projected_next_states.shape[-2]
            or projected_actions.shape[-2] != projected_next_states.shape[-2]
        ):
            raise ValueError(
                "projected JECC current states, actions, and next states must "
                "have the same agent count"
            )
        agents = projected_next_states.shape[-2]
        if focal.ndim < 1 or focal.shape[-1] != agents:
            raise ValueError(f"JECC focal mask must end in {agents}, got {focal.shape}")
        if active.ndim < 1 or active.shape[-1] != agents:
            raise ValueError(
                f"JECC active mask must end in {agents}, got {active.shape}"
            )
        target_prefix = _longest_compatible_prefix(
            projected_current_states.shape[:-2],
            projected_actions.shape[:-2],
            projected_next_states.shape[:-2],
            focal.shape[:-1],
            active.shape[:-1],
        )
        current_states = _expand_projected_agents(
            projected_current_states, target_prefix
        )
        actions = _expand_projected_agents(projected_actions, target_prefix)
        next_states = _expand_projected_agents(projected_next_states, target_prefix)
        active = _expand_agent_mask(active, target_prefix, agents).astype(bool)
        focal = _expand_agent_mask(focal, target_prefix, agents).astype(f32)
        focal = focal * active.astype(f32)

        members = jnp.concatenate([current_states, actions, next_states], axis=-1)
        members = self.sub("member_fusion", nn.Linear, self.width, winit=self.winit)(
            members
        )
        members = self.sub("member_norm", nn.Norm, self.norm)(members)
        members = members * active[..., None].astype(members.dtype)
        focal_weight = focal / jnp.maximum(focal.sum(axis=-1, keepdims=True), 1.0)
        focal_weight = focal_weight.astype(members.dtype)
        query_members = self.sub("focal_norm", nn.Norm, self.norm)(current_states)
        query_members = query_members * active[..., None].astype(query_members.dtype)
        focal_content = (query_members * focal_weight[..., None]).sum(axis=-2)

        horizon_queries = self.value(
            "horizon_queries",
            nn.init("trunc_normal"),
            (len(self.horizons), self.width),
            f32,
        )
        horizon_queries = nn.cast(
            jnp.broadcast_to(
                horizon_queries,
                (*target_prefix, len(self.horizons), self.width),
            )
        )
        queries = focal_content[..., None, :] + horizon_queries
        tokens = jnp.concatenate([queries, members], axis=-2)
        valid = jnp.concatenate(
            [
                jnp.ones((*target_prefix, len(self.horizons)), bool),
                active,
            ],
            axis=-1,
        )
        tokens = self.sub(
            "outcome_transformer",
            _EquivariantTransformer,
            width=self.width,
            heads=self.heads,
            layers=self.layers,
            ffup=self.ffup,
            act=self.act,
            norm=self.norm,
            winit=self.winit,
        )(tokens, valid)
        outcome = self.sub("output", nn.Linear, self.outcome_dim, winit=self.winit)(
            tokens[..., : len(self.horizons), :]
        )
        return _unit_length(outcome)


class OutcomeUtility(nj.Module):
    """Shared distributional decoder from outcome embeddings to return."""

    layers: int = 2
    units: int = 256
    bins: int = 255
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"
    outscale: float = 0.0

    def __init__(self, **kwargs):
        if self.layers < 1 or self.units < 1 or self.bins < 2:
            raise ValueError("JECC utility head dimensions must be positive")
        self.scalar = elements.Space(np.float32, ())
        del kwargs

    def __call__(self, outcomes, bdims=None):
        """Return a symexp-twohot distribution over ``outcomes[...,128]``."""

        if outcomes.ndim < 2:
            raise ValueError(
                f"JECC utility outcomes must be [...,D], got {outcomes.shape}"
            )
        bdims = outcomes.ndim - 1 if bdims is None else int(bdims)
        head = self.sub(
            "head",
            embodied.jax.MLPHead,
            self.scalar,
            layers=self.layers,
            units=self.units,
            act=self.act,
            norm=self.norm,
            output="symexp_twohot",
            outscale=self.outscale,
            winit=self.winit,
            bins=self.bins,
        )
        return head(nn.cast(_unit_length(outcomes)), bdims)


__all__ = [
    "OutcomeEncoder",
    "OutcomePredictor",
    "OutcomeUtility",
]
