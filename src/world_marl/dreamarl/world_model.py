"""Stochastic joint-action-conditioned JEPA world model for DreaMARL."""

from __future__ import annotations

import math
from typing import NamedTuple

from flax import linen as nn
import jax
import jax.numpy as jnp

from world_marl.dreamarl.config import DreaMARLConfig, EncoderConfig
from world_marl.dreamarl.contracts import JaxMultiAgentSequenceBatch
from world_marl.dreamarl.distributions import two_hot_mean
from world_marl.dreamarl.temporal import (
    CausalKVTransformer,
    TemporalCache,
    TemporalLayer,
)


def _logit(probability: float) -> float:
    return math.log(probability) - math.log1p(-probability)


class WorldState(NamedTuple):
    """One coherent latent state for all agents in a batch."""

    temporal: TemporalCache
    hidden: jax.Array
    stochastic: jax.Array
    agent_alive: jax.Array
    action_mask: jax.Array


class TransitionPrediction(NamedTuple):
    """Predictions and temporal input produced by one complete joint action."""

    pair: jax.Array
    context: jax.Array
    next_embedding: jax.Array
    team_reward_logits: jax.Array
    team_reward: jax.Array
    agent_reward_logits: jax.Array
    agent_reward: jax.Array
    continuation_logit: jax.Array
    next_alive_logit: jax.Array
    next_action_mask_logit: jax.Array


class SequencePrediction(NamedTuple):
    prior_logits: jax.Array
    posterior_logits: jax.Array
    stochastic: jax.Array
    hidden: jax.Array
    next_embedding: jax.Array
    team_reward_logits: jax.Array
    team_reward: jax.Array
    agent_reward_logits: jax.Array
    agent_reward: jax.Array
    continuation_logit: jax.Array
    next_alive_logit: jax.Array
    next_action_mask_logit: jax.Array


class SequenceInference(NamedTuple):
    final_state: WorldState
    predictions: SequencePrediction


class SharedObservationEncoder(nn.Module):
    """One parameter-shared encoder for every active agent."""

    config: EncoderConfig

    @nn.compact
    def __call__(self, observations: jax.Array) -> jax.Array:
        cfg = self.config
        leading = observations.shape[:-1]
        if cfg.kind == "vector":
            x = observations.astype(jnp.float32)
            x = x.reshape((*leading, -1))
            for index in range(cfg.vector_layers):
                x = nn.Dense(cfg.vector_hidden_dim, name=f"dense_{index}")(x)
                x = nn.silu(nn.RMSNorm(name=f"norm_{index}")(x))
        elif cfg.kind == "image":
            if observations.ndim < 4:
                raise ValueError("image observations must end in [height,width,channel]")
            image_leading = observations.shape[:-3]
            x = observations.astype(jnp.float32)
            if jnp.issubdtype(observations.dtype, jnp.integer):
                x = x / 255.0
            x = x.reshape((-1, *observations.shape[-3:]))
            depth = cfg.cnn_depth
            for index in range(cfg.cnn_blocks):
                x = nn.Conv(
                    depth * (2**index),
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding="SAME",
                    name=f"conv_{index}",
                )(x)
                x = nn.silu(nn.GroupNorm(name=f"conv_norm_{index}")(x))
            x = x.reshape((*image_leading, -1))
        else:  # pragma: no cover - validated by EncoderConfig typing.
            raise ValueError(f"unsupported observation kind {cfg.kind!r}")
        x = nn.Dense(cfg.embedding_dim, name="embedding")(x)
        return nn.RMSNorm(name="embedding_norm")(x)


class CrossAgentTransition(nn.Module):
    """Masked interaction block conditioned on the complete joint action."""

    config: DreaMARLConfig

    def setup(self) -> None:
        cfg = self.config
        dyn = cfg.dynamics
        self.input_projection = nn.Dense(dyn.model_dim, name="input_projection")
        self.layers = tuple(
            TemporalLayer(
                dyn.model_dim,
                dyn.cross_agent_heads,
                dyn.mlp_ratio,
                name=f"cross_layer_{index}",
            )
            for index in range(dyn.cross_agent_layers)
        )
        self.output_norm = nn.RMSNorm(name="output_norm")
        if dyn.use_agent_identity:
            self.agent_identity = self.param(
                "agent_identity",
                nn.initializers.normal(0.02),
                (cfg.max_agents, dyn.model_dim),
            )
        else:
            self.agent_identity = None

    def __call__(
        self,
        beliefs: jax.Array,
        actions: jax.Array,
        agent_alive: jax.Array,
    ) -> jax.Array:
        cfg = self.config
        if beliefs.shape[:2] != actions.shape or actions.shape != agent_alive.shape:
            raise ValueError("belief, action, and alive axes must agree")
        if beliefs.shape[1] != cfg.max_agents:
            raise ValueError("agent axis must equal configured max_agents")
        action = jax.nn.one_hot(actions, cfg.action_dim, dtype=jnp.float32)
        x = self.input_projection(jnp.concatenate([beliefs, action], axis=-1))
        if self.agent_identity is not None:
            x = x + self.agent_identity[None]
        x = jnp.where(agent_alive[..., None], x, 0.0)
        mask = (
            agent_alive[:, None, :, None]
            & agent_alive[:, None, None, :]
        )
        for layer in self.layers:
            x = layer(x, mask)
            x = jnp.where(agent_alive[..., None], x, 0.0)
        return jnp.where(
            agent_alive[..., None], self.output_norm(x), 0.0
        )


class DreaMARLWorldModel(nn.Module):
    """Final DreaMARL world-model core, independent of baseline repositories."""

    config: DreaMARLConfig

    def setup(self) -> None:
        cfg = self.config
        dyn = cfg.dynamics
        self.encoder = SharedObservationEncoder(cfg.encoder, name="encoder")
        self.temporal = CausalKVTransformer(
            pair_dim=cfg.temporal_pair_dim,
            model_dim=dyn.model_dim,
            num_layers=dyn.num_layers,
            num_heads=dyn.num_heads,
            mlp_ratio=dyn.mlp_ratio,
            context_length=dyn.context_length,
            name="temporal",
        )
        self.cross_agent = CrossAgentTransition(cfg, name="cross_agent")
        self.prior_hidden = nn.Dense(dyn.model_dim, name="prior_hidden")
        self.prior_out = nn.Dense(cfg.stochastic_dim, name="prior_out")
        self.posterior_hidden = nn.Dense(dyn.model_dim, name="posterior_hidden")
        self.posterior_out = nn.Dense(cfg.stochastic_dim, name="posterior_out")
        self.embedding_predictor = nn.Sequential(
            [
                nn.Dense(dyn.model_dim, name="embedding_hidden"),
                nn.silu,
                nn.Dense(cfg.encoder.embedding_dim, name="embedding_out"),
            ],
            name="embedding_predictor",
        )
        self.agent_reward_head = nn.Dense(
            cfg.distribution.bins,
            kernel_init=nn.initializers.zeros_init(),
            name="agent_reward",
        )
        self.team_reward_head = nn.Dense(
            cfg.distribution.bins,
            kernel_init=nn.initializers.zeros_init(),
            name="team_reward",
        )
        self.continuation_head = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.constant(
                _logit(cfg.dynamics.initial_continuation)
            ),
            name="continuation",
        )
        self.agent_alive_head = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.constant(
                _logit(cfg.dynamics.initial_agent_alive)
            ),
            name="agent_alive",
        )
        self.action_mask_head = nn.Dense(
            cfg.action_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.constant(
                _logit(cfg.dynamics.initial_action_legal)
            ),
            name="action_mask",
        )

    def __call__(
        self,
        batch: JaxMultiAgentSequenceBatch,
        key: jax.Array,
    ) -> SequenceInference:
        return self.observe_sequence(batch, key)

    def encode(self, observations: jax.Array) -> jax.Array:
        return self.encoder(observations)

    def initial(self, batch_size: int) -> WorldState:
        cfg = self.config
        dyn = cfg.dynamics
        return WorldState(
            temporal=self.temporal.initial(batch_size * cfg.max_agents),
            hidden=jnp.zeros(
                (batch_size, cfg.max_agents, dyn.model_dim), jnp.float32
            ),
            stochastic=jnp.zeros(
                (
                    batch_size,
                    cfg.max_agents,
                    dyn.stochastic_variables,
                    dyn.stochastic_classes,
                ),
                jnp.float32,
            ),
            agent_alive=jnp.ones((batch_size, cfg.max_agents), bool),
            action_mask=jnp.ones(
                (batch_size, cfg.max_agents, cfg.action_dim), bool
            ),
        )

    def posterior_logits(
        self, hidden: jax.Array, embedding: jax.Array
    ) -> jax.Array:
        cfg = self.config
        dyn = cfg.dynamics
        x = jnp.concatenate([hidden, embedding], axis=-1)
        x = nn.silu(self.posterior_hidden(x))
        return self.posterior_out(x).reshape(
            (*hidden.shape[:-1], dyn.stochastic_variables, dyn.stochastic_classes)
        )

    def prior_logits(self, hidden: jax.Array) -> jax.Array:
        dyn = self.config.dynamics
        x = nn.silu(self.prior_hidden(hidden))
        return self.prior_out(x).reshape(
            (*hidden.shape[:-1], dyn.stochastic_variables, dyn.stochastic_classes)
        )

    def sample_stochastic(self, logits: jax.Array, key: jax.Array) -> jax.Array:
        classes = self.config.dynamics.stochastic_classes
        probs = jax.nn.softmax(logits, axis=-1)
        probs = (1.0 - self.config.dynamics.unimix) * probs + (
            self.config.dynamics.unimix / classes
        )
        sample = jax.random.categorical(key, jnp.log(probs), axis=-1)
        hard = jax.nn.one_hot(sample, classes, dtype=probs.dtype)
        return probs + jax.lax.stop_gradient(hard - probs)

    def belief(self, state: WorldState) -> jax.Array:
        stochastic = state.stochastic.reshape(
            (*state.stochastic.shape[:2], self.config.stochastic_dim)
        )
        return jnp.concatenate([state.hidden, stochastic], axis=-1)

    def infer(
        self,
        temporal: TemporalCache,
        previous_pair: jax.Array,
        observations: jax.Array,
        is_first: jax.Array,
        agent_alive: jax.Array,
        action_mask: jax.Array,
        key: jax.Array,
    ) -> tuple[WorldState, jax.Array, jax.Array]:
        cfg = self.config
        batch = observations.shape[0]
        temporal, hidden = self.temporal.step(
            temporal,
            previous_pair.reshape((batch * cfg.max_agents, -1)),
            jnp.repeat(is_first, cfg.max_agents),
        )
        hidden = hidden.reshape((batch, cfg.max_agents, -1))
        embedding = self.encode(observations)
        prior = self.prior_logits(hidden)
        posterior = self.posterior_logits(hidden, embedding)
        stochastic = self.sample_stochastic(posterior, key)
        stochastic = jnp.where(agent_alive[..., None, None], stochastic, 0.0)
        if action_mask.shape[-1] == 0:
            action_mask = jnp.ones(
                (*agent_alive.shape, cfg.action_dim), dtype=bool
            )
        state = WorldState(
            temporal=temporal,
            hidden=jnp.where(agent_alive[..., None], hidden, 0.0),
            stochastic=stochastic,
            agent_alive=agent_alive,
            action_mask=action_mask,
        )
        return state, prior, posterior

    def transition(
        self, state: WorldState, actions: jax.Array
    ) -> TransitionPrediction:
        cfg = self.config
        belief = self.belief(state)
        context = self.cross_agent(belief, actions, state.agent_alive)
        action = jax.nn.one_hot(actions, cfg.action_dim, dtype=jnp.float32)
        pair = jnp.concatenate(
            [
                state.stochastic.reshape(
                    (*state.stochastic.shape[:2], cfg.stochastic_dim)
                ),
                action,
                context,
            ],
            axis=-1,
        )
        pair = jnp.where(state.agent_alive[..., None], pair, 0.0)
        active = state.agent_alive.astype(jnp.float32)
        denominator = jnp.maximum(active.sum(axis=1, keepdims=True), 1.0)
        pooled = (context * active[..., None]).sum(axis=1) / denominator
        global_context = jnp.broadcast_to(
            pooled[:, None], context.shape
        )
        slot_identity = jnp.broadcast_to(
            jnp.eye(cfg.max_agents, dtype=context.dtype)[None],
            (*context.shape[:2], cfg.max_agents),
        )
        lifecycle_context = jnp.concatenate(
            [context, global_context, slot_identity], axis=-1
        )
        team_reward_logits = self.team_reward_head(pooled)
        agent_reward_logits = self.agent_reward_head(context)
        return TransitionPrediction(
            pair=pair,
            context=context,
            next_embedding=self.embedding_predictor(context),
            team_reward_logits=team_reward_logits,
            team_reward=two_hot_mean(
                team_reward_logits, cfg.distribution
            ),
            agent_reward_logits=agent_reward_logits,
            agent_reward=two_hot_mean(
                agent_reward_logits, cfg.distribution
            ),
            continuation_logit=self.continuation_head(pooled)[..., 0],
            next_alive_logit=self.agent_alive_head(lifecycle_context)[..., 0],
            next_action_mask_logit=self.action_mask_head(lifecycle_context),
        )

    def imagine_step(
        self,
        state: WorldState,
        actions: jax.Array,
        key: jax.Array,
    ) -> tuple[WorldState, TransitionPrediction]:
        cfg = self.config
        prediction = self.transition(state, actions)
        batch = actions.shape[0]
        temporal, hidden = self.temporal.step(
            state.temporal,
            prediction.pair.reshape((batch * cfg.max_agents, -1)),
            jnp.zeros((batch * cfg.max_agents,), bool),
        )
        hidden = hidden.reshape((batch, cfg.max_agents, -1))
        stochastic = self.sample_stochastic(self.prior_logits(hidden), key)
        next_alive = jax.nn.sigmoid(prediction.next_alive_logit) >= 0.5
        predicted_mask = prediction.next_action_mask_logit >= 0.0
        has_legal = jnp.any(predicted_mask, axis=-1, keepdims=True)
        predicted_mask = jnp.where(has_legal, predicted_mask, True)
        predicted_mask = predicted_mask & next_alive[..., None]
        next_state = WorldState(
            temporal=temporal,
            hidden=jnp.where(next_alive[..., None], hidden, 0.0),
            stochastic=jnp.where(next_alive[..., None, None], stochastic, 0.0),
            agent_alive=next_alive,
            action_mask=predicted_mask,
        )
        return next_state, prediction

    def observe_sequence(
        self,
        batch: JaxMultiAgentSequenceBatch,
        key: jax.Array,
    ) -> SequenceInference:
        cfg = self.config
        batch_size = batch.observations.shape[1]
        initial = self.initial(batch_size)
        zero_pair = jnp.zeros(
            (batch_size, cfg.max_agents, cfg.temporal_pair_dim), jnp.float32
        )
        keys = jax.random.split(key, batch.observations.shape[0])

        def step(carry, inputs):
            previous_state, previous_pair = carry
            observations, actions, first, alive, mask, sample_key = inputs
            state, prior, posterior = self.infer(
                previous_state.temporal,
                previous_pair,
                observations,
                first,
                alive,
                mask,
                sample_key,
            )
            prediction = self.transition(state, actions)
            outputs = SequencePrediction(
                prior_logits=prior,
                posterior_logits=posterior,
                stochastic=state.stochastic,
                hidden=state.hidden,
                next_embedding=prediction.next_embedding,
                team_reward_logits=prediction.team_reward_logits,
                team_reward=prediction.team_reward,
                agent_reward_logits=prediction.agent_reward_logits,
                agent_reward=prediction.agent_reward,
                continuation_logit=prediction.continuation_logit,
                next_alive_logit=prediction.next_alive_logit,
                next_action_mask_logit=prediction.next_action_mask_logit,
            )
            return (state, prediction.pair), outputs

        inputs = (
            batch.observations,
            batch.actions,
            batch.is_first,
            batch.agent_alive,
            batch.action_mask,
            keys,
        )
        # Materialize all parameterized submodules before entering lax.scan;
        # Flax parameter creation is an initialization side effect and cannot
        # occur for the first time inside a JAX control-flow trace.
        carry, first_prediction = step(
            (initial, zero_pair),
            jax.tree.map(lambda value: value[0], inputs),
        )
        (final_state, _), remaining = jax.lax.scan(
            step,
            carry,
            jax.tree.map(lambda value: value[1:], inputs),
        )
        predictions = jax.tree.map(
            lambda first, rest: jnp.concatenate([first[None], rest], axis=0),
            first_prediction,
            remaining,
        )
        return SequenceInference(final_state=final_state, predictions=predictions)


def categorical_probabilities(logits: jax.Array, unimix: float) -> jax.Array:
    """Return numerically stable categorical probabilities with uniform mix."""

    classes = logits.shape[-1]
    probs = jax.nn.softmax(logits, axis=-1)
    return (1.0 - unimix) * probs + unimix / classes
