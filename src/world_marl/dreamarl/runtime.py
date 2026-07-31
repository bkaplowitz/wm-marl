"""Fused JAX collection runtime shared by training and evaluation."""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from world_marl.dreamarl.config import DreaMARLConfig
from world_marl.dreamarl.contracts import JaxMultiAgentSequenceBatch
from world_marl.dreamarl.control import SharedActor
from world_marl.dreamarl.environments import MultiAgentAdapter
from world_marl.dreamarl.world_model import DreaMARLWorldModel, WorldState


class DriverState(NamedTuple):
    env_state: Any
    observations: jax.Array
    agent_alive: jax.Array
    action_mask: jax.Array
    is_first: jax.Array
    episode_step: jax.Array
    episode_returns: jax.Array
    episode_length: jax.Array
    key: jax.Array


class PolicyContext(NamedTuple):
    world_state: WorldState
    previous_pair: jax.Array
    model_is_first: jax.Array


class CollectionMetrics(NamedTuple):
    completed: jax.Array
    episode_returns: jax.Array
    episode_lengths: jax.Array


class RandomCollectionOutput(NamedTuple):
    driver: DriverState
    transitions: JaxMultiAgentSequenceBatch
    metrics: CollectionMetrics


class PolicyCollectionOutput(NamedTuple):
    driver: DriverState
    context: PolicyContext
    transitions: JaxMultiAgentSequenceBatch
    metrics: CollectionMetrics


class DreaMARLRuntime:
    """Environment-neutral collection logic with a concrete adapter boundary."""

    def __init__(
        self,
        adapter: MultiAgentAdapter,
        config: DreaMARLConfig,
    ) -> None:
        if adapter.spec.num_agents != config.max_agents:
            raise ValueError("adapter agent count does not match model config")
        if adapter.spec.action_dim != config.action_dim:
            raise ValueError("adapter action count does not match model config")
        self.adapter = adapter
        self.config = config
        self.world_model = DreaMARLWorldModel(config)
        self.actor = SharedActor(config)

    @partial(jax.jit, static_argnums=(0, 1))
    def initialize_driver(self, num_envs: int, key: jax.Array) -> DriverState:
        reset_key, state_key = jax.random.split(key)
        reset = self.adapter.reset(jax.random.split(reset_key, num_envs))
        return DriverState(
            env_state=reset.env_state,
            observations=reset.observations,
            agent_alive=reset.agent_alive,
            action_mask=reset.action_mask,
            is_first=jnp.ones((num_envs,), bool),
            episode_step=jnp.zeros((num_envs,), jnp.int32),
            episode_returns=jnp.zeros(
                (num_envs, self.config.max_agents), jnp.float32
            ),
            episode_length=jnp.zeros((num_envs,), jnp.int32),
            key=state_key,
        )

    @partial(jax.jit, static_argnums=(0, 2))
    def collect_random(
        self,
        driver: DriverState,
        time_steps: int,
    ) -> RandomCollectionOutput:
        """Collect legal uniformly random joint actions in one scan."""

        def step(current: DriverState, _):
            next_key, action_key = jax.random.split(current.key)
            logits = jnp.where(current.action_mask, 0.0, -1e30)
            actions = jax.random.categorical(action_key, logits, axis=-1)
            next_driver, transition, metric = self._environment_step(
                current, actions, next_key
            )
            return next_driver, (transition, metric)

        driver, (transitions, metrics) = jax.lax.scan(
            step, driver, xs=None, length=time_steps
        )
        return RandomCollectionOutput(driver, transitions, metrics)

    @partial(jax.jit, static_argnums=0)
    def initialize_policy_context(
        self,
        driver: DriverState,
        world_model_params: Any,
    ) -> PolicyContext:
        batch = driver.observations.shape[0]
        return PolicyContext(
            world_state=self.world_model.apply(
                {"params": world_model_params},
                batch,
                method=self.world_model.initial,
            ),
            previous_pair=jnp.zeros(
                (
                    batch,
                    self.config.max_agents,
                    self.config.temporal_pair_dim,
                ),
                jnp.float32,
            ),
            model_is_first=jnp.ones((batch,), bool),
        )

    @partial(jax.jit, static_argnums=(0, 5, 6))
    def collect_policy(
        self,
        driver: DriverState,
        context: PolicyContext,
        world_model_params: Any,
        actor_params: Any,
        time_steps: int,
        deterministic: bool,
    ) -> PolicyCollectionOutput:
        """Collect one decentralized policy with recurrent world-state caches."""

        def step(carry, _):
            current, policy_context = carry
            next_key, infer_key, action_key = jax.random.split(current.key, 3)
            world_state, _, _ = self.world_model.apply(
                {"params": world_model_params},
                policy_context.world_state.temporal,
                policy_context.previous_pair,
                current.observations,
                policy_context.model_is_first,
                current.agent_alive,
                current.action_mask,
                infer_key,
                method=self.world_model.infer,
            )
            belief = self.world_model.apply(
                {"params": world_model_params},
                world_state,
                method=self.world_model.belief,
            )
            logits = self.actor.apply(
                {"params": actor_params},
                belief,
                current.agent_alive,
                current.action_mask,
            )
            sampled = jax.random.categorical(action_key, logits, axis=-1)
            actions = jnp.where(
                deterministic, jnp.argmax(logits, axis=-1), sampled
            )
            prediction = self.world_model.apply(
                {"params": world_model_params},
                world_state,
                actions,
                method=self.world_model.transition,
            )
            next_driver, transition, metric = self._environment_step(
                current, actions, next_key
            )
            next_context = PolicyContext(
                world_state=world_state,
                previous_pair=prediction.pair,
                model_is_first=next_driver.is_first,
            )
            return (next_driver, next_context), (transition, metric)

        (driver, context), (transitions, metrics) = jax.lax.scan(
            step, (driver, context), xs=None, length=time_steps
        )
        return PolicyCollectionOutput(driver, context, transitions, metrics)

    def _environment_step(
        self,
        current: DriverState,
        actions: jax.Array,
        next_key: jax.Array,
    ) -> tuple[
        DriverState,
        JaxMultiAgentSequenceBatch,
        CollectionMetrics,
    ]:
        num_envs = current.observations.shape[0]
        step_key, reset_key, state_key = jax.random.split(next_key, 3)
        transition = self.adapter.step(
            jax.random.split(step_key, num_envs),
            current.env_state,
            actions,
        )
        step_count = current.episode_step + 1
        time_limit = step_count >= self.adapter.spec.max_episode_steps
        is_last = transition.is_terminal | time_limit
        reset = self.adapter.reset(jax.random.split(reset_key, num_envs))
        next_observations = transition.next_observations
        next_returns = current.episode_returns + transition.rewards
        next_lengths = current.episode_length + 1
        metric = CollectionMetrics(
            completed=is_last,
            episode_returns=next_returns,
            episode_lengths=next_lengths,
        )
        driver = DriverState(
            env_state=_tree_select(
                is_last, reset.env_state, transition.env_state
            ),
            observations=_select(
                is_last, reset.observations, next_observations
            ),
            agent_alive=_select(
                is_last, reset.agent_alive, transition.next_agent_alive
            ),
            action_mask=_select(
                is_last, reset.action_mask, transition.next_action_mask
            ),
            is_first=is_last,
            episode_step=jnp.where(is_last, 0, step_count),
            episode_returns=jnp.where(
                is_last[:, None], 0.0, next_returns
            ),
            episode_length=jnp.where(is_last, 0, next_lengths),
            key=state_key,
        )
        record = JaxMultiAgentSequenceBatch(
            observations=current.observations,
            next_observations=next_observations,
            actions=actions,
            rewards=transition.rewards,
            team_rewards=_active_agent_mean(
                transition.rewards, current.agent_alive
            ),
            is_first=current.is_first,
            is_last=is_last,
            is_terminal=transition.is_terminal,
            valid=jnp.ones((num_envs,), bool),
            agent_alive=current.agent_alive,
            next_agent_alive=transition.next_agent_alive,
            action_mask=current.action_mask,
            next_action_mask=transition.next_action_mask,
        )
        return driver, record, metric


def _select(mask: jax.Array, on_true: jax.Array, on_false: jax.Array) -> jax.Array:
    expanded = mask.reshape((mask.shape[0], *([1] * (on_true.ndim - 1))))
    return jnp.where(expanded, on_true, on_false)


def _tree_select(mask: jax.Array, on_true: Any, on_false: Any) -> Any:
    return jax.tree.map(
        lambda true, false: _select(mask, true, false), on_true, on_false
    )


def _active_agent_mean(values: jax.Array, alive: jax.Array) -> jax.Array:
    weights = alive.astype(values.dtype)
    return jnp.sum(values * weights, axis=-1) / jnp.maximum(
        jnp.sum(weights, axis=-1), 1.0
    )
