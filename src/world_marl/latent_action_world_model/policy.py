"""Existing CNN PPO over decoded latent-world-model simulator pixels.

This module intentionally delegates action selection, GAE, and optimization to
``world_marl.algs.ippo``. It adds only a ``jax.lax.scan`` simulator collector;
the categorical six-action policy and ``ppo_update`` implementation are not
forked or modified.
"""

from collections.abc import Callable
from typing import Any

from flax.training.train_state import TrainState
import jax

from world_marl.algs.ippo import (
    IPPOConfig,
    RolloutBatch,
    apply_actor_critic,
    ppo_update,
    select_actions,
)


def collect_simulator_rollout(
    policy_state: TrainState,
    initial_state: Any,
    simulator_step: Callable,
    rng: jax.Array,
    *,
    horizon: int,
) -> tuple[Any, RolloutBatch, jax.Array]:
    step_rngs = jax.random.split(rng, horizon)

    def step(simulator_state, step_rng):
        action_rng, model_rng = jax.random.split(step_rng)
        observations = simulator_state.pixels[:, -1]
        actions, log_probs, values = select_actions(
            policy_state,
            action_rng,
            observations,
        )
        next_state, rewards, dones, _ = simulator_step(
            simulator_state,
            actions,
            model_rng,
        )
        transition = {
            "observations": observations,
            "actions": actions,
            "log_probs": log_probs,
            "rewards": rewards,
            "dones": dones.astype(rewards.dtype),
            "values": values,
        }
        return next_state, transition

    final_state, transitions = jax.lax.scan(step, initial_state, step_rngs)
    rollout = RolloutBatch(**transitions)
    _, last_values = apply_actor_critic(policy_state, final_state.pixels[:, -1])
    return final_state, rollout, last_values


def simulator_ppo_update(
    policy_state: TrainState,
    initial_state: Any,
    simulator_step: Callable,
    rng: jax.Array,
    *,
    horizon: int,
    config: IPPOConfig,
) -> tuple[TrainState, Any, RolloutBatch, dict[str, jax.Array]]:
    rollout_rng, update_rng = jax.random.split(rng)
    final_state, rollout, last_values = collect_simulator_rollout(
        policy_state,
        initial_state,
        simulator_step,
        rollout_rng,
        horizon=horizon,
    )
    policy_state, metrics = ppo_update(
        policy_state,
        rollout,
        last_values,
        update_rng,
        config,
    )
    return policy_state, final_state, rollout, metrics


def scan_simulator_ppo_updates(
    policy_state: TrainState,
    initial_state: Any,
    simulator_step: Callable,
    rng: jax.Array,
    *,
    updates: int,
    horizon: int,
    config: IPPOConfig,
) -> tuple[TrainState, Any, RolloutBatch, dict[str, jax.Array]]:
    update_rngs = jax.random.split(rng, updates)

    def update(carry, update_rng):
        current_policy, simulator_state = carry
        current_policy, simulator_state, rollout, metrics = simulator_ppo_update(
            current_policy,
            simulator_state,
            simulator_step,
            update_rng,
            horizon=horizon,
            config=config,
        )
        return (current_policy, simulator_state), (rollout, metrics)

    (policy_state, final_state), (rollouts, metrics) = jax.lax.scan(
        update,
        (policy_state, initial_state),
        update_rngs,
    )
    return policy_state, final_state, rollouts, metrics
