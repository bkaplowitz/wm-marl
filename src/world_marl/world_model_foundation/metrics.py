from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

METRIC_KEYS = frozenset(
    {
        "reconstruction_loss",
        "observation_prediction_loss",
        "token_prediction_loss",
        "reward_loss",
        "continue_loss",
        "rollout_loss",
        "rollout_return",
        "real_env_return",
        "bridge_accuracy",
        "latent_action_usage",
    }
)


@partial(jax.jit, static_argnames=("target_episodes",))
def scan_episode_summaries(
    rewards: jax.Array,
    dones: jax.Array,
    *,
    target_episodes: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    dones = jnp.asarray(dones, dtype=bool)
    if rewards.ndim != 2 or dones.shape != rewards.shape:
        raise ValueError("scanned rewards and dones must have shape (time, env)")

    num_envs = rewards.shape[1]

    def step(carry, inputs):
        episode_returns, episode_lengths = carry
        step_rewards, step_dones = inputs
        episode_returns = episode_returns + step_rewards
        episode_lengths = episode_lengths + 1
        completed_returns = jnp.where(step_dones, episode_returns, 0.0)
        completed_lengths = jnp.where(step_dones, episode_lengths, 0)
        carry = (
            jnp.where(step_dones, 0.0, episode_returns),
            jnp.where(step_dones, 0, episode_lengths),
        )
        return carry, (completed_returns, completed_lengths)

    initial = (
        jnp.zeros((num_envs,), dtype=jnp.float32),
        jnp.zeros((num_envs,), dtype=jnp.int32),
    )
    _, (return_events, length_events) = jax.lax.scan(
        step,
        initial,
        (rewards, dones),
    )
    done_flat = dones.reshape((-1,))
    event_indices = jnp.nonzero(
        done_flat,
        size=target_episodes,
        fill_value=0,
    )[0]
    return (
        return_events.reshape((-1,))[event_indices],
        length_events.reshape((-1,))[event_indices],
        jnp.sum(done_flat, dtype=jnp.int32),
    )


def scanned_episode_metrics(
    rewards: Any,
    dones: Any,
    *,
    target_episodes: int,
    policy_source: str,
    arrival_aligned: bool = False,
) -> list[dict[str, float | str]]:
    episode_returns, episode_lengths, completed_count = jax.device_get(
        scan_episode_summaries(
            rewards,
            dones,
            target_episodes=target_episodes,
        )
    )
    if int(completed_count) < target_episodes:
        raise RuntimeError(
            f"scan completed {int(completed_count)} episodes, expected {target_episodes}"
        )
    return [
        {
            "episode": episode,
            "return": float(episode_returns[episode]),
            "length": float(
                max(
                    int(episode_lengths[episode]) - int(arrival_aligned),
                    0,
                )
            ),
            "policy_source": policy_source,
            "evaluation_execution": "jax_scan",
        }
        for episode in range(target_episodes)
    ]
