"""Executable foundation gate for the DreaMARL multi-agent contract."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from world_marl.dreamarl.collect import collect_coin_game_sequence
from world_marl.dreamarl.contracts import sequence_batch_to_jax


def verify_foundation(
    *,
    output: Path | None = None,
    time_steps: int = 32,
    num_envs: int = 8,
    max_cycles: int = 8,
    seed: int = 0,
) -> dict[str, object]:
    """Verify collection, lifecycle semantics, and on-device joint tensors."""

    batch = collect_coin_game_sequence(
        time_steps=time_steps,
        num_envs=num_envs,
        max_cycles=max_cycles,
        seed=seed,
    )
    jax_batch = sequence_batch_to_jax(batch)

    @jax.jit
    def summarize(values):
        alive_rewards = values.rewards * values.agent_alive
        return (
            alive_rewards.sum(),
            values.team_rewards.sum(),
            values.is_last.sum(),
            values.is_terminal.sum(),
        )

    reward_sum, team_reward_sum, last_count, terminal_count = summarize(jax_batch)
    if not jnp.allclose(reward_sum, team_reward_sum):
        raise AssertionError("team reward must equal the collected per-agent sum")
    expected_boundaries = time_steps // max_cycles * num_envs
    if int(last_count) != expected_boundaries:
        raise AssertionError(
            f"expected {expected_boundaries} time-limit cuts, got {int(last_count)}"
        )
    if int(terminal_count) != 0:
        raise AssertionError("CoinGame time-limit cuts must not be terminals")

    result = {
        "name": "DreaMARL foundation gate",
        "passed": True,
        "backend": jax.default_backend(),
        "time_steps": batch.time_steps,
        "num_envs": batch.num_envs,
        "num_agents": batch.num_agents,
        "agent_ids": list(batch.agent_ids),
        "observation_shape": list(batch.observations.shape),
        "action_shape": list(batch.actions.shape),
        "is_last_count": int(last_count),
        "is_terminal_count": int(terminal_count),
        "reward_sum": float(reward_sum),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
