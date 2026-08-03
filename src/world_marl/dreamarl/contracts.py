"""Machine-checkable contracts for the maintained DreaMARL architecture."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import DreaMARLRunSpec


def verify_run_contract(spec: "DreaMARLRunSpec") -> dict[str, object]:
    """Record the architectural invariants attached to every launch."""

    if spec.num_agents < 1:
        raise ValueError("num_agents must be positive")
    values = asdict(spec)
    return {
        "contract_version": 2,
        "num_agents": spec.num_agents,
        "policy_information": "focal observation, focal action history, local carry",
        "policy_peer_access": False,
        "world_state_axis": "one environment state with an explicit agent axis",
        "world_action_conditioning": "synchronous joint action",
        "imagined_actor_state": "joint prior directly predicts each local belief",
        "world_reward_prediction": "complete per-agent reward vector",
        "continuation_prediction": "one joint continuation",
        "critic_information": "joint latent state during centralized training",
        "imagination_atomicity": "all local actions sampled before one joint advance",
        "single_agent_status": "valid reduction, not a numerical parity constraint",
        "resolved_task": values["task"],
    }
