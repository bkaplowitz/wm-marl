"""Machine-checkable contract for the maintained DreaMARL architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import DreaMARLRunSpec


def verify_run_contract(spec: "DreaMARLRunSpec") -> dict[str, object]:
    """Describe the selected first-party temporal backend and shared algorithm."""

    return {
        "contract_version": 26,
        "marl_architecture": "joint-action-conditioned local JEPA",
        "num_agents": spec.num_agents,
        "foundation": ("danijar/dreamerv3@e3f02248693a79dc8b0ebd62c93683888ddaccfe"),
        "world_model_contract": spec._training_state,
        "world_state_axis": ("one categorical latent and temporal state per agent"),
        "world_action_conditioning": (
            "each local transition receives its own action and a "
            "permutation-equivariant summary of peer latent-action tokens"
        ),
        "parameter_sharing": "one parameter tree shared over the folded agent axis",
        "agent_axis_adapter": "[B,T,A,...] <-> [B*A,T,...]",
        "agent_axis_metadata": (
            "agent_present, agent_alive, and optional discrete action_mask"
        ),
        "policy_information": "one agent's local predicted latent feature",
        "policy_peer_access": False,
        "policy_state_supervision": "the locked local latent and JEPA objectives",
        "critic_information": "one agent's local predicted latent feature",
        "imagination": "synchronous joint-action-conditioned local rollouts",
        "imagination_atomicity": (
            "all decentralized actions are sampled before one joint-conditioned step"
        ),
        "actor_imagination_horizon": 15,
        "temporal_context": 64,
        "training_replay_sampling": "uniform sequences",
        "training_replay_state": (
            "one local Transformer history plus its peer interaction input"
        ),
        "world_model_objective": "embedding",
        "embedding_target": "ema",
        "embedding_loss": "cosine",
        "posterior_jepa": True,
        "dynamics_jepa": True,
        "spatial_jepa": True,
        "spatial_mask_ratio": 0.5,
        "spatial_mask_topology": "fixed_count",
        "visual_encoder": "simple",
        "sigreg": True,
        "sigreg_aggregation": "pooled",
        "decoder_role": "absent; targets are fixed EMA encoder representations",
        "world_reward_prediction": "one local reward from each local latent state",
        "continuation_prediction": "one continuation from each local latent state",
        "execution": "decentralized",
        "single_agent_status": "same local outputs, losses, gradients, and updates",
        "resolved_task": spec.task,
    }
