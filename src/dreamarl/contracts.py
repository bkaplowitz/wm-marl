"""Machine-checkable contract for the maintained DreaMARL architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import DreaMARLRunSpec


def verify_run_contract(spec: "DreaMARLRunSpec") -> dict[str, object]:
    """Describe the selected first-party temporal backend and shared algorithm."""

    return {
        "contract_version": 37,
        "marl_stage": spec.marl_stage,
        "marl_architecture": spec._marl_architecture,
        "num_agents": spec.num_agents,
        "foundation": ("danijar/dreamerv3@e3f02248693a79dc8b0ebd62c93683888ddaccfe"),
        "world_model_contract": spec._training_state,
        "world_state_axis": ("one categorical latent and temporal state per agent"),
        "world_action_conditioning": (
            "each local transition receives only its own previous latent and action"
        ),
        "parameter_sharing": "one parameter tree shared over the folded agent axis",
        "agent_axis_adapter": "[B,T,A,...] <-> [B*A,T,...]",
        "agent_axis_metadata": (
            "agent_present, agent_alive, and optional discrete action_mask"
        ),
        "policy_information": "one observation-local latent history per agent",
        "policy_peer_access": False,
        "policy_state_supervision": "the shared local latent and JEPA objectives",
        "critic_information": "one observation-local latent history per agent",
        "imagination": "synchronized independent local rollouts",
        "imagination_atomicity": (
            "team starts remain grouped while every transition uses only its own action"
        ),
        "actor_imagination_horizon": 15,
        "temporal_context": 64,
        "training_replay_sampling": "uniform sequences",
        "training_replay_state": (
            "128-step loss-excluded reconstruction of local Transformer history "
            "with absolute RoPE position"
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
        "sigreg_aggregation": "per_agent",
        "decoder_role": "absent; targets are fixed EMA encoder representations",
        "world_reward_prediction": "one local reward from each local latent state",
        "continuation_prediction": "one continuation from each local latent state",
        "execution": "strict decentralized parameter-shared actors",
        "agent_axis_jepa": (
            (
                "whole-agent-masked prediction of complete fixed-width EMA team "
                "slots at the current timestep plus joint-action-conditioned "
                "one-step future team-slot prediction, with agent-relative "
                "balanced slot-to-agent matching, hidden-agent coverage, and "
                "slot anti-collapse regularization"
                if spec.agent_jepa_future_scale > 0.0
                else "detached whole-agent-masked prediction of current complete "
                "fixed-width EMA team slots"
            )
            if spec.marl_stage == "b1" and spec.num_agents > 1
            else "disabled"
        ),
        "agent_jepa_local_model_gradient": (
            "stopped"
            if spec.marl_stage == "b1" and spec.num_agents > 1
            else "disabled"
        ),
        "agent_jepa_future_horizon": (
            1
            if spec.marl_stage == "b1"
            and spec.num_agents > 1
            and spec.agent_jepa_future_scale > 0.0
            else 0
        ),
        "agent_jepa_future_action_conditioning": (
            "aligned per-agent joint replay action, training only"
            if spec.marl_stage == "b1"
            and spec.num_agents > 1
            and spec.agent_jepa_future_scale > 0.0
            else "disabled"
        ),
        "team_teacher": (
            "training-only EMA set encoder over all active local EMA embeddings"
            if spec.marl_stage == "b1" and spec.num_agents > 1
            else "disabled"
        ),
        "team_teacher_execution_access": False,
        "single_agent_status": "same local outputs, losses, gradients, and updates",
        "environment_seed_mode": (
            "construction-time Lab2D seed stream; Shimmy reset(seed) is ignored"
            if spec.task.startswith("meltingpot_")
            else "suite construction seed"
        ),
        "environment_reproducibility": (
            "construction_seed_controlled_not_trajectory_deterministic"
            if spec.task.startswith("meltingpot_")
            else "suite_seed_semantics"
        ),
        "meltingpot_reward_mode": (
            "collective" if spec.task.startswith("meltingpot_") else "native"
        ),
        "resolved_task": spec.task,
    }
