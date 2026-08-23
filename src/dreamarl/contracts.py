"""Machine-checkable contract for the maintained DreaMARL architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import DreaMARLRunSpec


def verify_run_contract(spec: "DreaMARLRunSpec") -> dict[str, object]:
    """Describe the selected first-party temporal backend and shared algorithm."""

    return {
        "contract_version": 46,
        "marl_stage": spec.marl_stage,
        "marl_stage_status": spec._marl_stage_status,
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
            "agent_present, agent_alive, controllable_alive, and optional "
            "discrete action_mask"
        ),
        "policy_information": "one observation-local latent history per agent",
        "policy_peer_access": False,
        "policy_state_supervision": "the shared local latent and JEPA objectives",
        "critic_information": (
            "synchronized stopped local posterior states through agent attention"
            if spec.marl_stage == "ctde" and spec.num_agents > 1
            else (
                "focal local latent plus a stopped-gradient JEPA-derived active-team belief"
                if spec.marl_stage == "b2" and spec.num_agents > 1
                else "one observation-local latent history per agent"
            )
        ),
        "imagination": (
            "training-only joint-JEPA local-observation prediction completed by the "
            "unchanged local posterior"
            if spec.marl_stage == "ctde" and spec.num_agents > 1
            else "synchronized independent local rollouts"
        ),
        "imagination_atomicity": (
            "team starts remain grouped; actors sample independently, then the joint "
            "simulator consumes the synchronized action set"
            if spec.marl_stage == "ctde" and spec.num_agents > 1
            else "team starts remain grouped while every transition uses only its own action"
        ),
        "actor_imagination_horizon": 15,
        "optimizer_updates_per_environment_step": spec.train_ratio / (16 * 64),
        "behavior_objective": spec.behavior_objective,
        "optimizer_topology": spec.behavior_optimizer,
        "ppo_epochs": spec.ppo_epochs if spec.behavior_objective == "ppo" else 0,
        "ppo_clip": spec.ppo_clip if spec.behavior_objective == "ppo" else None,
        "repval_grad": spec.repval_grad,
        "temporal_context": 64,
        "training_replay_sampling": (
            "all training sequences weighted by 0.9998^age"
            if spec.replay_sampling == "recent"
            else "uniform sequences"
        ),
        "training_replay_state": (
            "128-step loss-excluded reconstruction of local and joint Transformer "
            "history with absolute RoPE position"
            if spec.marl_stage == "ctde" and spec.num_agents > 1
            else "128-step loss-excluded reconstruction of local Transformer "
            "history with absolute RoPE position"
        ),
        "world_model_objective": "embedding",
        "embedding_target": "ema",
        "embedding_loss": "cosine",
        "posterior_jepa": True,
        "dynamics_jepa": True,
        "spatial_jepa": not spec.task.startswith("smac_"),
        "spatial_mask_ratio": 0.5,
        "spatial_mask_topology": "fixed_count",
        "visual_encoder": "simple",
        "sigreg": True,
        "sigreg_aggregation": "per_agent",
        "decoder_role": "absent; targets are fixed EMA encoder representations",
        "world_reward_prediction": (
            "joint simulator reward from synchronized local states and actions"
            if spec.marl_stage == "ctde" and spec.num_agents > 1
            else "one local reward from each local latent state"
        ),
        "continuation_prediction": (
            "joint simulator continuation used authoritatively in imagination"
            if spec.marl_stage == "ctde" and spec.num_agents > 1
            else "one continuation from each local latent state"
        ),
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
            if spec.marl_stage in {"b1", "b2"} and spec.num_agents > 1
            else "disabled"
        ),
        "agent_jepa_local_model_gradient": (
            "stopped"
            if spec.marl_stage in {"b1", "b2"} and spec.num_agents > 1
            else "disabled"
        ),
        "agent_jepa_future_horizon": (
            1
            if spec.marl_stage in {"b1", "b2"}
            and spec.num_agents > 1
            and spec.agent_jepa_future_scale > 0.0
            else 0
        ),
        "agent_jepa_future_action_conditioning": (
            "aligned per-agent joint replay action, training only"
            if spec.marl_stage in {"b1", "b2"}
            and spec.num_agents > 1
            and spec.agent_jepa_future_scale > 0.0
            else "disabled"
        ),
        "team_teacher": (
            "training-only EMA set encoder over all active local EMA embeddings"
            if spec.marl_stage in {"b1", "b2"} and spec.num_agents > 1
            else "disabled"
        ),
        "team_teacher_execution_access": False,
        "central_critic": (
            spec.marl_stage in {"b2", "ctde"} and spec.num_agents > 1
        ),
        "central_critic_team_belief": (
            "eight 256-wide slots from predicted EMA local embeddings and local "
            "causal histories; identically reconstructed in replay and imagination"
            if spec.marl_stage == "b2" and spec.num_agents > 1
            else "disabled"
        ),
        "central_critic_gradient_boundary": (
            "critic gradients stop at the team belief and local world state"
            if spec.marl_stage == "b2" and spec.num_agents > 1
            else (
                "critic gradients stop at every synchronized local posterior state"
                if spec.marl_stage == "ctde" and spec.num_agents > 1
                else "not applicable"
            )
        ),
        "ctde_enabled": spec.marl_stage == "ctde" and spec.num_agents > 1,
        "ctde": spec._ctde_manifest if spec.marl_stage == "ctde" else "disabled",
        "ctde_execution_boundary": (
            "policy synchronization contains enc, dyn, and pol only; joint simulator "
            "and centralized critic are absent from collection and evaluation"
            if spec.marl_stage == "ctde" and spec.num_agents > 1
            else "not applicable"
        ),
        "ctde_joint_target": (
            "next stopped EMA local observation embedding plus stopped online "
            "posterior-interface alignment"
            if spec.marl_stage == "ctde" and spec.num_agents > 1
            else "not applicable"
        ),
        "ctde_actor_state": (
            "existing local temporal proposal completed by the existing local "
            "posterior from the joint model's predicted local observation embedding"
            if spec.marl_stage == "ctde" and spec.num_agents > 1
            else "not applicable"
        ),
        "ctde_critic_action_leakage": False,
        "jecc_enabled": spec.marl_stage == "jecc" and spec.num_agents > 1,
        "jecc": spec._jecc_manifest if spec.marl_stage == "jecc" else "disabled",
        "jecc_base_model_authority": (
            "unchanged B0 local world model, actor, critic, and imagination; the "
            "B0 advance-prior-complete transition is the one-step intervention "
            "operator and receives no JECC gradients"
            if spec.marl_stage == "jecc" and spec.num_agents > 1
            else "not applicable"
        ),
        "jecc_optimizer_ownership": (
            "outcome encoder, outcome predictor, and outcome utility only; the EMA "
            "outcome teacher and every B0 module are excluded"
            if spec.marl_stage == "jecc" and spec.num_agents > 1
            else "not applicable"
        ),
        "jecc_gradient_boundary": (
            "counterfactual credit enters the local actor only through a stopped "
            "score-function advantage"
            if spec.marl_stage == "jecc" and spec.num_agents > 1
            else "not applicable"
        ),
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
        "smac_protocol": (
            "SMAC-v1, SC2 4.10, difficulty 7, continuing episodes, shared team "
            "reward; training reward unchanged; legacy and corrected combat outcomes "
            "logged separately"
            if spec.task.startswith("smac_")
            else "not applicable"
        ),
        "resolved_task": spec.task,
    }
