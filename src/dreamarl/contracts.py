"""Machine-checkable contract for canonical DreaMARL runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import DreaMARLRunSpec


def verify_run_contract(spec: "DreaMARLRunSpec") -> dict[str, object]:
    """Describe the local execution boundary and selected training model."""

    return {
        "contract_version": 1,
        "algorithm": spec.algorithm,
        "num_agents": spec.num_agents,
        "foundation": "danijar/dreamerv3@e3f02248693a79dc8b0ebd62c93683888ddaccfe",
        "agent_axis": "[B,T,A,...] <-> [B*A,T,...]",
        "parameter_sharing": "one parameter tree shared over the folded agent axis",
        "world_model": {
            "objective": "EMA-target cosine joint-embedding prediction",
            "decoder": False,
            "local_transition_inputs": "own previous latent and action only",
            "posterior_context": "strict-causal local history",
            "replay_burn_in": 192,
            "sigreg_aggregation": "per_agent",
        },
        "execution": {
            "mode": "strict_decentralized",
            "policy_modules": spec.to_dict()["policy_modules"],
            "policy_peer_access": False,
            "runtime_communication": False,
        },
        "training": {
            "optimizer_topology": spec.optimizer_topology,
            "actor_objective": "score_function_reinforce",
            "replay_sampling": spec.effective_replay_sampling,
            "train_ratio": spec.effective_train_ratio,
        },
        "ctde": {
            **spec.ctde_manifest,
            "training_only": True,
            "joint_inputs": (
                "synchronized stopped local posterior states, aligned joint "
                "actions, roster/liveness masks, and replay-prefix history"
            ),
            "joint_target": "next stopped EMA local observation embedding",
            "actor_state": (
                "local temporal proposal completed by the local posterior from "
                "the joint model's predicted local observation embedding"
            ),
            "critic": (
                "permutation-equivariant attention over synchronized stopped "
                "local posterior states before the sampled joint action"
            ),
            "critic_action_leakage": False,
        },
        "evaluation": {
            **spec.to_dict()["evaluation_protocol"],
            "primary_smac_metric": "fixed deterministic battle win rate",
        },
        "resolved_task": spec.task,
    }
