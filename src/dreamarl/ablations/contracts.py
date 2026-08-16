"""Machine-readable contracts for non-canonical DreaMARL controls."""

from __future__ import annotations

from dataclasses import asdict

from .config import AblationRunSpec


def verify_run_contract(spec: AblationRunSpec) -> dict[str, object]:
    values = asdict(spec)
    return {
        "contract_version": 14,
        "num_agents": spec.num_agents,
        "foundation": ("danijar/dreamerv3@e3f02248693a79dc8b0ebd62c93683888ddaccfe"),
        "world_model_contract": spec._training_state,
        "policy_peer_access": False,
        "execution": "decentralized",
        "world_model_objective": spec.world_model_objective,
        "embedding_target": spec.embedding_target,
        "embedding_loss": spec.embedding_loss,
        "posterior_jepa": spec.posterior_jepa,
        "dynamics_jepa": spec.dynamics_jepa,
        "spatial_jepa": spec.spatial_jepa,
        "spatial_mask_ratio": spec.spatial_mask_ratio,
        "spatial_mask_topology": spec.spatial_mask_topology,
        "visual_encoder": spec.visual_encoder,
        "sigreg": spec.sigreg,
        "sigreg_aggregation": spec.sigreg_aggregation,
        "decoder_role": (
            "observation reconstruction attribution control"
            if spec.world_model_objective == "reconstruction"
            else (
                "absent; visual targets are fixed EMA encoder representations"
                if spec.embedding_target == "ema"
                else "absent; visual targets are full-gradient online embeddings"
            )
        ),
        "single_agent_status": (
            "same architecture and schedule with an identity agent-axis fold"
        ),
        "resolved_task": values["task"],
    }
