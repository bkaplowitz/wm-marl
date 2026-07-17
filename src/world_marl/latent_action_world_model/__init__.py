"""Repository integration for source-derived latent-action world models."""

from world_marl.latent_action_world_model.bridge import (
    ExpertActionBridge,
    load_expert_bridge,
    sample_real_actions,
)
from world_marl.latent_action_world_model.heads import (
    RewardContinueHeads,
    decode_reward,
    reward_continue_loss,
)
from world_marl.latent_action_world_model.replay import (
    BackendSequenceBatch,
    TransitionBatch,
    pair_valid_transitions,
    to_backend_sequence,
)
from world_marl.latent_action_world_model.simulator import (
    JafarSimulatorState,
    JasmineSimulatorState,
)

__all__ = [
    "BackendSequenceBatch",
    "ExpertActionBridge",
    "JafarSimulatorState",
    "JasmineSimulatorState",
    "RewardContinueHeads",
    "TransitionBatch",
    "decode_reward",
    "load_expert_bridge",
    "pair_valid_transitions",
    "reward_continue_loss",
    "sample_real_actions",
    "to_backend_sequence",
]
