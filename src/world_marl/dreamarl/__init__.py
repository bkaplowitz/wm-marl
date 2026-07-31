"""DreaMARL: coherent multi-agent JEPA world-model reinforcement learning."""

from world_marl.dreamarl.contracts import (
    JaxMultiAgentSequenceBatch,
    MultiAgentSequenceBatch,
    sequence_batch_to_jax,
    stack_agent_actions,
)

__all__ = [
    "JaxMultiAgentSequenceBatch",
    "MultiAgentSequenceBatch",
    "sequence_batch_to_jax",
    "stack_agent_actions",
]
