from __future__ import annotations

from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter
from world_marl.evaluation import constant_policy, evaluate_policy_host


def test_evaluation_loop_with_fixed_dummy_policy(dummy_env_factory):
    adapter = MeltingPotVectorAdapter(num_envs=1, env_factory=dummy_env_factory)
    try:
        result = evaluate_policy_host(adapter, constant_policy(action=1), episodes=2)
        assert result.episodes == 2
        assert result.mean_return_per_agent == 3.0
        assert result.returns.shape == (2, 2)
    finally:
        adapter.close()
