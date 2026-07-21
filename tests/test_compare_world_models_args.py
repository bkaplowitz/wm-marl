from __future__ import annotations

import sys

from world_marl.scripts import compare_world_models


def test_wide_rollout_aliases_parse(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_world_models",
            "--flow-types",
            "transformer",
            "--fit-steps",
            "100000",
            "--chunk-steps",
            "5000",
            "--heldout-seeds",
            "1",
            "--rollout-envs",
            "6000",
        ],
    )

    args = compare_world_models.parse_args()

    assert args.flow_types == ["transformer"]
    assert args.fit_steps == 100000
    assert args.chunk_steps == 5000
    assert args.heldout_random_rollouts == 1
    assert args.heldout_initial_rollouts == 1
    assert args.num_envs == 6000
