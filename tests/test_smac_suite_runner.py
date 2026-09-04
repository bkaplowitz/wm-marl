"""Configuration gates for the fixed annealed SMAC campaign."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path


def _suite_module():
    path = Path(__file__).parents[1] / "scripts" / "run_annealed_smac_suite.py"
    module_spec = importlib.util.spec_from_file_location("annealed_suite", path)
    assert module_spec and module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def test_suite_contains_each_requested_map_and_seed_once() -> None:
    suite = _suite_module()
    runs = [run for queue in suite.SLOT_QUEUES.values() for run in queue]

    assert len(runs) == 30
    assert len({(run.map_name, run.seed) for run in runs}) == 30
    assert Counter(run.map_name for run in runs) == Counter(
        {name: 3 for name in suite.MAPS}
    )
    assert all(
        {run.seed for run in runs if run.map_name == name} == set(suite.SEEDS)
        for name in suite.MAPS
    )


def test_suite_uses_established_map_specific_imagination_horizons() -> None:
    suite = _suite_module()

    expected = {
        "3m": 15,
        "8m": 8,
        "MMM": 8,
        "2s_vs_1sc": 15,
        "3s_vs_4z": 15,
        "3s_vs_5z": 15,
        "2c_vs_64zg": 15,
        "5m_vs_6m": 15,
        "MMM2": 8,
        "corridor": 8,
    }

    assert {name: values[2] for name, values in suite.MAPS.items()} == expected


def test_suite_profiles_resolve_to_the_fixed_annealed_baseline() -> None:
    suite = _suite_module()

    for queue in suite.SLOT_QUEUES.values():
        for run in queue:
            resolved = suite.validate_profile(run)
            assert resolved["death_masking"]
            assert resolved["entropy_enabled"]
            assert resolved["entropy_decay_steps"] == int(0.8 * run.steps)
            assert resolved["actor_lr"] == 1e-5
            assert resolved["plan_aggregation"] == "mean"
            assert resolved["collection_unimix"] == 0.0
            assert resolved["policy_unimix"] == 0.0
            assert resolved["world_model_unimix"] == 0.01
