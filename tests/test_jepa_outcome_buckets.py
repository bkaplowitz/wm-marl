from __future__ import annotations

import numpy as np

from world_marl.scripts.diagnose_jepa_outcome_buckets import (
    average_ranks,
    bucket_contrast,
    evenly_spaced_context_steps,
    outcome_bucket,
    spearman,
)


def test_outcome_bucket_uses_explicit_failure_and_success_boundaries():
    kwargs = {"failure_threshold": 100.0, "success_threshold": 900.0}
    assert outcome_bucket(99.9, **kwargs) == "failure"
    assert outcome_bucket(100.0, **kwargs) == "intermediate"
    assert outcome_bucket(899.9, **kwargs) == "intermediate"
    assert outcome_bucket(900.0, **kwargs) == "success"


def test_even_context_steps_avoid_terminal_horizon():
    steps = evenly_spaced_context_steps(
        max_cycles=1000,
        context_window=8,
        max_horizon=8,
        count=4,
    )
    assert len(steps) == 4
    assert min(steps) >= 7
    assert max(steps) + 8 < 1000


def test_average_ranks_and_spearman_handle_ties():
    ranks = average_ranks(np.asarray([3.0, 1.0, 1.0, 2.0]))
    np.testing.assert_allclose(ranks, [3.0, 0.5, 0.5, 2.0])
    assert spearman(np.asarray([1.0, 2.0, 3.0]), np.asarray([3.0, 2.0, 1.0])) == -1.0


def test_bucket_contrast_reports_intermediate_minus_success():
    by_bucket = {
        "intermediate": {
            "by_horizon": {
                "4": {
                    "reward_step_mae_mean": 0.20,
                    "latent_cosine_mean": 0.80,
                }
            }
        },
        "success": {
            "by_horizon": {
                "4": {
                    "reward_step_mae_mean": 0.05,
                    "latent_cosine_mean": 0.95,
                }
            }
        },
    }
    contrast = bucket_contrast(by_bucket)["4"]
    assert np.isclose(contrast["reward_step_mae_mean"], 0.15)
    assert np.isclose(contrast["latent_cosine_mean"], -0.15)
    assert contrast["top1_regret_mean"] is None
