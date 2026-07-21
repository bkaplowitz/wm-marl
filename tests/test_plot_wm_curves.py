"""Curve-extraction tests for plot_wm_curves."""

from world_marl.scripts.plot_wm_curves import (
    arm_aggregate,
    jepa_curve,
    sorted_arm_summaries,
)


def test_arm_aggregate_mean_band_and_final_over_seeds():
    records = [
        {"points": [(0.0, -500.0), (100.0, -100.0)]},
        {"points": [(0.0, -300.0), (100.0, -50.0)]},
    ]
    curve, final_mean = arm_aggregate(records)
    assert curve[0] == {"x": 0.0, "mean": -400.0, "low": -500.0, "high": -300.0}
    assert curve[1] == {"x": 100.0, "mean": -75.0, "low": -100.0, "high": -50.0}
    assert final_mean == -75.0


def test_sorted_arm_summaries_orders_by_final_return():
    arms = {
        "model-free": [{"points": [(0.0, -200.0), (100.0, -120.0)]}],
        "jepa": [{"points": [(0.0, -500.0), (100.0, -36.0)]}],
        "llada2": [{"points": [(0.0, -220.0), (100.0, -81.0)]}],
    }
    summaries = sorted_arm_summaries(arms)
    assert [name for name, _, _ in summaries] == ["jepa", "llada2", "model-free"]
    assert summaries[0][2] == -36.0


def test_jepa_curve_reads_budgets_nested_under_args():
    config = {
        "action_mode": "continuous",
        "args": {"num_envs": 16, "collect_steps": 7000, "online_collect_steps": 3000},
    }
    outcome = {
        "policy_initial_mean": -500.0,
        "online_policy_champion_returns": [-100.0],
        "final_policy_eval_mean": -50.0,
    }
    curve = jepa_curve(outcome, config)
    assert curve == [(0.0, -500.0), (160000.0, -100.0), (160000.0, -50.0)]
