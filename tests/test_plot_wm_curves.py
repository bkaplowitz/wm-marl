"""Curve-extraction tests for plot_wm_curves."""

from world_marl.scripts.plot_wm_curves import jepa_curve


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
