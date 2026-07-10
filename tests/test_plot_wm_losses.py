"""Extraction tests for plot_wm_losses (run-generated metrics.jsonl curves)."""

import json

from world_marl.scripts.plot_wm_losses import (
    build_loss_figure,
    genwm_loss_points,
    jepa_loss_points,
    loss_axis_scale,
)


def _write_metrics(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records))


def test_genwm_loss_points_cumulative_segments(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics(
        metrics,
        [
            {"note": "backfilled from console.log"},
            {"phase": "run 0 fit", "step": 1, "total": 4, "wm_loss": 10.0},
            {"phase": "run 0 fit", "step": 3, "total": 4, "wm_loss": 8.0},
            {
                "phase": "run 0 policy",
                "step": 1,
                "total": 3,
                "total_loss": 2.0,
                "entropy": 1.0,
            },
            {
                "phase": "run 0 policy",
                "step": 3,
                "total": 3,
                "total_loss": 1.0,
                "entropy": 0.5,
            },
            {"phase": "run 0 online 0 fit", "step": 1, "total": 2, "wm_loss": 6.0},
            {
                "phase": "run 0 online 0 policy",
                "step": 2,
                "total": 3,
                "total_loss": 0.5,
            },
            {"phase": "run 0 online 1 fit", "step": 2, "total": 2, "wm_loss": 5.0},
        ],
    )
    losses = genwm_loss_points(metrics)
    assert losses["wm_loss"] == [(1, 10.0), (3, 8.0), (5, 6.0), (8, 5.0)]
    assert losses["ppo_loss"] == [(1, 2.0), (3, 1.0), (5, 0.5)]


def test_genwm_loss_points_model_free_and_non_ppo_records(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics(
        metrics,
        [
            {
                "phase": "run 0 genie",
                "step": 1,
                "total": 5,
                "genie_total_loss": 4.0,
                "genie_recon_loss": 3.0,
            },
            {
                "phase": "run 0 model-free",
                "step": 6,
                "total": 54,
                "total_loss": 77.4,
                "entropy": 2.8,
            },
            {
                "phase": "run 0 model-free",
                "step": 11,
                "total": 54,
                "total_loss": 70.7,
                "entropy": 2.7,
            },
        ],
    )
    losses = genwm_loss_points(metrics)
    assert losses["ppo_loss"] == [(6, 77.4), (11, 70.7)]
    assert losses["wm_loss"] == []


def test_loss_axis_scale_log_when_all_positive():
    arms = {
        "a": [{"seed": "s0", "point": 1, "value": 0.05}],
        "b": [{"seed": "s0", "point": 1, "value": 80.0}],
    }
    scale, low, high = loss_axis_scale(arms)
    assert scale == "log"
    assert low <= 0.05
    assert high >= 80.0


def test_loss_axis_scale_symlog_when_nonpositive():
    arms = {
        "a": [
            {"seed": "s0", "point": 1, "value": -0.5},
            {"seed": "s0", "point": 2, "value": 3.0},
        ]
    }
    scale, low, high = loss_axis_scale(arms)
    assert scale == "symlog"
    assert low <= -0.5
    assert high >= 3.0


def test_build_loss_figure_shared_log_axes():
    import matplotlib.pyplot as plt

    arms = {
        "a": [
            {"seed": "s0", "point": 1, "value": 10.0},
            {"seed": "s0", "point": 2, "value": 0.05},
        ],
        "b": [
            {"seed": "s0", "point": 1, "value": 80.0},
            {"seed": "s0", "point": 2, "value": 0.5},
        ],
    }
    fig, axes = build_loss_figure(
        arms, env="brax_reacher", metric_kind="ppo_loss", jepa_metric="m"
    )
    try:
        assert {axis.get_yscale() for axis in axes} == {"log"}
        limits = {axis.get_ylim() for axis in axes}
        assert len(limits) == 1
        ((low, high),) = limits
        assert low <= 0.05
        assert high >= 80.0
    finally:
        plt.close(fig)


def test_jepa_loss_points_sequential_index(tmp_path):
    run_dir = tmp_path / "brax_jepa_x" / "none" / "run_000"
    run_dir.mkdir(parents=True)
    rows = [
        {"phase": "world_model", "model/total_loss": 3.0},
        {"phase": "policy_selection", "candidate": 1},
        {"phase": "online_candidate_refit_checkpoint", "model/total_loss": 2.0},
    ]
    (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    points = jepa_loss_points(tmp_path, metric="model/total_loss")
    assert points == [(1, 3.0), (2, 2.0)]
