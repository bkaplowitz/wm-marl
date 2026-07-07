"""Parsing tests for plot_wm_losses (console-log and metrics.jsonl loss curves)."""

import json

from world_marl.scripts.plot_wm_losses import (
    jepa_loss_points,
    parse_console_losses,
)


def test_parse_console_losses_cumulative_segments(tmp_path):
    console = tmp_path / "console.log"
    console.write_text(
        "\n".join(
            [
                "[genwm] fitting",
                "[run 0 fit] step 1/4 wm_loss=10.0",
                "[run 0 fit] step 3/4 wm_loss=8.0",
                "[run 0 policy] step 1/3 ppo_loss=2.0 entropy=1.0",
                "[run 0 policy] step 3/3 ppo_loss=1.0 entropy=0.5",
                "[run 0 online 0 fit] step 1/2 wm_loss=6.0",
                "[run 0 online 0 policy] step 2/3 ppo_loss=0.5 entropy=0.1",
                "[run 0 online 1 fit] step 2/2 wm_loss=5.0",
            ]
        )
    )
    losses = parse_console_losses(console)
    assert losses["wm_loss"] == [(1, 10.0), (3, 8.0), (5, 6.0), (8, 5.0)]
    assert losses["ppo_loss"] == [(1, 2.0), (3, 1.0), (5, 0.5)]


def test_parse_console_losses_model_free_updates(tmp_path):
    console = tmp_path / "console.log"
    console.write_text(
        "\n".join(
            [
                "[run 0 model-free] update 6/54 ppo_loss=77.4 entropy=2.8",
                "[run 0 model-free] update 11/54 ppo_loss=70.7 entropy=2.7",
            ]
        )
    )
    losses = parse_console_losses(console)
    assert losses["ppo_loss"] == [(6, 77.4), (11, 70.7)]
    assert losses["wm_loss"] == []


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
