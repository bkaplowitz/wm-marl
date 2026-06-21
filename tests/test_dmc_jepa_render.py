from __future__ import annotations

import json

import numpy as np

from world_marl.scripts.render_dmc_jepa_policy import resolve_run_source, save_gif


def test_resolve_run_source_selects_best_main_run(tmp_path):
    experiment = tmp_path / "dmc_jepa_20260621T000000Z"
    best = experiment / "none" / "run_001"
    other = experiment / "none" / "run_000"
    control = experiment / "no-action-world-model" / "run_000"
    for run_dir in (best, other, control):
        (run_dir / "checkpoint").mkdir(parents=True)
        (run_dir / "checkpoint" / "checkpoint.msgpack").write_bytes(b"fake")

    summary = {
        "runs": [
            {
                "run_index": 0,
                "control": "none",
                "run_dir": str(other),
                "policy_trained_mean": 20.0,
            },
            {
                "run_index": 1,
                "control": "none",
                "run_dir": str(best),
                "policy_trained_mean": 40.0,
            },
            {
                "run_index": 0,
                "control": "no-action-world-model",
                "run_dir": str(control),
                "policy_trained_mean": 100.0,
            },
        ],
    }
    (experiment / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    source = resolve_run_source(experiment)

    assert source.run_dir == best
    assert source.checkpoint_dir == best / "checkpoint"
    assert source.control == "none"
    assert source.run_index == 1


def test_save_gif_writes_animation(tmp_path):
    frames = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.full((8, 8, 3), 255, dtype=np.uint8),
    ]
    output = tmp_path / "rollout.gif"

    save_gif(frames, output, fps=10)

    assert output.exists()
    assert output.stat().st_size > 0
