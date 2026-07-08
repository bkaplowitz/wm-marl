from __future__ import annotations

import csv
import json

from world_marl.scripts.compare_visual_wm import main


def test_compare_visual_wm_aggregates_summary_artifacts(tmp_path) -> None:
    dreamer = tmp_path / "dreamer" / "summary.json"
    genie = tmp_path / "genie2" / "summary.json"
    dreamer.parent.mkdir()
    genie.parent.mkdir()
    dreamer.write_text(
        json.dumps(
            {
                "model": "dreamer_v3_baseline",
                "status": "ok",
                "final_loss": 1.5,
                "learning_gate_passed": True,
            }
        )
    )
    genie.write_text(
        json.dumps(
            {
                "model": "genie2_continuous_jax",
                "status": "learning_gate_failed",
                "final_loss": 2.5,
                "learning_gate_passed": False,
            }
        )
    )

    exit_code = main(
        [
            "--summary",
            str(dreamer),
            "--summary",
            str(genie),
            "--out-dir",
            str(tmp_path / "comparison"),
        ]
    )

    assert exit_code == 0
    rows = json.loads((tmp_path / "comparison" / "comparison.json").read_text())
    assert [row["model"] for row in rows] == [
        "dreamer_v3_baseline",
        "genie2_continuous_jax",
    ]
    with (tmp_path / "comparison" / "comparison.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["status"] == "ok"
    assert csv_rows[1]["learning_gate_passed"] == "False"
