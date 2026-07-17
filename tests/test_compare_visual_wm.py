from __future__ import annotations

import csv
import json
import subprocess

from world_marl.scripts import compare_visual_wm
from world_marl.scripts.compare_visual_wm import build_arm_command, main


def _summary(model: str, status: str, return_value: float) -> dict:
    return {
        "schema_version": 1,
        "model": model,
        "env": "synthetic:image-grid",
        "seed": 0,
        "status": status,
        "metrics": {
            "random_return": 0.25,
            "learned_simulator_return": return_value - 0.1,
            "bridged_real_return": return_value,
            "final_dynamics_loss": 1.5,
        },
    }


def test_compare_visual_wm_aggregates_jafar_and_jasmine_summaries(tmp_path) -> None:
    paths = []
    for model, status, return_value in (
        ("jafar", "ok", 0.5),
        ("jasmine", "learning_gate_failed", 0.75),
    ):
        path = tmp_path / model / "summary.json"
        path.parent.mkdir()
        path.write_text(json.dumps(_summary(model, status, return_value)))
        paths.append(path)

    exit_code = main(
        [
            "--summary",
            str(paths[0]),
            "--summary",
            str(paths[1]),
            "--out-dir",
            str(tmp_path / "comparison"),
        ]
    )

    assert exit_code == 0
    rows = json.loads((tmp_path / "comparison" / "comparison.json").read_text())
    assert [row["model"] for row in rows] == ["jafar", "jasmine"]
    assert rows[1]["bridged_real_return"] == 0.75
    with (tmp_path / "comparison" / "comparison.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["status"] == "ok"
    assert csv_rows[1]["bridged_real_return"] == "0.75"


def test_compare_visual_wm_builds_jafar_and_jasmine_cli_commands(tmp_path) -> None:
    calibration = tmp_path / "expert.npz"
    common = {
        "env": "playground-vision:CartpoleBalance",
        "collect_steps": 16,
        "num_envs": 2,
        "max_cycles": 100,
        "train_steps": 2,
        "policy_train_steps": 2,
        "eval_episodes": 1,
        "allow_fail": True,
        "expert_calibration": calibration,
        "seed": 3,
        "image_size": 64,
        "dmc_camera_id": 1,
        "dmc_workers": 2,
        "brax_backend": "mjx",
    }
    jafar = build_arm_command("jafar", out_dir=tmp_path / "jafar", **common)
    jasmine = build_arm_command("jasmine", out_dir=tmp_path / "jasmine", **common)

    assert jafar[:3] == ["uv", "run", "world-marl-train-jafar"]
    assert jasmine[:3] == ["uv", "run", "world-marl-train-jasmine"]
    for command in (jafar, jasmine):
        assert command[command.index("--expert-calibration") + 1] == str(calibration)
        assert command[command.index("--tokenizer-steps") + 1] == "2"
        assert command[command.index("--lam-steps") + 1] == "2"
        assert command[command.index("--dynamics-steps") + 1] == "2"


def test_compare_visual_wm_dispatches_both_source_arms(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run(command, check=False):
        del check
        calls.append(list(command))
        arm = "jafar" if "world-marl-train-jafar" in command else "jasmine"
        out_dir = tmp_path / "comparison" / arm
        out_dir.mkdir(parents=True)
        (out_dir / "summary.json").write_text(
            json.dumps(_summary(arm, "ok", 0.5 if arm == "jafar" else 0.75))
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(compare_visual_wm.subprocess, "run", fake_run)
    calibration = tmp_path / "expert.npz"
    exit_code = main(
        [
            "--arm",
            "jafar",
            "--arm",
            "jasmine",
            "--env",
            "synthetic:image-grid",
            "--expert-calibration",
            str(calibration),
            "--out-dir",
            str(tmp_path / "comparison"),
            "--train-steps",
            "2",
            "--policy-train-steps",
            "1",
            "--allow-fail",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 2
    rows = json.loads((tmp_path / "comparison" / "comparison.json").read_text())
    assert [row["model"] for row in rows] == ["jafar", "jasmine"]
