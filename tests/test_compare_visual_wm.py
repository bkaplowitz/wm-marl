from __future__ import annotations

import csv
import json
import subprocess

from world_marl.scripts import compare_visual_wm
from world_marl.scripts.compare_visual_wm import build_arm_command, main


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


def test_compare_visual_wm_builds_real_cli_commands(tmp_path) -> None:
    dreamer = build_arm_command(
        "dreamer_v3_baseline",
        env="brax:reacher",
        out_dir=tmp_path / "dreamer",
        collect_steps=4,
        num_envs=2,
        max_cycles=4,
        train_steps=2,
        policy_train_steps=2,
        eval_episodes=1,
        allow_fail=True,
        seed=3,
        image_size=32,
        dmc_camera_id=1,
        dmc_workers=2,
        brax_backend="mjx",
    )
    genie2 = build_arm_command(
        "genie2_continuous_jax",
        env="brax:reacher",
        out_dir=tmp_path / "genie2",
        collect_steps=4,
        num_envs=2,
        max_cycles=4,
        train_steps=2,
        policy_train_steps=2,
        eval_episodes=1,
        allow_fail=True,
        seed=3,
        image_size=32,
        dmc_camera_id=1,
        dmc_workers=2,
        brax_backend="mjx",
    )

    assert dreamer[:3] == ["uv", "run", "world-marl-train-dreamer-v3-baseline"]
    assert genie2[:3] == ["uv", "run", "world-marl-train-genie2-continuous-jax"]
    assert "genie-like-jax" not in " ".join(dreamer + genie2)
    assert "codex/" not in " ".join(dreamer + genie2)
    assert dreamer[dreamer.index("--seed") + 1] == "3"
    assert dreamer[dreamer.index("--image-size") + 1] == "32"
    assert dreamer[dreamer.index("--dmc-camera-id") + 1] == "1"
    assert dreamer[dreamer.index("--dmc-workers") + 1] == "2"
    assert dreamer[dreamer.index("--brax-backend") + 1] == "mjx"


def test_compare_visual_wm_dispatches_real_cli_outputs(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run(command, check=False):
        del check
        calls.append(list(command))
        arm = "dreamer_v3_baseline"
        if "world-marl-train-genie2-continuous-jax" in command:
            arm = "genie2_continuous_jax"
        out_dir = tmp_path / "comparison" / arm
        out_dir.mkdir(parents=True)
        (out_dir / "summary.json").write_text(
            json.dumps(
                {
                    "model": arm,
                    "env": "brax:reacher",
                    "status": "ok",
                    "final_loss": 1.0,
                    "learning_gate_passed": True,
                    "real_env_return": -0.5 if arm == "dreamer_v3_baseline" else None,
                    "real_env_bridged_return": (
                        -0.25 if arm == "genie2_continuous_jax" else None
                    ),
                }
            )
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(compare_visual_wm.subprocess, "run", fake_run)

    exit_code = main(
        [
            "--arm",
            "dreamer_v3_baseline",
            "--arm",
            "genie2_continuous_jax",
            "--env",
            "brax:reacher",
            "--out-dir",
            str(tmp_path / "comparison"),
            "--collect-steps",
            "4",
            "--num-envs",
            "2",
            "--max-cycles",
            "4",
            "--train-steps",
            "2",
            "--policy-train-steps",
            "2",
            "--eval-episodes",
            "1",
            "--allow-fail",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 2
    rows = json.loads((tmp_path / "comparison" / "comparison.json").read_text())
    assert [row["model"] for row in rows] == [
        "dreamer_v3_baseline",
        "genie2_continuous_jax",
    ]
    assert rows[0]["env"] == "brax:reacher"
    assert rows[1]["real_env_bridged_return"] == -0.25
