from __future__ import annotations

import json
from pathlib import Path

from world_marl.scripts import jafar_jasmine_quality


def test_quality_commands_use_three_fixed_seeds_and_equal_source_budgets(
    tmp_path,
) -> None:
    calibration = tmp_path / "expert.npz"
    args = jafar_jasmine_quality.parse_args(
        [
            "--expert-calibration",
            str(calibration),
            "--out-dir",
            str(tmp_path / "quality"),
            "--dry-run",
        ]
    )

    commands = jafar_jasmine_quality.build_commands(args)

    assert len(commands) == 6
    assert {(record.arm, record.seed) for record in commands} == {
        (arm, seed) for arm in ("jafar", "jasmine") for seed in (0, 1, 2)
    }
    for record in commands:
        assert record.command[0].endswith(f"world-marl-train-{record.arm}")
        assert _value(record.command, "--model-size") == "source"
        assert _value(record.command, "--env") == "playground-vision:CartpoleBalance"
        assert _value(record.command, "--image-size") == "64"
        assert _value(record.command, "--batch-size") == "48"
        assert _value(record.command, "--num-envs") == "48"
        assert _value(record.command, "--tokenizer-steps") == "1"
        assert _value(record.command, "--lam-steps") == "1"
        assert _value(record.command, "--dynamics-steps") == "1"
        assert _value(record.command, "--reward-continue-steps") == "1"
        assert _value(record.command, "--policy-train-steps") == "1"


def test_quality_dry_run_writes_reproducible_manifest(tmp_path) -> None:
    out_dir = tmp_path / "quality"
    result = jafar_jasmine_quality.main(
        [
            "--expert-calibration",
            str(tmp_path / "expert.npz"),
            "--out-dir",
            str(out_dir),
            "--dry-run",
        ]
    )

    assert result == 0
    manifest = json.loads((out_dir / "dry_run.json").read_text())
    assert manifest["environment"] == "playground-vision:CartpoleBalance"
    assert manifest["seeds"] == [0, 1, 2]
    assert manifest["warp_vision_runtime"]["playground"] == "0.2.0"
    assert manifest["warp_vision_runtime"]["jax"] == "0.4.36"
    assert manifest["warp_vision_runtime"]["flax"] == "0.10.4"
    assert len(manifest["runs"]) == 6
    assert all(run["budgets"] == manifest["equal_budgets"] for run in manifest["runs"])


def test_quality_summary_aggregates_required_metrics(tmp_path) -> None:
    run_dirs: list[Path] = []
    for arm_index, arm in enumerate(("jafar", "jasmine")):
        for seed in (0, 1, 2):
            run_dir = tmp_path / arm / f"seed-{seed}"
            run_dir.mkdir(parents=True)
            metrics = {
                "status": "ok",
                "random_return": float(seed),
                "learned_simulator_return": float(seed + 1),
                "bridged_real_return": float(seed + 2),
                "final_tokenizer_loss": 4.0,
                "final_lam_loss": 3.0,
                "final_dynamics_loss": 2.0,
                "final_reward_continue_loss": 1.0,
                "updates_per_second": 5.0 + arm_index,
                "jax_platform": "gpu",
            }
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {"model": arm, "seed": seed, "status": "ok", "metrics": metrics}
                )
            )
            (run_dir / "code_usage.json").write_text(
                json.dumps({"training_transition_counts": [1, 1, 1, 1, 1, 1]})
            )
            run_dirs.append(run_dir)

    summary = jafar_jasmine_quality.aggregate_runs(run_dirs)

    assert summary["status"] == "ok"
    assert summary["models"]["jafar"]["seeds"] == [0, 1, 2]
    assert summary["models"]["jasmine"]["metrics"]["bridged_real_return"]["mean"] == 3.0
    assert summary["models"]["jafar"]["code_coverage"] == [6, 6, 6]
    assert summary["models"]["jasmine"]["gpu_runs"] == 3


def _value(command: tuple[str, ...], option: str) -> str:
    return command[command.index(option) + 1]
