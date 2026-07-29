from __future__ import annotations

from world_marl.scripts.run_jepatransformer_phase1_eval_lane import _command, main


def test_phase1_eval_commands_use_fixed_latest_policy_protocol(tmp_path):
    dreamer = _command("dreamerv3", tmp_path / "dreamer")
    ne = _command("nedreamer", tmp_path / "ne")
    for command in (dreamer, ne):
        assert command[command.index("--episodes") + 1] == "20"
        assert command[command.index("--eval-seed") + 1] == "10000"
    assert dreamer[dreamer.index("--envs") + 1] == "4"
    assert ne[ne.index("--device") + 1] == "cuda:0"


def test_phase1_eval_lane_can_select_one_implementation(tmp_path, monkeypatch):
    lane_pid = tmp_path / "lane.pid"
    lane_pid.write_text("99999999", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        "world_marl.scripts.run_jepatransformer_phase1_eval_lane._training_completed",
        lambda run_dir: calls.append(run_dir) or False,
    )

    assert main(
        [
            "--gpu",
            "0",
            "--seed",
            "0",
            "--lane-pid-file",
            str(lane_pid),
            "--output-root",
            str(tmp_path),
            "--implementations",
            "dreamerv3",
        ]
    ) == 1
    assert len(calls) == 2
    assert all("dreamerv3" in str(path) for path in calls)
