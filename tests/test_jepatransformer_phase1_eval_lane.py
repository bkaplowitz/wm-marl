from __future__ import annotations

from world_marl.scripts.run_jepatransformer_phase1_eval_lane import _command


def test_phase1_eval_commands_use_fixed_latest_policy_protocol(tmp_path):
    dreamer = _command("dreamerv3", tmp_path / "dreamer")
    ne = _command("nedreamer", tmp_path / "ne")
    for command in (dreamer, ne):
        assert command[command.index("--episodes") + 1] == "20"
        assert command[command.index("--eval-seed") + 1] == "10000"
    assert dreamer[dreamer.index("--envs") + 1] == "4"
    assert ne[ne.index("--device") + 1] == "cuda:0"
