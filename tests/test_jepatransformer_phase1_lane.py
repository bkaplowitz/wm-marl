from __future__ import annotations

from world_marl.scripts.run_jepatransformer_phase1_lane import _command


def test_phase1_dreamer_command_is_visual_and_uses_official_budget(tmp_path):
    command = _command(
        "dreamerv3",
        run_dir=tmp_path / "dreamer",
        task="dmc_walker_walk",
        seed=1,
        project="project",
        entity="entity",
    )
    assert command[command.index("--observation-mode") + 1] == "vision"
    assert "--official-budget" in command
    assert command[command.index("--seed") + 1] == "1"


def test_phase1_ne_command_uses_pinned_native_budget(tmp_path):
    command = _command(
        "nedreamer",
        run_dir=tmp_path / "ne",
        task="dmc_cheetah_run",
        seed=0,
        project="project",
        entity="entity",
    )
    assert command[command.index("--total-env-steps") + 1] == "1100000"
    assert command[command.index("--device") + 1] == "cuda:0"
