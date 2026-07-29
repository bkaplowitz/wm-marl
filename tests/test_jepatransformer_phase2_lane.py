from world_marl.scripts.run_jepatransformer_phase2_lane import (
    TASKS,
    evaluation_command,
    training_command,
)


def test_phase2_gate_has_pre_registered_tasks():
    assert TASKS == (
        "dmc_reacher_easy",
        "dmc_walker_walk",
        "dmc_cheetah_run",
        "dmc_hopper_hop",
    )


def test_phase2_training_command_uses_fixed_visual_budget(tmp_path):
    command = training_command(
        run_dir=tmp_path / "run",
        task="dmc_hopper_hop",
        seed=1,
        project="project",
        entity="entity",
    )
    assert command[command.index("--total-env-steps") + 1] == "250000"
    assert command[command.index("--seed") + 1] == "1"
    assert command[command.index("--platform") + 1] == "cuda"


def test_phase2_evaluation_is_fixed_latest_policy(tmp_path):
    command = evaluation_command(tmp_path / "run")
    assert command[command.index("--episodes") + 1] == "20"
    assert command[command.index("--eval-seed") + 1] == "10000"
