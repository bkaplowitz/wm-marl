from world_marl.scripts.run_jepatransformer_m3_lane import (
    TASKS,
    evaluation_command,
    training_command,
)


def test_m3_lane_uses_registered_budget_and_latest_policy_evaluation(tmp_path):
    run_dir = tmp_path / "dmc_reacher_easy" / "seed_1"
    train = training_command(
        run_dir=run_dir,
        task="dmc_reacher_easy",
        seed=1,
        project="world-marl",
        entity="osaze-obahor",
    )
    evaluate = evaluation_command(run_dir)
    assert train[train.index("--total-env-steps") + 1] == "250000"
    assert train[train.index("--experiment-dir") + 1] == str(run_dir)
    assert "world_marl.scripts.eval_dmc_jepa_transformer" in evaluate
    assert evaluate[evaluate.index("--episodes") + 1] == "20"
    assert evaluate[evaluate.index("--eval-seed") + 1] == "10000"


def test_m3_lane_keeps_the_registered_task_order():
    assert TASKS == (
        "dmc_reacher_easy",
        "dmc_walker_walk",
        "dmc_cheetah_run",
        "dmc_hopper_hop",
    )
