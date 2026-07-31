import subprocess

from world_marl.scripts.run_jepatransformer_m3_lane import (
    TASKS,
    evaluation_command,
    main,
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


def test_m3_lane_accepts_a_registered_task_subset(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert (
        main(
            [
                "--gpu",
                "0",
                "--seed",
                "0",
                "--tasks",
                "dmc_walker_walk",
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert len(calls) == 2
    assert "dmc_walker_walk" in calls[0]
    assert "world_marl.scripts.eval_dmc_jepa_transformer" in calls[1]


def test_m3_lane_stops_after_a_failed_training_run(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert (
        main(
            [
                "--gpu",
                "0",
                "--seed",
                "0",
                "--tasks",
                "dmc_walker_walk",
                "dmc_cheetah_run",
                "--output-root",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert len(calls) == 1
