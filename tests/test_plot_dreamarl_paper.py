from dreamarl.baselines.dreamerv3.artifacts import EpisodeScore
from dreamarl.scripts.plot_dreamarl_paper import RunArtifact, aggregate_curves


def test_aggregate_curves_bins_each_seed_before_aggregation(tmp_path):
    runs = [
        RunArtifact(
            "dmc_reacher_easy",
            seed,
            tmp_path / f"seed{seed}",
            (
                EpisodeScore(1_000, 100.0 + 20 * seed),
                EpisodeScore(9_000, 300.0 + 20 * seed),
                EpisodeScore(15_000, 500.0 + 20 * seed),
            ),
            {},
        )
        for seed in (0, 1)
    ]

    curve = aggregate_curves(runs, bins=2, max_steps=20_000)["dmc_reacher_easy"]

    assert curve == [
        {"env_steps": 10_000, "mean": 210.0, "std": 10.0, "seeds": 2},
        {"env_steps": 20_000, "mean": 510.0, "std": 10.0, "seeds": 2},
    ]
