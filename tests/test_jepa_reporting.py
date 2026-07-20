from __future__ import annotations

import numpy as np

from world_marl.jepa.reporting import dreamer_style_training_score


def _phase(
    iteration: int,
    *,
    train_env_steps: int,
    returns: list[float],
    finish_steps: list[int],
) -> dict:
    return {
        "iteration": iteration,
        "actor_replay": {
            "returns": returns,
            "lengths": [1_000] * len(returns),
            "episode_finish_train_env_steps": finish_steps,
            "train_replay_total_env_steps": train_env_steps,
        },
    }


def test_dreamer_score_uses_disjoint_bins_and_final_three_bin_means():
    history = [
        _phase(
            1,
            train_env_steps=20_000,
            returns=[100.0, 300.0, 400.0],
            finish_steps=[5_000, 9_000, 15_000],
        ),
        _phase(
            2,
            train_env_steps=50_000,
            returns=[900.0, 1_000.0, 950.0],
            finish_steps=[35_000, 39_000, 49_000],
        ),
    ]

    score = dreamer_style_training_score(
        history,
        window_env_steps=10_000,
        budget_env_steps=50_000,
        final_bins=3,
    )

    assert score["budget_reached"]
    assert score["final_train_env_step"] == 50_000
    assert [item["bin_end_env_step"] for item in score["curve"]] == [
        10_000,
        20_000,
        40_000,
        50_000,
    ]
    assert [item["mean_return"] for item in score["curve"]] == [
        200.0,
        400.0,
        950.0,
        950.0,
    ]
    assert score["selected_bin_end_env_steps"] == [20_000, 40_000, 50_000]
    assert score["selected_bin_means"] == [400.0, 950.0, 950.0]
    assert np.isclose(score["mean_return"], (400.0 + 950.0 + 950.0) / 3.0)
    assert score["episodes"] == 4
    assert score["episode_mean_return"] == 812.5


def test_dreamer_score_budget_comes_from_replay_not_last_episode_finish():
    score = dreamer_style_training_score(
        [
            _phase(
                1,
                train_env_steps=500_000,
                returns=[950.0],
                finish_steps=[490_000],
            )
        ],
        window_env_steps=10_000,
        budget_env_steps=500_000,
    )

    assert score["budget_reached"]
    assert score["final_train_env_step"] == 500_000
    assert score["final_episode_finish_env_step"] == 490_000
    assert score["mean_return"] == 950.0


def test_dreamer_score_can_be_disabled():
    score = dreamer_style_training_score(
        [],
        window_env_steps=10_000,
        budget_env_steps=0,
    )

    assert not score["enabled"]
    assert score["curve"] == []
    assert score["mean_return"] is None
