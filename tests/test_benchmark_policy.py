from world_marl.scripts.benchmark_policy import loss_at_episode_checkpoints


def test_loss_at_episode_checkpoints_uses_first_row_at_or_after_budget():
    rows = [
        {
            "update": 1,
            "cumulative_real_episodes": 3,
            "ppo/total_loss": 9.0,
            "ppo/actor_loss": 1.0,
            "ppo/value_loss": 2.0,
            "ppo/entropy": 0.1,
        },
        {
            "update": 2,
            "cumulative_real_episodes": 7,
            "ppo/total_loss": 5.0,
            "ppo/actor_loss": 0.5,
            "ppo/value_loss": 1.5,
            "ppo/entropy": 0.2,
        },
    ]

    result = loss_at_episode_checkpoints(rows, [1, 5, 10])

    assert result == {
        "1": {
            "checkpoint": 1,
            "actual_real_episodes": 3,
            "update": 1,
            "ppo/total_loss": 9.0,
            "ppo/actor_loss": 1.0,
            "ppo/value_loss": 2.0,
            "ppo/entropy": 0.1,
        },
        "5": {
            "checkpoint": 5,
            "actual_real_episodes": 7,
            "update": 2,
            "ppo/total_loss": 5.0,
            "ppo/actor_loss": 0.5,
            "ppo/value_loss": 1.5,
            "ppo/entropy": 0.2,
        },
        "10": None,
    }
