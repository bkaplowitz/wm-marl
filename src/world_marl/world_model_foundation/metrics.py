from __future__ import annotations

METRIC_KEYS = frozenset(
    {
        "reconstruction_loss",
        "observation_prediction_loss",
        "token_prediction_loss",
        "reward_loss",
        "continue_loss",
        "rollout_loss",
        "rollout_return",
        "real_env_return",
        "bridge_accuracy",
        "latent_action_usage",
    }
)
