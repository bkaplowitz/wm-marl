from __future__ import annotations

from world_marl.scripts.write_dmc_vector_launcher import (
    COMMON_PARAMS,
    PRESETS,
    params_to_shell_args,
    step_accounting,
    write_launcher,
)


def test_launcher_serializes_wandb_tracking_controls():
    command = params_to_shell_args(
        {
            "wandb_project": "world-marl",
            "wandb_entity": "osaze-obahor",
            "wandb_tags": ("jepa", "reacher"),
            "wandb_videos": True,
            "wandb_video_frame_stride": 4,
        }
    )

    assert "--wandb-project" in command
    assert "world-marl" in command
    assert "--wandb-tags" in command
    assert "jepa" in command
    assert "reacher" in command
    assert "--wandb-videos" in command
    assert "--wandb-video-frame-stride" in command


def test_launcher_syncs_tracking_extra_when_enabled(tmp_path):
    write_launcher(
        tmp_path,
        [{"task": "reacher/easy", "seed": 0, "short": "reacher_easy_seed0"}],
        ["0"],
        sync=True,
        tracking=True,
    )

    launcher = (tmp_path / "launcher.sh").read_text()
    assert "--extra dmc --extra cuda12 --extra tracking" in launcher


def test_dreamer_parity_100k_preset_is_fixed_budget_and_latest_policy():
    params = {**COMMON_PARAMS, **PRESETS["jepa_dreamer_parity_100k"]}
    accounting = step_accounting(params)
    command = params_to_shell_args(params)

    assert accounting["train_replay_env_steps"] == 99_584
    assert accounting["train_plus_validation_env_steps"] == 100_864
    assert accounting["world_model_updates"] == 99_584
    assert accounting["policy_updates"] == 99_584
    assert accounting["world_model_replay_ratio"] == 1024.0
    assert params["policy_gradient_mode"] == "reinforce"
    assert params["policy_return_normalization"] == "ema-percentile"
    assert params["policy_replay_critic_return_mode"] == "lambda"
    assert params["policy_replay_critic_all_steps"]
    assert params["value_clip"] == 100.0
    assert params["online_checkpoint_interval"] == 5
    assert params["isolated_rng_streams"]
    assert params["deterministic_compute"]
    assert params["final_policy_eval_seed"] == 9_000_000
    assert params["wandb_video_every_phases"] == 10
    assert not params["policy_eval_during_training"]
    assert not params["online_policy_champion"]
    assert not params["online_candidate_refit"]
    assert "--no-policy-eval-during-training" in command
    assert "--no-online-policy-champion" in command
    assert "--no-online-freeze-encoder" in command
    assert "--no-online-reset-replay-env" in command
    assert "--value-clip" in command
    assert "100.0" in command
    assert "--online-checkpoint-interval" in command
    assert "--isolated-rng-streams" in command
    assert "--deterministic-compute" in command
    assert "--final-policy-eval-seed" in command


def test_dreamer_parity_500k_preset_stays_below_training_data_budget():
    params = {**COMMON_PARAMS, **PRESETS["jepa_dreamer_parity_500k"]}
    accounting = step_accounting(params)

    assert accounting["train_replay_env_steps"] == 496_896
    assert accounting["train_plus_validation_env_steps"] == 498_176
