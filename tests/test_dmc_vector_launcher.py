from __future__ import annotations

import sys

from world_marl.scripts import write_dmc_vector_launcher as launcher
from world_marl.scripts.write_dmc_vector_launcher import (
    PRESETS,
    params_to_shell_args,
    step_accounting,
    write_launcher,
)


def test_launcher_serializes_tracking_controls():
    command = params_to_shell_args(
        {
            "wandb_project": "world-marl",
            "wandb_entity": "osaze-obahor",
            "wandb_tags": ("jepa", "reacher"),
            "wandb_videos": True,
        }
    )

    assert "--wandb-project" in command
    assert "world-marl" in command
    assert "--wandb-tags" in command
    assert "jepa" in command
    assert "reacher" in command
    assert "--wandb-videos" in command


def test_launcher_serializes_canonical_replay_and_evaluation_controls():
    command = params_to_shell_args(
        {
            "online_recent_world_model_fraction": 0.5,
            "online_recent_world_model_until_env_steps": 50_000,
            "policy_reset_start_fraction": 0.05,
            "policy_reset_start_max_age": 63,
            "online_recent_replay_steps": 320,
            "online_recent_replay_max_oversample": 10.0,
            "curve_eval_interval_env_steps": 50_000,
            "curve_eval_episodes": 20,
            "curve_eval_seed": 9_000_000,
        }
    )

    tokens = command.replace("\\\n", " ").split()
    assert tokens[tokens.index("--online-recent-world-model-fraction") + 1] == "0.5"
    assert (
        tokens[tokens.index("--online-recent-world-model-until-env-steps") + 1]
        == "50000"
    )
    assert tokens[tokens.index("--policy-reset-start-fraction") + 1] == "0.05"
    assert tokens[tokens.index("--policy-reset-start-max-age") + 1] == "63"
    assert tokens[tokens.index("--online-recent-replay-steps") + 1] == "320"
    assert tokens[tokens.index("--online-recent-replay-max-oversample") + 1] == "10.0"
    assert tokens[tokens.index("--curve-eval-interval-env-steps") + 1] == "50000"
    assert tokens[tokens.index("--curve-eval-episodes") + 1] == "20"
    assert tokens[tokens.index("--curve-eval-seed") + 1] == "9000000"


def test_launcher_serializes_budget_relative_encoder_freeze():
    command = params_to_shell_args(
        {
            "online_freeze_encoder_after_env_steps": 100_000,
        }
    )

    assert command.replace("\\\n", " ").split() == [
        "--online-freeze-encoder-after-env-steps",
        "100000",
    ]


def test_launcher_serializes_explicit_reporting_budget():
    command = params_to_shell_args(
        {
            "dreamer_report_budget_env_steps": 150_000,
        }
    )

    assert command.replace("\\\n", " ").split() == [
        "--dreamer-report-budget-env-steps",
        "150000",
    ]


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


def test_launcher_pins_generation_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out_root = tmp_path / "runs"
    out_root.mkdir()

    write_launcher(
        out_root,
        [{"task": "reacher/easy", "seed": 0, "short": "reacher_easy_seed0"}],
        ["0"],
        sync=False,
        tracking=False,
    )

    script = (out_root / "launcher.sh").read_text()
    assert f"DEFAULT_REPO_ROOT={tmp_path}" in script
    assert 'REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"' in script
    assert 'cd "$REPO_ROOT"' in script


def test_launcher_can_pin_an_explicit_repo_root(tmp_path):
    out_root = tmp_path / "runs"
    repo_root = tmp_path / "checkout"
    out_root.mkdir()
    repo_root.mkdir()

    write_launcher(
        out_root,
        [{"task": "reacher/easy", "seed": 0, "short": "reacher_easy_seed0"}],
        ["0"],
        sync=False,
        tracking=False,
        repo_root=repo_root,
    )

    script = (out_root / "launcher.sh").read_text()
    assert f"DEFAULT_REPO_ROOT={repo_root}" in script


def test_maintained_presets_are_small_and_unambiguous():
    assert set(PRESETS) == {"smoke", "jepa_100k", "jepa_200k", "jepa_500k"}
    forbidden = {
        "policy_selection_interval",
        "policy_confirmation_episodes",
        "online_policy_champion",
        "online_candidate_refit",
        "policy_hard_start_max_steps",
        "policy_actor_cvar_coef",
        "policy_action_bound_coef",
    }
    for params in PRESETS.values():
        assert forbidden.isdisjoint(params)


def test_100k_preset_matches_the_reset_rich_interleaved_contract():
    params = PRESETS["jepa_100k"]
    accounting = step_accounting(params)

    assert accounting["train_replay_env_steps"] == 98_304
    assert accounting["validation_replay_env_steps"] == 1_280
    assert accounting["train_plus_validation_env_steps"] == 99_584
    assert accounting["world_model_updates"] == 94_464
    assert accounting["policy_updates"] == 47_872
    assert params["collect_steps"] == 320
    assert params["initial_reset_interval"] == 80
    assert params["initial_random_action_hold_steps"] == 1
    assert params["online_iterations"] == 91
    assert params["online_collect_steps"] == 64
    assert params["online_train_steps"] == 1_024
    assert params["online_policy_train_steps"] == 512
    assert params["online_policy_actor_update_interval"] == 2
    assert params["online_policy_actor_update_interval_start_env_steps"] == 10_000
    assert params["online_freeze_encoder_after_env_steps"] == 20_275
    assert params["online_recent_replay_steps"] == 320
    assert params["online_recent_world_model_fraction"] == 0.5
    assert params["online_recent_world_model_until_env_steps"] == 10_000
    assert params["online_recent_replay_max_oversample"] == 10.0
    assert params["policy_reset_start_fraction"] == 0.1
    assert params["policy_reset_start_fraction_start_env_steps"] == 40_346
    assert params["policy_reset_start_max_age"] == 63
    assert params["policy_actor_kl_coef"] == 1.0
    assert params["policy_actor_kl_target_per_dim"] == 0.1
    assert params["policy_actor_kl_reference_interval"] == 512
    assert params["value_clip_schedule_start_env_steps"] == 30_106
    assert params["value_clip_schedule_end_env_steps"] == 50_176
    assert accounting["actor_updates"] == 1_280 + 5 * 512 + 86 * 256
    assert accounting["critic_updates"] == 1_280 + 91 * 512


def test_actor_update_interval_override_is_accounted_separately(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_dmc_vector_launcher.py",
            "--out-root",
            str(tmp_path),
            "--preset",
            "jepa_100k",
            "--tasks",
            "reacher/easy",
            "--seeds",
            "1",
            "--online-policy-actor-update-interval",
            "2",
            "--online-policy-actor-update-interval-start-env-steps",
            "50000",
        ],
    )

    args = launcher.parse_args()
    params = dict(launcher.PRESETS[args.preset])
    launcher.apply_optional_overrides(args, params)
    accounting = launcher.step_accounting(params)

    assert params["online_policy_actor_update_interval"] == 2
    assert params["online_policy_actor_update_interval_start_env_steps"] == 50_000
    assert accounting["critic_updates"] == 1_280 + 91 * 512
    assert accounting["actor_updates"] == 1_280 + 44 * 512 + 47 * 256


def test_500k_preset_matches_the_current_running_model():
    params = PRESETS["jepa_500k"]
    accounting = step_accounting(params)

    assert accounting["train_replay_env_steps"] == 499_712
    assert accounting["validation_replay_env_steps"] == 1_280
    assert accounting["train_plus_validation_env_steps"] == 500_992
    assert accounting["world_model_updates"] == 495_872
    assert accounting["policy_updates"] == 248_576
    assert accounting["actor_updates"] == 136_192
    assert accounting["critic_updates"] == 248_576
    assert params["online_iterations"] == 483
    assert params["online_checkpoint_interval"] == 16
    assert params["validation_seed"] == 1_000_042
    assert params["final_policy_eval_seed"] == 9_000_000
    assert params["final_policy_eval_episodes"] == 100


def test_200k_preset_scales_training_milestones_with_the_fixed_budget():
    params = PRESETS["jepa_200k"]
    accounting = step_accounting(params)

    assert accounting["train_replay_env_steps"] == 199_680
    assert accounting["validation_replay_env_steps"] == 1_280
    assert accounting["train_plus_validation_env_steps"] == 200_960
    assert accounting["world_model_updates"] == 195_840
    assert accounting["policy_updates"] == 98_560
    assert params["dreamer_report_budget_env_steps"] == 200_000
    assert params["online_policy_actor_update_interval_start_env_steps"] == 20_000
    assert params["online_freeze_encoder_after_env_steps"] == 40_550
    assert params["online_recent_world_model_until_env_steps"] == 20_000
    assert params["policy_reset_start_fraction_start_env_steps"] == 80_691
    assert params["value_clip_schedule_start_env_steps"] == 60_211
    assert params["value_clip_schedule_end_env_steps"] == 100_352
    assert accounting["actor_updates"] == 1_280 + 15 * 512 + 175 * 256


def test_launcher_can_disable_value_clipping():
    command = params_to_shell_args(
        {
            "value_clip": 0.0,
        }
    )

    assert command.replace("\\\n", " ").split() == [
        "--value-clip",
        "0.0",
    ]


def test_launcher_serializes_value_clip_schedule():
    command = params_to_shell_args(
        {
            "value_clip": 100.0,
            "value_clip_final": 200.0,
            "value_clip_schedule_start_env_steps": 100_000,
            "value_clip_schedule_end_env_steps": 200_000,
        }
    )

    assert command.replace("\\\n", " ").split() == [
        "--value-clip",
        "100.0",
        "--value-clip-final",
        "200.0",
        "--value-clip-schedule-start-env-steps",
        "100000",
        "--value-clip-schedule-end-env-steps",
        "200000",
    ]


def test_launcher_serializes_training_snapshot_controls():
    command = params_to_shell_args(
        {
            "training_snapshot_env_steps": [150_528, 200_704],
            "resume_training_snapshot": "/tmp/snapshot_150528",
        }
    )

    assert command.replace("\\\n", " ").split() == [
        "--training-snapshot-env-steps",
        "150528",
        "200704",
        "--resume-training-snapshot",
        "/tmp/snapshot_150528",
    ]


def test_launcher_serializes_actor_kl_controls():
    command = params_to_shell_args(
        {
            "policy_actor_kl_coef": 1.0,
            "policy_actor_kl_target_per_dim": 0.01,
            "policy_actor_kl_reference_interval": 64,
        }
    )

    assert command.replace("\\\n", " ").split() == [
        "--policy-actor-kl-coef",
        "1.0",
        "--policy-actor-kl-target-per-dim",
        "0.01",
        "--policy-actor-kl-reference-interval",
        "64",
    ]


def test_500k_preset_locks_current_architecture_and_control_stack():
    params = PRESETS["jepa_500k"]

    assert params["latent_dim"] == 128
    assert params["model_dim"] == 128
    assert params["num_layers"] == 2
    assert params["num_heads"] == 4
    assert params["context_window"] == 8
    assert params["model_horizon"] == 5
    assert params["imag_horizon"] == 15
    assert params["actor_hidden_dim"] == 64
    assert params["critic_hidden_dim"] == 64
    assert params["actor_num_layers"] == 3
    assert params["critic_num_layers"] == 3
    assert "policy_gradient_mode" not in params
    assert "policy_return_mode" not in params
    assert "policy_return_normalization" not in params
    assert "actor_entropy_mode" not in params
    assert params["actor_entropy_coef"] == 3e-3
    assert params["value_clip"] == 100.0
    assert params["value_clip_final"] == 333.0
    assert params["value_clip_schedule_start_env_steps"] == 150_528
    assert params["value_clip_schedule_end_env_steps"] == 250_880
    assert params["policy_actor_kl_coef"] == 1.0
    assert params["policy_actor_kl_target_per_dim"] == 0.1
    assert params["policy_actor_kl_reference_interval"] == 512
    assert params["target_critic_ema_decay"] == 0.98
    assert params["policy_replay_critic_loss_coef"] == 0.3
    assert params["policy_slow_value_regularization_coef"] == 1.0
    assert params["online_freeze_encoder_after_env_steps"] == 101_376
    assert params["online_recent_world_model_fraction"] == 0.5
    assert params["online_recent_world_model_until_env_steps"] == 50_000
    assert params["policy_reset_start_fraction"] == 0.1
    assert params["policy_reset_start_fraction_start_env_steps"] == 201_728
    assert params["model_grad_clip_norm"] == 0.0
    assert params["actor_grad_clip_norm"] == 10.0
    assert params["critic_grad_clip_norm"] == 100.0


def test_canonical_command_contains_no_selection_or_hard_start_flags():
    command = params_to_shell_args(PRESETS["jepa_500k"])

    assert "--policy-selection" not in command
    assert "--champion" not in command
    assert "--candidate" not in command
    assert "--hard-start" not in command
    assert "--final-policy-eval-seed" in command
    assert "--isolated-rng-streams" in command
    assert "--deterministic-compute" in command
