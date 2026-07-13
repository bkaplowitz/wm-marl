from __future__ import annotations

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


def test_launcher_serializes_entropy_decay_controls():
    command = params_to_shell_args(
        {
            "actor_entropy_coef": 3e-3,
            "actor_entropy_final_coef": 3e-4,
            "actor_entropy_decay_start_env_steps": 300_000,
            "actor_entropy_decay_end_env_steps": 500_000,
        }
    )

    tokens = command.replace("\\\n", " ").split()
    assert tokens[tokens.index("--actor-entropy-coef") + 1] == "0.003"
    assert tokens[tokens.index("--actor-entropy-final-coef") + 1] == "0.0003"
    assert tokens[tokens.index("--actor-entropy-decay-start-env-steps") + 1] == "300000"
    assert tokens[tokens.index("--actor-entropy-decay-end-env-steps") + 1] == "500000"


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


def test_maintained_presets_are_small_and_unambiguous():
    assert set(PRESETS) == {"smoke", "jepa_100k", "jepa_500k"}
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
    assert params["online_iterations"] == 91
    assert params["online_collect_steps"] == 64
    assert params["online_train_steps"] == 1_024
    assert params["online_policy_train_steps"] == 512


def test_500k_preset_matches_the_current_running_model():
    params = PRESETS["jepa_500k"]
    accounting = step_accounting(params)

    assert accounting["train_replay_env_steps"] == 497_664
    assert accounting["validation_replay_env_steps"] == 1_280
    assert accounting["train_plus_validation_env_steps"] == 498_944
    assert accounting["world_model_updates"] == 493_824
    assert accounting["policy_updates"] == 247_552
    assert params["online_iterations"] == 481
    assert params["online_checkpoint_interval"] == 16
    assert params["validation_seed"] == 1_000_042
    assert params["final_policy_eval_seed"] == 9_000_000
    assert params["final_policy_eval_episodes"] == 20


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
    assert params["policy_gradient_mode"] == "reinforce"
    assert params["policy_return_mode"] == "lambda"
    assert params["policy_return_normalization"] == "ema-percentile"
    assert params["actor_entropy_mode"] == "tanh-normal"
    assert params["actor_entropy_coef"] == 3e-3
    assert params["value_clip"] == 100.0
    assert params["target_critic_ema_decay"] == 0.98
    assert params["policy_replay_critic_loss_coef"] == 0.3
    assert params["policy_slow_value_regularization_coef"] == 1.0
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
