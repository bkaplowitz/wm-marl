from __future__ import annotations

import argparse
import json
import sys

from world_marl.jepa.config import smoke_jepa_config
from world_marl.scripts import optuna_dmc_jepa, train_dmc_jepa


class _CanonicalTrial:
    def suggest_categorical(self, name, choices):
        preferred = {
            "learning_rate": 4e-5,
            "actor_learning_rate": 4e-5,
            "actor_entropy_coef": 3e-3,
            "policy_actor_kl_coef": 1.0,
            "online_policy_actor_update_interval": 2,
            "actor_grad_clip_norm": 10.0,
        }[name]
        assert preferred in choices
        return preferred


def test_hpo_command_uses_locked_h8_trainer_contract(monkeypatch, tmp_path):
    params = optuna_dmc_jepa.sample_params(
        _CanonicalTrial(),
        base_params=smoke_jepa_config(),
    )
    args = argparse.Namespace(task="reacher/easy", seed=2)
    command = optuna_dmc_jepa.build_command(
        args,
        params,
        tmp_path / "trial_0000",
    )

    assert command[2] == "world_marl.scripts.train_dmc_jepa_bootstrap"
    assert "--model-horizon" in command
    assert command[command.index("--model-horizon") + 1] == "8"
    assert "--dynamics-ensemble-size" not in command
    assert "--online-policy-trust-coef" not in command

    monkeypatch.setattr(sys, "argv", [command[2], *command[3:]])
    resolved = train_dmc_jepa.parse_args()
    assert resolved.env == "dmc:reacher/easy"
    assert resolved.seed == 2
    assert resolved.model_horizon == 8
    assert resolved.open_loop_horizon == 8
    assert resolved.imag_horizon == 15


def test_extract_progress_reads_canonical_metrics(tmp_path):
    run_dir = tmp_path / "run" / "dmc_jepa_stamp" / "run_000"
    run_dir.mkdir(parents=True)
    rows = [
        {
            "phase": "online_actor_replay",
            "budget/train_env_steps": 10_000,
            "report": {"episode_return_mean": 123.0},
        },
        {
            "phase": "paper_online_score_bin",
            "budget/train_env_steps": 20_000,
            "paper/online_return_mean": 456.0,
        },
    ]
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    step, value, row = optuna_dmc_jepa.extract_progress(tmp_path)

    assert step == 20_000
    assert value == 456.0
    assert row["phase"] == "paper_online_score_bin"
