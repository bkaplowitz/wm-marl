"""Protect the fixed comparisons, source isolation, and final-only selection."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"
for name in ["run_ppo_correction_screen", "run_ppo_self_fed_screen"]:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
screen = sys.modules["run_ppo_self_fed_screen"]
base = screen.base


def baseline_evidence(root, wins=(0.1, 0.1, 0.2, 0.2), returns=(10, 10, 10, 10)):
    paths = []
    for index, (scale, seed) in enumerate([(0.0, 0), (0.0, 1), (0.3, 0), (0.3, 1)]):
        arm = "team_return_anchor" if scale else "team_return"
        path = root / "runs" / f"{arm}-2s3z-seed{seed}"
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(
            json.dumps(
                {
                    "source_sha256": "baseline",
                    "protocol": {"train_steps": 50000},
                    "configuration": {
                        "replay_value_scale": scale,
                        "seed": seed,
                        "task": "smac_2s3z",
                    },
                }
            )
        )
        outcome = {
            "completed": True,
            "summary": {
                "evaluation_protocol": {"episodes": 128},
                "win_rate": wins[index],
                "return_mean": returns[index],
            },
        }
        (path / "outcome.json").write_text(json.dumps(outcome))
        paths.append(path)
    return paths


def test_all_six_comparisons_preserve_horizons_seeds_and_disable_consumer_kl():
    args = SimpleNamespace(python=Path("/runtime/python"))
    expected = [
        (5, 0, True),
        (2, 0, False),
        (5, 1, True),
        (2, 1, False),
        (5, 0, True),
        (5, 123, False),
    ]
    for slot, (horizon, seed, recurrent) in enumerate(expected):
        run = screen.run_spec(slot, 0.3, 0.1)
        train = base.train_command(args, run, Path("/train"))
        final = base.eval_command(args, run, Path("/final"), Path("/checkpoint"))
        assert run.seed == seed and run.recurrent == recurrent
        for command in [train, final]:
            assert command[command.index("--agent.imag_length") + 1] == str(horizon)
            assert command[command.index("--agent.ppo.replay_value_scale") + 1] == "0.3"
            if recurrent:
                assert (
                    command[
                        command.index("--agent.marl.ctde.self_fed.consumer_kl_scale")
                        + 1
                    ]
                    == "0.0"
                )
                begin = command.index("--agent.marl.ctde.self_fed.horizons")
                assert command[begin + 1 : begin + 4] == ["2", "4", "5"]
            else:
                assert not any("self_fed" in arg for arg in command)
        assert train[train.index("--run.steps") + 1] == "50000"
        assert final[final.index("--run.eval_eps") + 1] == "128"
        assert final[final.index("--run.eval_worker_offset") + 1] == "100000"


def test_anchor_selection_uses_both_final_seeds_then_return_tiebreak(tmp_path):
    baseline_evidence(tmp_path, wins=(0.1, 0.3, 0.2, 0.2), returns=(12, 12, 10, 10))
    result = screen.select_anchor(tmp_path, "baseline")
    assert result["selected_replay_value_scale"] == 0.0
    assert len(result["outcomes"]) == 4 and not result["provisional"]


def test_anchor_selection_refuses_failed_or_missing_peer_and_wrong_source(tmp_path):
    paths = baseline_evidence(tmp_path)
    with pytest.raises(ValueError, match="mismatched"):
        screen.select_anchor(tmp_path, "wrong hash")
    (paths[-1] / "outcome.json").write_text('{"completed":false}')
    with pytest.raises(ValueError, match="mismatched"):
        screen.select_anchor(tmp_path, "baseline")
    (paths[-1] / "outcome.json").unlink()
    with pytest.raises(FileNotFoundError):
        screen.select_anchor(tmp_path, "baseline")


def test_full_anchor_tie_retains_current_setting(tmp_path):
    baseline_evidence(tmp_path, wins=(0.2,) * 4)
    assert (
        screen.select_anchor(tmp_path, "baseline")["selected_replay_value_scale"] == 0.3
    )


def prepare_args(tmp_path, slot):
    for name in ["baseline", "optional"]:
        source = tmp_path / name / "src/majepa"
        source.mkdir(parents=True)
        (source / "agent.py").write_text(name)
    baseline = tmp_path / "baseline"
    optional = tmp_path / "optional"
    return SimpleNamespace(
        source=optional,
        expected_source_sha256=base.source_fingerprint(optional),
        baseline_source=baseline,
        expected_baseline_source_sha256=base.source_fingerprint(baseline),
        self_fed_scale=0.1,
        candidate_anchor_scale=0.3,
        validate_only=True,
        disabled_parity_record=None,
        slot=slot,
    )


@pytest.mark.parametrize("slot,recurrent", [(0, True), (1, False), (5, False)])
def test_control_source_is_prior_corrected_package(tmp_path, slot, recurrent):
    args = prepare_args(tmp_path, slot)
    run, evidence = screen.prepare(args)
    assert args.source == tmp_path / ("optional" if recurrent else "baseline")
    assert run.recurrent == recurrent
    assert evidence["anchor_decision"]["provisional"]


def test_provisional_anchor_cannot_launch_training(tmp_path):
    args = prepare_args(tmp_path, 0)
    args.validate_only = False
    with pytest.raises(ValueError, match="validation only"):
        screen.prepare(args)


def test_parity_evidence_must_execute_both_bound_sources():
    record = {
        "passed": True,
        "baseline_source_sha256": "old",
        "optional_source_sha256": "new",
        "test_command": "paired CPU test",
        "parity_kind": "cross_source_executed_train",
    }
    assert screen.validate_parity(record, "old", "new") == record
    with pytest.raises(ValueError):
        screen.validate_parity(record, "old", "different")
    with pytest.raises(ValueError):
        screen.validate_parity(
            {**record, "parity_kind": "within_source_config_modes"}, "old", "new"
        )


def test_wandb_names_use_new_profile_seed_instead_of_correction_slot_default():
    run = screen.run_spec(5, 0.3, 0.1)
    args = SimpleNamespace(
        slot=5,
        run_spec=run,
        wandb_project="project",
        wandb_entity="entity",
        wandb_group="new-screen",
        wandb_run_prefix="new-prefix",
        expected_source_sha256="hash",
        screen_label="Recurrent screen",
    )
    env = base.phase_environment(args, Path("/results/final128"), "final128", {})
    assert env["WANDB_NAME"] == "new-prefix-corrected_h5-2s3z-seed123-final128"
    assert env["WANDB_RUN_ID"] == "new-prefix-s5-final128"
