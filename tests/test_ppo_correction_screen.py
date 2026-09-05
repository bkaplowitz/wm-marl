"""Boundaries that protect an ongoing run during the correction comparison."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


SCRIPT = Path(__file__).parents[1] / "scripts/run_ppo_correction_screen.py"
SPEC = importlib.util.spec_from_file_location("correction_screen", SCRIPT)
screen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = screen
SPEC.loader.exec_module(screen)


def test_old_reference_does_not_receive_new_configuration_fields():
    args = SimpleNamespace(python=Path("/runtime/python"))
    old = screen.train_command(args, screen.SLOTS[5], Path("/results/train"))
    corrected = screen.train_command(args, screen.SLOTS[1], Path("/results/train"))
    assert "--agent.ppo.replay_value_scale" not in old
    assert corrected[corrected.index("--agent.ppo.replay_value_scale") + 1] == "0.3"
    assert old[old.index("--run.steps") + 1] == "50000"


def test_evaluation_uses_unchanged_held_out_protocol():
    args = SimpleNamespace(python=Path("/runtime/python"))
    command = screen.eval_command(
        args, screen.SLOTS[4], Path("/results/final"), Path("/checkpoint")
    )
    for flag, value in {
        "--run.eval_worker_offset": "100000",
        "--run.eval_eps": "128",
        "--run.envs": "4",
        "--run.eval_policy_mode": "eval",
    }.items():
        assert command[command.index(flag) + 1] == value


def test_fingerprint_covers_package_content_and_names_not_generated_cache(tmp_path):
    package = tmp_path / "src/majepa"
    package.mkdir(parents=True)
    source = package / "agent.py"
    source.write_text("old")
    before = screen.source_fingerprint(tmp_path)
    (package / "cache.pyc").write_bytes(b"cache")
    assert screen.source_fingerprint(tmp_path) == before
    source.write_text("new")
    assert screen.source_fingerprint(tmp_path) != before


def test_wait_includes_predecessor_during_gpu_free_transition(monkeypatch, tmp_path):
    # A training child can exit before its supervisor starts final128.
    # Require both the supervisor and GPU to be idle for a continuous interval.
    args = SimpleNamespace(wait_pid=[100], gpu=0, idle_seconds=2, poll_seconds=1)
    clock = [0]
    statuses = []
    monkeypatch.setattr(screen.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(screen.time, "time", lambda: clock[0])
    monkeypatch.setattr(
        screen.time, "sleep", lambda _: clock.__setitem__(0, clock[0] + 1)
    )
    monkeypatch.setattr(screen, "pid_alive", lambda pid: clock[0] < 2)
    monkeypatch.setattr(
        screen, "gpu_processes", lambda gpu: [222] if clock[0] == 3 else []
    )
    monkeypatch.setattr(
        screen, "atomic_json", lambda path, value: statuses.append(value)
    )
    screen.wait_for_gpu(args, tmp_path)
    assert clock[0] == 6
    assert statuses[1]["predecessor_pids"] == [100]
    assert statuses[3]["compute_pids"] == [222]
    assert statuses[-1]["idle_seconds"] == 2


def test_gpu_process_filter_selects_assigned_gpu_only(monkeypatch):
    def output(command, text):
        return (
            "GPU-second\n"
            if "--query-gpu=uuid" in command
            else "GPU-first, 11\nGPU-second, 22\n"
        )

    monkeypatch.setattr(screen.subprocess, "check_output", output)
    assert screen.gpu_processes(1) == [22]


def test_optional_wandb_logging_has_unique_ids_for_train_and_evaluation():
    args = SimpleNamespace(
        python=Path("/runtime/python"),
        slot=1,
        wandb_project="majepa-ppo-treatments",
        wandb_entity="osaze-obahor",
        wandb_group="ma-jepa-ppo-correction-2663ae5-20260905",
        wandb_run_prefix="corr-2663-20260905",
        expected_source_sha256="a" * 64,
    )
    train = screen.train_command(args, screen.SLOTS[1], Path("/results/train"))
    final = screen.eval_command(
        args, screen.SLOTS[1], Path("/results/final"), Path("/checkpoint")
    )
    for command in (train, final):
        start = command.index("--logger.outputs")
        assert command[start + 1 : start + 3] == ["jsonl", "wandb"]
    before = {"WANDB_MODE": "online", "OTHER": "preserved"}
    train_env = screen.phase_environment(args, Path("/results/train"), "train", before)
    final_env = screen.phase_environment(
        args, Path("/results/final"), "final128", before
    )
    assert train_env["WANDB_RUN_ID"] == "corr-2663-20260905-s1-train"
    assert final_env["WANDB_RUN_ID"] == "corr-2663-20260905-s1-final128"
    assert train_env["WANDB_RESUME"] == "never"
    assert train_env["WANDB_RUN_GROUP"] == args.wandb_group
    assert before == {"WANDB_MODE": "online", "OTHER": "preserved"}


def test_disabled_logging_does_not_inherit_another_run_id(monkeypatch):
    args = SimpleNamespace(
        source=Path("/source"),
        external=Path("/external"),
        sc2=Path("/sc2"),
        gpu=0,
        portserver_address="@local",
    )
    monkeypatch.setenv("WANDB_RUN_ID", "unrelated")
    env = screen.execution_environment(args)
    assert env["WANDB_MODE"] == "disabled"
    assert "WANDB_RUN_ID" not in env
