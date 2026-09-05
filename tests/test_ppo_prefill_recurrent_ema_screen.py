"""Verify the user-selected six arms and preserve existing runner defaults."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

SCRIPTS = Path(__file__).parents[1] / "scripts"
for name in (
    "run_ppo_correction_screen",
    "run_ppo_self_fed_screen",
    "run_ppo_prefill_recurrent_ema_screen",
):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
screen = sys.modules["run_ppo_prefill_recurrent_ema_screen"]


def value(command, flag):
    assert command.count(flag) == 1
    return command[command.index(flag) + 1]


def test_six_arms_share_prefill_and_fixed_evaluation():
    args = SimpleNamespace(python=Path("/runtime/python"))
    for slot in range(6):
        run = screen.run_spec(slot, 0.1)
        assert run.map_name == ("3m", "2s3z", "8m")[slot % 3]
        assert run.seed == 0 and run.num_agents == (3, 5, 8)[slot % 3]
        train = screen.base.train_command(args, run, Path("/train"))
        final = screen.base.eval_command(args, run, Path("/final"), Path("/checkpoint"))
        for command in (train, final):
            assert value(command, "--agent.imag_length") == "5"
            assert value(command, "--agent.ppo.replay_value_scale") == "0.3"
            assert value(command, "--agent.opt.warmup") == "0"
            assert value(command, "--agent.marl.ctde.opt.warmup") == "0"
            assert value(command, "--agent.slowvalue.rate") == (
                "1.0" if slot < 3 else "0.02"
            )
            if slot < 3:
                assert (
                    value(command, "--agent.marl.ctde.self_fed.consumer_kl_scale")
                    == "0.0"
                )
                assert value(command, "--agent.marl.ctde.self_fed.scale") == "0.1"
            else:
                assert not any("self_fed" in field for field in command)
        assert value(train, "--run.world_model_start_step") == "5000"
        assert value(train, "--run.ppo_start_step") == "5000"
        assert value(train, "--run.steps") == "50000"
        assert value(train, "--run.envs") == "1"
        assert value(final, "--run.eval_eps") == "128"
        assert value(final, "--run.eval_worker_offset") == "100000"


def test_existing_correction_profile_keeps_original_startup():
    args = SimpleNamespace(python=Path("/runtime/python"))
    command = screen.base.train_command(args, screen.base.SLOTS[0], Path("/train"))
    assert value(command, "--run.world_model_start_step") == "0"
    assert "--agent.opt.warmup" not in command
    assert "--agent.slowvalue.rate" not in command


def test_wandb_has_unique_phase_and_slot_identity():
    for slot in range(6):
        args = SimpleNamespace(
            slot=slot,
            run_spec=screen.run_spec(slot, 0.1),
            wandb_project="majepa-ppo-treatments",
            wandb_entity="osaze-obahor",
            wandb_group="ma-jepa-prefill-recurrent-vs-ema-20260905",
            wandb_run_prefix="pre-rec-ema-20260905",
            expected_source_sha256="pinned",
        )
        for phase in ("train", "final128"):
            env = screen.base.phase_environment(args, Path("/run"), phase, {})
            assert env["WANDB_RUN_ID"] == f"pre-rec-ema-20260905-s{slot}-{phase}"
            assert args.run_spec.name in env["WANDB_NAME"]
            assert env["WANDB_RESUME"] == "never"
