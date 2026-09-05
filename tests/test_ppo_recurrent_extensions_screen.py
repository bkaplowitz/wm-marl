"""Each new run changes exactly one experimental axis."""

from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import run_ppo_recurrent_extensions as screen


def get(command, flag):
    assert command.count(flag) == 1
    return command[command.index(flag) + 1]


def test_four_independent_extensions_keep_budget_and_final_evaluation():
    args = SimpleNamespace(python=Path("python"))
    for slot in range(4):
        run = screen.run_spec(slot)
        assert run.map_name == "2s3z" and run.seed == 0
        assert (
            sum(
                [
                    run.fresh_history,
                    run.bptt_steps != 1,
                    run.envs != 1,
                    run.slowvalue_rate != 1,
                ]
            )
            == 1
        )
        train = screen.base.train_command(args, run, Path("train"))
        final = screen.base.eval_command(args, run, Path("eval"), Path("checkpoint"))
        assert get(train, "--run.envs") == ("16" if slot == 2 else "1")
        assert get(final, "--run.envs") == "4"
        assert get(train, "--run.train_ratio") == "128"
        assert get(train, "--run.steps") == "50000"
        assert get(train, "--run.world_model_start_step") == "5000"
        assert get(final, "--run.eval_eps") == "128"
        assert get(train, "--agent.slowvalue.rate") == ("0.5" if slot == 3 else "1.0")
