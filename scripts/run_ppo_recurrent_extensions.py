#!/usr/bin/env python3
"""Four individual recurrent extensions and two paired combinations on 2s3z."""

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

import run_ppo_correction_screen as base
import run_ppo_prefill_recurrent_ema_screen as prefill


@dataclass(frozen=True)
class RunSpec(prefill.RunSpec):
    envs: int = 1
    fresh_history: bool = False
    bptt_steps: int = 1

    @property
    def configuration_flags(self):
        return super().configuration_flags + (
            "--agent.marl.ctde.self_fed.fresh_history",
            str(self.fresh_history),
            "--agent.marl.ctde.self_fed.bptt_steps",
            str(self.bptt_steps),
        )


def run_spec(slot):
    options = {
        0: ("recurrent_fresh_history", {"fresh_history": True}),
        1: ("recurrent_bptt2", {"bptt_steps": 2}),
        2: ("recurrent_env16", {"envs": 16}),
        3: ("recurrent_ema05", {"slowvalue_rate": 0.5}),
        4: ("recurrent_fresh_bptt2", {"fresh_history": True, "bptt_steps": 2}),
        5: ("recurrent_env16_ema05", {"envs": 16, "slowvalue_rate": 0.5}),
    }
    arm, changes = options[slot]
    values = dict(
        arm=arm,
        map_name="2s3z",
        seed=0,
        replay_value_scale=0.3,
        imag_length=5,
        recurrent=True,
        self_fed_scale=0.1,
        slowvalue_rate=1.0,
    )
    return RunSpec(**{**values, **changes})


def validate_profile(args, run, env):
    common = base.validate_profile(args, run, env)
    commands = {
        "train": base.train_command(args, run, Path("unused"))[3:],
        "final128": base.eval_command(args, run, Path("unused"), Path("checkpoint"))[
            3:
        ],
    }
    output = subprocess.check_output(
        [
            str(args.python),
            "-c",
            prefill.EXTRA_VALIDATION,
            str(args.source),
            json.dumps(commands),
        ],
        cwd=args.source,
        env={**env, "JAX_PLATFORMS": "cpu"},
        text=True,
    )
    actual = json.loads(output.strip().splitlines()[-1])
    shared = dict(
        imag_length=5,
        num_agents=5,
        slowvalue_rate=run.slowvalue_rate,
        self_fed=dict(
            enabled=True,
            horizons=[2, 4, 5],
            anchors=8,
            scale=0.1,
            consumer_kl_scale=0.0,
            trajectory_kl_scale=0.0,
            fresh_history=run.fresh_history,
            bptt_steps=run.bptt_steps,
        ),
    )
    expected = {
        "train": dict(
            **shared,
            steps=50000,
            envs=run.envs,
            curve_interval=5000,
            curve_eps=32,
            curve_seed_offset=50000,
            checkpoint_at_curve_eval=False,
        ),
        "final128": dict(
            **shared, envs=4, episodes=128, seed_offset=100000, policy_mode="eval"
        ),
    }
    if actual != expected:
        raise ValueError(f"Extension configuration mismatch: {actual} != {expected}")
    return dict(common, phases=actual)


def main():
    parser = base.argument_parser()
    args = parser.parse_args()
    if args.slot not in range(6):
        parser.error("Only six extension slots exist")
    args.screen_label = "Recurrent extensions"
    base.run_screen(
        args,
        run=run_spec(args.slot),
        profile_validator=validate_profile,
        extra_manifest={
            "screen": "recurrent_extensions_20260905",
            "reference_run": "pre-rec-ema-20260905-s1-train",
            "interpretation": "Individual changes in slots0-3; paired combinations in slots4-5. 2s3z seed0 screening only.",
        },
    )


if __name__ == "__main__":
    main()
