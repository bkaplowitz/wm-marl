#!/usr/bin/env python3
"""Launch the user-selected six-run prefill recurrent-versus-EMA screen.

Three maps, seed 0, fixed 50k training/final128. Shared replay prefill, zero
world-optimizer warmups and factual critic anchor .3. Recurrent loss and slow
critic EMA are separate candidate packages; this is not a factorial comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess

import run_ppo_correction_screen as base
import run_ppo_self_fed_screen as recurrent


@dataclass(frozen=True)
class RunSpec(recurrent.RunSpec):
    slowvalue_rate: float
    world_model_start_step: int = 5000
    world_warmup: int = 0

    @property
    def num_agents(self):
        return {"3m": 3, "2s3z": 5, "8m": 8}[self.map_name]

    @property
    def configuration_flags(self):
        return super().configuration_flags + (
            "--agent.opt.warmup",
            "0",
            "--agent.marl.ctde.opt.warmup",
            "0",
            "--agent.slowvalue.rate",
            str(self.slowvalue_rate),
        )


def run_spec(slot, scale):
    if slot not in range(6):
        raise ValueError("Require slot 0 through 5")
    use_recurrent = slot < 3
    task = ("3m", "2s3z", "8m")[slot % 3]
    return RunSpec(
        "prefill_recurrent" if use_recurrent else "prefill_ema",
        task,
        0,
        0.3,
        5,
        use_recurrent,
        scale if use_recurrent else 0.0,
        1.0 if use_recurrent else 0.02,
    )


EXTRA_VALIDATION = recurrent.EXTRA_VALIDATION.replace(
    "'imag_length':int(c.agent.imag_length),",
    "'slowvalue_rate':float(c.agent.slowvalue.rate),'num_agents':int(c.agent.num_agents),'imag_length':int(c.agent.imag_length),",
)


def validate_profile(args, run, env):
    resolved = base.validate_profile(args, run, env)
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
            EXTRA_VALIDATION,
            str(args.source),
            json.dumps(commands),
        ],
        env={**env, "JAX_PLATFORMS": "cpu"},
        cwd=args.source,
        text=True,
    )
    actual = json.loads(output.strip().splitlines()[-1])
    sf = (
        {
            "enabled": True,
            "horizons": [2, 4, 5],
            "anchors": 8,
            "scale": run.self_fed_scale,
            "consumer_kl_scale": 0.0,
        }
        if run.recurrent
        else None
    )
    common = {
        "imag_length": 5,
        "self_fed": sf,
        "slowvalue_rate": run.slowvalue_rate,
        "num_agents": run.num_agents,
    }
    expected = {
        "train": {
            **common,
            "steps": 50000,
            "envs": 1,
            "curve_interval": 5000,
            "curve_eps": 32,
            "curve_seed_offset": 50000,
            "checkpoint_at_curve_eval": False,
        },
        "final128": {
            **common,
            "episodes": 128,
            "envs": 4,
            "seed_offset": 100000,
            "policy_mode": "eval",
        },
    }
    if actual != expected:
        raise RuntimeError(
            f"Prefill screen configuration mismatch: {actual} != {expected}"
        )
    return {**resolved, "phases": actual}


def prepare(args):
    if not math.isfinite(args.self_fed_scale) or args.self_fed_scale <= 0:
        raise ValueError("Require finite, positive validated recurrent scale")
    optional_source, optional_hash = args.source.absolute(), args.expected_source_sha256
    baseline_source = args.baseline_source.absolute()
    for source, expected in (
        (optional_source, optional_hash),
        (baseline_source, args.expected_baseline_source_sha256),
    ):
        if base.source_fingerprint(source) != expected:
            raise ValueError(f"Source fingerprint mismatch: {source}")
    parity = recurrent.validate_parity(
        json.loads(args.disabled_parity_record.read_text()),
        args.expected_baseline_source_sha256,
        optional_hash,
    )
    decision = recurrent.select_anchor(
        args.baseline_results_root, args.expected_baseline_source_sha256
    )
    if decision["selected_replay_value_scale"] != 0.3:
        raise ValueError("Completed evidence does not support the fixed .3 anchor")
    run = run_spec(args.slot, args.self_fed_scale)
    args.source = optional_source if run.recurrent else baseline_source
    args.expected_source_sha256 = (
        optional_hash if run.recurrent else args.expected_baseline_source_sha256
    )
    args.screen_label = "Prefill recurrent versus EMA screen"
    return run, {
        "screen": "prefill_recurrent_vs_ema",
        "anchor_decision": decision,
        "interpretation": "Two candidate packages; recurrence and EMA are not independently identified without a matched no-recurrence/rate1 control.",
        "source_comparison": {
            "baseline_source": str(baseline_source),
            "baseline_source_sha256": args.expected_baseline_source_sha256,
            "optional_source": str(optional_source),
            "optional_source_sha256": optional_hash,
            "disabled_parity": parity,
        },
    }


def main():
    parser = base.argument_parser()
    parser.description = __doc__
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--expected-baseline-source-sha256", required=True)
    parser.add_argument("--baseline-results-root", type=Path, required=True)
    parser.add_argument("--self-fed-scale", type=float, required=True)
    parser.add_argument("--disabled-parity-record", type=Path, required=True)
    args = parser.parse_args()
    run, provenance = prepare(args)
    base.run_screen(
        args, run=run, profile_validator=validate_profile, extra_manifest=provenance
    )


if __name__ == "__main__":
    main()
