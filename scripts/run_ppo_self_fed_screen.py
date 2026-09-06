#!/usr/bin/env python3
"""Run the recurrent-H5 versus corrected-H2 screen using pinned source packages.

No anchor is selected from partial curves. Actual training requires all four
corrected baseline final128 outcomes and explicit disabled-parity evidence.
--validate-only --candidate-anchor-scale allows provisional config checks.
The correction runner retains process ownership, idle waiting, and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess

import run_ppo_correction_screen as base


@dataclass(frozen=True)
class RunSpec(base.RunSpec):
    imag_length: int
    recurrent: bool
    self_fed_scale: float

    @property
    def configuration_flags(self):
        if not self.recurrent:
            return ()  # Old corrected source has no optional self-fed fields.
        prefix = "--agent.marl.ctde.self_fed."
        return (
            prefix + "enabled",
            "True",
            prefix + "horizons",
            "2",
            "4",
            "5",
            prefix + "anchors",
            "8",
            prefix + "scale",
            str(self.self_fed_scale),
            prefix + "consumer_kl_scale",
            "0.0",
        )


def run_spec(slot, anchor, scale):
    profiles = {
        0: ("recurrent_h5", "2s3z", 0, 5, True),
        1: ("corrected_h2", "2s3z", 0, 2, False),
        2: ("recurrent_h5", "2s3z", 1, 5, True),
        3: ("corrected_h2", "2s3z", 1, 2, False),
        4: ("recurrent_h5", "3m", 0, 5, True),
        5: ("corrected_h5", "2s3z", 123, 5, False),
    }
    arm, task, seed, horizon, recurrent = profiles[slot]
    return RunSpec(
        arm, task, seed, anchor, horizon, recurrent, scale if recurrent else 0.0
    )


def select_anchor(root, expected_hash):
    """Choose on mean final128 wins, then return; exact tie retains scale0.3."""
    records = []
    for scale, arm in [(0.0, "team_return"), (0.3, "team_return_anchor")]:
        for seed in (0, 1):
            run = root / "runs" / f"{arm}-2s3z-seed{seed}"
            manifest = json.loads((run / "manifest.json").read_text())
            outcome = json.loads((run / "outcome.json").read_text())
            summary = outcome.get("summary", {})
            config = manifest.get("configuration", {})
            if (
                outcome.get("completed") is not True
                or summary.get("evaluation_protocol", {}).get("episodes") != 128
                or manifest.get("protocol", {}).get("train_steps") != 50000
                or manifest.get("source_sha256") != expected_hash
                or config.get("replay_value_scale") != scale
                or config.get("seed") != seed
                or config.get("task") != "smac_2s3z"
            ):
                raise ValueError(
                    f"Missing or mismatched fixed-final baseline evidence: {run.name}"
                )
            win_rate, returns = (
                float(summary["win_rate"]),
                float(summary["return_mean"]),
            )
            if not 0 <= win_rate <= 1 or not math.isfinite(returns):
                raise ValueError("Invalid baseline final metrics")
            records.append(
                {
                    "run": run.name,
                    "replay_value_scale": scale,
                    "seed": seed,
                    "win_rate": win_rate,
                    "return_mean": returns,
                    "outcome_path": str(run / "outcome.json"),
                }
            )

    def score(scale):
        peers = [row for row in records if row["replay_value_scale"] == scale]
        return (
            sum(row["win_rate"] for row in peers) / 2,
            sum(row["return_mean"] for row in peers) / 2,
            scale == 0.3,
        )

    selected = max((0.0, 0.3), key=score)
    return {
        "selected_replay_value_scale": selected,
        "provisional": False,
        "rule": "mean final128 win rate, then mean final return, then retain0.3",
        "source_sha256": expected_hash,
        "outcomes": records,
    }


def validate_parity(record, baseline_hash, optional_hash):
    if (
        record.get("passed") is not True
        or record.get("parity_kind") != "cross_source_executed_train"
        or record.get("baseline_source_sha256") != baseline_hash
        or record.get("optional_source_sha256") != optional_hash
        or not record.get("test_command")
    ):
        raise ValueError("Disabled-parity evidence does not match both source packages")
    return record


EXTRA_VALIDATION = """
import json,pathlib,sys
import elements,majepa
from majepa.main import _load_configs,_resolve_config_profiles
if not pathlib.Path(majepa.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve()):
 raise RuntimeError('Package import escaped pinned source')
out={}
for phase,flags in json.loads(sys.argv[2]).items():
 parsed,other=elements.Flags(configs=['smac_vector','ma_jepa']).parse_known(flags)
 c=elements.Flags(_resolve_config_profiles(_load_configs(),parsed.configs)).parse(other)
 sf=c.agent.marl.ctde.self_fed if hasattr(c.agent.marl.ctde,'self_fed') else None
 out[phase]={'imag_length':int(c.agent.imag_length),'self_fed':dict(sf) if sf is not None else None}
 if phase=='train':out[phase].update(steps=int(c.run.steps),envs=int(c.run.envs),curve_interval=int(c.run.curve_eval_interval),curve_eps=int(c.run.curve_eval_eps),curve_seed_offset=int(c.run.curve_eval_seed_offset),checkpoint_at_curve_eval=bool(c.run.checkpoint_at_curve_eval))
 else:out[phase].update(episodes=int(c.run.eval_eps),envs=int(c.run.envs),seed_offset=int(c.run.eval_worker_offset),policy_mode=str(c.run.eval_policy_mode))
print(json.dumps(out))
"""


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
    self_fed = (
        {
            "enabled": True,
            "horizons": [2, 4, 5],
            "anchors": 8,
            "scale": run.self_fed_scale,
            "consumer_kl_scale": 0.0,
            "trajectory_kl_scale": 0.0,
        }
        if run.recurrent
        else None
    )
    expected = {
        "train": {
            "imag_length": run.imag_length,
            "self_fed": self_fed,
            "steps": 50000,
            "envs": 1,
            "curve_interval": 5000,
            "curve_eps": 32,
            "curve_seed_offset": 50000,
            "checkpoint_at_curve_eval": False,
        },
        "final128": {
            "imag_length": run.imag_length,
            "self_fed": self_fed,
            "episodes": 128,
            "envs": 4,
            "seed_offset": 100000,
            "policy_mode": "eval",
        },
    }
    if actual != expected:
        raise RuntimeError(
            f"Recurrent screen configuration mismatch: {actual} != {expected}"
        )
    return {**resolved, "phases": actual}


def prepare(args):
    if not math.isfinite(args.self_fed_scale) or args.self_fed_scale <= 0:
        raise ValueError(
            "Specify a finite, positive recurrent loss scale after validation"
        )
    optional_source, optional_hash = args.source.absolute(), args.expected_source_sha256
    baseline_source = args.baseline_source.absolute()
    for source, expected in [
        (optional_source, optional_hash),
        (baseline_source, args.expected_baseline_source_sha256),
    ]:
        if base.source_fingerprint(source) != expected:
            raise ValueError(f"Source fingerprint mismatch: {source}")
    if args.candidate_anchor_scale is not None:
        if not args.validate_only:
            raise ValueError(
                "A candidate anchor is permitted for configuration validation only"
            )
        decision = {
            "selected_replay_value_scale": args.candidate_anchor_scale,
            "provisional": True,
            "rule": "configuration check only; no experiment selection",
        }
    else:
        if args.baseline_results_root is None:
            raise ValueError(
                "Actual launch requires completed baseline final128 results"
            )
        decision = select_anchor(
            args.baseline_results_root, args.expected_baseline_source_sha256
        )
    parity = None
    if args.disabled_parity_record:
        parity = validate_parity(
            json.loads(args.disabled_parity_record.read_text()),
            args.expected_baseline_source_sha256,
            optional_hash,
        )
    elif not args.validate_only:
        raise ValueError("Actual launch requires source-bound disabled-parity evidence")
    run = run_spec(
        args.slot, decision["selected_replay_value_scale"], args.self_fed_scale
    )
    args.source = optional_source if run.recurrent else baseline_source
    args.expected_source_sha256 = (
        optional_hash if run.recurrent else args.expected_baseline_source_sha256
    )
    args.screen_label = "Recurrent-H5 versus corrected-H2 screen"
    provenance = {
        "screen": "recurrent_h5_and_corrected_h2",
        "anchor_decision": decision,
        "source_comparison": {
            "baseline_source": str(baseline_source),
            "baseline_source_sha256": args.expected_baseline_source_sha256,
            "optional_source": str(optional_source),
            "optional_source_sha256": optional_hash,
            "disabled_parity": parity,
        },
    }
    return run, provenance


def main():
    parser = base.argument_parser()
    parser.description = __doc__
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--expected-baseline-source-sha256", required=True)
    parser.add_argument("--baseline-results-root", type=Path)
    parser.add_argument("--self-fed-scale", type=float, required=True)
    parser.add_argument("--candidate-anchor-scale", type=float, choices=(0.0, 0.3))
    parser.add_argument("--disabled-parity-record", type=Path)
    args = parser.parse_args()
    run, provenance = prepare(args)
    base.run_screen(
        args, run=run, profile_validator=validate_profile, extra_manifest=provenance
    )


if __name__ == "__main__":
    main()
