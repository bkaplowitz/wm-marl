#!/usr/bin/env python3
"""Eight collection environments with the otherwise unchanged recurrent reference."""

import run_ppo_correction_screen as base
import run_ppo_recurrent_capacity_screen as capacity


def main():
    parser = base.argument_parser()
    args = parser.parse_args()
    if args.slot != 0:
        parser.error("Only slot 0 exists")
    run = capacity.RunSpec(
        arm="recurrent_env8", map_name="2s3z", seed=0,
        replay_value_scale=0.3, imag_length=5, recurrent=True,
        self_fed_scale=0.1, slowvalue_rate=1.0, envs=8,
    )
    args.screen_label = "Recurrent eight-environment screen"
    base.run_screen(
        args, run=run, profile_validator=capacity.validate_profile,
        extra_manifest={
            "screen": "recurrent_env8_20260905",
            "reference_run": "pre-rec-ema-20260905-s1-train",
            "comparison_run": "rec-ext-20260905-s2-train",
            "interpretation": "Only collection environment count changes from reference: 1 to 8. Compare with the separate 16-env arm. 2s3z seed0 screening only.",
        },
    )


if __name__ == "__main__":
    main()
