#!/usr/bin/env python3
"""Isolate world-model learning rate, actor size, and their interaction."""

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

import run_ppo_correction_screen as base
import run_ppo_recurrent_extensions as recurrent


@dataclass(frozen=True)
class RunSpec(recurrent.RunSpec):
    world_lr: float = 4e-5
    actor_layers: int = 3
    actor_units: int = 1024

    @property
    def configuration_flags(self):
        return super().configuration_flags + (
            "--agent.opt.lr", str(self.world_lr),
            "--agent.marl.ctde.opt.lr", str(self.world_lr),
            "--agent.policy.layers", str(self.actor_layers),
            "--agent.policy.units", str(self.actor_units),
        )


def run_spec(slot):
    arms = {
        0: ("recurrent_wmlr1e4", dict(world_lr=1e-4)),
        1: ("recurrent_actor2x256", dict(actor_layers=2, actor_units=256)),
        2: ("recurrent_wmlr1e4_actor2x256", dict(
            world_lr=1e-4, actor_layers=2, actor_units=256)),
    }
    arm, changes = arms[slot]
    return RunSpec(
        arm=arm, map_name="2s3z", seed=0, replay_value_scale=0.3,
        imag_length=5, recurrent=True, self_fed_scale=0.1,
        slowvalue_rate=1.0, **changes,
    )


def validate_profile(args, run, env):
    resolved = recurrent.validate_profile(args, run, env)
    commands = {
        "train": base.train_command(args, run, Path("unused"))[3:],
        "final128": base.eval_command(args, run, Path("unused"), Path("checkpoint"))[3:],
    }
    code = """
import json, sys, elements
from majepa.main import _load_configs, _resolve_config_profiles
result = {}
for phase, flags in json.loads(sys.argv[1]).items():
    parsed, other = elements.Flags(configs=['smac_vector','ma_jepa']).parse_known(flags)
    c = elements.Flags(_resolve_config_profiles(_load_configs(), parsed.configs)).parse(other)
    result[phase] = dict(local_lr=float(c.agent.opt.lr), joint_lr=float(c.agent.marl.ctde.opt.lr),
                        actor_layers=int(c.agent.policy.layers), actor_units=int(c.agent.policy.units),
                        actor_lr=float(c.agent.ppo.actor_lr), critic_lr=float(c.agent.ppo.critic_lr))
print(json.dumps(result))
"""
    output = subprocess.check_output(
        [str(args.python), "-c", code, json.dumps(commands)],
        cwd=args.source, env={**env, "JAX_PLATFORMS": "cpu"}, text=True,
    )
    actual = json.loads(output.strip().splitlines()[-1])
    expected = dict(local_lr=run.world_lr, joint_lr=run.world_lr,
                    actor_layers=run.actor_layers, actor_units=run.actor_units,
                    actor_lr=3e-5, critic_lr=3e-5)
    if actual != {phase: expected for phase in commands}:
        raise ValueError(f"Capacity configuration mismatch: {actual}")
    return dict(resolved, capacity=actual)


def main():
    parser = base.argument_parser()
    args = parser.parse_args()
    if args.slot not in range(3):
        parser.error("Require one of three capacity slots")
    args.screen_label = "Recurrent learning-rate and actor-capacity screen"
    base.run_screen(
        args, run=run_spec(args.slot), profile_validator=validate_profile,
        extra_manifest={
            "screen": "recurrent_capacity_20260905",
            "reference_run": "pre-rec-ema-20260905-s1-train",
            "interpretation": "Three cells of a 2x2 comparison; existing recurrent reference is the fourth. 2s3z seed0 screening only.",
        },
    )


if __name__ == "__main__":
    main()
