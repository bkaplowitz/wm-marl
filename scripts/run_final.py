"""Run one final-suite setting and its dedicated 100-episode evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
MAPS = {
    "2m_vs_1z": (2, 100000), "2s_vs_1sc": (2, 100000),
    "2s3z": (5, 100000), "3m": (3, 100000), "3s_vs_3z": (3, 100000),
    "3s_vs_4z": (3, 100000), "8m": (8, 100000), "MMM": (10, 100000),
    "so_many_baneling": (7, 100000), "3s_vs_5z": (3, 200000),
    "2c_vs_64zg": (2, 200000), "corridor": (6, 400000),
}
ARMS = {"jema": (1.0, 0.1), "outcomes": (1.0, 0.0),
        "jepa_only": (0.0, 0.1), "detached": (0.0, 0.0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", choices=MAPS, default="2s3z")
    parser.add_argument("--seed", type=int, default=1302)
    parser.add_argument("--arm", choices=ARMS, default="jema")
    parser.add_argument("--output", type=Path, help="New output directory")
    options = parser.parse_args()
    if not os.environ.get("SC2PATH"):
        parser.error("Set SC2PATH to your StarCraft II 4.10.0 installation")
    name = f"{options.arm}-{options.map}-s{options.seed}-{uuid.uuid4().hex[:10]}"
    output = (options.output or ROOT / "runs" / name).resolve()
    output.mkdir(parents=True, exist_ok=False)
    agents, steps = MAPS[options.map]
    args = json.loads((ROOT / "scripts/train_args.json").read_text())
    for key, value in zip(("outcome_input_scale", "outcome_jepa_scale"), ARMS[options.arm]):
        args[args.index("--agent.world_model_gradients." + key) + 1] = str(value)
    args += ["--task", "smac_" + options.map, "--seed", str(options.seed),
             "--agent.num_agents", str(agents), "--run.steps", str(steps)]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}/src:{ROOT}/external/dreamerv3"
    env.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/majepa_jax_cache")
    if env.get("WANDB_PROJECT"):
        args.insert(args.index("--logger.outputs") + 2, "wandb")
    (output / "arguments.json").write_text(json.dumps(args, indent=2))

    def run(phase, extra):
        phase_env = env.copy()
        phase_env.update(WANDB_RUN_ID=name + "-" + phase, WANDB_RESUME="never",
                         WANDB_NAME=name + "-" + phase, WANDB_JOB_TYPE=phase)
        subprocess.run([sys.executable, "-m", "majepa.main", *args,
                        "--logdir", str(output / phase), *extra],
                       env=phase_env, cwd=ROOT, check=True)

    run("train", [])
    checkpoint_root = output / "train" / "ckpt"
    checkpoint = checkpoint_root / (checkpoint_root / "latest").read_text().strip()
    if not (checkpoint / "done").exists():
        raise RuntimeError(f"Incomplete checkpoint: {checkpoint}")
    run("final100", ["--script", "eval_only", "--run.from_checkpoint", str(checkpoint),
         "--run.eval_worker_offset", "100000", "--run.eval_eps", "100", "--run.envs", "4",
         "--run.eval_policy_mode", "eval", "--run.world_model_start_step", "0",
         "--run.curve_eval_interval", "0", "--run.eval_envs", "1", "--jax.precompile", "False"])


if __name__ == "__main__":
    main()
