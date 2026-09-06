"""Bounded frozen-checkpoint diagnostics, separate from all training queues."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path("/workspace/majepa_seed_diagnostics_20260906")
SOURCE = Path("/workspace/ma_jepa_recurrent_extensions_f2273e9")
MATRIX = Path("/workspace/majepa_bptt2_matrix_20260905/runs")
PYTHON = "/workspace/majepa-runtime/bin/python"
GPU = "2"
CASES = [("3s_vs_3z", seed) for seed in (0, 1, 2)] + [
    ("3s_vs_5z", seed) for seed in (0, 2)
]


def write(name, value):
    path = ROOT / name
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def execute(name, command, env):
    started = time.time()
    with (ROOT / f"{name}.log").open("x") as log:
        proc = subprocess.Popen(command, env=env, cwd=SOURCE, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        write("status.json", {"phase": name, "pid": proc.pid, "started_at": started})
        code = proc.wait()
    result = {"name": name, "command": command, "returncode": code,
              "started_at": started, "finished_at": time.time()}
    write(f"{name}.execution.json", result)
    if code:
        raise RuntimeError(f"Diagnostic {name} failed with exit code {code}")
    return result


def main():
    ROOT.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPU,
               XLA_PYTHON_CLIENT_MEM_FRACTION="0.14",
               XLA_PYTHON_CLIENT_PREALLOCATE="false",
               PYTHONPATH=f"{SOURCE}/src:/workspace/external/dreamerv3",
               SC2PATH="/workspace/StarCraftII",
               PORTSERVER_ADDRESS="@majepa-seed-diagnostics-20260906",
               PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               WANDB_MODE="disabled", OMP_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    write("manifest.json", {
        "created_at": time.time(), "source": str(SOURCE), "gpu": GPU,
        "training_source_sha256": "af6e6d23ab98e60d893071054942889ca73c14d0d9071505c4a9ee886df46b5a",
        "cases": CASES, "episodes_per_case": 32, "policy_mode": "eval_sample",
        "environment_seed": 900000, "diagnostic_seed": 9001,
        "gpu_memory_fraction": 0.14, "training_mutations": False,
        "common_world_bank": "Equal episode counts from all three 3s_vs_3z policies",
        "world_roots": 96, "world_horizons": [1, 2, 4, 5, 8],
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    portlog = (ROOT / "portserver.log").open("x")
    port = subprocess.Popen([
        PYTHON, "/workspace/majepa-runtime/bin/portserver.py",
        "--portserver_static_pool", "56000-56499", "--portserver_address",
        env["PORTSERVER_ADDRESS"],
    ], env=env, stdout=portlog, stderr=subprocess.STDOUT, start_new_session=True)
    results = []
    try:
        time.sleep(1)
        if port.poll() is not None:
            raise RuntimeError("Diagnostic portserver did not start")
        for map_name, seed in CASES:
            name = f"calibration-{map_name}-s{seed}"
            run = MATRIX / f"bptt2-{map_name}-seed{seed}"
            saved = run / "train/run"
            ckpt = saved / "ckpt" / (saved / "ckpt/latest").read_text().strip()
            results.append(execute(name, [
                PYTHON, str(SOURCE / "scripts/evaluate_value_calibration.py"),
                "--config", str(saved / "config.yaml"), "--checkpoint", str(ckpt),
                "--training-manifest", str(run / "manifest.json"),
                "--output", str(ROOT / name), "--episodes", "32", "--envs", "1",
                "--eval-seed", "800000", "--worker-offset", "100000",
                "--diagnostic-seed", "9001", "--policy-mode", "eval_sample",
                "--platform", "cuda",
            ], env))
        bank = ROOT / "common_3s_vs_3z"
        bank.mkdir()
        for seed in (0, 1, 2):
            os.link(ROOT / f"calibration-3s_vs_3z-s{seed}/episodes.npz",
                    bank / f"policy-seed{seed}.npz")
        for seed in (0, 1, 2):
            name = f"common-world-3s_vs_3z-s{seed}"
            saved = MATRIX / f"bptt2-3s_vs_3z-seed{seed}/train/run"
            results.append(execute(name, [
                PYTHON, str(SOURCE / "scripts/audit_causal_simulator.py"),
                "--run", str(saved), "--replay", str(bank),
                "--output", str(ROOT / name), "--external", "/workspace/external/dreamerv3",
                "--platform", "cuda", "--roots", "96", "--max-episodes", "96",
                "--max-chunks", "3", "--horizons", "1", "2", "4", "5", "8",
                "--seed", "2718",
            ], env))
        write("done.json", {"finished_at": time.time(), "results": results})
        write("status.json", {"phase": "complete", "finished_at": time.time()})
    except Exception as error:
        write("status.json", {"phase": "failed", "error": repr(error),
                              "finished_at": time.time()})
        raise
    finally:
        port.terminate()
        port.wait(timeout=10)
        portlog.close()


if __name__ == "__main__":
    main()
