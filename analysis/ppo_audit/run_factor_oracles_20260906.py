"""Follow the paired audit with separate observation/liveness interventions."""

import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path("/workspace/majepa_seed_diagnostics_20260906")
SOURCE = Path("/workspace/ma_jepa_recurrent_extensions_f2273e9")
TOOLS = Path("/workspace/majepa_seed_diagnostic_tools_20260906")


def main():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="2", XLA_PYTHON_CLIENT_MEM_FRACTION="0.14",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               PYTHONPATH=f"{SOURCE}/src:/workspace/external/dreamerv3")
    results = []
    for seed in (0, 1, 2):
        name = f"factor-oracles-3s_vs_3z-s{seed}"
        run = Path("/workspace/majepa_bptt2_matrix_20260905/runs") / f"bptt2-3s_vs_3z-seed{seed}/train/run"
        command = ["/workspace/majepa-runtime/bin/python", str(TOOLS / "scripts/audit_causal_simulator.py"),
                   "--run", str(run), "--replay", str(ROOT / "common_3s_vs_3z"),
                   "--output", str(ROOT / name), "--external", "/workspace/external/dreamerv3",
                   "--platform", "cuda", "--roots", "96", "--max-episodes", "96",
                   "--max-chunks", "3", "--horizons", "1", "2", "4", "5", "8",
                   "--seed", "2718", "--factor-oracles"]
        started = time.time()
        with (ROOT / f"{name}.log").open("x") as log:
            child = subprocess.Popen(command, env=env, cwd=SOURCE, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            (ROOT / "factor_status.json").write_text(json.dumps({"phase": name,
                "pid": child.pid, "started_at": started}, indent=2) + "\n")
            code = child.wait()
        row = {"name": name, "command": command, "returncode": code,
               "started_at": started, "finished_at": time.time()}
        (ROOT / f"{name}.execution.json").write_text(json.dumps(row, indent=2) + "\n")
        results.append(row)
        if code:
            raise RuntimeError(f"{name} exited with {code}")
    (ROOT / "factor_status.json").write_text(json.dumps({"phase": "complete",
        "finished_at": time.time(), "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
