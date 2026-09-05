#!/usr/bin/env python3
"""Bounded full-shape learner gate on copied frozen replay/checkpoint inputs.

No environment collection or checkpoint saves. The canonical runtime loads the
frozen checkpoint and updates only its disposable in-memory model/replay copy.
Run only after a GPU has explicitly been assigned for this gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, values):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(values, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def gate_config(saved, output, *, enabled=True):
    """Keep all saved model, batch, replay and PPO settings except this treatment."""
    import elements

    config = elements.Config(saved)
    if config.task != "smac_2s3z":
        raise ValueError("This production gate is pinned to the 2s3z profile")
    if config.replay.sampling != "recent_world_uniform_behavior":
        raise ValueError("The gate requires the maintained independent replay views")
    if config.jax.platform not in {"cuda", "gpu"}:
        raise ValueError("The saved production config must select a GPU platform")
    overrides = {
        "logdir": str(output / "runtime"),
        "jax.precompile": False,
        "jax.policy_devices": [0],
        "jax.train_devices": [0],
        "jax.expect_devices": 0,
        "jax.profiler": False,
        "agent.marl.ctde.self_fed.enabled": bool(enabled),
        "agent.marl.ctde.self_fed.horizons": [2, 4, 5],
        "agent.marl.ctde.self_fed.anchors": 8,
        "agent.marl.ctde.self_fed.scale": 0.1,
        "agent.marl.ctde.self_fed.consumer_kl_scale": 0.0,
    }
    return elements.Config({**config.flat, **overrides}), overrides


def copy_inputs(run, checkpoint, output):
    """Copy every input before runtime construction; never use source as logdir."""
    run, checkpoint, output = (
        Path(value).resolve() for value in (run, checkpoint, output)
    )
    if output.is_relative_to(run) or output.is_relative_to(checkpoint):
        raise ValueError("Gate output must be outside the preserved inputs")
    if output.exists():
        raise FileExistsError(output)
    paths = [
        run / "config.yaml",
        checkpoint / "agent.pkl",
        checkpoint / "done",
        checkpoint / "step.pkl",
    ]
    replay_files = sorted((run / "replay").glob("*.npz"))
    if not replay_files or any(not path.is_file() for path in paths):
        raise ValueError(
            "The gate requires a completed checkpoint, step, config and replay chunks"
        )
    paths += replay_files
    output.mkdir(parents=True, exist_ok=False)
    copied_checkpoint = output / "inputs" / "checkpoint"
    copied_replay = output / "runtime" / "replay"
    copied_checkpoint.mkdir(parents=True)
    copied_replay.mkdir(parents=True)
    manifest = []
    for source in paths:
        if source == run / "config.yaml":
            target = output / "inputs" / "config.yaml"
        elif source.parent == checkpoint:
            target = copied_checkpoint / source.name
        else:
            target = copied_replay / source.name
        before = digest_file(source)
        shutil.copy2(source, target)
        if digest_file(target) != before:
            raise ValueError(f"Copy verification failed: {source}")
        manifest.append(
            {
                "source": str(source),
                "copy": str(target),
                "sha256": before,
                "bytes": source.stat().st_size,
            }
        )
    write_json(output / "input_manifest.json", manifest)
    return copied_checkpoint, manifest


def parse_process_memory(text, pid):
    total = 0
    for line in text.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) >= 2 and values[0] == str(pid):
            total += int(values[1])
    return total


class ProcessMemorySampler:
    """Sample our process allocation, independently of JAX allocator counters."""

    def __init__(self, output=None):
        self.stop = threading.Event()
        self.peak_mib = 0
        self.samples = 0
        self.errors = []
        self.output = output
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    capture_output=True,
                    timeout=5,
                    check=True,
                )
                self.peak_mib = max(
                    self.peak_mib, parse_process_memory(result.stdout, os.getpid())
                )
                self.samples += 1
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                if len(self.errors) < 3:
                    self.errors.append(str(error))
            if self.output and self.samples % 4 == 0:
                write_json(self.output, self.result())
            self.stop.wait(0.25)

    def result(self):
        return {
            "process_peak_mib_sampled": self.peak_mib,
            "samples": self.samples,
            "interval_seconds": 0.25,
            "errors": self.errors,
        }


def scalar_metrics(metrics):
    import numpy as np

    result = {}
    for name, value in metrics.items():
        array = np.asarray(value)
        if not np.isfinite(array.astype(np.float32)).all():
            raise ValueError(f"Nonfinite learner metric: {name}")
        if array.ndim == 0:
            result[name] = float(array)
    return result


def require_update_metrics(metrics, *, enabled):
    for group in ("local_world", "joint_world"):
        if metrics[f"opt/{group}/skipped"] != 0:
            raise ValueError(f"Nonfinite or skipped {group} update")
    if enabled:
        for horizon in (2, 4, 5):
            if metrics[f"ctde/self_fed_train_h{horizon}/team_count"] <= 0:
                raise ValueError(f"The real batch had no valid H{horizon} endpoints")


def run(args):
    import pickle
    from ruamel.yaml import YAML

    output = args.output.expanduser().resolve()
    copied_checkpoint, inputs = copy_inputs(args.run, args.checkpoint, output)
    saved = YAML(typ="safe").load((output / "inputs" / "config.yaml").read_text())
    config, overrides = gate_config(saved, output, enabled=not args.disabled_control)
    config.save(str(output / "gate_config.yaml"))
    with (copied_checkpoint / "step.pkl").open("rb") as stream:
        environment_step = int(pickle.load(stream))
    source_root = Path(__file__).resolve().parents[1]
    source_files = sorted((source_root / "src" / "majepa").rglob("*.py"))
    source_files += [
        source_root / "src" / "majepa" / "configs.yaml",
        Path(__file__).resolve(),
    ]
    provenance = {
        "command": sys.argv,
        "pid": os.getpid(),
        "python": sys.executable,
        "started_unix": time.time(),
        "environment_step": environment_step,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_files": {
            str(path.relative_to(source_root)): digest_file(path)
            for path in source_files
        },
        "overrides": overrides,
        "requested_updates": args.updates,
        "checkpoint_saves": 0,
        "environment_collection_steps": 0,
        "replay_note": "Frozen serialized replay copied before load; archive codes may predate final in-memory replay refreshes.",
    }
    try:
        provenance["git_commit"] = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        provenance["git_commit"] = None
    write_json(output / "provenance.json", provenance)
    if args.validate_only:
        write_json(
            output / "result.json", {"status": "validated_only", "gpu_execution": False}
        )
        return

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if len(visible.split(",")) != 1 or visible in {"", "-1"}:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one assigned GPU")

    sampler = ProcessMemorySampler(output / "memory.json")
    sampler.thread.start()
    result = {"status": "running", "updates": [], "input_files_unchanged": None}
    write_json(output / "result.json", result)
    try:
        import elements
        import importlib.metadata
        import importlib
        import jax
        import jax.numpy as jnp
        import numpy as np
        from majepa.main import make_agent, make_replay, make_stream
        from majepa.train import _with_prefixed_batch

        result["runtime"] = {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "ninjax")
        }
        result["runtime_files"] = {}
        for name in (
            "embodied.jax.agent",
            "embodied.jax.internal",
            "embodied.core.replay",
        ):
            path = Path(importlib.import_module(name).__file__).resolve()
            result["runtime_files"][str(path)] = digest_file(path)
        result["stage"] = "initializing"
        write_json(output / "result.json", result)
        start = time.perf_counter()
        # Let the canonical constructor configure XLA/transfer/preallocation
        # flags before any JAX backend initialization.
        agent = make_agent(config)
        jax.block_until_ready(agent.params)
        devices = jax.local_devices(backend="gpu")
        if len(devices) != 1:
            raise ValueError("Expose exactly one explicitly assigned GPU for this gate")
        result["device"] = str(devices[0])
        result["device_kind"] = devices[0].device_kind
        result["initialize_seconds"] = time.perf_counter() - start
        result["memory_after_initialize"] = devices[0].memory_stats()
        result["stage"] = "loading_checkpoint"
        write_json(output / "result.json", result)
        start = time.perf_counter()
        loader = elements.Checkpoint()
        loader.agent = agent
        loader.load(str(copied_checkpoint), keys=["agent"])
        jax.block_until_ready(agent.params)
        result["checkpoint_load_seconds"] = time.perf_counter() - start
        result["updates_before"] = int(agent.n_updates)
        result["actions_before"] = int(agent.n_actions)
        result["stage"] = "loading_replay"
        write_json(output / "result.json", result)

        replay = make_replay(config, "replay")
        replay.load()
        if len(replay) < config.batch_size * config.batch_length:
            raise ValueError("Copied replay does not provide a production batch")
        result["replay_items"] = len(replay)
        paired = _with_prefixed_batch(
            make_stream(config, replay, "train_world"),
            make_stream(config, replay, "train_behavior"),
            "_behavior_replay/",
        )

        def stamped():
            for batch in paired:
                batch["_environment_step"] = np.full(
                    batch["is_first"].shape, environment_step, np.int32
                )
                yield batch

        batches = iter(agent.stream(stamped()))
        carry = agent.init_train(config.batch_size)
        finite = jax.jit(
            lambda values: jnp.stack(
                [jnp.isfinite(value).all() for value in jax.tree.leaves(values)]
            ).all()
        )
        if not bool(jax.device_get(finite(agent.params))):
            raise ValueError("Loaded parameters/optimizer state are nonfinite")
        for index in range(args.updates):
            batch = next(batches)
            result["stage"] = f"update_{index}"
            write_json(output / "result.json", result)
            start = time.perf_counter()
            carry, _, _ = agent.train(carry, dict(batch))
            jax.block_until_ready((agent.params, carry))
            seconds = time.perf_counter() - start
            # Canonical train returns previous-call metrics. Drain THIS call's
            # asynchronous result directly; never run an extra update to flush.
            metrics = scalar_metrics(agent._take_outs(agent.pending_mets))
            agent.pending_mets = None
            outputs = agent._take_outs(agent.pending_outs)
            agent.pending_outs = None
            if "replay" in outputs:
                replay.update(outputs["replay"])  # Disposable in-memory copy.
            state_finite = bool(jax.device_get(finite(agent.params)))
            entry = {
                "index": index,
                "seconds": seconds,
                "includes_first_compile": index == 0,
                "all_parameters_and_optimizer_finite": state_finite,
                "metrics": metrics,
                "memory": devices[0].memory_stats(),
            }
            result["updates"].append(entry)
            result["sampled_memory"] = sampler.result()
            write_json(output / "result.json", result)
            if not state_finite:
                raise ValueError("Update produced nonfinite parameters/optimizer state")
            require_update_metrics(metrics, enabled=not args.disabled_control)
        result["updates_after"] = int(agent.n_updates)
        result["actions_after"] = int(agent.n_actions)
        if result["updates_after"] - result["updates_before"] != args.updates:
            raise ValueError("Unexpected learner update count")
        if result["actions_before"] != result["actions_after"]:
            raise ValueError("Gate unexpectedly executed actor/environment actions")
        result["status"] = "passed"
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        sampler.stop.set()
        sampler.thread.join(timeout=6)
        result["sampled_memory"] = sampler.result()
        result["input_files_unchanged"] = all(
            digest_file(item["source"]) == item["sha256"] for item in inputs
        )
        if not result["input_files_unchanged"]:
            result["status"] = "failed"
        result["finished_unix"] = time.time()
        write_json(output / "result.json", result)
    if not result["input_files_unchanged"]:
        raise ValueError("Preserved input files changed during gate")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--disabled-control", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.updates <= 5:
        parser.error("The gate is bounded to one through five learner updates")
    run(args)


if __name__ == "__main__":
    main()
