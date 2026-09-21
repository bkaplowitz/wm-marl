"""Restartable RunPod campaigns with reusable GPU workers and bounded GPU spend."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, suppress
import fcntl
import hashlib
import io
import json
import math
import netrc
import os
from pathlib import Path
import posixpath
import re
import shlex
import subprocess
import sys
import tarfile
import time
import uuid

IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
REPO = Path(__file__).resolve().parents[2]
CAPACITY_MESSAGE = (
    "There are no longer any instances available with the requested specifications"
)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", value
    ):
        raise ValueError(f"invalid identifier: {value!r}")
    return value


def positive_finite(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("expected a positive finite number")
    return value


def gpu_options(manifest):
    options = ["NVIDIA L40", "NVIDIA L40S"]
    if manifest.get("allow_a100", False):
        options += ["NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"]
    return options


def expected_updates(config):
    import elements

    if config["run.envs"] != 1:
        raise ValueError(
            "campaign exact-budget validation currently requires run.envs=1"
        )
    steps = int(config["run.steps"])
    if steps % 10:
        raise ValueError(
            "run.steps must be a multiple of the driver's 10-step collection block"
        )
    start = int(config["run.world_model_start_step"])
    batch_steps = int(config["batch_size"]) * int(config["batch_length"])
    eligibility = (
        batch_steps
        + int(config["consec_train"]) * int(config["batch_length"])
        + int(config["replay_context"])
        - 1
    )
    if start < eligibility or start >= steps or config["replay.size"] < batch_steps:
        raise ValueError(
            "prefill must cover replay eligibility and precede the final step"
        )
    schedule = elements.when.Ratio(
        positive_finite(config["run.train_ratio"]) / batch_steps
    )
    return sum(schedule(step) for step in range(start, steps + 1))


def plan(spec, name):
    dependency = str(REPO / "external" / "dreamerv3")
    if dependency not in sys.path:
        sys.path.insert(0, dependency)
    from .main import _load_configs, _resolve_config_profiles

    identifier(name)
    allowed = {
        "profiles",
        "overrides",
        "maps",
        "seeds",
        "treatments",
        "run_order",
        "wandb",
        "budget",
        "max_gpu_hourly_rate",
        "max_hours",
        "max_gpus",
        "gpus_per_pod",
        "allow_a100",
        "placements",
        "storage",
        "predecessor",
        "image",
        "sc2path",
    }
    if set(spec) - allowed:
        raise ValueError(f"unknown campaign settings: {sorted(set(spec) - allowed)}")
    profiles = spec.get("profiles", ["smac_vector", "ma_jepa", "localmask_reference"])
    base = _resolve_config_profiles(_load_configs(), profiles)
    budget = positive_finite(spec["budget"])
    rate = positive_finite(spec["max_gpu_hourly_rate"])
    hours = positive_finite(spec.get("max_hours", 8))
    if hours > 8:
        raise ValueError("maximum pod runtime is eight hours")
    max_gpus, per_pod = spec.get("max_gpus", 4), spec.get("gpus_per_pod", 4)
    if (
        any(type(n) is not int or n < 1 for n in (max_gpus, per_pod))
        or per_pod > max_gpus
    ):
        raise ValueError("require positive integer gpus_per_pod <= max_gpus")
    if type(spec.get("allow_a100", False)) is not bool:
        raise ValueError("allow_a100 must be a boolean")
    destination = spec["wandb"]
    if set(destination) != {"entity", "project"}:
        raise ValueError("wandb requires explicit entity and project")
    for value in destination.values():
        identifier(value)
    placements = spec["placements"]
    if not placements:
        raise ValueError("at least one region/network-volume placement is required")
    for placement in placements:
        if set(placement) not in ({"region"}, {"region", "volume"}):
            raise ValueError("each placement requires region and an optional volume")
        for value in placement.values():
            identifier(value)
    storage = {"quota_gb": 300, "min_free_gb": 20, **spec.get("storage", {})}
    if set(storage) - {"quota_gb", "min_free_gb", "workspace_root"}:
        raise ValueError("unknown storage settings")
    if positive_finite(storage["min_free_gb"]) >= positive_finite(storage["quota_gb"]):
        raise ValueError("storage quota must exceed minimum free space")
    seeds, maps, treatments = spec["seeds"], spec["maps"], spec["treatments"]
    if (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be unique nonnegative integers")
    if not maps or not treatments:
        raise ValueError("maps and treatments cannot be empty")
    map_names = [identifier(m["name"]) for m in maps]
    treatment_names = [identifier(t["name"]) for t in treatments]
    if len(set(map_names)) != len(maps) or len(set(treatment_names)) != len(treatments):
        raise ValueError("duplicate map or treatment name")
    runs = []
    seen = set()
    for treatment in treatments:
        if set(treatment) - {"name", "overrides", "depends_on"}:
            raise ValueError("unknown treatment settings")
        dependencies = treatment.get("depends_on", [])
        if any(dep not in seen for dep in dependencies):
            raise ValueError("treatment dependencies must precede their dependents")
        seen.add(treatment["name"])
    run_order = spec.get("run_order", "treatment_first")
    if run_order not in ("treatment_first", "seed_first"):
        raise ValueError("run_order must be treatment_first or seed_first")
    combinations = (
        ((treatment, seed) for seed in seeds for treatment in treatments)
        if run_order == "seed_first"
        else ((treatment, seed) for treatment in treatments for seed in seeds)
    )
    for treatment, seed in combinations:
        for mapping in maps:
            if set(mapping) != {"name", "agents", "steps"}:
                raise ValueError("each map requires name, agents and steps")
            if (
                type(mapping["agents"]) is not int
                or mapping["agents"] < 2
                or type(mapping["steps"]) is not int
                or mapping["steps"] <= 0
            ):
                raise ValueError("invalid map agents or steps")
            run_name = f"{treatment['name']}-{mapping['name']}-seed{seed}"
            overrides = {
                **spec.get("overrides", {}),
                **treatment.get("overrides", {}),
            }
            unknown = overrides.keys() - base.flat.keys()
            if unknown:
                raise ValueError(f"unknown config overrides: {sorted(unknown)}")
            reserved = {
                "task",
                "seed",
                "logdir",
                "script",
                "agent.num_agents",
                "run.steps",
                "run.from_checkpoint",
            }
            if overrides.keys() & reserved:
                raise ValueError(
                    "map/seed/path settings belong in the campaign specification"
                )
            config = base.update(overrides).update(
                {
                    "task": "smac_" + mapping["name"],
                    "seed": seed,
                    "agent.num_agents": mapping["agents"],
                    "run.steps": mapping["steps"],
                    "logdir": f"/RUN/{run_name}/train",
                    "script": "train",
                    "run.from_checkpoint": "",
                    "run.final_save": True,
                    "run.checkpoint_at_curve_eval": True,
                    "run.eval_eps": 100,
                    "run.eval_policy_mode": "eval",
                    "jax.policy_devices": [0],
                    "jax.train_devices": [0],
                    "logger.outputs": ["jsonl", "wandb"],
                }
            )
            flat = json.loads(json.dumps(dict(config.flat)))
            runs.append(
                {
                    "name": run_name,
                    "config": flat,
                    "expected_updates": expected_updates(flat),
                    "depends_on": [
                        f"{dep}-{mapping['name']}-seed{seed}" for dep in dependencies
                    ],
                }
            )
    if len({r["name"] for r in runs}) != len(runs):
        raise ValueError("map and treatment identifiers produce duplicate run names")
    groups = [(m["name"], seed) for seed in seeds for m in maps]
    pod_count = min(math.ceil(min(max_gpus, len(runs)) / per_pod), len(groups))
    allocations = []
    remaining = max_gpus
    for i in range(pod_count):
        assigned = set(groups[i::pod_count])
        run_names = [
            r["name"]
            for r in runs
            if (r["config"]["task"][5:], r["config"]["seed"]) in assigned
        ]
        count = min(per_pod, remaining)
        if "gpus_per_pod" not in spec:
            count = min(count, len(run_names))
        remaining -= count
        allocations.append(
            {"name": f"pod{i}", "gpu_count": count, "run_names": run_names}
        )
    if sum(a["gpu_count"] for a in allocations) * rate * hours > budget - 0.5:
        raise ValueError(
            "GPU budget cannot cover full allocation including shutdown allowance"
        )
    return {
        "version": 2,
        "campaign": name,
        "image": spec.get("image", IMAGE),
        "wandb": destination,
        "budget": budget,
        "rate": rate,
        "max_hours": hours,
        "max_gpus": max_gpus,
        "gpus_per_pod": per_pod,
        "allow_a100": spec.get("allow_a100", False),
        "placements": placements,
        "storage": storage,
        "predecessor": spec.get("predecessor"),
        "sc2path": spec.get("sc2path"),
        "runs": runs,
        "allocations": allocations,
        "jobs": [],
    }


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


@contextmanager
def locked_manifest(directory):
    with (directory / "manifest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads((directory / "manifest.json").read_text())
        yield manifest
        write_json(directory / "manifest.json", manifest)


def save_job(directory, job):
    with locked_manifest(directory) as manifest:
        saved = next(
            item for item in manifest["jobs"] if item["wandb_id"] == job["wandb_id"]
        )
        status = saved.get("status")
        saved.update(job)
        if status in ("stop_requested", "stopped"):
            saved["status"] = status
        stopped = saved["status"] in ("stop_requested", "stopped")
    if stopped and job["status"] != "failed":
        raise RuntimeError("campaign monitor stopped this job")


def accounted_cost(manifest, now):
    for job in manifest["jobs"]:
        if job.get("status") == "not_created":
            continue
        elapsed = max(0, job.get("stopped_at", now) - job["created_at"])
        rate = max(manifest["rate"] * job["gpu_count"], job.get("actual_rate", 0))
        job["observed_cost"] = max(job.get("observed_cost", 0), rate * elapsed / 3600)
    return sum(
        max(job["reserved_cost"], job["observed_cost"]) for job in manifest["jobs"]
    )


def reserve(manifest, name, hours, *, now=None):
    allocation = next((p for p in manifest["allocations"] if p["name"] == name), None)
    if allocation is None:
        raise ValueError("unknown campaign allocation")
    if any(
        job["name"] == name and job["status"] != "not_created"
        for job in manifest["jobs"]
    ):
        raise ValueError(
            "job already reserved; reconcile manifest rather than recreate"
        )
    if not 0 < hours <= manifest["max_hours"]:
        raise ValueError("hours must be positive and at most eight")
    allocated = sum(
        job["gpu_count"] for job in manifest["jobs"] if "stopped_at" not in job
    )
    if allocated + allocation["gpu_count"] > manifest["max_gpus"]:
        raise ValueError("maximum concurrent campaign GPUs exceeded")
    now = time.time() if now is None else now
    # Reserve the full allocation even when the local monitor disappears.
    reserved = accounted_cost(manifest, now)
    cost = hours * manifest["rate"] * allocation["gpu_count"]
    if reserved + cost > manifest["budget"] - 0.5:
        raise ValueError("GPU budget exhausted (including shutdown allowance)")
    job = {
        **allocation,
        "created_at": now,
        "deadline": now + hours * 3600,
        "reserved_cost": cost,
        "status": "creating",
        "wandb_id": uuid.uuid4().hex[:12],
    }
    manifest["jobs"].append(job)
    return job


def api(path, *, method="GET"):
    parts = path.split("/")
    if len(parts) not in (2, 3) or parts[0] != "pods":
        raise ValueError("only exact pod reads and stops are supported")
    command = ["runpodctl", "--output", "json", "pod"]
    if method == "GET" and len(parts) == 2:
        return json.loads(
            subprocess.check_output(command + ["get", parts[1]], text=True)
        )
    if method == "POST" and parts[2:] == ["stop"]:
        subprocess.run(command + ["stop", parts[1]], check=True)
        return {}
    raise ValueError("unsupported pod operation")


def ssh(pod, directory):
    return [
        "ssh",
        "-i",
        str(Path.home() / ".ssh/runpod_key"),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={directory / 'known_hosts'}",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=3",
        "-p",
        str(pod["ssh"]["port"]),
        "root@" + pod["ssh"]["ip"],
    ]


def remote(command, script, **kwargs):
    return subprocess.run(
        [*command, "bash -lc " + shlex.quote(script)], check=True, text=True, **kwargs
    )


def _source_path(value):
    relative = Path(value)
    if (
        not value
        or relative.is_absolute()
        or ".." in relative.parts
        or ".git" in relative.parts
    ):
        raise ValueError(f"unsafe source path: {value}")
    path = REPO / relative
    try:
        resolved = path.resolve(strict=False).relative_to(REPO.resolve())
    except ValueError as exc:
        raise ValueError(f"unsafe source path: {value}") from exc
    if ".git" in resolved.parts:
        raise ValueError(f"unsafe source path: {value}")
    return relative.as_posix(), path


def _snapshot(paths):
    snapshot = {}
    for name in paths:
        name, path = _source_path(name)
        try:
            stat = path.lstat()
        except FileNotFoundError:
            continue
        mode = stat.st_mode & 0o777
        if path.is_symlink():
            link = os.readlink(path)
            try:
                target = path.resolve(strict=False).relative_to(REPO.resolve())
            except ValueError as exc:
                raise ValueError(f"unsafe source symlink: {name}") from exc
            if Path(link).is_absolute() or ".git" in target.parts:
                raise ValueError(f"unsafe source symlink: {name}")
            snapshot[name] = ("symlink", mode, link.encode())
        elif path.is_file():
            snapshot[name] = ("file", mode, path.read_bytes())
        else:
            raise ValueError(f"source path is not a file: {name}")
    return snapshot


def _add_snapshot(target, snapshot):
    for name, (kind, mode, data) in snapshot.items():
        member = tarfile.TarInfo(name)
        member.mode = mode
        member.mtime = 0
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = data.decode()
            target.addfile(member)
        else:
            member.size = len(data)
            target.addfile(member, io.BytesIO(data))


def _add_archive(target, data, prefix=""):
    with tarfile.open(fileobj=io.BytesIO(data)) as source:
        for member in source:
            parts = Path(member.name).parts
            if member.name.startswith("/") or ".." in parts or ".git" in parts:
                raise ValueError(f"unsafe archived source path: {member.name}")
            if member.issym() or member.islnk():
                linked = posixpath.normpath(
                    posixpath.join(posixpath.dirname(member.name), member.linkname)
                )
                if (
                    member.linkname.startswith("/")
                    or linked == ".."
                    or linked.startswith("../")
                    or ".git" in Path(linked).parts
                ):
                    raise ValueError(f"unsafe archived source symlink: {member.name}")
            member.name = prefix + member.name
            target.addfile(
                member, source.extractfile(member) if member.isfile() else None
            )


def _git_names(command):
    data = subprocess.check_output(command, cwd=REPO)
    return [name.decode() for name in data.split(b"\0") if name]


def freeze(directory, *, working_tree=False, dreamerv3_source=None, include_source=()):
    include_source = tuple(_source_path(str(path))[0] for path in include_source)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "ls-tree", revision, "external/dreamerv3"],
        cwd=REPO,
        text=True,
    ).split()
    if len(tree) < 3 or tree[0] != "160000":
        raise ValueError("DreamerV3 gitlink is missing")
    pinned = tree[2]
    dependency = Path(dreamerv3_source or REPO / "external/dreamerv3")
    archive_path = directory / "source.tar.gz"

    if not working_tree:
        if include_source:
            raise ValueError("--include-source requires --working-tree")
        dirty = subprocess.check_output(
            ["git", "diff", "HEAD", "--name-only"], cwd=REPO, text=True
        )
        if dirty.strip():
            raise ValueError("commit implementation or use --working-tree")
        root_archive = subprocess.check_output(["git", "archive", revision], cwd=REPO)
        dependency_archive = subprocess.check_output(
            ["git", "archive", pinned], cwd=dependency
        )
        with tarfile.open(archive_path, "w:gz") as target:
            _add_archive(target, root_archive)
            _add_archive(target, dependency_archive, "external/dreamerv3/")
        metadata = {
            "base_commit": revision,
            "dreamerv3_revision": pinned,
            "changed_files": [],
            "deleted_paths": [],
        }
    else:
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO)
        tracked_paths = []
        for entry in tracked.split(b"\0"):
            if not entry:
                continue
            name = entry.decode()
            if name != "external/dreamerv3" and not name.startswith(
                "external/dreamerv3/"
            ):
                tracked_paths.append(name)
        paths = tuple(dict.fromkeys([*tracked_paths, *include_source]))
        snapshot = _snapshot(paths)
        missing = [name for name in include_source if name not in snapshot]
        if missing:
            raise ValueError(f"included source path does not exist: {missing[0]}")
        patch = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=REPO)
        deleted = _git_names(
            ["git", "diff", "HEAD", "--name-only", "--diff-filter=D", "-z"]
        )
        changed = _git_names(
            [
                "git",
                "diff",
                "HEAD",
                "--name-only",
                "--diff-filter=ACMRTUXB",
                "-z",
            ]
        )
        changed = list(dict.fromkeys([*changed, *include_source]))
        changed_files = [
            {
                "path": name,
                "sha256": hashlib.sha256(snapshot[name][2]).hexdigest(),
            }
            for name in changed
            if name in snapshot
        ]
        metadata = {
            "base_commit": revision,
            "dreamerv3_revision": pinned,
            "changed_files": changed_files,
            "deleted_paths": deleted,
            "git_diff_sha256": hashlib.sha256(patch).hexdigest(),
        }
        dependency_archive = subprocess.check_output(
            ["git", "archive", pinned], cwd=dependency
        )
        with tarfile.open(archive_path, "w:gz") as target:
            _add_snapshot(target, snapshot)
            _add_archive(target, dependency_archive, "external/dreamerv3/")
            _add_snapshot(
                target,
                {
                    ".majepa-source/git-diff.patch": ("file", 0o644, patch),
                    ".majepa-source/metadata.json": (
                        "file",
                        0o644,
                        (
                            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
                        ).encode(),
                    ),
                },
            )
        if (
            snapshot != _snapshot(paths)
            or patch
            != subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=REPO)
            or tracked != subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO)
            or revision
            != subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip()
        ):
            archive_path.unlink(missing_ok=True)
            raise RuntimeError("source changed while freezing")

    return {
        "revision": revision,
        "base_commit": revision,
        "dreamerv3_revision": pinned,
        "changed_source_files": metadata["changed_files"],
        "deleted_source_paths": metadata["deleted_paths"],
        "source_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }


def initialize(
    directory, *, spec, working_tree=False, dreamerv3_source=None, include_source=()
):
    manifest = plan(spec, directory.name)
    directory.mkdir(parents=True, exist_ok=False)
    if working_tree:
        include_source = tuple(
            dict.fromkeys(
                [
                    *include_source,
                    "src/majepa/campaign.py",
                    "src/majepa/campaign_queue.py",
                    "src/majepa/tracking.py",
                ]
            )
        )
    source = freeze(
        directory,
        working_tree=working_tree,
        dreamerv3_source=dreamerv3_source,
        include_source=include_source,
    )
    with tarfile.open(directory / "source.tar.gz") as archive:
        for relative in (
            "src/majepa/campaign.py",
            "src/majepa/campaign_queue.py",
            "src/majepa/tracking.py",
            "src/majepa/main.py",
            "src/majepa/configs.yaml",
        ):
            member = archive.extractfile(relative)
            if member is None or member.read() != (REPO / relative).read_bytes():
                raise ValueError(
                    f"frozen source differs from resolved launcher source: {relative}"
                )
        (directory / "controller.py").write_bytes(
            archive.extractfile("src/majepa/campaign.py").read()
        )
    if plan(spec, directory.name) != manifest:
        raise RuntimeError("configuration changed while freezing source")
    write_json(directory / "spec.json", spec)
    manifest.update(source)
    for run in manifest["runs"]:
        run.update(
            wandb_id=uuid.uuid4().hex[:12], evaluation_wandb_id=uuid.uuid4().hex[:12]
        )
    manifest["plan_sha256"] = manifest_fingerprint(manifest)
    write_json(directory / "manifest.json", manifest)
    write_json(
        directory / "resolved-configs.json",
        {r["name"]: r["config"] for r in manifest["runs"]},
    )
    return manifest


def manifest_fingerprint(manifest):
    immutable = {k: v for k, v in manifest.items() if k not in ("jobs", "plan_sha256")}
    return hashlib.sha256(json.dumps(immutable, sort_keys=True).encode()).hexdigest()


def verify_manifest(manifest):
    if manifest.get("version") != 2 or manifest.get(
        "plan_sha256"
    ) != manifest_fingerprint(manifest):
        raise ValueError(
            "campaign plan changed or is obsolete; initialize a new campaign"
        )


def startup_command(deadline):
    code = (
        "import os,subprocess,time\n"
        f"time.sleep(max(0,{deadline!r}-time.time()-30))\n"
        "while True:\n"
        ' for command in (["runpodctl","stop","pod",os.environ["RUNPOD_POD_ID"]],'
        '["runpodctl","pod","stop",os.environ["RUNPOD_POD_ID"]]):\n'
        "  try:\n"
        "   if subprocess.run(command,timeout=30).returncode == 0: raise SystemExit(0)\n"
        "  except (OSError,subprocess.TimeoutExpired): pass\n"
        " time.sleep(5)\n"
    )
    script = f"nohup python -c {shlex.quote(code)} >/tmp/majepa-runtime-limit.log 2>&1 </dev/null & exec /start.sh"
    return shlex.join(["bash", "-lc", script])


def bootstrap_script(out, *, stage_assets=False):
    return "\n".join(
        [
            "set -euo pipefail",
            "trap "
            + shlex.quote(
                f"python {shlex.quote(out + '/campaign.py')} stop --directory {shlex.quote(out)}"
            )
            + " EXIT",
            f"cd {shlex.quote(out)}",
            "trap "
            + shlex.quote(
                "printf '%s\n' "
                + shlex.quote('{"completed":false,"reason":"bootstrap failed"}')
                + " > "
                + shlex.quote(out + "/outcome.json")
            )
            + " ERR",
            "mkdir repo; tar -xzf source.tar.gz -C repo",
            "python -m pip install uv",
            "cd repo",
            *(
                [
                    "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y unzip"
                ]
                if stage_assets
                else []
            ),
            "export UV_PROJECT_ENVIRONMENT=/opt/majepa-venv",
            "uv sync --locked --python 3.11 --extra dev --extra smac --extra cuda12",
            'export PYTHONPATH="$PWD/src:$PWD/external/dreamerv3"',
            *(
                [
                    '"$UV_PROJECT_ENVIRONMENT/bin/python" -m majepa.campaign_assets '
                    + shlex.quote(out + "/assets")
                ]
                if stage_assets
                else []
            ),
            "trap - ERR",
            f'"$UV_PROJECT_ENVIRONMENT/bin/python" -m majepa.campaign run --directory {shlex.quote(out)}',
        ]
    )


def launch(directory, manifest, name, hours, *, placement=None, gpu_id=None):
    auth = netrc.netrc().authenticators("api.wandb.ai")
    if not auth:
        raise ValueError("W&B netrc credentials are required")
    public_key = (Path.home() / ".ssh/runpod_key.pub").read_text().strip()
    if (
        "source_sha256" in manifest
        and hashlib.sha256((directory / "source.tar.gz").read_bytes()).hexdigest()
        != manifest["source_sha256"]
    ):
        raise ValueError("frozen source archive checksum mismatch")
    with locked_manifest(directory) as current:
        gpu_id = gpu_id or gpu_options(current)[0]
        placement = placement or current["placements"][0]
        if gpu_id not in gpu_options(current):
            raise ValueError("unsupported campaign GPU")
        job = reserve(current, name, hours).copy()
        job.update(gpu_id=gpu_id, **placement)
        current["jobs"][-1].update(job)
        manifest = current
    command = [
        "runpodctl",
        "--output",
        "json",
        "pod",
        "create",
        "--name",
        manifest["campaign"] + "-" + name,
        "--image",
        manifest["image"],
        "--gpu-id",
        gpu_id,
        "--gpu-count",
        str(job["gpu_count"]),
        "--cloud-type",
        "SECURE",
        *(
            ["--network-volume-id", job["volume"]]
            if job.get("volume")
            else ["--volume-in-gb", str(math.ceil(manifest["storage"]["quota_gb"]))]
        ),
        "--data-center-ids",
        job["region"],
        "--container-disk-in-gb",
        "30",
        "--volume-mount-path",
        "/workspace",
        "--ports",
        "22/tcp",
        "--docker-args",
        startup_command(job["deadline"]),
        "--env",
        json.dumps({"PUBLIC_KEY": public_key}),
    ]
    try:
        payload = json.loads(
            subprocess.check_output(command, text=True, stderr=subprocess.PIPE)
        )
        job["pod_id"] = payload["id"] if "id" in payload else payload["pod"]["id"]
        job["status"] = "provisioning"
        save_job(directory, job)
        print(f"Created {job['pod_id']}; waiting for SSH", file=sys.stderr, flush=True)
        wait_until = min(job["deadline"], time.time() + 900)
        while time.time() < wait_until:
            pod = api("pods/" + job["pod_id"])
            actual_rate = pod.get("costPerHr")
            if (
                actual_rate is not None
                and math.isfinite(float(actual_rate))
                and float(actual_rate) > 0
            ):
                job["actual_rate"] = float(actual_rate)
            if (
                actual_rate is None
                or not math.isfinite(float(actual_rate))
                or float(actual_rate) > manifest["rate"] * job["gpu_count"]
                or float(actual_rate) <= 0
            ):
                raise ValueError(
                    f"pod GPU hourly rate exceeds reservation or is unavailable: {actual_rate}"
                )
            if (pod.get("ssh") or {}).get("ip") and pod["ssh"].get("port"):
                try:
                    remote(ssh(pod, directory), "true", capture_output=True)
                    break
                except subprocess.CalledProcessError:
                    pass
            time.sleep(10)
        else:
            raise TimeoutError("pod SSH unavailable after 15 minutes")
        connection = ssh(pod, directory)
        out = f"/workspace/{manifest['campaign']}/{name}-{job['wandb_id']}"
        job["remote_dir"] = out
        save_job(directory, job)
        remote(
            connection,
            f"mkdir -p {shlex.quote(out)}; test ! -e {shlex.quote(out + '/job.json')}",
        )
        data = {**manifest, "job": job}
        remote(
            connection,
            f"cat > {shlex.quote(out + '/job.json')}",
            input=json.dumps(data),
        )
        with tarfile.open(directory / "source.tar.gz") as source:
            runner = source.extractfile("src/majepa/campaign.py").read().decode()
        remote(connection, f"cat > {shlex.quote(out + '/campaign.py')}", input=runner)
        remote(
            connection,
            f"nohup python {shlex.quote(out + '/campaign.py')} watchdog --directory {shlex.quote(out)} > {shlex.quote(out + '/watchdog.log')} 2>&1 < /dev/null &",
        )
        login, _, password = auth
        remote(
            connection,
            "umask 077; cat > /root/.netrc",
            input=f"machine api.wandb.ai\nlogin {login}\npassword {password}\n",
        )
        with (directory / "source.tar.gz").open("rb") as source:
            subprocess.run(
                [*connection, "cat > " + shlex.quote(out + "/source.tar.gz")],
                stdin=source,
                check=True,
            )
        bootstrap = bootstrap_script(out, stage_assets=not job.get("volume"))
        remote(
            connection, f"cat > {shlex.quote(out + '/bootstrap.sh')}", input=bootstrap
        )
        started = remote(
            connection,
            f"nohup bash {shlex.quote(out + '/bootstrap.sh')} > {shlex.quote(out + '/job.log')} 2>&1 < /dev/null & echo $!",
            capture_output=True,
        )
        job["bootstrap_pid"] = int(started.stdout.strip())
        remote(connection, f"kill -0 {job['bootstrap_pid']}", capture_output=True)
        job["process_verified_at"] = time.time()
        job["status"] = "running"
        save_job(directory, job)
    except BaseException:
        job["status"] = "failed" if job.get("pod_id") else "creation_uncertain"
        save_job(directory, job)
        if job.get("pod_id"):
            api("pods/" + job["pod_id"] + "/stop", method="POST")
        raise


def monitor(directory, manifest):
    accounted_cost(manifest, time.time())
    budget_exhausted = (
        sum(j.get("observed_cost", 0) for j in manifest["jobs"])
        >= manifest["budget"] - 0.5
    )
    for job in manifest["jobs"]:
        if not job.get("pod_id") or "stopped_at" in job:
            continue
        pod = api("pods/" + job["pod_id"])
        if pod.get("desiredStatus") == "EXITED":
            job["stopped_at"] = time.time()
            job["status"] = "stopped"
        elif budget_exhausted or time.time() >= job["deadline"] - 30:
            api("pods/" + job["pod_id"] + "/stop", method="POST")
            job["status"] = "stop_requested"
    accounted_cost(manifest, time.time())
    write_json(directory / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def own_stop(pod_id):
    if not re.fullmatch("[a-zA-Z0-9]+", pod_id):
        raise ValueError("invalid own pod ID")
    subprocess.run(
        [
            "bash",
            "-ec",
            ". /etc/rp_environment; "
            'test "$RUNPOD_POD_ID" = "$1"; '
            'runpodctl stop pod "$1" || runpodctl pod stop "$1"',
            "own-stop",
            pod_id,
        ],
        check=True,
        timeout=60,
    )


def watchdog(directory):
    job = json.loads((directory / "job.json").read_text())["job"]
    while (
        time.time() < job["deadline"] - 30 and not (directory / "outcome.json").exists()
    ):
        time.sleep(min(5, max(0, job["deadline"] - 30 - time.time())))
    if not (directory / "outcome.json").exists():
        with suppress(OSError):
            write_json(
                directory / "outcome.json",
                {"completed": False, "reason": "runtime limit"},
            )
    while True:
        try:
            own_stop(job["pod_id"])
            return
        except Exception as exc:
            print(f"self-stop failed: {type(exc).__name__}", flush=True)
            time.sleep(5)


def discover_sc2(map_names=("2s3z",)):
    candidates = [Path(os.environ["SC2PATH"])] if os.environ.get("SC2PATH") else []
    for root, dirs, files in os.walk("/workspace"):
        if "StarCraftII" in dirs:
            candidates.append(Path(root) / "StarCraftII")
            dirs.remove("StarCraftII")
        dirs[:] = [
            name
            for name in dirs
            if name not in (".venv", ".git", "wandb", "replay", "ckpt")
        ]
        if len(Path(root).parts) >= 6:
            dirs.clear()
    valid = [
        p
        for p in candidates
        if (p / "Versions/Base75689/SC2_x64").is_file()
        and all(
            (p / f"Maps/SMAC_Maps/{identifier(name)}.SC2Map").is_file()
            for name in map_names
        )
    ]
    if not valid:
        raise FileNotFoundError(
            f"existing StarCraft II 4.10 Base75689 and SMAC maps {map_names} not found"
        )
    versions = [
        int(p.name[4:])
        for p in (valid[0] / "Versions").glob("Base*")
        if p.name[4:].isdigit()
    ]
    if max(versions) != 75689:
        raise ValueError(
            "SC2 installation has a newer default version than required 4.10"
        )
    return valid[0]


def list_pods():
    payload = json.loads(
        subprocess.check_output(
            ["runpodctl", "--output", "json", "pod", "list", "--all"], text=True
        )
    )
    return payload if isinstance(payload, list) else payload["pods"]


def reconcile_rejection(directory, name, output):
    if CAPACITY_MESSAGE not in output:
        raise RuntimeError(
            "creation result is uncertain; inspect exact pod name before retrying"
        )
    pods = list_pods()
    with locked_manifest(directory) as manifest:
        matches = [
            p for p in pods if p.get("name") == manifest["campaign"] + "-" + name
        ]
        if matches:
            raise RuntimeError("a matching pod exists; refusing automatic recreation")
        job = next(j for j in reversed(manifest["jobs"]) if j["name"] == name)
        if job["status"] != "creation_uncertain" or job.get("pod_id"):
            raise RuntimeError("only unallocated capacity rejections can be reconciled")
        job.update(
            status="not_created",
            stopped_at=job["created_at"],
            reserved_cost=0.0,
            observed_cost=0.0,
            confirmed_absent_at=time.time(),
            failure_reason=CAPACITY_MESSAGE,
        )
        write_json(
            directory / (job["wandb_id"] + "-reconciliation.json"),
            {
                "exact_name": manifest["campaign"] + "-" + name,
                "checked_at": time.time(),
                "matches": [],
                "total_pods_checked": len(pods),
            },
        )


def collect_status(directory, manifest):
    for job in manifest["jobs"]:
        if job.get("status") in ("not_created", "creation_uncertain") or not job.get(
            "remote_dir"
        ):
            continue
        if "stopped_at" in job:
            continue
        try:
            pod = api("pods/" + job["pod_id"])
            out = job["remote_dir"]
            script = (
                "import json,os,pathlib\n"
                f"root=pathlib.Path({out!r})\n"
                "def read(name):\n"
                " p=root/name\n"
                " return json.loads(p.read_text()) if p.exists() else {}\n"
                "def alive(pid):\n"
                " if not isinstance(pid,int) or pid<1: return False\n"
                " try: os.kill(pid,0); return True\n"
                " except ProcessLookupError: return False\n"
                "q=read('queue.json')\n"
                "pids=[q.get('supervisor_pid')]+[w.get('pid') for w in q.get('workers',{}).values()]\n"
                "pids += [j.get('child_pid') for j in q.get('jobs',[])]\n"
                f"print(json.dumps(dict(queue=q,outcome=read('outcome.json'),bootstrap_alive=alive({job.get('bootstrap_pid')!r}),"
                "processes={str(p):alive(p) for p in pids if p})))\n"
            )
            result = remote(
                ssh(pod, directory),
                "python -c " + shlex.quote(script),
                capture_output=True,
            )
            evidence = json.loads(result.stdout)
            queue = evidence.pop("queue")
            write_json(directory / (job["name"] + "-queue.json"), queue)
            job["queue_status"] = {
                j["name"]: j["status"] for j in queue.get("jobs", [])
            }
            job["checked_at"] = time.time()
            job["process_health"] = evidence
            if time.time() - job.get("process_verified_at", time.time()) >= 120:
                job["two_minute_check"] = {
                    "checked_at": time.time(),
                    "supervisor_alive": evidence["processes"].get(
                        str(queue.get("supervisor_pid")), False
                    ),
                    "bootstrap_alive": evidence["bootstrap_alive"],
                    "outcome": evidence["outcome"],
                }
            if evidence["outcome"]:
                job["outcome"] = evidence["outcome"]
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            job["status_check_error"] = str(exc)


def verify_wandb(directory, manifest):
    import wandb

    client = wandb.Api(timeout=30)
    evidence = {}
    for run in manifest["runs"]:
        if not run.get("wandb_id"):
            continue
        try:
            prefix = f"{manifest['wandb']['entity']}/{manifest['wandb']['project']}/"
            remote_run = client.run(prefix + run["wandb_id"])
            actual = json.loads(json.dumps(remote_run.config))
            missing = object()

            def value(key):
                current = actual
                for part in key.split("."):
                    if not isinstance(current, dict) or part not in current:
                        return missing
                    current = current[part]
                return current

            mismatch = [
                k for k, v in run["config"].items() if k != "logdir" and value(k) != v
            ]
            losses = {
                k: v
                for k, v in remote_run.summary.items()
                if k
                in ("train/opt/loss", "train/ppo/actor/loss", "train/ppo/critic/loss")
            }
            source = remote_run.summary.get("artifacts/source_verified") is True
            config = remote_run.summary.get("artifacts/config_verified") is True
            current = {
                "state": remote_run.state,
                "config_present": bool(remote_run.config),
                "config_mismatches": mismatch,
                "source_verified": source,
                "config_artifact_verified": config,
                "step": remote_run.summary.get("_step"),
                "losses_finite": bool(losses)
                and all(math.isfinite(float(v)) for v in losses.values()),
                "url": remote_run.url,
                "verified": not mismatch and source and config,
                "completed": False,
            }
            if remote_run.summary.get("artifacts/checkpoint_verified") is True:
                evaluation = client.run(prefix + run["evaluation_wandb_id"])
                current["completed"] = (
                    current["verified"]
                    and remote_run.state == "finished"
                    and evaluation.state == "finished"
                    and evaluation.summary.get("artifacts/evaluation_verified") is True
                )
                current["evaluation_url"] = evaluation.url
            evidence[run["name"]] = current
        except Exception as exc:
            evidence[run["name"]] = {
                "verified": False,
                "completed": False,
                "error": str(exc),
            }
    write_json(
        directory / "wandb-verification.json",
        {"checked_at": time.time(), "runs": evidence},
    )
    return evidence


def controller(directory, *, once=False):
    # A controller lock prevents concurrent launchers from choosing the same allocation.
    with (directory / "controller.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            with locked_manifest(directory) as manifest:
                verify_manifest(manifest)
                monitor(directory, manifest)
                collect_status(directory, manifest)
            if any(
                j["status"] in ("creating", "creation_uncertain")
                for j in manifest["jobs"]
            ):
                raise RuntimeError(
                    "unreconciled creation; refusing to create another pod"
                )
            verification = directory / "wandb-verification.json"
            if any(j.get("pod_id") for j in manifest["jobs"]) and (
                not verification.exists()
                or time.time() - verification.stat().st_mtime >= 120
            ):
                verify_wandb(directory, manifest)
            reserved = {
                j["name"] for j in manifest["jobs"] if j["status"] != "not_created"
            }
            pending = next(
                (p for p in manifest["allocations"] if p["name"] not in reserved), None
            )
            if pending:
                launched = False
                for gpu in gpu_options(manifest):
                    for placement in manifest["placements"]:
                        print(
                            f"Allocating {pending['gpu_count']} x {gpu} in {placement['region']}",
                            flush=True,
                        )
                        try:
                            launch(
                                directory,
                                manifest,
                                pending["name"],
                                manifest["max_hours"],
                                placement=placement,
                                gpu_id=gpu,
                            )
                            launched = True
                            break
                        except subprocess.CalledProcessError as exc:
                            reconcile_rejection(
                                directory,
                                pending["name"],
                                (exc.stderr or "") + (exc.output or ""),
                            )
                    if launched:
                        break
                if not launched:
                    print(
                        "No matching capacity; queue preserved for next attempt",
                        flush=True,
                    )
            elif all("stopped_at" in j for j in manifest["jobs"]):
                results = verify_wandb(directory, manifest)
                if len(results) != len(manifest["runs"]) or not all(
                    r.get("completed") for r in results.values()
                ):
                    raise RuntimeError(
                        "pods stopped with incomplete or unverified runs; inspect wandb-verification.json"
                    )
                print("All campaign runs and final artifacts verified", flush=True)
                return
            if once:
                return
            time.sleep(30)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "init",
            "launch",
            "monitor",
            "status",
            "verify",
            "run",
            "watchdog",
            "stop",
        ],
    )
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--working-tree", action="store_true")
    parser.add_argument("--dreamerv3-source", type=Path)
    parser.add_argument("--include-source", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    directory = args.directory.expanduser().resolve()
    identifier(directory.name)
    if args.action == "init":
        if args.spec is None:
            parser.error("init requires --spec")
        spec = json.loads(args.spec.read_text())
        if args.dry_run:
            print(json.dumps(plan(spec, directory.name), indent=2))
        else:
            initialize(
                directory,
                spec=spec,
                working_tree=args.working_tree,
                dreamerv3_source=args.dreamerv3_source,
                include_source=args.include_source,
            )
        return
    manifest_file = directory / "manifest.json"
    if args.dry_run or args.action == "status":
        print(manifest_file.read_text())
        return
    if args.action == "run":
        from majepa.campaign_queue import run

        run(directory)
    elif args.action == "watchdog":
        watchdog(directory)
    elif args.action == "stop":
        own_stop(json.loads((directory / "job.json").read_text())["job"]["pod_id"])
    elif args.action == "launch":
        controller(directory, once=args.once)
    elif args.action == "verify":
        manifest = json.loads(manifest_file.read_text())
        print(json.dumps(verify_wandb(directory, manifest), indent=2))
    else:
        while True:
            with locked_manifest(directory) as manifest:
                monitor(directory, manifest)
                collect_status(directory, manifest)
            if args.once:
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
