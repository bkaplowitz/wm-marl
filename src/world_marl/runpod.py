"""Run wm-marl jobs on a fresh Runpod pod and clean it up."""

from __future__ import annotations

import argparse
import json
import netrc
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_COMPARE_ARGS = [
    "--flow-types",
    "transformer",
    "--fit-steps",
    "100000",
    "--chunk-steps",
    "5000",
    "--heldout-seeds",
    "1",
    "--rollout-envs",
    "6000",
]
DEFAULT_BENCHMARK_ARGS = [
    "--model-flow-types",
    "transformer",
    "discrete",
]
DEFAULT_XLA_FLAGS = "--xla_gpu_enable_triton_gemm=false"


def default_train_args(args: argparse.Namespace) -> list[str]:
    warmup_updates = 0 if args.no_policy_warmstart else args.policy_warmstart_updates
    return [
        "--algorithm",
        "ippo",
        "--substrate",
        "coins",
        "--num-envs",
        "4",
        "--rollout-steps",
        "128",
        "--total-env-steps",
        "50000",
        "--eval-episodes",
        "50",
        "--num-runs",
        "3",
        "--seed",
        "0",
        "--max-cycles",
        "1000",
        "--min-improvement",
        "0.2",
        "--negative-control",
        "freeze-policy",
        "--prefit-world-model",
        "--wm-random-rollouts",
        "2000",
        "--wm-initial-rollouts",
        "5000",
        "--wm-fit-steps",
        str(args.prefit_train_steps),
        "--wm-learning-rate",
        "0.001",
        "--wm-hidden-dim",
        "256",
        "--wm-integration-steps",
        "10",
        "--wm-flow-type",
        "transformer",
        "--wm-policy-warmup-updates",
        str(warmup_updates),
        "--learning-rate",
        "0.0005",
        "--gamma",
        "0.99",
        "--gae-lambda",
        "0.95",
        "--clip-eps",
        "0.2",
        "--ent-coef",
        "0.01",
        "--vf-coef",
        "0.5",
        "--max-grad-norm",
        "0.5",
        "--update-epochs",
        "4",
        "--num-minibatches",
        "4",
        "--activation",
        "relu",
        "--wandb-project",
        "wm-marl",
    ]


@dataclass(frozen=True)
class JobSpec:
    """Specification for a job to run on a Runpod pod."""

    remote_out_dir: str
    local_out_dir: Path
    command: list[str]


@dataclass(frozen=True)
class SshInfo:
    """SSH information for a Runpod pod."""

    user: str
    host: str
    port: int | None
    key_path: Path

    @property
    def target(self) -> str:
        """The target for SSHing into the pod."""
        return f"{self.user}@{self.host}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job",
        choices=(
            "train-e2e",
            "compare-world-models",
            "benchmark-policy",
            "compare-single-wm",
            "frontier-world-model-quality",
            "optuna-single-genwm",
        ),
        default="compare-world-models",
        help="Remote wm-marl job to run.",
    )
    parser.add_argument("--name-prefix", default="wm-marl")
    parser.add_argument("--template-id", default="runpod-torch-v240")
    parser.add_argument("--gpu-id", default="NVIDIA GeForce RTX 5090")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument(
        "--cloud-type", default="SECURE", choices=("SECURE", "COMMUNITY")
    )
    parser.add_argument("--volume-gb", type=int, default=100)
    parser.add_argument("--container-disk-gb", type=int, default=50)
    parser.add_argument("--volume-mount-path", default="/workspace")
    parser.add_argument("--ports", default="22/tcp")
    parser.add_argument(
        "--auto-stop-hours",
        type=float,
        default=12.0,
        help="Set Runpod auto-stop this many hours from launch; use 0 to disable.",
    )
    parser.add_argument(
        "--stop-after",
        help="Runpod auto-stop datetime, e.g. 2026-06-25T03:00:00Z",
    )
    parser.add_argument(
        "--terminate-after",
        help="Runpod auto-terminate datetime, e.g. 2026-06-25T03:00:00Z",
    )
    parser.add_argument(
        "--terminate-after-hours",
        type=float,
        default=0.0,
        help="Set Runpod auto-terminate this many hours from launch; "
        "use 0 to disable. Billing backstop for abandoned detached jobs.",
    )
    parser.add_argument(
        "--detach-poll-seconds",
        type=int,
        default=300,
        help="Seconds between completion-marker polls. The remote job always "
        "runs under nohup with output to job.log, so a dropped local "
        "connection never kills it; polling just resumes.",
    )
    parser.add_argument("--detach-timeout-hours", type=float, default=96.0)
    parser.add_argument(
        "--sync-extra",
        action="append",
        default=[],
        help="Additional uv extras to install on the pod beyond dev+cuda12 "
        "(repeatable), e.g. --sync-extra brax.",
    )
    parser.add_argument("--ssh-key", default="~/.ssh/runpod_key")
    parser.add_argument("--ssh-timeout-seconds", type=int, default=900)
    parser.add_argument("--ssh-poll-seconds", type=int, default=15)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--remote-repo-dir", default="/root/wm-marl")
    parser.add_argument("--remote-out-root", default="/workspace/outputs/wm_marl")
    parser.add_argument("--local-out-root", default="runs/runpod")
    parser.add_argument("--skip-uv-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prefit-train-steps",
        type=int,
        default=100000,
        help="Alias for train-e2e --wm-fit-steps.",
    )
    parser.add_argument(
        "--policy-warmstart-updates",
        type=int,
        default=3,
        help="Alias for train-e2e --wm-policy-warmup-updates.",
    )
    parser.add_argument(
        "--no-policy-warmstart",
        action="store_true",
        help="Set train-e2e --wm-policy-warmup-updates 0.",
    )
    parser.add_argument(
        "job_args",
        nargs=argparse.REMAINDER,
        help="Arguments after '--' are appended to the selected wm-marl command.",
    )
    args = parser.parse_args()
    if args.job_args and args.job_args[0] == "--":
        args.job_args = args.job_args[1:]
    if "--out-dir" in args.job_args or "--out-root" in args.job_args:
        parser.error(
            "do not pass --out-dir/--out-root; this wrapper manages "
            "remote/local outputs"
        )
    if args.gpu_count < 1:
        parser.error("--gpu-count must be >= 1")
    if args.volume_gb < 0:
        parser.error("--volume-gb must be >= 0")
    if args.container_disk_gb < 1:
        parser.error("--container-disk-gb must be >= 1")
    if args.auto_stop_hours < 0:
        parser.error("--auto-stop-hours must be >= 0")
    if args.prefit_train_steps < 1:
        parser.error("--prefit-train-steps must be >= 1")
    if args.policy_warmstart_updates < 0:
        parser.error("--policy-warmstart-updates must be >= 0")
    return args


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def required_sync_extras(args: argparse.Namespace) -> list[str]:
    extras = list(args.sync_extra)
    if args.job == "frontier-world-model-quality" and "dmc" not in extras:
        extras.append("dmc")
    return extras


def write_manifest(local_out_dir: Path, manifest: dict[str, Any]) -> Path:
    local_out_dir.mkdir(parents=True, exist_ok=True)
    path = local_out_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def main() -> int:
    args = parse_args()
    sync_extras = required_sync_extras(args)
    repo_root = Path(args.repo_root).expanduser().resolve()
    ssh_key = Path(args.ssh_key).expanduser().resolve()
    run_id = utc_stamp()
    pod_name = f"{args.name_prefix}-{args.job}-{run_id}"
    job = build_job_spec(args, run_id)

    create_cmd = build_create_pod_cmd(args, pod_name)
    if args.dry_run:
        print_dry_run(
            create_cmd=create_cmd,
            pod_name=pod_name,
            job=job,
            repo_root=repo_root,
            ssh_key=ssh_key,
            remote_repo_dir=args.remote_repo_dir,
            skip_uv_sync=args.skip_uv_sync,
            sync_extras=sync_extras,
        )
        return 0

    require_local_tools(["runpodctl", "ssh", "rsync"])
    require_repo(repo_root)
    ensure_runpod_api_key(repo_root)
    if not ssh_key.exists():
        raise SystemExit(f"SSH key not found: {ssh_key}")
    push_netrc = command_uses_wandb(job.command)
    if push_netrc:
        read_wandb_netrc_entry()

    pod_id: str | None = None
    manifest: dict[str, Any] = {
        "pod_name": pod_name,
        "pod_id": None,
        "status": "creating",
        "job": args.job,
        "gpu_id": args.gpu_id,
        "gpu_count": args.gpu_count,
        "cloud_type": args.cloud_type,
        "command": job.command,
        "remote_repo_dir": args.remote_repo_dir,
        "remote_out_dir": job.remote_out_dir,
        "local_out_dir": str(job.local_out_dir),
        "created_at": run_id,
        "finished_at": None,
    }
    detached_running = False
    try:
        print(f"creating Runpod pod: {pod_name}", flush=True)
        created = run_json(create_cmd)
        pod_id = extract_pod_id(created)
        manifest.update(pod_id=pod_id, status="running")
        manifest_path = write_manifest(job.local_out_dir, manifest)
        print(f"created pod: {pod_id} (manifest: {manifest_path})", flush=True)

        ssh_info = wait_for_ssh(pod_id, ssh_key, args)
        ensure_remote_rsync(ssh_info)
        sync_repo(repo_root, args.remote_repo_dir, ssh_info)
        if push_netrc:
            push_wandb_netrc(ssh_info)
        start_remote_job_detached(
            args.remote_repo_dir,
            job.command,
            ssh_info,
            args.skip_uv_sync,
            sync_extras,
            job.remote_out_dir,
        )
        detached_running = True
        manifest.update(status="detached-running")
        write_manifest(job.local_out_dir, manifest)
        job_status = wait_for_detached_job(pod_id, ssh_key, args, job.remote_out_dir)
        detached_running = False
        ssh_info = get_ssh_info(pod_id, ssh_key)
        download_outputs(job.remote_out_dir, job.local_out_dir, ssh_info)
        if job_status == "failed":
            print("remote job failed; see job.log in outputs", file=sys.stderr)
            stop_for_inspection(pod_id, job.remote_out_dir)
            manifest.update(status="job-failed-pod-stopped", finished_at=utc_stamp())
            write_manifest(job.local_out_dir, manifest)
            return 1

    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr, flush=True)
        if pod_id and detached_running:
            print(
                f"detached job is still running on pod {pod_id}; NOT stopping it. "
                f"Watch {job.remote_out_dir}/job.log and delete the pod when done.",
                file=sys.stderr,
                flush=True,
            )
            manifest.update(status="detached-monitoring-abandoned")
            write_manifest(job.local_out_dir, manifest)
            return 130
        if pod_id:
            stop_for_inspection(pod_id, job.remote_out_dir)
            manifest.update(status="stopped-for-inspection", finished_at=utc_stamp())
            write_manifest(job.local_out_dir, manifest)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        if pod_id and detached_running:
            print(
                f"monitoring lost but the detached job may still be running on "
                f"pod {pod_id}; NOT stopping it. The --terminate-after backstop "
                "bounds billing. Re-check the pod manually.",
                file=sys.stderr,
                flush=True,
            )
            manifest.update(status="detached-monitoring-lost")
            write_manifest(job.local_out_dir, manifest)
            return 1
        if pod_id:
            stop_for_inspection(pod_id, job.remote_out_dir)
            manifest.update(status="stopped-for-inspection", finished_at=utc_stamp())
            write_manifest(job.local_out_dir, manifest)
        return 1

    deleted = bool(pod_id) and delete_pod(pod_id)
    manifest.update(
        status="completed-pod-deleted" if deleted else "completed-delete-failed",
        finished_at=utc_stamp(),
    )
    write_manifest(job.local_out_dir, manifest)
    if not deleted:
        print(
            f"warning: runpodctl could not delete pod {pod_id}; delete it manually",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(f"downloaded artifacts to {job.local_out_dir}", flush=True)
    return 0


def ensure_runpod_api_key(repo_root: Path) -> None:
    """Ensure the Runpod API key is set."""
    if os.environ.get("RUNPOD_API_KEY"):
        return
    env_path = repo_root / ".env"
    if env_path.exists():
        value = read_dotenv_value(env_path, "RUNPOD_API_KEY")
        if value:
            os.environ["RUNPOD_API_KEY"] = value
            return
    raise SystemExit(
        "RUNPOD_API_KEY is not set; export it or add RUNPOD_API_KEY=... to .env"
    )


def read_dotenv_value(env_path: Path, key: str) -> str | None:
    """Read a value from a .env file."""
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        name, sep, value = line.partition("=")
        if sep and name.strip() == key:
            return strip_dotenv_quotes(value.strip())
    return None


def strip_dotenv_quotes(value: str) -> str:
    """Strip quotes from a value read from a .env file."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def build_job_spec(args: argparse.Namespace, run_id: str) -> JobSpec:
    """Build a job specification for a Runpod pod."""
    remote_out_dir = f"{args.remote_out_root.rstrip('/')}/{args.job}/{run_id}"
    local_out_dir = Path(args.local_out_root).expanduser().resolve() / args.job / run_id
    if args.job == "train-e2e":
        job_args = [
            *default_train_args(args),
            *args.job_args,
            "--out-dir",
            remote_out_dir,
        ]
        return JobSpec(
            remote_out_dir=remote_out_dir,
            local_out_dir=local_out_dir,
            command=["uv", "run", "world-marl-train-e2e", *job_args],
        )
    if args.job == "benchmark-policy":
        train_args = [*default_train_args(args), *args.job_args]
        job_args = [
            *DEFAULT_BENCHMARK_ARGS,
            "--out-dir",
            remote_out_dir,
            "--",
            *train_args,
        ]
        return JobSpec(
            remote_out_dir=remote_out_dir,
            local_out_dir=local_out_dir,
            command=["uv", "run", "world-marl-benchmark-policy", *job_args],
        )
    if args.job == "compare-single-wm":
        job_args = [*args.job_args, "--out-dir", remote_out_dir]
        return JobSpec(
            remote_out_dir=remote_out_dir,
            local_out_dir=local_out_dir,
            command=[
                "env",
                f"XLA_FLAGS={DEFAULT_XLA_FLAGS}",
                "uv",
                "run",
                "world-marl-compare-single-wm",
                *job_args,
            ],
        )
    if args.job == "frontier-world-model-quality":
        job_args = [*args.job_args, "--out-dir", remote_out_dir]
        return JobSpec(
            remote_out_dir=remote_out_dir,
            local_out_dir=local_out_dir,
            command=[
                "env",
                f"XLA_FLAGS={DEFAULT_XLA_FLAGS}",
                "MUJOCO_GL=egl",
                "uv",
                "run",
                "world-marl-frontier-wm-quality",
                *job_args,
            ],
        )
    if args.job == "optuna-single-genwm":
        job_args = [*args.job_args, "--out-root", remote_out_dir]
        return JobSpec(
            remote_out_dir=remote_out_dir,
            local_out_dir=local_out_dir,
            command=[
                "env",
                f"XLA_FLAGS={DEFAULT_XLA_FLAGS}",
                "uv",
                "run",
                "world-marl-optuna-single-genwm",
                *job_args,
            ],
        )
    job_args = [*(args.job_args or DEFAULT_COMPARE_ARGS), "--out-dir", remote_out_dir]
    return JobSpec(
        remote_out_dir=remote_out_dir,
        local_out_dir=local_out_dir,
        command=[
            "env",
            f"XLA_FLAGS={DEFAULT_XLA_FLAGS}",
            "uv",
            "run",
            "python",
            "-m",
            "world_marl.scripts.compare_world_models",
            *job_args,
        ],
    )


def read_public_key(ssh_key: str) -> str:
    """Read the public key from the SSH key file."""
    pub_path = Path(str(Path(ssh_key).expanduser()) + ".pub")
    if not pub_path.exists():
        raise SystemExit(
            f"SSH public key not found: {pub_path}; needed to inject PUBLIC_KEY into the pod"
        )
    return pub_path.read_text(encoding="utf-8").strip()


def build_create_pod_cmd(args: argparse.Namespace, pod_name: str) -> list[str]:
    """Build the command to create a Runpod pod."""
    cmd = [
        "runpodctl",
        "--output",
        "json",
        "pod",
        "create",
        "--name",
        pod_name,
        "--template-id",
        args.template_id,
        "--gpu-id",
        args.gpu_id,
        "--gpu-count",
        str(args.gpu_count),
        "--cloud-type",
        args.cloud_type,
        "--volume-in-gb",
        str(args.volume_gb),
        "--container-disk-in-gb",
        str(args.container_disk_gb),
        "--volume-mount-path",
        args.volume_mount_path,
        "--ports",
        args.ports,
        "--env",
        json.dumps({"PUBLIC_KEY": read_public_key(args.ssh_key)}),
    ]
    terminate_after = args.terminate_after
    if not terminate_after and args.terminate_after_hours > 0:
        terminate_at = datetime.now(timezone.utc) + timedelta(
            hours=args.terminate_after_hours
        )
        terminate_after = terminate_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    stop_after = args.stop_after
    if not stop_after and not terminate_after and args.auto_stop_hours > 0:
        stop_at = datetime.now(timezone.utc) + timedelta(hours=args.auto_stop_hours)
        stop_after = stop_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    if stop_after:
        cmd.extend(["--stop-after", stop_after])
    if terminate_after:
        cmd.extend(["--terminate-after", terminate_after])
    return cmd


def run_json(cmd: list[str]) -> Any:
    """Run a command and return the JSON output."""
    result = run(cmd, capture_output=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"command did not return JSON: {shlex.join(cmd)}\n{result.stdout}"
        ) from exc


def run(
    cmd: list[str],
    *,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command and print it to the console."""
    print(f"+ {shlex.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def extract_pod_id(payload: Any) -> str:
    """Extract the pod ID from the Runpod API response."""
    if isinstance(payload, dict):
        for key in ("id", "pod_id", "podId"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        pod = payload.get("pod")
        if isinstance(pod, dict):
            return extract_pod_id(pod)
    raise RuntimeError(f"could not extract pod id from runpodctl output: {payload!r}")


def require_local_tools(names: list[str]) -> None:
    """Require the local tools to be present."""
    missing = [name for name in names if not shutil_which(name)]
    if missing:
        raise SystemExit(f"missing required command(s): {', '.join(missing)}")


def shutil_which(name: str) -> str | None:
    """Find a command in the PATH."""
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(entry) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def require_repo(repo_root: Path) -> None:
    """Require the repository to be valid."""
    pyproject = repo_root / "pyproject.toml"
    package_dir = repo_root / "src" / "world_marl"
    if not pyproject.exists() or not package_dir.is_dir():
        raise SystemExit(f"{repo_root} does not look like the wm-marl repo root")


def wait_for_ssh(pod_id: str, ssh_key: Path, args: argparse.Namespace) -> SshInfo:
    """Wait for the SSH to become ready for a Runpod pod."""
    deadline = time.monotonic() + args.ssh_timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            info = get_ssh_info(pod_id, ssh_key)
            probe_ssh(info)
            print(f"ssh ready: {info.target}", flush=True)
            return info
        except Exception as exc:
            last_error = str(exc)
            print(f"waiting for ssh: {last_error}", flush=True)
            time.sleep(args.ssh_poll_seconds)
    raise RuntimeError(f"SSH did not become ready for pod {pod_id}: {last_error}")


def get_ssh_info(pod_id: str, fallback_key: Path) -> SshInfo:
    """Get the SSH information for a Runpod pod."""
    return parse_direct_ssh_info(fetch_pod_rest(pod_id), fallback_key)


def fetch_pod_rest(pod_id: str) -> dict[str, Any]:
    """Fetch the pod information from the Runpod API."""
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    request = urllib.request.Request(
        f"https://rest.runpod.io/v1/pods/{pod_id}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected pod payload for {pod_id}: {payload!r}")
    return payload


def parse_direct_ssh_info(pod: dict[str, Any], key_path: Path) -> SshInfo:
    """Parse the SSH information for a Runpod pod."""
    public_ip = pod.get("publicIp")
    port_mappings = pod.get("portMappings") or {}
    public_port = port_mappings.get("22")
    if not public_ip or not public_port:
        status = pod.get("desiredStatus", "unknown")
        raise RuntimeError(
            f"ssh endpoint not ready for pod {pod.get('id')}: status={status}, "
            f"publicIp={public_ip!r}, port22={public_port!r}"
        )
    return SshInfo(
        user="root", host=str(public_ip), port=int(public_port), key_path=key_path
    )


def ssh_base(info: SshInfo) -> list[str]:
    """Build the base SSH command for a Runpod pod."""
    cmd = [
        "ssh",
        "-i",
        str(info.key_path),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
    ]
    if info.port is not None:
        cmd.extend(["-p", str(info.port)])
    cmd.append(info.target)
    return cmd


def probe_ssh(info: SshInfo) -> None:
    """Probe the SSH connection to a Runpod pod."""
    run([*ssh_base(info), "true"], capture_output=True)


def ssh_script_command(info: SshInfo, script: str) -> list[str]:
    """SSH command running a multi-line script through a quoted ``bash -lc``.

    ssh space-joins the remote command args into one string for the remote
    login shell, so the script must be quoted client-side — otherwise the
    remote shell re-parses it word by word (``bash -lc mkdir -p ...`` runs a
    bare ``mkdir`` and executes the remaining lines outside ``bash -lc``).
    """
    return [*ssh_base(info), f"bash -lc {shlex.quote(script)}"]


def command_uses_wandb(command: list[str]) -> bool:
    """True when the job command enables W&B logging (``--wandb-project``)."""
    return any(
        token == "--wandb-project" or token.startswith("--wandb-project=")
        for token in command
    )


def read_wandb_netrc_entry() -> str:
    """Read the local api.wandb.ai netrc entry, failing fast when absent."""
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        raise SystemExit(
            "job uses wandb but no local ~/.netrc exists; run 'wandb login' first"
        )
    auth = netrc.netrc(netrc_path).authenticators("api.wandb.ai")
    if auth is None:
        raise SystemExit(
            "job uses wandb but ~/.netrc has no api.wandb.ai entry; "
            "run 'wandb login' first"
        )
    login, _, password = auth
    return f"machine api.wandb.ai\n  login {login}\n  password {password}\n"


def push_wandb_netrc(info: SshInfo) -> None:
    """Append the local api.wandb.ai netrc entry to the pod's ~/.netrc.

    The entry travels over ssh stdin only — ``run()`` prints argv, so the key
    must never be part of the command line (or the manifest).
    """
    entry = read_wandb_netrc_entry()
    cmd = ssh_script_command(info, "umask 077; cat >> ~/.netrc")
    print("+ (writing api.wandb.ai netrc entry to pod over ssh stdin)", flush=True)
    subprocess.run(cmd, check=True, text=True, input=entry)


def ensure_remote_rsync(info: SshInfo) -> None:
    """Ensure remote transfer and headless EGL runtime packages are installed."""
    script = (
        "if ! command -v rsync >/dev/null 2>&1 || "
        "! test -e /usr/lib/x86_64-linux-gnu/libEGL.so.1; then "
        "apt-get update && apt-get install -y rsync libegl1; fi"
    )
    run([*ssh_base(info), script])


def sync_repo(repo_root: Path, remote_repo_dir: str, info: SshInfo) -> None:
    """Sync the repository to a Runpod pod."""
    run([*ssh_base(info), f"mkdir -p {shlex.quote(remote_repo_dir)}"])
    ssh_cmd = [
        "ssh",
        "-i",
        str(info.key_path),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
    ]
    if info.port is not None:
        ssh_cmd.extend(["-p", str(info.port)])
    run(
        [
            "rsync",
            "-az",
            "--delete",
            "--exclude",
            ".git/",
            "--exclude",
            ".venv/",
            "--exclude",
            "runs/",
            "--exclude",
            "__pycache__/",
            "--exclude",
            ".env",
            "-e",
            shlex.join(ssh_cmd),
            f"{repo_root}/",
            f"{info.target}:{remote_repo_dir}/",
        ]
    )


def remote_job_script(
    remote_repo_dir: str,
    job_command: list[str],
    skip_uv_sync: bool,
    sync_extras: list[str],
) -> str:
    """Build the setup + job script executed on the pod."""
    commands = [
        "set -euo pipefail",
        f"cd {shlex.quote(remote_repo_dir)}",
        "python -m pip install -U uv",
    ]
    if not skip_uv_sync:
        extras = "".join(
            f" --extra {extra}" for extra in ("dev", "cuda12", *sync_extras)
        )
        commands.append(f"uv sync --python 3.11{extras}")
    if "dmc" in sync_extras:
        commands.extend(
            [
                'PLAYGROUND_SITE=$(uv run python -c "import sysconfig; '
                "print(sysconfig.get_path('purelib'))\")",
                'MENAGERIE_DIR="$PLAYGROUND_SITE/mujoco_playground/'
                'external_deps/mujoco_menagerie"',
                'mkdir -p "$MENAGERIE_DIR"',
            ]
        )
    commands.extend(
        [
            "uv run world-marl-verify-install",
            'uv run python -c "import jax; devs = jax.devices(); '
            "assert any(d.platform == 'gpu' for d in devs), "
            "f'no GPU visible to JAX (silent CPU fallback): {devs}'; "
            "print('jax devices:', devs)\"",
            shlex.join(job_command),
        ]
    )
    return "\n".join(commands)


def start_remote_job_detached(
    remote_repo_dir: str,
    job_command: list[str],
    info: SshInfo,
    skip_uv_sync: bool,
    sync_extras: list[str],
    remote_out_dir: str,
) -> None:
    """Start the job under nohup on the pod and return immediately.

    The job writes ``JOB_DONE`` or ``JOB_FAILED`` into ``remote_out_dir`` when
    it exits; ``wait_for_detached_job`` polls for those markers. Console output
    goes to ``job.log`` in the same directory.
    """
    script = remote_job_script(remote_repo_dir, job_command, skip_uv_sync, sync_extras)
    out = remote_out_dir.rstrip("/")
    runner = f"{out}/runner.sh"
    wrapper = (
        f"mkdir -p {shlex.quote(out)}\n"
        f"cat > {shlex.quote(runner)} <<'WM_MARL_RUNNER'\n"
        f"{script}\n"
        "WM_MARL_RUNNER\n"
        f"rm -f {shlex.quote(out + '/JOB_DONE')} {shlex.quote(out + '/JOB_FAILED')}\n"
        f"nohup bash -c 'bash {shlex.quote(runner)} "
        f"&& touch {shlex.quote(out + '/JOB_DONE')} "
        f"|| touch {shlex.quote(out + '/JOB_FAILED')}' "
        f"> {shlex.quote(out + '/job.log')} 2>&1 &\n"
        "echo detached job started"
    )
    run(ssh_script_command(info, wrapper))


def wait_for_detached_job(
    pod_id: str,
    ssh_key: Path,
    args: argparse.Namespace,
    remote_out_dir: str,
) -> str:
    """Poll the pod until the detached job writes its completion marker.

    Returns ``"done"`` or ``"failed"``. Transient SSH/network errors are
    logged and retried — they must not be treated as job failure.
    """
    out = remote_out_dir.rstrip("/")
    probe = (
        f"if [ -f {shlex.quote(out + '/JOB_DONE')} ]; then echo STATUS:done; "
        f"elif [ -f {shlex.quote(out + '/JOB_FAILED')} ]; then echo STATUS:failed; "
        "else echo STATUS:running; fi; "
        f"tail -n 3 {shlex.quote(out + '/job.log')} 2>/dev/null || true"
    )
    deadline = time.monotonic() + args.detach_timeout_hours * 3600.0
    while time.monotonic() < deadline:
        time.sleep(args.detach_poll_seconds)
        try:
            info = get_ssh_info(pod_id, ssh_key)
            result = run(ssh_script_command(info, probe), capture_output=True)
        except Exception as exc:
            print(f"poll failed (will retry): {exc}", flush=True)
            continue
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
        print(f"[{stamp}] {result.stdout.strip()}", flush=True)
        for line in result.stdout.splitlines():
            if line.strip() == "STATUS:done":
                return "done"
            if line.strip() == "STATUS:failed":
                return "failed"
    raise RuntimeError(
        f"detached job did not finish within {args.detach_timeout_hours}h; "
        f"pod {pod_id} is still running"
    )


def download_outputs(remote_out_dir: str, local_out_dir: Path, info: SshInfo) -> None:
    """Download the outputs from a Runpod pod."""
    local_out_dir.mkdir(parents=True, exist_ok=True)
    ssh_cmd = [
        "ssh",
        "-i",
        str(info.key_path),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
    ]
    if info.port is not None:
        ssh_cmd.extend(["-p", str(info.port)])
    run(
        [
            "rsync",
            "-az",
            "-e",
            shlex.join(ssh_cmd),
            f"{info.target}:{remote_out_dir}/",
            f"{local_out_dir}/",
        ]
    )


def stop_for_inspection(pod_id: str, remote_out_dir: str) -> None:
    """Stop the pod and print a message to inspect the remote outputs after restart."""
    print(
        f"stopping pod {pod_id}; inspect remote outputs at {remote_out_dir} after restart",
        file=sys.stderr,
        flush=True,
    )
    run(["runpodctl", "pod", "stop", pod_id], check=False)


def delete_pod(pod_id: str) -> bool:
    """Delete the pod and return True if successful."""
    result = run(["runpodctl", "pod", "delete", pod_id], check=False)
    return result.returncode == 0


def print_dry_run(
    *,
    create_cmd: list[str],
    pod_name: str,
    job: JobSpec,
    repo_root: Path,
    ssh_key: Path,
    remote_repo_dir: str,
    skip_uv_sync: bool,
    sync_extras: list[str],
) -> None:
    """Print a dry run of the pod creation and job execution."""
    print(f"pod name: {pod_name}")
    print(f"repo root: {repo_root}")
    print(f"ssh key: {ssh_key}")
    print(f"remote repo: {remote_repo_dir}")
    print(f"remote outputs: {job.remote_out_dir}")
    print(f"local outputs: {job.local_out_dir}")
    print(f"manifest: {job.local_out_dir / 'manifest.json'}")
    print("\ncommands:")
    print(shlex.join(create_cmd))
    print("GET https://rest.runpod.io/v1/pods/<pod-id>  (publicIp + portMappings[22])")
    print("ssh -i <key> root@<publicIp> -p <port22> true")
    print("ssh <pod-ssh-target> 'command -v rsync || apt-get install -y rsync'")
    print(f"rsync repo to <pod-ssh-target>:{remote_repo_dir}/")
    if skip_uv_sync:
        print("skip uv sync")
    else:
        extras = "".join(
            f" --extra {extra}" for extra in ("dev", "cuda12", *sync_extras)
        )
        print(f"uv sync --python 3.11{extras}")
    print("uv run world-marl-verify-install")
    print("assert jax.devices() shows a GPU (fail fast on silent CPU fallback)")
    print(f"nohup the job on the pod; poll {job.remote_out_dir}/JOB_DONE|JOB_FAILED")
    print(shlex.join(job.command))
    print(f"rsync <pod-ssh-target>:{job.remote_out_dir}/ {job.local_out_dir}/")
    print("success cleanup: runpodctl pod delete <pod-id>")
    print("failure cleanup: runpodctl pod stop <pod-id>")


if __name__ == "__main__":
    raise SystemExit(main())
