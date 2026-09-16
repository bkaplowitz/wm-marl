import errno
import json

import elements
import pytest

from majepa import campaign
from majepa.main import _load_configs, _resolve_config_profiles


def test_campaign_configuration_and_eval_preserve_architecture():
    base = _resolve_config_profiles(
        _load_configs(), ["defaults", "smac_vector", "ma_jepa"]
    )
    configs = [
        elements.Flags(base).parse(
            campaign.training_args(seed=0, samples=n, logdir="/tmp/train")
        )
        for n in (1, 2)
    ]
    changed = {k for k in configs[0].flat if configs[0].flat[k] != configs[1].flat[k]}
    assert changed == {"agent.imag_action_samples"}
    config = configs[1]
    assert config.agent.dyn.parallel_transformer.deter == 4096
    assert config.agent.policy.units == 512
    assert config.agent.loss_scales.ctde_multistep_jepa_action == 0.1
    assert config.agent.loss_scales.ctde_posterior_alignment == 0.05
    assert config.run.steps == 50000
    assert config.run.replay_stream_mode == "snapshot_staggered"
    assert config.agent.marl.ctde.teammate_belief.enabled is False
    assert config.agent.collection_unimix == 0
    assert config.agent.ppo.entropy_coefficient == 0.003
    evaluated = elements.Flags(base).parse(
        campaign.training_args(
            seed=0, samples=2, logdir="/tmp/eval", checkpoint="/tmp/checkpoint"
        )
    )
    assert evaluated.agent == config.agent
    assert evaluated.run.eval_eps == 100
    assert evaluated.run.envs == 4
    assert evaluated.run.eval_worker_offset == 100000


def test_budget_reservation_counts_setup_and_refuses_duplicates():
    manifest = {"jobs": [], "budget": 50, "rate": 1.59}
    campaign.reserve(manifest, "smoke", 0.25, now=100)
    for job in campaign.JOBS[:4]:
        if job == campaign.JOBS[3]:
            with pytest.raises(ValueError, match="four"):
                campaign.reserve(manifest, job, 5, now=100)
            manifest["jobs"][0]["stopped_at"] = 200
        campaign.reserve(manifest, job, 5, now=100)
    with pytest.raises(ValueError, match="already"):
        campaign.reserve(manifest, campaign.JOBS[0], 1, now=110)
    manifest["jobs"][1]["stopped_at"] = 200
    campaign.reserve(manifest, campaign.JOBS[4], 5, now=200)
    manifest["jobs"][2]["stopped_at"] = 200
    with pytest.raises(ValueError, match="budget"):
        campaign.reserve(manifest, campaign.JOBS[5], 8, now=200)
    with pytest.raises(ValueError, match="hours"):
        campaign.reserve(manifest, campaign.JOBS[5], 9, now=200)


def test_dry_run_has_no_files_or_processes(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run executed a command")

    monkeypatch.setattr(campaign.subprocess, "run", forbidden)
    path = tmp_path / "absent"
    campaign.main(["init", "--directory", str(path), "--dry-run"])
    assert not path.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["jobs"] == list(campaign.JOBS)
    assert plan["max_concurrent"] == 4
    assert all(
        command[0] == "/opt/majepa-venv/bin/python"
        for command in plan["training_commands"].values()
    )


def test_bootstrap_installs_and_runs_from_container_environment():
    script = campaign.bootstrap_script("/workspace/campaign/smoke")
    lines = script.splitlines()
    environment = "export UV_PROJECT_ENVIRONMENT=/opt/majepa-venv"
    sync = "uv sync --locked --python 3.11 --extra dev --extra smac --extra cuda12"
    run = '"$UV_PROJECT_ENVIRONMENT/bin/python" -m majepa.campaign run --directory /workspace/campaign/smoke'
    assert environment in lines
    assert lines.index(environment) < lines.index(sync) < lines.index(run)
    assert 'export PYTHONPATH="$PWD/src:$PWD/external/dreamerv3"' in lines
    assert ".venv/bin/python" not in script


def test_startup_watchdog_is_independent_and_preserves_image_start():
    command = campaign.startup_command(200)
    assert "/start.sh" in command
    assert "RUNPOD_POD_ID" in command
    assert "RUNPOD_API_KEY" not in command
    assert "200" in command


def test_uncertain_creation_is_persisted_and_never_retried(tmp_path, monkeypatch):
    class Credentials:
        def authenticators(self, host):
            return ("user", None, "secret")

    monkeypatch.setattr(campaign.netrc, "netrc", Credentials)
    monkeypatch.setenv("RUNPOD_API_KEY", "secret")
    monkeypatch.setattr(campaign.Path, "home", lambda: tmp_path)
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh/runpod_key.pub").write_text("ssh-ed25519 PUBLIC")
    monkeypatch.setattr(
        campaign.subprocess, "check_output", lambda *a, **kw: "uncertain response"
    )
    manifest = {
        "campaign": "test",
        "jobs": [],
        "rate": 1.59,
        "budget": 50,
        "volume": "volume",
        "region": "region",
    }
    campaign.write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(json.JSONDecodeError):
        campaign.launch(tmp_path, manifest, "smoke", 0.5)
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["jobs"][0]["status"] == "creation_uncertain"
    assert "secret" not in (tmp_path / "manifest.json").read_text()
    with pytest.raises(ValueError, match="already"):
        campaign.launch(tmp_path, saved, "smoke", 0.5)


def test_monitor_only_stops_exact_expired_campaign_pod(tmp_path, monkeypatch):
    operations = []

    def api(path, *, method="GET"):
        operations.append((path, method))
        return {"desiredStatus": "RUNNING"}

    monkeypatch.setattr(campaign, "api", api)
    manifest = {
        "rate": 1.59,
        "budget": 50,
        "jobs": [
            {"pod_id": "newpod", "created_at": 0, "deadline": 1, "reserved_cost": 1.59}
        ],
    }
    campaign.monitor(tmp_path, manifest)
    assert operations == [("pods/newpod", "GET"), ("pods/newpod/stop", "POST")]
    assert manifest["jobs"][0]["status"] == "stop_requested"


@pytest.mark.parametrize("outcome_writable", [True, False])
@pytest.mark.parametrize("stage", ["bootstrap", "training"])
def test_bootstrap_failure_stops_even_without_outcome(
    tmp_path, monkeypatch, outcome_writable, stage
):
    import os
    import subprocess
    import sys

    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(campaign.REPO / 'src')!r})\n"
        "from majepa import campaign\n"
        f"campaign.own_stop = lambda pod: Path({str(tmp_path / 'stop.log')!r}).write_text(pod)\n"
        "campaign.main(sys.argv[2:])\n"
    )
    python.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    campaign.write_json(tmp_path / "job.json", {"job": {"pod_id": "recordedpod"}})
    if not outcome_writable:
        (tmp_path / "outcome.json").mkdir()

    script = campaign.bootstrap_script(str(tmp_path))
    if stage == "bootstrap":
        script = script.replace("mkdir repo; tar -xzf source.tar.gz -C repo", "false")
    else:
        script = script.replace(
            "mkdir repo; tar -xzf source.tar.gz -C repo", "mkdir repo"
        )
        script = script.replace("python -m pip install uv", ":")
        script = script.replace(
            "uv sync --locked --python 3.11 --extra dev --extra smac --extra cuda12",
            ":",
        )
        script = script.replace(
            '"$UV_PROJECT_ENVIRONMENT/bin/python" -m majepa.campaign run --directory',
            "false",
        )
        if outcome_writable:
            campaign.write_json(
                tmp_path / "outcome.json",
                {"completed": False, "error": "training failed"},
            )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode != 0
    assert (tmp_path / "stop.log").read_text() == "recordedpod"
    if outcome_writable:
        outcome = json.loads((tmp_path / "outcome.json").read_text())
        assert outcome["completed"] is False
        if stage == "training":
            assert outcome["error"] == "training failed"


def test_budget_admission_accounts_for_smoke_overrun():
    manifest = {"jobs": [], "budget": 50, "rate": 1.59}
    smoke = campaign.reserve(manifest, "smoke", 0.5, now=0)
    smoke["stopped_at"] = 7200
    for name in campaign.JOBS[:5]:
        job = campaign.reserve(manifest, name, 5, now=7200)
        job["stopped_at"] = 7201
    with pytest.raises(ValueError, match="budget"):
        campaign.reserve(manifest, campaign.JOBS[-1], 5, now=7201)


def test_monitor_stops_all_exact_campaign_pods_on_aggregate_overrun(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(campaign.time, "time", lambda: 7200)
    operations = []

    def api(path, *, method="GET"):
        operations.append((path, method))
        return {"desiredStatus": "RUNNING"}

    monkeypatch.setattr(campaign, "api", api)
    manifest = {
        "rate": 1.59,
        "budget": 50,
        "jobs": [
            {
                "name": "smoke",
                "created_at": 0,
                "stopped_at": 7200,
                "reserved_cost": 0.795,
            },
            *[
                {
                    "name": name,
                    "pod_id": "new" + str(i),
                    "created_at": 7200,
                    "deadline": 25200,
                    "reserved_cost": 7.95,
                }
                for i, name in enumerate(campaign.JOBS)
            ],
        ],
    }
    campaign.monitor(tmp_path, manifest)
    assert {path for path, method in operations if method == "POST"} == {
        f"pods/new{i}/stop" for i in range(6)
    }
    assert all(job["status"] == "stop_requested" for job in manifest["jobs"][1:])


def test_monitor_and_duplicate_admission_work_while_launch_is_pending(
    tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    class Credentials:
        def authenticators(self, host):
            return ("user", None, "secret")

    monkeypatch.setattr(campaign.netrc, "netrc", Credentials)
    monkeypatch.setattr(campaign.Path, "home", lambda: tmp_path)
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh/runpod_key.pub").write_text("ssh-ed25519 PUBLIC")
    monkeypatch.setattr(campaign.time, "time", lambda: 100)
    entered, release = threading.Event(), threading.Event()

    def create(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        raise RuntimeError("uncertain creation")

    monkeypatch.setattr(campaign.subprocess, "check_output", create)
    operations = []

    def api(path, *, method="GET"):
        operations.append((path, method))
        return {"desiredStatus": "RUNNING"}

    monkeypatch.setattr(campaign, "api", api)
    campaign.write_json(
        tmp_path / "manifest.json",
        {
            "campaign": "test",
            "rate": 1.59,
            "budget": 50,
            "volume": "volume",
            "region": "region",
            "jobs": [
                {
                    "name": campaign.JOBS[0],
                    "pod_id": "existingcampaignpod",
                    "created_at": 0,
                    "deadline": 100,
                    "reserved_cost": 1.59,
                }
            ],
        },
    )
    argv = ["--directory", str(tmp_path)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        launch = pool.submit(campaign.main, ["launch", *argv, "--job", "smoke"])
        try:
            assert entered.wait(2)
            pool.submit(campaign.main, ["monitor", *argv, "--once"]).result(timeout=2)
            with pytest.raises(ValueError, match="already"):
                campaign.main(["launch", *argv, "--job", "smoke"])
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="uncertain creation"):
            launch.result(timeout=2)
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["jobs"][0]["status"] == "stop_requested"
    assert saved["jobs"][1]["status"] == "creation_uncertain"
    assert ("pods/existingcampaignpod/stop", "POST") in operations


def test_self_stop_loads_pod_credentials_and_checks_recorded_identity(
    tmp_path, monkeypatch
):
    import os
    import subprocess

    environment = tmp_path / "rp_environment"
    environment.write_text(
        "export RUNPOD_POD_ID=recordedpod\nexport RUNPOD_API_KEY=podscoped\n"
    )
    cli = tmp_path / "runpodctl"
    cli.write_text(
        '#!/bin/sh\n[ "$RUNPOD_API_KEY" = podscoped ] || exit 1\nprintf "%s\\n" "$*" > "$STOP_LOG"\n'
    )
    cli.chmod(0o755)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setenv("STOP_LOG", str(tmp_path / "stop.log"))
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    run = subprocess.run

    def on_pod(command, **kwargs):
        command = [
            part.replace("/etc/rp_environment", str(environment)) for part in command
        ]
        return run(command, **kwargs)

    monkeypatch.setattr(campaign.subprocess, "run", on_pod)
    campaign.own_stop("recordedpod")
    assert (tmp_path / "stop.log").read_text() == "stop pod recordedpod\n"
    with pytest.raises(subprocess.CalledProcessError):
        campaign.own_stop("otherpod")
    assert (tmp_path / "stop.log").read_text() == "stop pod recordedpod\n"


def test_smoke_records_manifest_pod_id_without_ssh_environment(tmp_path, monkeypatch):
    stops = []
    monkeypatch.setattr(campaign, "own_stop", stops.append)
    monkeypatch.setattr(campaign.os, "environ", campaign.os.environ.copy())
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    monkeypatch.setattr(campaign, "discover_sc2", lambda: tmp_path)
    monkeypatch.setattr(
        campaign.shutil,
        "disk_usage",
        lambda path: type("Disk", (), {"free": 30 * 1024**3})(),
    )
    campaign.write_json(
        tmp_path / "job.json",
        {
            "campaign": "test",
            "job": {"name": "smoke", "pod_id": "recordedpod", "wandb_id": "smokeid"},
        },
    )

    def smoke(directory):
        assert campaign.os.environ["RUNPOD_POD_ID"] == "recordedpod"

    monkeypatch.setattr(campaign, "smoke", smoke)
    campaign.run_job(tmp_path)
    assert json.loads((tmp_path / "outcome.json").read_text())["completed"] is True
    assert stops == ["recordedpod"]


@pytest.mark.parametrize("error_number", [errno.ENOSPC, errno.EDQUOT])
@pytest.mark.parametrize("action", ["run", "watchdog"])
def test_volume_write_failure_does_not_prevent_own_stop(
    tmp_path, monkeypatch, error_number, action
):
    campaign.write_json(
        tmp_path / "job.json",
        {
            "campaign": "test",
            "job": {
                "name": "samples2-seed0",
                "pod_id": "recordedpod",
                "wandb_id": "trainid",
                "deadline": 0,
            },
        },
    )
    stops = []
    monkeypatch.setattr(campaign, "own_stop", stops.append)

    def unavailable(*args):
        raise OSError(error_number, "volume full")

    def failed_training(command, **kwargs):
        assert command[1:3] == ["-m", "majepa.main"]
        raise campaign.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(campaign, "write_json", unavailable)
    monkeypatch.setattr(campaign.os, "environ", campaign.os.environ.copy())
    monkeypatch.setattr(campaign, "discover_sc2", lambda: tmp_path)
    monkeypatch.setattr(
        campaign.shutil,
        "disk_usage",
        lambda path: type("Disk", (), {"free": 30 * 1024**3})(),
    )
    monkeypatch.setattr(campaign.subprocess, "run", failed_training)
    if action == "run":
        with pytest.raises(OSError, match="volume full"):
            campaign.run_job(tmp_path)
    else:
        campaign.watchdog(tmp_path)
    assert stops == ["recordedpod"]
