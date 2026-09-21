import json

import pytest

from majepa import campaign


def specification(**changes):
    return {
        "wandb": {"entity": "test-entity", "project": "test-project"},
        "budget": 50,
        "max_gpu_hourly_rate": 1.0,
        "max_hours": 8,
        "placements": [{"region": "TEST-1", "volume": "testvolume"}],
        "maps": [{"name": "2s3z", "agents": 5, "steps": 50000}],
        "seeds": [0, 1, 2],
        "treatments": [{"name": "reference", "overrides": {}}],
        **changes,
    }


def test_plan_resolves_actual_configs_and_reuses_gpus():
    spec = specification(
        treatments=[
            {"name": "reference", "overrides": {}},
            {
                "name": "jointgradient",
                "overrides": {"agent.world_model_gradients.joint_prediction": True},
                "depends_on": ["reference"],
            },
        ]
    )
    plan = campaign.plan(spec, "test")
    assert len(plan["runs"]) == 6
    assert len(plan["allocations"]) == 1
    assert plan["allocations"][0]["gpu_count"] == 4
    first = plan["runs"][0]
    assert first["expected_updates"] == 5626
    assert first["config"]["agent.ppo.entropy_coefficient"] == 0.003
    assert first["config"]["agent.dyn.parallel_transformer.deter"] == 4096
    assert plan["runs"][3]["depends_on"] == ["reference-2s3z-seed0"]


def test_seed_first_run_order_prioritizes_every_treatment():
    treatments = [
        {"name": name, "overrides": {}}
        for name in ("first", "second", "third", "fourth")
    ]
    plan = campaign.plan(
        specification(treatments=treatments, run_order="seed_first"), "test"
    )
    assert [run["name"] for run in plan["runs"]] == [
        f"{treatment}-2s3z-seed{seed}"
        for seed in (0, 1, 2)
        for treatment in ("first", "second", "third", "fourth")
    ]


def test_single_gpu_topology_keeps_dependencies_on_same_pod():
    spec = specification(
        gpus_per_pod=1,
        treatments=[
            {"name": "reference", "overrides": {}},
            {"name": "treatment", "overrides": {}, "depends_on": ["reference"]},
        ],
    )
    plan = campaign.plan(spec, "test")
    assert len(plan["allocations"]) == 3
    assert all(p["gpu_count"] == 1 for p in plan["allocations"])
    for p in plan["allocations"]:
        assert len(p["run_names"]) == 2


def test_explicit_pod_size_is_preserved_with_fewer_jobs():
    result = campaign.plan(specification(gpus_per_pod=4), "test")
    assert len(result["allocations"]) == 1
    assert result["allocations"][0]["gpu_count"] == 4
    assert len(result["allocations"][0]["run_names"]) == 3
    with pytest.raises(ValueError, match="budget"):
        campaign.plan(specification(gpus_per_pod=4, budget=25), "test")


def test_gpu_fallback_requires_explicit_a100_opt_in():
    plan = campaign.plan(specification(), "test")
    assert campaign.gpu_options(plan) == ["NVIDIA L40", "NVIDIA L40S"]
    plan["allow_a100"] = True
    assert campaign.gpu_options(plan)[2:] == [
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA A100 80GB PCIe",
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"budget": float("nan")},
        {"max_gpu_hourly_rate": -1},
        {"gpus_per_pod": 0},
        {"seeds": [0, 0]},
        {"overrides": {"agent.does_not_exist": 1}},
        {"overrides": {"run.envs": 2}},
        {"treatments": [{"name": "../bad", "overrides": {}}]},
    ],
)
def test_invalid_spec_fails_before_any_cloud_operation(change):
    with pytest.raises((ValueError, KeyError, TypeError)):
        campaign.plan(specification(**change), "test")


def test_budget_reserves_all_gpus_and_setup_and_rejects_duplicate():
    plan = campaign.plan(specification(), "test")
    pod = campaign.reserve(plan, "pod0", 8, now=100)
    assert pod["reserved_cost"] == 24
    assert pod["deadline"] == 28900
    with pytest.raises(ValueError, match="already"):
        campaign.reserve(plan, "pod0", 8, now=100)
    pod["actual_rate"] = 3.5
    assert campaign.accounted_cost(plan, 36100) == 35


def test_rejects_campaign_that_cannot_reserve_full_budget():
    with pytest.raises(ValueError, match="budget"):
        campaign.plan(specification(budget=20), "test")


def test_dry_run_is_read_only_and_shows_resolved_jobs(tmp_path, monkeypatch, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(specification()))
    monkeypatch.setattr(
        campaign.subprocess, "run", lambda *a, **kw: pytest.fail("process")
    )
    monkeypatch.setattr(
        campaign.subprocess, "check_output", lambda *a, **kw: pytest.fail("process")
    )
    target = tmp_path / "campaign"
    campaign.main(
        ["init", "--directory", str(target), "--spec", str(spec), "--dry-run"]
    )
    result = json.loads(capsys.readouterr().out)
    assert len(result["runs"]) == 3
    assert not target.exists()


def test_capacity_rejection_requires_exact_absence_before_fallback(
    tmp_path, monkeypatch
):
    manifest = campaign.plan(specification(), "test")
    campaign.reserve(manifest, "pod0", 8, now=100)
    manifest["jobs"][0]["status"] = "creation_uncertain"
    campaign.write_json(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(
        campaign, "list_pods", lambda: [{"name": "test-pod0", "id": "exists"}]
    )
    with pytest.raises(RuntimeError, match="matching pod"):
        campaign.reconcile_rejection(tmp_path, "pod0", campaign.CAPACITY_MESSAGE)
    monkeypatch.setattr(campaign, "list_pods", lambda: [])
    campaign.reconcile_rejection(tmp_path, "pod0", campaign.CAPACITY_MESSAGE)
    after = json.loads((tmp_path / "manifest.json").read_text())
    assert after["jobs"][0]["status"] == "not_created"
    assert campaign.accounted_cost(after, 1000) == 0


def test_monitor_stops_only_expired_pod_and_keeps_siblings(tmp_path, monkeypatch):
    plan = campaign.plan(specification(gpus_per_pod=1), "test")
    first = campaign.reserve(plan, "pod0", 1, now=100)
    second = campaign.reserve(plan, "pod1", 8, now=100)
    first["pod_id"], second["pod_id"] = "first", "second"
    calls = []
    monkeypatch.setattr(campaign.time, "time", lambda: 4000)
    monkeypatch.setattr(
        campaign,
        "api",
        lambda path, **kw: calls.append((path, kw)) or {"desiredStatus": "RUNNING"},
    )
    campaign.monitor(tmp_path, plan)
    assert ("pods/first/stop", {"method": "POST"}) in calls
    assert not any(path == "pods/second/stop" for path, _ in calls)


def test_bootstrap_reuses_runtime_and_supervisor_owns_stop():
    bootstrap = campaign.bootstrap_script("/workspace/test/pod0")
    assert "uv sync --locked" in bootstrap
    assert "-m majepa.campaign run" in bootstrap
    assert "campaign.py stop" in bootstrap
    assert "RUNPOD_API_KEY" not in campaign.startup_command(1234)


def test_saved_asset_staging_dry_run_and_existing_directory(tmp_path, monkeypatch):
    from majepa import campaign_assets

    def unexpected(*args, **kwargs):
        pytest.fail("staging must not execute in dry-run or overwrite existing assets")

    monkeypatch.setattr(campaign_assets.subprocess, "run", unexpected)
    root = tmp_path / "assets"
    commands = campaign_assets.stage(root, dry_run=True)
    assert len(commands) == 4
    assert commands[1][-1] == str(root)
    assert not root.exists()
    root.mkdir()
    with pytest.raises(FileExistsError):
        campaign_assets.stage(root)


def test_rejects_ambiguous_run_names_and_inexact_step_budget():
    with pytest.raises(ValueError, match="duplicate run"):
        campaign.plan(
            specification(
                seeds=[0],
                maps=[
                    {"name": "c", "agents": 3, "steps": 50000},
                    {"name": "b-c", "agents": 3, "steps": 50000},
                ],
                treatments=[
                    {"name": "a-b", "overrides": {}},
                    {"name": "a", "overrides": {}},
                ],
            ),
            "test",
        )
    with pytest.raises(ValueError, match="10-step"):
        campaign.plan(
            specification(maps=[{"name": "2s3z", "agents": 5, "steps": 50001}]), "test"
        )
    with pytest.raises(ValueError, match="prefill"):
        campaign.plan(
            specification(overrides={"run.world_model_start_step": 1216}), "test"
        )


def test_campaign_uses_baseline_checkpoint_policy():
    result = campaign.plan(specification(), "test")
    assert result["runs"][0]["config"]["run.checkpoint_at_curve_eval"] is False


def test_fingerprint_excludes_live_state_but_catches_config_changes():
    plan = campaign.plan(specification(), "test")
    plan["plan_sha256"] = campaign.manifest_fingerprint(plan)
    campaign.reserve(plan, "pod0", 8, now=100)
    campaign.verify_manifest(plan)
    plan["runs"][0]["config"]["seed"] = 100
    with pytest.raises(ValueError, match="changed"):
        campaign.verify_manifest(plan)


def test_controller_refuses_interrupted_creation_before_another_allocation(
    tmp_path, monkeypatch
):
    plan = campaign.plan(specification(gpus_per_pod=1), "test")
    plan["plan_sha256"] = campaign.manifest_fingerprint(plan)
    campaign.reserve(plan, "pod0", 8)
    campaign.write_json(tmp_path / "manifest.json", plan)
    monkeypatch.setattr(
        campaign, "launch", lambda *a, **kw: pytest.fail("duplicate allocation")
    )
    campaign.controller  # Only local manifest reads allowed in this state.
    with pytest.raises(RuntimeError, match="unreconciled"):
        campaign.controller(tmp_path, once=True)


def test_verified_wandb_config_normalizes_sequence_types(tmp_path, monkeypatch):
    import elements
    import types
    import wandb

    plan = campaign.plan(specification(seeds=[0]), "test")
    run = plan["runs"][0]
    run.update(wandb_id="train", evaluation_wandb_id="eval")
    remote = types.SimpleNamespace(
        config={
            **dict(elements.Config(run["config"])),
            "campaign_provenance": {"changed_files": []},
        },
        state="running",
        url="https://example.test/run",
        summary={
            "artifacts/source_verified": True,
            "artifacts/config_verified": True,
            "train/opt/loss": 1.0,
            "_step": 5000,
        },
    )
    monkeypatch.setattr(
        wandb, "Api", lambda **kw: types.SimpleNamespace(run=lambda path: remote)
    )
    result = campaign.verify_wandb(tmp_path, plan)[run["name"]]
    assert result["verified"] is True
    assert result["config_mismatches"] == []
    assert result["completed"] is False


def test_collect_results_requires_verified_fixed_100_and_aggregates():
    import elements
    import types

    plan = campaign.plan(
        specification(
            treatments=[
                {"name": "alpha", "overrides": {}},
                {
                    "name": "beta",
                    "overrides": {
                        "agent.world_model_gradients.joint_prediction": True,
                        "agent.world_model_gradients.joint_prediction_scale": 0.1,
                    },
                },
            ]
        ),
        "test",
    )
    remote = {}
    scores = {
        "alpha": [(50, 1.0), (60, 2.0), (70, 3.0)],
        "beta": [(60, 2.0), (80, 3.0), (70, 4.0)],
    }
    for run in plan["runs"]:
        treatment = run["name"].split("-2s3z-")[0]
        seed = run["config"]["seed"]
        run.update(
            wandb_id=f"train-{treatment}-{seed}",
            evaluation_wandb_id=f"eval-{treatment}-{seed}",
        )
        remote[run["wandb_id"]] = types.SimpleNamespace(
            config=dict(elements.Config(run["config"])),
            state="finished",
            summary={
                "artifacts/source_verified": True,
                "artifacts/config_verified": True,
                "artifacts/checkpoint_verified": True,
            },
            url=f"https://example.test/{run['wandb_id']}",
        )
        wins, mean_return = scores[treatment][seed]
        remote[run["evaluation_wandb_id"]] = types.SimpleNamespace(
            state="finished",
            summary={
                "artifacts/evaluation_verified": True,
                "final_eval/episodes": 100,
                "final_eval/wins": wins,
                "final_eval/win_rate": wins / 100,
                "final_eval/return_mean": mean_return,
            },
            url=f"https://example.test/{run['evaluation_wandb_id']}",
        )
    client = types.SimpleNamespace(run=lambda path: remote[path.rsplit("/", 1)[-1]])

    result = campaign.collect_results(plan, client)

    alpha = next(item for item in result["aggregates"] if item["treatment"] == "alpha")
    assert alpha["mean_win_rate"] == pytest.approx(0.6)
    assert alpha["sample_sd_win_rate"] == pytest.approx(0.1)
    assert alpha["mean_return"] == pytest.approx(2.0)
    paired = result["paired_differences"][0]
    assert (paired["left"], paired["right"], paired["seeds"]) == (
        "alpha",
        "beta",
        [0, 1, 2],
    )
    assert paired["mean_win_rate_difference"] == pytest.approx(-0.1)

    remote[plan["runs"][0]["evaluation_wandb_id"]].summary[
        "final_eval/episodes"
    ] = 32
    with pytest.raises(ValueError, match="100 episodes"):
        campaign.collect_results(plan, client)


def test_compare_campaign_configs_matches_baseline_and_enforces_allowlist():
    reference = campaign.plan(specification(seeds=[0]), "reference")
    target = campaign.plan(
        specification(
            seeds=[0],
            treatments=[
                {
                    "name": "joint",
                    "overrides": {
                        "agent.world_model_gradients.joint_prediction": True,
                        "agent.world_model_gradients.joint_prediction_scale": 0.1,
                        "agent.loss_scales.ctde_multistep_jepa_action": 0.0,
                    },
                }
            ],
        ),
        "target",
    )
    allowed = {
        "agent.world_model_gradients.joint_prediction",
        "agent.world_model_gradients.joint_prediction_scale",
    }

    result = campaign.compare_campaign_configs(target, reference, allowed)

    assert {item["key"] for item in result["differences"]} == {
        *allowed,
        "agent.loss_scales.ctde_multistep_jepa_action",
    }
    assert [item["key"] for item in result["unexpected"]] == [
        "agent.loss_scales.ctde_multistep_jepa_action"
    ]
    assert all(
        item["reference_run"] == "reference-2s3z-seed0"
        for item in result["differences"]
    )


@pytest.mark.parametrize("observed_rate", [2.5, 4.0])
@pytest.mark.parametrize("network_volume", [True, False])
def test_complete_mocked_pod_launch_stages_one_queue_for_three_gpus(
    tmp_path, monkeypatch, observed_rate, network_volume
):
    import io
    import tarfile
    import types

    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh/runpod_key.pub").write_text("ssh-ed25519 public")
    monkeypatch.setattr(campaign.Path, "home", lambda: home)
    monkeypatch.setattr(
        campaign.netrc,
        "netrc",
        lambda: types.SimpleNamespace(
            authenticators=lambda _: ("user", None, "SECRET")
        ),
    )
    placement = {"region": "TEST-1"}
    if network_volume:
        placement["volume"] = "testvolume"
    manifest = campaign.plan(specification(placements=[placement]), "test")
    campaign.write_json(tmp_path / "manifest.json", manifest)
    with tarfile.open(tmp_path / "source.tar.gz", "w:gz") as archive:
        code = b"# dummy frozen runner\n"
        info = tarfile.TarInfo("src/majepa/campaign.py")
        info.size = len(code)
        archive.addfile(info, io.BytesIO(code))
    commands, staged = [], []

    def create(command, **kwargs):
        commands.append(command)
        return '{"id":"newpod"}'

    monkeypatch.setattr(campaign.subprocess, "check_output", create)
    monkeypatch.setattr(
        campaign.subprocess, "run", lambda *a, **kw: types.SimpleNamespace(returncode=0)
    )
    pod_calls = []

    def pod_api(path, **kwargs):
        pod_calls.append((path, kwargs))
        return {"costPerHr": observed_rate, "ssh": {"ip": "127.0.0.1", "port": 22}}

    monkeypatch.setattr(campaign, "api", pod_api)

    def remote(connection, script, **kwargs):
        staged.append((script, kwargs.get("input")))
        return types.SimpleNamespace(stdout="1234\n")

    monkeypatch.setattr(campaign, "remote", remote)
    if observed_rate > 3:
        with pytest.raises(ValueError, match="hourly rate"):
            campaign.launch(tmp_path, manifest, "pod0", 8)
        failed = json.loads((tmp_path / "manifest.json").read_text())["jobs"][0]
        assert failed["actual_rate"] == observed_rate
        assert ("pods/newpod/stop", {"method": "POST"}) in pod_calls
        return
    campaign.launch(tmp_path, manifest, "pod0", 8)
    create_command = commands[0]
    assert create_command[create_command.index("--gpu-count") + 1] == "3"
    assert create_command[create_command.index("--gpu-id") + 1] == "NVIDIA L40"
    if network_volume:
        assert "--network-volume-id" in create_command
    else:
        assert "--network-volume-id" not in create_command
        assert create_command[create_command.index("--volume-in-gb") + 1] == "300"
    bootstrap = next(
        data
        for script, data in staged
        if "cat >" in script and "bootstrap.sh" in script
    )
    assert ("majepa.campaign_assets" in bootstrap) is not network_volume
    assert ("apt-get install -y unzip" in bootstrap) is not network_volume
    assert "SECRET" not in str(commands)
    saved = json.loads((tmp_path / "manifest.json").read_text())
    allocation = saved["jobs"][0]
    assert allocation["status"] == "running"
    assert allocation["bootstrap_pid"] == 1234
    assert allocation["actual_rate"] == 2.5
    payload = next(
        json.loads(data)
        for script, data in staged
        if "cat >" in script and "job.json" in script
    )
    assert len(payload["runs"]) == 3
    assert payload["job"]["gpu_count"] == 3
    assert any(script == "kill -0 1234" for script, _ in staged)


def test_120_second_check_rejects_stale_queue_pids(tmp_path, monkeypatch):
    import types

    plan = campaign.plan(specification(), "test")
    pod = campaign.reserve(plan, "pod0", 8, now=100)
    pod.update(
        pod_id="id",
        remote_dir="/workspace/test",
        bootstrap_pid=12,
        process_verified_at=100,
    )
    monkeypatch.setattr(campaign.time, "time", lambda: 250)
    monkeypatch.setattr(campaign, "api", lambda *a: {})
    monkeypatch.setattr(campaign, "ssh", lambda *a: [])
    stale = {
        "queue": {"supervisor_pid": 22, "jobs": []},
        "outcome": {},
        "bootstrap_alive": False,
        "processes": {"22": False},
    }
    monkeypatch.setattr(
        campaign,
        "remote",
        lambda *a, **kw: types.SimpleNamespace(stdout=json.dumps(stale)),
    )
    campaign.collect_status(tmp_path, plan)
    assert pod["two_minute_check"]["supervisor_alive"] is False
    assert pod["two_minute_check"]["bootstrap_alive"] is False


def test_capacity_fallback_tries_l40_everywhere_then_l40s(tmp_path, monkeypatch):
    plan = campaign.plan(
        specification(
            seeds=[0],
            placements=[
                {"region": "UK-1", "volume": "uk"},
                {"region": "EU-1", "volume": "eu"},
            ],
        ),
        "test",
    )
    plan["plan_sha256"] = campaign.manifest_fingerprint(plan)
    campaign.write_json(tmp_path / "manifest.json", plan)
    attempts = []

    def allocate(directory, manifest, name, hours, *, placement, gpu_id):
        attempts.append((gpu_id, placement["region"]))
        with campaign.locked_manifest(directory) as current:
            job = campaign.reserve(current, name, hours)
            job["status"] = "creation_uncertain" if len(attempts) < 3 else "running"
        if len(attempts) < 3:
            raise campaign.subprocess.CalledProcessError(
                1, ["create"], stderr=campaign.CAPACITY_MESSAGE
            )

    monkeypatch.setattr(campaign, "launch", allocate)
    monkeypatch.setattr(campaign, "list_pods", lambda: [])
    campaign.controller(tmp_path, once=True)
    assert attempts == [
        ("NVIDIA L40", "UK-1"),
        ("NVIDIA L40", "EU-1"),
        ("NVIDIA L40S", "UK-1"),
    ]
    after = json.loads((tmp_path / "manifest.json").read_text())
    assert [j["status"] for j in after["jobs"]] == [
        "not_created",
        "not_created",
        "running",
    ]
    assert campaign.accounted_cost(after, campaign.time.time()) == 8


def test_dryrun_subprocess_does_not_need_pytest_dependency_path(tmp_path):
    import os
    import subprocess
    import sys

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(specification(seeds=[0])))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "majepa.campaign",
            "init",
            "--directory",
            str(tmp_path / "dry-run"),
            "--spec",
            str(spec),
            "--dry-run",
        ],
        env={**os.environ, "PYTHONPATH": str(campaign.REPO / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert len(json.loads(result.stdout)["runs"]) == 1
    assert not (tmp_path / "dry-run").exists()
