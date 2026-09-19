"""Campaign provenance and verified W&B artifact uploads."""

import hashlib
import json
import os
from pathlib import Path


def recorded_config(config):
    result = dict(config)
    manifest = os.environ.get("MAJEPA_CAMPAIGN_MANIFEST")
    if not manifest:
        return result
    if "wandb" not in config.logger.outputs:
        raise ValueError("campaign recording requires the wandb logger")
    if os.environ.get("WANDB_MODE", "online") != "online":
        raise ValueError("campaign recording requires online W&B")
    if config.script == "train" and not config.run.final_save:
        raise ValueError("campaign recording requires run.final_save")
    manifest_data = json.loads(Path(manifest).read_text())
    provenance = {
        key: manifest_data[key]
        for key in (
            "campaign",
            "source_sha256",
            "base_commit",
            "dreamerv3_revision",
            "changed_source_files",
            "deleted_source_paths",
        )
        if key in manifest_data
    }
    if "job" in manifest_data:
        provenance["job"] = {
            key: manifest_data["job"][key]
            for key in ("pod_id", "name")
            if key in manifest_data["job"]
        }
    if not provenance.get("base_commit"):
        raise ValueError("campaign recording requires base_commit provenance")
    with Path(os.environ["MAJEPA_SOURCE_ARCHIVE"]).open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != provenance.get("source_sha256"):
        raise ValueError("campaign source archive hash does not match provenance")
    result["campaign_provenance"] = provenance
    return result


def record_campaign_start(config, provenance):
    receipts = {
        "source": upload_verified_artifact(
            "source", [os.environ["MAJEPA_SOURCE_ARCHIVE"]], metadata=provenance
        ),
        "config": upload_verified_artifact(
            "config",
            [
                Path(config.logdir) / "config.yaml",
                os.environ["MAJEPA_CAMPAIGN_MANIFEST"],
            ],
            metadata=provenance,
        ),
    }
    path = Path(config.logdir) / "artifact_verification.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipts, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def upload_verified_artifact(kind, paths, metadata=None):
    import wandb

    run = wandb.run
    if run is None:
        raise RuntimeError("campaign artifact requires an active W&B run")
    run.summary[f"artifacts/{kind}_verified"] = False
    artifact = wandb.Artifact(f"{run.id}-{kind}", type=kind, metadata=metadata)
    names = set()
    for value in paths:
        path = Path(str(value))
        if path.name in names:
            raise ValueError(f"duplicate artifact path name: {path.name}")
        names.add(path.name)
        if path.is_dir():
            artifact.add_dir(str(path), name=path.name)
        elif path.is_file():
            artifact.add_file(str(path), name=path.name)
        else:
            raise FileNotFoundError(path)
    expected = {name: entry.digest for name, entry in artifact.manifest.entries.items()}
    if not expected:
        raise ValueError("cannot verify an empty artifact")
    uploaded = run.log_artifact(artifact).wait(timeout=600)
    qualified_name = f"{run.entity}/{run.project}/{uploaded.name}"
    remote = wandb.Api().artifact(qualified_name)
    actual = {name: entry.digest for name, entry in remote.manifest.entries.items()}
    if (
        remote.state != "COMMITTED"
        or remote.digest != uploaded.digest
        or actual != expected
    ):
        raise RuntimeError(f"remote artifact verification failed: {qualified_name}")
    run.summary[f"artifacts/{kind}_verified"] = True
    return {
        "artifact": qualified_name,
        "digest": remote.digest,
        "files": sorted(actual),
        "verified": True,
    }


def publish_run_artifact(entity, project, run_id, kind, paths, metadata=None):
    """Resume a completed child run to upload an artifact, then close that run."""
    import wandb

    if wandb.run is not None:
        raise RuntimeError("artifact publisher cannot replace an active W&B run")
    run = wandb.init(
        entity=entity, project=project, id=run_id, resume="must", mode="online"
    )
    exit_code = 1
    try:
        receipt = upload_verified_artifact(kind, paths, metadata=metadata)
        exit_code = 0
        return receipt
    finally:
        run.finish(exit_code=exit_code)
