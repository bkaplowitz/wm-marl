# Source provenance

Commit `0fd31b3` imported the MA-JEPA source from the pod snapshot listed below.
The current `clean_jepa` adds configurable two-action imagination to that source.
This port changes the package fingerprint and does not deploy or restart pod jobs.

## Imported executable source

- Pod source: `/workspace/ma_jepa_staggered_replay_20260915`
- Deployed base commit recorded by the pod: `6261429abc4489871cc54c036c7871e65777a13e`
- Executable source fingerprint: `711bfd0a89a270ee16dc4089248c3f53956bddd17ec8757ad9d71124a586ad9f`
- Fingerprint algorithm: SHA-256 over sorted paths under `src/majepa`, relative to
  the source root; each path, file bytes, and a trailing NUL are included. Files
  with `.py`, `.yaml`, or `.yml` suffixes are hashed.
- Package files covered by the fingerprint: 47.

`SOURCE_SHA256` stores the imported fingerprint and `DEPLOYED_COMMIT` stores the
pod's recorded base commit. These files remain records of that baseline, not of
the modified package. The import recorded the same manifest for the pod's
fixed-start and entropy queue launchers.

The branch retains the pinned Python dependency lockfile from the cleaned
checkout. The two-action port retains the pod configuration and runtime modules,
adding the sample-count option and its rollout/training behavior. Generated
caches, queue logs, checkpoints, and one-off launch outputs are not part of the
branch.

## What this establishes

Checking out `0fd31b3` restores the imported MA-JEPA package snapshot. The current
branch includes subsequent changes and must not be identified as the source of
the recorded pod results. The deployed commit identifies the snapshot's recorded
origin, while the fingerprint verifies its imported package files.

This parity check does not claim bit-for-bit training reproducibility. Replay
sampling and accelerator scheduling can still affect a run unless the launch
protocol also fixes their ordering. It also does not change or restart any live
pod job.

## Verification

From the repository root, recompute the package fingerprint with:

```bash
python - <<'PY'
from pathlib import Path
import hashlib

root = Path('.')
files = sorted(
    path for path in (root / 'src' / 'majepa').rglob('*')
    if path.is_file()
    and path.suffix in {'.py', '.yaml', '.yml'}
    and not path.name.startswith('._')
)
digest = hashlib.sha256()
for path in files:
    digest.update(str(path.relative_to(root)).encode() + b'\0')
    digest.update(path.read_bytes() + b'\0')
print(digest.hexdigest())
PY
```

At imported commit `0fd31b3`, the output must equal `SOURCE_SHA256`. With the
two-action port applied, a different fingerprint is expected.
