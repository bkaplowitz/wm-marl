# Source provenance

`clean_jepa` is the source checkout for the MA-JEPA learner used by the
currently running pod jobs. The pod source was copied into this checkout before
this branch was pushed; the live pod directory was left unchanged so active
jobs keep using their original snapshot.

## Canonical executable source

- Pod source: `/workspace/ma_jepa_staggered_replay_20260915`
- Deployed base commit recorded by the pod: `6261429abc4489871cc54c036c7871e65777a13e`
- Executable source fingerprint: `711bfd0a89a270ee16dc4089248c3f53956bddd17ec8757ad9d71124a586ad9f`
- Fingerprint algorithm: SHA-256 over sorted paths under `src/majepa`, relative to
  the source root; each path, file bytes, and a trailing NUL are included. Files
  with `.py`, `.yaml`, or `.yml` suffixes are hashed.
- Package files covered by the fingerprint: 47.

`SOURCE_SHA256` stores the expected fingerprint and `DEPLOYED_COMMIT` stores the
pod's recorded base commit. Recomputing the hash over this checkout gives the
same 47-file fingerprint. The active queue launchers read the same manifest from
the pod source before starting a job, so fixed-start and entropy queue jobs use a
single executable snapshot.

The branch retains the pinned Python dependency lockfile from the cleaned
checkout. The executable package, configuration, and runtime modules are the
ones from the pod snapshot; generated caches, queue logs, checkpoints, and
one-off launch outputs are not part of the branch.

## What this establishes

A fresh clone of `clean_jepa`, with its submodules initialized and its locked
environment installed, has the same MA-JEPA package source as the current pod
snapshot. The commit hash and source fingerprint are independent checks: the
commit identifies the snapshot's recorded origin, while the fingerprint verifies
the files actually used by the learner.

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

The output must equal the value in `SOURCE_SHA256`.
