# Campaign Reporting Design and Implementation Plan

## Design

Extend `python -m majepa.campaign` with two read-only actions instead of adding a
new launcher or report framework:

- `results` reads the existing manifest, fetches its exact W&B train/evaluation
  run IDs, rejects incomplete or unverified fixed-100 evaluations, and emits
  per-seed, aggregate, and paired-treatment results as Markdown or JSON.
- `diff` compares resolved run configs from two existing manifests, matching by
  task and seed, and fails when differences fall outside an explicit key
  allowlist.

Use only the standard library and the existing W&B dependency. Keep collection,
comparison, and rendering as small functions in `campaign.py` so mocked tests do
not need network access.

## Implementation

1. Add focused tests for verified result aggregation, invalid evaluation
   rejection, and allowlisted config comparison.
2. Add the minimal collection, aggregation, diff, and Markdown rendering
   functions to `campaign.py`.
3. Wire `results`, `diff`, `--format`, `--reference`, and
   `--allow-difference` into the existing command.
4. Run only the new targeted tests, then smoke-test the commands against an
   existing completed campaign artifact.

