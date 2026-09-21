# Baseline provenance

The branch was created from the exact package snapshot used by the September 21,
2026 control reproductions in W&B group `ma-jepa-local-kl-exploration-20260921`.

- Pod snapshot: `/workspace/majepa_kl_exploration_20260921/source`
- Executable package digest: `e4760f05e306704e2ba2ab2462cb4230da4ee35782a75ed91b12f7dbc1681798`
- Digest algorithm: SHA-256 over sorted paths and bytes under `src`, excluding
  `__pycache__`
- Reference runs: `ke21-control-repro1-*`

The imported source was then reduced to the one reproduced baseline profile.
The paired-RNG audit, entropy annealing, replay sweeps, and other treatments are
not part of this branch. Git history and W&B retain those experiments.
