# Terminal-Contract Candidate Result

## Decision

The terminal-contract candidate is **not promoted**. The cleaned canonical
baseline remains commit `82d671dbf3b961fd0807bd5136584b79ce3c79a2`.

## Protocol

- Environment: `dmc:reacher/easy`
- Budget: 199,680 training transitions per run
- Seeds: 1 and 2
- Evaluation: latest deterministic policy
- Curve evaluation: 20 episodes at fixed 50k intervals
- Final evaluation: 100 episodes
- Parent: `82d671dbf3b961fd0807bd5136584b79ce3c79a2`
- Candidate code: `e3a40f9f9f9bbe1f66efcd4e179b20f5bd5d3b68`
- Resolved manifests: identical except output paths
- Source isolation: checkout-specific `PYTHONPATH`, one GPU per variant

Remote artifacts:
`/workspace/jepa_terminal_contract_200k_isolated_20260718`

## Final Results

| Seed | Variant | Mean | Std | P10 | CVaR10 | Failure | Success |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Parent | 808.67 | 330.45 | 0.0 | 0.0 | 13% | 73% |
| 1 | Candidate | 809.64 | 293.43 | 131.5 | 26.5 | 9% | 57% |
| 2 | Parent | 921.60 | 104.62 | 854.1 | 733.4 | 1% | 74% |
| 2 | Candidate | 631.82 | 395.55 | 0.0 | 0.0 | 25% | 40% |

| Variant | Mean of means | Mean failure | Mean success | Mean P10 | Mean CVaR10 |
|---|---:|---:|---:|---:|---:|
| Parent | 865.14 | 7% | 73.5% | 427.05 | 366.70 |
| Candidate | 720.73 | 17% | 48.5% | 65.75 | 13.25 |

## Interpretation

The explicit replay contract passed unit and execution correctness checks, but
the bundled behavioral change was not robust. It improved the seed-1 lower
tail, delayed seed-2 learning, briefly recovered at 150k, and then suffered a
large latest-policy regression by 200k.

The failed bundle combined two questions that must be separated:

1. how replay represents boundaries, terminals, cuts, and physical successors;
2. whether DMC time-limit truncations should bootstrap under the chosen
   finite-episode control objective.

The next candidate must first make the representation explicit with exact
baseline behavior, then test boundary targets and truncation bootstrapping one
at a time.
