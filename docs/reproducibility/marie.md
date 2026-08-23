# MARIE Reproduction Notes

## Purpose

MARIE is retained as a peer-reviewed multi-agent world-model reference for
DreaMARL-CTDE. The upstream experiment asks whether the published low-data SMAC
behavior can be reproduced in the local hardware and software environment. It
is not part of DreaMARL and no MARIE source is imported by the maintained
algorithm.

Primary sources:

- [MARIE paper](https://arxiv.org/abs/2406.15836)
- [official MARIE repository](https://github.com/breez3young/MARIE)
- [official SMAC repository](https://github.com/oxwhirl/smac)

## Pinned protocol

| Component | Revision or version |
| --- | --- |
| MARIE | `5dc114f78e9f35389b843e05f01c455988451d0e` |
| SMAC | `d6aab33f76abc3849c50463a8592a84f59a5ef84` |
| StarCraft II | `4.10` |

The pinned MARIE revision is the paper-era source from April 2025. The launcher
checks the expected revision and rejects tracked upstream modifications.

The first comparison maps are `3m` and `3s_vs_4z`, each at 100,000 real
environment steps. They exercise basic coordination and the harder asymmetric
combat regime targeted by DreaMARL-CTDE. A one-seed run is only an
implementation and throughput gate; a paper comparison requires the complete
multi-seed protocol.

MARIE maps CLI seed `s` to internal seed `23 + 100s`. The launcher records both
values.

## Isolated runtime

The official source requires a legacy Python 3.10 environment. The tested
runtime used PyTorch 1.13.1 with CUDA 11.7, torchvision 0.14.1, Ray 2.7.2, the
pinned SMAC source, and StarCraft II 4.10. MARIE eagerly imports its Flatland
and MAMuJoCo adapters even for SMAC, so their import-time dependencies must
also be installed. Those adapters do not participate in the SMAC experiment.

Before a full reproduction, verify:

1. CUDA import and device discovery;
2. clean pinned MARIE and SMAC revisions;
3. import of the MARIE training entry point without source edits;
4. one real SMAC reset and environment step;
5. StarCraft II 4.10 visibility through `SC2PATH`;
6. artifact and W&B destinations.

## Invocation

```bash
MARIE_PYTHON=/path/to/python3.10 \
SC2PATH=/path/to/StarCraftII \
uv run dreamarl-train-marie \
  --map 3s_vs_4z \
  --seed 1 \
  --steps 100000 \
  --mode online
```

The adapter generates the unmodified upstream command:

```text
python train.py --n_workers 1 --env starcraft --env_name <map> \
  --seed <seed> --steps 100000 --mode online --tokenizer vq --decay 0.8 \
  --temperature 1.0 --sample_temp inf --ce_for_av
```

The upstream project logs to the W&B project `starcraft`. DreaMARL comparison
tools normalize completed artifacts afterward rather than modifying MARIE's
logger.

## Outcome of the local attempt

The local MARIE jobs initialized and trained without upstream source edits, but
throughput was too low for the planned comparison window. They were stopped
before the declared 100,000-step budgets. Consequently:

- the run is evidence that the pinned program can execute in the local SMAC
  environment;
- it is not a completed reproduction;
- its partial win-rate curve must not be compared with a completed DreaMARL run;
- it neither confirms nor refutes MARIE's published result.

This outcome is why the repository keeps MARIE as isolated comparison tooling
and a structural reference, while the maintained implementation work proceeds
in the substantially faster first-party DreaMARL runtime.

## Completion criteria for a future reproduction

A future reproduction can be described as successful only when:

1. every reported job reaches its exact real-step budget without source edits;
2. evaluation uses the paper's map, horizon, seed, and aggregation protocol;
3. source revisions, StarCraft II version, seed mapping, and step accounting are
   recoverable from manifests;
4. sustained evaluation behavior is reported rather than a selected peak;
5. the requested number of seeds is complete.

Until then, MARIE numbers should be cited from the paper and clearly separated
from locally reproduced results.
