# Official MARIE Reproduction Contract

## Purpose

MARIE is the first external multi-agent mechanism gate for DreaMARL. The
reference runs answer one question before any port is attempted:

> Can the official MARIE implementation reproduce its published low-data SMAC
> behavior in our compute environment?

The reference implementation is never imported by the maintained DreaMARL
runtime. It is executed as an isolated process and remains unmodified.

## Immutable Sources

| Component | Source | Revision |
|---|---|---|
| MARIE | `breez3young/MARIE` | `5dc114f78e9f35389b843e05f01c455988451d0e` |
| SMAC | `oxwhirl/smac` | `d6aab33f76abc3849c50463a8592a84f59a5ef84` |
| StarCraft II | Blizzard Linux package | `4.10` |

The MARIE revision is the paper-era source from April 2025, before later SMAX
support. The launcher rejects a different revision or tracked modifications.

Primary references:

- [MARIE paper](https://arxiv.org/abs/2406.15836)
- [official MARIE repository](https://github.com/breez3young/MARIE)
- [official SMAC repository](https://github.com/oxwhirl/smac)

## First Reproduction Gate

We use two SMAC maps whose paper protocol agrees with the released default
imagination horizon of 15:

| Map | Agents | Real steps | Paper result, four-seed mean win rate |
|---|---:|---:|---:|
| `3m` | 3 | 100,000 | 99.5% |
| `3s_vs_4z` | 3 | 100,000 | 73.0% |

This deliberately avoids maps for which the paper describes a horizon override
that the released CLI does not expose. The first gate uses one seed to verify
the implementation and learning signal. Replication across the paper's four
seeds follows only after this gate passes.

MARIE maps CLI seed `s` to the internal seed `23 + 100s`. Thus CLI seed 1 is
internal seed 123. Both values are recorded in every manifest.

## Exact Invocation

```bash
MARIE_PYTHON=/path/to/python3.10 \
SC2PATH=/path/to/StarCraftII \
uv run dreamarl-train-marie \
  --map 3m \
  --seed 1 \
  --steps 100000 \
  --mode online
```

The generated upstream command is:

```text
python train.py --n_workers 1 --env starcraft --env_name <map> \
  --seed 1 --steps 100000 --mode online --tokenizer vq --decay 0.8 \
  --temperature 1.0 --sample_temp inf --ce_for_av
```

The upstream source hard-codes the W&B project to `starcraft`; preserving that
name is part of running it untouched. DreaMARL comparisons will normalize the
completed artifacts afterward rather than patching MARIE's logger.

## Runtime

The official source requires an isolated Python 3.10 environment. The tested
runtime uses PyTorch 1.13.1 with CUDA 11.7, torchvision 0.14.1, Ray 2.7.2, the
pinned SMAC source, and StarCraft II 4.10. MARIE imports its Flatland and
MAMuJoCo adapters eagerly even for SMAC, so their legacy import dependencies
must also be present. These are packaging requirements only; neither adapter
participates in the SMAC experiment.

Before a paid run, the environment must pass:

1. CUDA import and device discovery;
2. import of the pinned MARIE entry point without source edits;
3. one real `3m` reset and environment step;
4. submodule revision and clean-tree verification;
5. W&B authentication when `--mode online` is used.

## Promotion Gate

The official reproduction passes when all of the following hold:

1. both jobs finish their exact 100,000-real-step budget without source edits;
2. evaluation win rate shows sustained learning rather than one isolated peak;
3. `3m` approaches the paper's near-solved regime;
4. `3s_vs_4z` shows the expected nontrivial learning signal;
5. step accounting, seed mapping, source revisions, and run artifacts are
   recoverable from manifests.

The first seed is an implementation gate, not a paper result. It cannot support
a claim about mean performance or variance.
