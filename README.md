# MA-JEPA baseline

This branch contains one maintained algorithm and one configuration: the
fixed4 MA-JEPA baseline reproduced on September 21, 2026. It is a decoder-free,
centralized-training/decentralized-execution world model for SMAC with imagined
PPO.

The executable actor receives only an agent's local observation and local
history. Training additionally uses a joint JEPA predictor and centralized
attention critic. The baseline uses:

- local action masks during imagination and Bernoulli availability sampling;
- no local reward or continuation heads;
- a 4096-dimensional deterministic state and a 32 x 64 categorical latent;
- a 3 x 512 actor and width-256 centralized critic;
- local and joint world-model learning rates of `1e-4`;
- actor and critic learning rates of `3e-5`;
- BPTT2, imagination horizon 5, PPO clip 0.2, and fixed entropy 0.003;
- 50/50 recent/uniform sampling for independent world and behavior replay views;
- fixed4 snapshot-staggered replay startup after a 5k prefill;
- one collection environment and no paired-RNG intervention.

The complete resolved configuration is [src/majepa/configs.yaml](src/majepa/configs.yaml).
There are no sweep or treatment profiles in this branch.

## Install

Use Python 3.11 from the repository root:

```bash
git submodule update --init --recursive
uv sync --locked --extra dev --extra smac --extra cuda12
export SC2PATH=/path/to/StarCraftII
```

## Reproduce the baseline

The launcher trains the final checkpoint and then evaluates it over 100 held-out
greedy episodes using a separate W&B run ID:

```bash
./scripts/run_baseline.sh
```

Its default is `2s3z`, seed 0, 5 agents, and 50k environment records. Override
only the map-specific fields when reproducing another run:

```bash
TASK=smac_3s_vs_4z NUM_AGENTS=3 STEPS=100000 SEED=1 \
RUN_NAME=jema-baseline-3s_vs_4z-s1 ./scripts/run_baseline.sh
```

The script refuses to reuse an existing run directory or W&B identity. It uses
`WANDB_ENTITY=osaze-obahor`, `WANDB_PROJECT=majepa-ppo-treatments`, and
`WANDB_RUN_GROUP=jema-baseline` unless overridden.

The Python entry point provides the same locked configuration:

```bash
uv run --no-sync majepa-train \
  --task smac_2s3z --num-agents 5 --seed 0 \
  --total-env-steps 50000 --experiment-dir ./runs/2s3z-s0
```

## Layout

```text
src/majepa/agent.py          local encoder, history model, actor, value models
src/majepa/marl/core.py      joint JEPA predictor and centralized critic
src/majepa/training/         JEPA objectives, BPTT2, and imagined PPO
src/majepa/replay.py         independent 50/50 replay views
src/majepa/train.py          fixed4 collection and learner scheduling
src/majepa/evaluation.py     curve and final-checkpoint evaluation
scripts/run_baseline.sh      exact train + final100 reproduction
```

See [architecture](docs/architecture.md) and [provenance](docs/provenance.md).
