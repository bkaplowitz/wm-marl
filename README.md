# MA-JEPA

Decoder-free multi-agent world modelling with a shared decentralized actor and
PPO on imagined team trajectories. `clean_jepa` contains the implementation used
by the September 12, 2026 size / learning-rate runs. It is a cleanup of the deployed
source, with explicit configurations and a reproducible evaluation entry point.

The maintained model has a local encoder and history for each agent, a shared
joint JEPA predictor during training, and a centralized critic. It learns future
embeddings and aligns the resulting local posterior distributions. There is no
observation decoder or learned teammate-belief module.

See [architecture](docs/architecture.md), [results and next experiments](docs/research-notes.md),
and [source provenance and validation](docs/provenance.md).

## Install

Use Python 3.11. Run these commands from the repository root:

```bash
git submodule update --init --recursive
uv sync --locked --extra dev --extra smac --extra cuda12
export PYTHONPATH="$PWD/external/dreamerv3${PYTHONPATH:+:$PYTHONPATH}"
export SC2PATH=/path/to/StarCraftII
```

Install StarCraft II and the SMAC maps separately, following the
[SMAC instructions](https://github.com/oxwhirl/smac#installation).
`external/dreamerv3` supplies the pinned Embodied/JAX runtime; its algorithms are
not alternative first-party learners. Keep this checkout and `PYTHONPATH` when
using either command-line entry point. CPU development needs only `--extra dev`;
CUDA training also requires a compatible NVIDIA driver.

## Train

The reference profile is WM4096, 32 categorical variables with 64 classes each,
actor 3 × 512, and world-model LR `1e-4`:

```bash
uv run --no-sync majepa-train \
  --configs reference \
  --task smac_2s3z \
  --agent.num_agents 5 \
  --seed 0 \
  --logdir ./runs/reference-2s3z-seed0
```

For `3s_vs_4z`, use `--task smac_3s_vs_4z --agent.num_agents 3`.
Use a fresh log directory for each independent run. The training runtime resumes
an existing checkpoint when reusing a training directory.

| Profile | Local deterministic width | Local and joint WM LR |
| --- | ---: | ---: |
| `reference` | 4096 | `1e-4` |
| `wm_lr15e5` | 4096 | `1.5e-4` |
| `wm2048` | 2048 | `1e-4` |
| `wm2048_lr15e5` | 2048 | `1.5e-4` |

These profiles retain the same 64 latent classes, actor, critic, loss weights,
and data budget. They are separate experiments; smaller width or higher LR is
not automatically promoted into the reference profile.

The default budget is 50,000 driver records including a 5,000-record replay
prefill, with one collection environment. Curve evaluations use 32 greedy
held-out episodes every 5,000 records. The driver clock includes reset records;
log `counters/environment_steps` as well when comparing sample budgets.
A 200k experiment uses `--run.steps 200000`; prefill remains explicitly 5k.

Configuration flags use dotted names from
[src/majepa/configs.yaml](src/majepa/configs.yaml). For example:

```bash
uv run --no-sync majepa-train --configs reference \
  --agent.opt.lr 0.0001 \
  --agent.marl.ctde.opt.lr 0.00015 \
  --agent.ppo.actor_lr 0.00003 \
  --agent.ppo.critic_lr 0.00003 \
  --logdir ./runs/separate-world-rates-seed0
```

Local JSON logs are enabled by default. For W&B, authenticate using `wandb login`,
set `WANDB_ENTITY`, `WANDB_PROJECT`, and optionally `WANDB_RUN_GROUP`, then add
`--logger.outputs jsonl wandb`. The resolved configuration is saved locally and
sent to W&B. Use distinct `WANDB_RUN_ID` / `WANDB_NAME` values for each training
and evaluation process, or leave them unset.

## Evaluate the final checkpoint

```bash
uv run --no-sync majepa-evaluate ./runs/reference-2s3z-seed0
```

This uses the saved training configuration and the latest complete checkpoint,
100 greedy episodes, up to four evaluation environments, and worker seed offset
100,000. Results go to `final100/evaluation_summary.json` and
`final100/evaluation_episodes.jsonl`, plus the configured loggers. Existing output
is not overwritten. Use `--output-dir` for a separate evaluation.

The final checkpoint is evaluated rather than selecting the best training-curve
point. `--episodes`, `--envs`, and `--seed-offset` can override the protocol.

## Code layout

```text
src/majepa/main.py          configuration, environments, logging, entry points
src/majepa/agent.py         shared local modules and state
src/majepa/marl/            team axes, joint simulator, centralized critic
src/majepa/models/          encoder, JEPA predictors, heads, latent distributions
src/majepa/world_model/     causal local transformer and posterior
src/majepa/training/        representation, self-fed learning, PPO, optimizers
src/majepa/replay.py        independent world-model and behavior replay views
src/majepa/train.py         collection, learner scheduling, checkpoints
src/majepa/evaluation.py    training-curve and final evaluation protocols
tests/                     focused algorithm and runtime correctness checks
```

Standard model, PPO, availability, and outcome metrics remain available. One-off
experiment launchers, remote queues, archived treatments, and standalone diagnostic
programs are outside this branch. Existing experiment data remains in W&B and the
original checkouts.

## Check

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
```

MIT. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
