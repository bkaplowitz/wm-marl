# World MARL

## Official DreamerV3 Baseline

The DreamerV3 baseline runs the pinned `danijar/dreamerv3` implementation as
an external process. `world_marl` does not reimplement or replace its RSSM,
replay, losses, actor, critic, or training loop. The integration owns only
environment setup, launch metadata, fixed-checkpoint evaluation, step
accounting, and normalized artifacts.

Initialize the pinned source and create its dependency-isolated environment:

```bash
git submodule update --init --recursive
uv sync --python 3.11 --extra dev
uv run world-marl-setup-dreamerv3 --accelerator cpu     # macOS/local smoke
uv run world-marl-setup-dreamerv3 --accelerator cuda12  # Linux GPU
```

Run the canonical 500K DMC Reacher Easy experiment:

```bash
uv run world-marl-train-dmc-dreamerv3 \
  --task dmc_reacher_easy \
  --seed 0 \
  --total-env-steps 500000 \
  --platform cuda \
  --save-every-seconds 300 \
  --wandb-project world-marl \
  --wandb-entity osaze-obahor
```

The model, optimizer, replay, and imagination settings come directly from the
upstream `dmc_proprio` config. Use `--official-budget` for its current 1.1M-step
preset. `--debug --platform cpu` is available only for installation smoke tests
and is not a learning result.

Evaluate the latest periodically saved checkpoint without selecting among
checkpoints:

```bash
uv run world-marl-eval-dmc-dreamerv3 \
  runs/dreamerv3/dmc_reacher_easy/seed_0/<timestamp> \
  --episodes 20 \
  --success-threshold 900
```

Each experiment contains the untouched upstream log directory plus:

- `launch.json`: exact command, upstream commit, configs, seed, and budget;
- `process.log`: complete upstream stdout/stderr;
- `normalized/training_episodes.jsonl`: online episode returns and steps;
- `normalized/training_curve.json`: 10K-transition bins;
- `normalized/official_reference.json`: bundled official five-seed curve;
- `normalized/training_curve.png`: this run against the official curve;
- `evaluation/*/evaluation_summary.json`: all held-out returns and separate
  train, evaluation, and total real-transition counts.

The canonical contract is recorded in
`configs/dreamerv3/dmc_proprio.yaml`. The pinned upstream revision is
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`.

## JaxMARL + MeltingPot POC

## Architecture

- Melting Pot / dmlab2d: Python-side substrate stepping.
- Shimmy: PettingZoo-style compatibility wrapper.
- `world_marl.envs.MeltingPotVectorAdapter`: batching, RGB normalization, reset,
  optional scalar-observation channels, step, auto-reset, and rollout-friendly
  tensors.
- JAX / Flax / Distrax / Optax: IPPO and MAPPO policies, GAE, and PPO updates.

## Setup

```bash
uv sync --python 3.11 --extra dev
```

For Melting Pot/JaxMARL work, install the Melting Pot extra:

```bash
uv sync --python 3.11 --extra dev --extra meltingpot
```

For DMC JEPA work, install only the DMC extra. This avoids pulling TensorFlow
through Melting Pot on small cloud disks:

```bash
uv sync --python 3.11 --extra dmc
```

The important optional Melting Pot pins are:

- `dm-meltingpot==2.4.0`
- `dmlab2d==1.0.0`
- `shimmy[meltingpot]==2.0.1`

The common JAX/JaxMARL pins are:

- `jaxmarl[algs]==0.1.0`
- `jax==0.4.36`
- `jaxlib==0.4.36`

### CUDA

```bash
uv sync --python 3.11 --extra dev --extra cuda12
```

DMC + CUDA on cloud:

```bash
uv sync --python 3.11 --extra dmc --extra cuda12
```

### Runpod launch

For a one-shot GPU run that cleans itself up, use the Runpod wrapper from the
repo root. The default job is the prefit `train_e2e` path:

```bash
uv run world-marl-runpod \
  --gpu-id "NVIDIA A40" \
  --prefit-train-steps 5000 \
  --policy-warmstart-updates 1
```

The wrapper creates a fresh pod, syncs this checkout to `/root/wm-marl`, installs
the CUDA dev environment, runs the selected wm-marl command, downloads artifacts
to `runs/runpod/<job>/<timestamp>/`, and then deletes the pod. As soon as the pod
exists, that directory holds a `manifest.json` recording the pod id, job command,
and lifecycle status (updated on completion or failure), so a killed local
process still leaves the pod id on disk for manual cleanup. If the remote
command fails or the local process is interrupted, it stops the pod instead and
prints the pod id plus remote output path for inspection. New pods also get a
12-hour auto-stop backstop by default; adjust it with `--auto-stop-hours` or
disable it with `--auto-stop-hours 0`.

The training defaults mirror the long prefit CoinGame run: IPPO, `coins`,
`--prefit-world-model`, `--wm-flow-type linear`, `--wm-fit-steps` from
`--prefit-train-steps`, and `--wm-policy-warmup-updates` from
`--policy-warmstart-updates`. Use `--no-policy-warmstart` to set the warmstart
updates to zero.

Pass extra train options after `--`; later duplicate argparse flags override the
wrapper defaults:

```bash
uv run world-marl-runpod \
  --gpu-id "NVIDIA A40" \
  --prefit-train-steps 10000 \
  -- --total-env-steps 100000 --num-runs 1
```

To run the standalone compare harness instead:

```bash
uv run world-marl-runpod \
  --job compare-world-models \
  --gpu-id "NVIDIA A40" \
  -- --fit-steps 20000 --flow-types discrete transformer linear
```

Preview the lifecycle without creating a pod:

```bash
uv run world-marl-runpod --dry-run
```

### Basic CMDs 

```bash
uv run world-marl-train-e2e \
  --algorithm mappo \
  --substrate coins \
  --num-envs 8 \
  --rollout-steps 128 \
  --total-env-steps 200000 \
  --eval-episodes 50 \
  --num-runs 1 \
  --max-cycles 500 \
  --observation-size 44 \
  --include-observation-scalars \
  --append-agent-id \
  --stochastic-eval \
  --learning-rate 0.00025 \
  --update-epochs 4 \
  --num-minibatches 8 \
  --ent-coef 0.02 \
  --negative-control freeze-policy \
  --min-improvement 0.2
```

```bash
uv run world-marl-train-e2e \
  --algorithm ippo \
  --substrate coins \
  --num-envs 4 \
  --rollout-steps 128 \
  --total-env-steps 100000 \
  --eval-episodes 50 \
  --num-runs 3
```

Each run writes:

- `config.json`
- `versions.json`
- `random_baseline.json`
- `initial_policy_evaluation.json`
- `metrics.jsonl`
- `returns.png`
- `checkpoint/checkpoint.msgpack`
- `checkpoint/metadata.json`
- `reload_evaluation.json`
- `outcome.json`

The top-level experiment directory streams one `progress.jsonl` row per
completed run (including the negative control) as it finishes, and writes
`summary.json` at the end.

Each `metrics.jsonl` row includes rollout diagnostics for debugging learning
failures:

- sampled action counts/frequencies, both aggregate and per agent;
- sampled-policy entropy, aggregate and per agent;
- rollout rewards and completed-episode returns split by agent;
- value prediction mean/std, GAE target mean/std, and value explained variance;
- generic info/event counters, including coin-related and `coin_consumed` keys
  if the Melting Pot wrapper exposes them.

### JEPA Model-Based RL

The maintained JEPA route learns action-conditioned latent dynamics and trains a
stochastic actor plus critic through imagined latent rollouts. It uses no
observation-reconstruction objective, no EMA target encoder, no real-environment
checkpoint search, and no task-specific failure buffer. An optional post-hoc
decoder probe can visualize frozen latent rollouts; it has no gradient path into
the trained model or policy.

Install DMC and run a local smoke test:

```bash
uv sync --extra dmc
uv run python -m world_marl.scripts.write_dmc_vector_launcher \
  --preset smoke \
  --tasks reacher/easy \
  --seeds 0 \
  --gpus 0 \
  --out-root runs/jepa_smoke
bash runs/jepa_smoke/launcher.sh
```

Generate the fixed 500K experiment:

```bash
uv run python -m world_marl.scripts.write_dmc_vector_launcher \
  --preset jepa_500k \
  --tasks reacher/easy cartpole/swingup finger/spin cheetah/run walker/walk \
  --seeds 0 1 2 3 4 \
  --gpus 0 1 \
  --out-root runs/jepa_500k
bash runs/jepa_500k/launcher.sh
```

The 500K preset uses 499,712 training transitions plus 1,280 held-out
world-model validation transitions. Final 100-episode evaluation is tracked
separately. All W&B curves use actual training environment steps on the x-axis,
and the reported final score comes from the deterministic latest policy.

For the model, objective, schedule, and reproducibility contract, see
[`src/world_marl/jepa/ARCHITECTURE.md`](src/world_marl/jepa/ARCHITECTURE.md).

### Config files & wandb

Instead of a long flag list, load defaults from a YAML file. Keys are the argparse
dest names (underscores); explicit CLI flags still override, and an unknown key is a
hard error.

```bash
uv run world-marl-train-e2e --config configs/train_e2e.example.yaml
uv run world-marl-train-e2e --config configs/train_e2e.example.yaml --wm-fit-steps 5000
```

Add `--wandb` to mirror every `metrics.jsonl` row to Weights & Biases (all runs of an
experiment share a `group`; the local artifacts above are still written):

```bash
uv run world-marl-train-e2e --config configs/train_e2e.example.yaml --wandb \
  --wandb-project world-marl
```

Authenticate once with `wandb login` (or set `WANDB_API_KEY`); use `WANDB_MODE=offline`
to log without a network connection.

### Single-agent adapters & the JEPA world model

Three single-agent adapters expose the same vector-env contract as the
Melting Pot and CoinGame adapters (observations `[env, agent, ...]` with a
singleton agent axis):

- `GymnaxVectorAdapter` — discrete-action Gymnax tasks. Wired into
  `world-marl-train-e2e`: pass `--substrate gymnax:<env_name>` (e.g.
  `gymnax:CartPole-v1`) to train IPPO/MAPPO with the vector/MLP policy path.
- `BraxVectorAdapter` / `DMCVectorAdapter` — continuous-control Brax and
  DeepMind Control Suite tasks. These are library adapters (not wired into
  the discrete-action `train_e2e` pipeline); their heavy dependencies are
  optional extras:

```bash
uv sync --extra brax   # brax
uv sync --extra dmc    # dm-control
```

Additional single-agent and world-model tools are available separately from
the canonical JEPA launcher:

```bash
uv run world-marl-train-dmc-jepa --env dmc:cartpole-balance   # canonical JEPA model-based RL trainer
uv run world-marl-train-brax-ppo --env brax:reacher           # model-free Brax PPO baseline (brax.training)
uv run world-marl-plot-brax-diagnostics                       # compare JEPA vs PPO runs from their run dirs
uv run world-marl-optuna-dmc-jepa --task cartpole/swingup       # optional JEPA HPO diagnostics (needs --extra hpo)
uv run world-marl-plot-jepa-decoder --run-dir <run>            # plot optional frozen-latent decoder artifacts
uv run world-marl-write-dmc-vector-launcher                   # emit tmux/pod launcher scripts for sweeps
```

### CEM-MPC single-agent arm

CEM-MPC is an alternative policy optimizer to PPO, using the cross-entropy method for model-predictive control inside the learned world model. Instead of amortizing a policy network, it plans a short action sequence at each decision point using receding-horizon MPC and executes only the first action. Use `--policy-optimizer cem` with `train_single_genwm` to compare planning vs amortized-PPO learning on single-agent environments; CEM requires a learned world model and is not compatible with `--arm model-free`.

```bash
uv run python -m world_marl.scripts.train_single_genwm \
  --env brax:reacher --arm discrete-transformer \
  --policy-optimizer cem \
  --cem-horizon 5 --cem-samples 64 --cem-topk 8 --cem-iters 3
```

Key CEM hyperparameters: `--cem-samples` (population per iteration), `--cem-topk` (elite fraction), `--cem-iters` (CEM iterations per solve), `--cem-horizon` (planning horizon in steps), `--cem-receding-horizon` (enable receding-horizon MPC; on by default). The CEM solver is JIT-compiled (`cem_solve`) and exposed in `world_marl.genwm` alongside `CEMConfig`, `CEMPlanner`, `make_genwm_plan_fn`, `discounted_return`, and `sample_candidates`.

## Tests

```bash
uv run pytest
```
