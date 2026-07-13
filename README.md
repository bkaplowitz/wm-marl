# JaxMARL + MeltingPot POC

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

The important pins are:

- `dm-meltingpot==2.4.0`
- `dmlab2d==1.0.0`
- `shimmy[meltingpot]==2.0.1`
- `jaxmarl[algs]==0.1.0`
- `jax==0.4.36`
- `jaxlib==0.4.36`

### CUDA

```bash
uv sync --python 3.11 --extra dev --extra cuda12
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

`world_marl.jepa` is a SIGReg-JEPA latent world model for continuous control:
a shared-encoder self-predictive transformer that fits
`p(z_next, reward, continue | z, continuous_action)` with a sketched
isotropic-Gaussian (LeJEPA-style) collapse regularizer, plus replay,
open-loop evaluation, and latent-policy training utilities
(`src/world_marl/jepa/ARCHITECTURE.md` has the full design). Its pass/fail
gates and accounting live in `world_marl.jepa.validation`.

The end-to-end DMC/Brax validation harness that drives it:

```bash
uv run world-marl-train-dmc-jepa --env dmc:cartpole-balance   # fit + gate a JEPA world model (+ optional latent policy)
uv run world-marl-train-brax-ppo --env brax:reacher           # model-free Brax PPO baseline (brax.training)
uv run world-marl-plot-brax-diagnostics                       # compare JEPA vs PPO runs from their run dirs
uv run world-marl-optuna-dmc-jepa --task dmc:cartpole-balance # Optuna HPO over train-dmc-jepa trials (needs --extra hpo)
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
