#!/usr/bin/env bash
set -euo pipefail

# Exact fixed4 MA-JEPA baseline used by the September 21 control reruns.
# Override values through environment variables, for example:
#   TASK=smac_3s_vs_4z NUM_AGENTS=3 STEPS=100000 SEED=1 ./scripts/run_baseline.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
TASK="${TASK:-smac_2s3z}"
NUM_AGENTS="${NUM_AGENTS:-5}"
SEED="${SEED:-0}"
STEPS="${STEPS:-50000}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_NAME="${RUN_NAME:-jema-baseline-${TASK#smac_}-s${SEED}-${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/runs/${RUN_NAME}}"
TRAIN_DIR="${LOG_ROOT}/train/run"
EVAL_DIR="${LOG_ROOT}/final100/run"

: "${SC2PATH:?Set SC2PATH to the StarCraft II installation}"
export PYTHONPATH="${ROOT}/src:${ROOT}/external/dreamerv3${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_ENTITY="${WANDB_ENTITY:-osaze-obahor}"
export WANDB_PROJECT="${WANDB_PROJECT:-majepa-ppo-treatments}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-jema-baseline}"
export WANDB_RESUME=never

if [[ -e "${LOG_ROOT}" ]]; then
  echo "Refusing to reuse existing run directory: ${LOG_ROOT}" >&2
  exit 2
fi

mkdir -p "${TRAIN_DIR}" "${EVAL_DIR}"

export WANDB_RUN_ID="${RUN_NAME}-train"
export WANDB_NAME="${WANDB_RUN_ID}"
export WANDB_JOB_TYPE=train
"${PYTHON}" -m majepa.main \
  --configs baseline \
  --task "${TASK}" \
  --agent.num_agents "${NUM_AGENTS}" \
  --seed "${SEED}" \
  --run.steps "${STEPS}" \
  --logdir "${TRAIN_DIR}" \
  --logger.outputs jsonl wandb

CHECKPOINT_ROOT="${TRAIN_DIR}/ckpt"
CHECKPOINT="${CHECKPOINT_ROOT}/$(<"${CHECKPOINT_ROOT}/latest")"
test -f "${CHECKPOINT}/done"

export WANDB_RUN_ID="${RUN_NAME}-final100"
export WANDB_NAME="${WANDB_RUN_ID}"
export WANDB_JOB_TYPE=final100
"${PYTHON}" -m majepa.main \
  --configs baseline \
  --task "${TASK}" \
  --agent.num_agents "${NUM_AGENTS}" \
  --seed "${SEED}" \
  --script eval_only \
  --run.from_checkpoint "${CHECKPOINT}" \
  --run.eval_worker_offset 100000 \
  --run.eval_eps 100 \
  --run.envs 4 \
  --run.eval_policy_mode eval \
  --run.steps "${STEPS}" \
  --run.world_model_start_step 0 \
  --run.curve_eval_interval 0 \
  --run.eval_envs 1 \
  --jax.precompile False \
  --logdir "${EVAL_DIR}" \
  --logger.outputs jsonl wandb
