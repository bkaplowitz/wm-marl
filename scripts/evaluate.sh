#!/usr/bin/env bash
set -euo pipefail

# Reload the training configuration; change only evaluation execution settings.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
TRAIN_DIR="${1:?Usage: scripts/evaluate.sh TRAIN_DIR}"
: "${SC2PATH:?Set SC2PATH to the StarCraft II installation}"
CHECKPOINT_ROOT="${TRAIN_DIR}/ckpt"
CHECKPOINT="${CHECKPOINT_ROOT}/$(<"${CHECKPOINT_ROOT}/latest")"
test -f "${CHECKPOINT}/done"
test -f "${TRAIN_DIR}/config.yaml"
EVAL_TAG="$(date -u +%Y%m%dT%H%M%SZ)-$$"
EVAL_DIR="${EVAL_DIR:-${TRAIN_DIR}/../final100-${EVAL_TAG}}"
if [[ -e "${EVAL_DIR}" ]]; then
  echo "Refusing to reuse evaluation directory: ${EVAL_DIR}" >&2
  exit 2
fi
export PYTHONPATH="${ROOT}/src:${ROOT}/external/dreamerv3${PYTHONPATH:+:${PYTHONPATH}}"
OUTPUTS=(jsonl)
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  OUTPUTS+=(wandb)
  export WANDB_RESUME=never
  export WANDB_RUN_ID="${EVAL_RUN_NAME:-eval-${EVAL_TAG}}"
  export WANDB_NAME="${WANDB_RUN_ID}"
  export WANDB_JOB_TYPE=final100
fi
"${PYTHON}" -m majepa.main \
  --config "${TRAIN_DIR}/config.yaml" \
  --script eval_only --run.from_checkpoint "${CHECKPOINT}" \
  --run.eval_worker_offset 100000 --run.eval_eps 100 --run.envs 4 \
  --run.eval_policy_mode eval --run.world_model_start_step 0 \
  --run.curve_eval_interval 0 --run.eval_envs 1 --jax.precompile False \
  --logdir "${EVAL_DIR}" --logger.outputs "${OUTPUTS[@]}"
