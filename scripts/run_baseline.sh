#!/usr/bin/env bash
set -euo pipefail

# Train once, then evaluate the saved checkpoint on 100 held-out episodes.
# CLI overrides are forwarded to training and saved for the final evaluation.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
TASK="${TASK:-smac_2s3z}"
NUM_AGENTS="${NUM_AGENTS:-5}"
SEED="${SEED:-0}"
STEPS="${STEPS:-50000}"
CONFIG="${CONFIG:-baseline}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_NAME="${RUN_NAME:-${CONFIG}-${TASK#smac_}-s${SEED}-${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/runs/${RUN_NAME}}"
TRAIN_DIR="${LOG_ROOT}/train"

: "${SC2PATH:?Set SC2PATH to the StarCraft II installation}"
export PYTHONPATH="${ROOT}/src:${ROOT}/external/dreamerv3${PYTHONPATH:+:${PYTHONPATH}}"
OUTPUTS=(jsonl)
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  OUTPUTS+=(wandb)
  export WANDB_RESUME=never
  export WANDB_RUN_ID="${RUN_NAME}-train"
  export WANDB_NAME="${WANDB_RUN_ID}"
  export WANDB_JOB_TYPE=train
fi
if [[ -e "${LOG_ROOT}" ]]; then
  echo "Refusing to reuse existing run directory: ${LOG_ROOT}" >&2
  exit 2
fi
mkdir -p "${LOG_ROOT}"

"${PYTHON}" -m majepa.main \
  --configs "${CONFIG}" \
  --task "${TASK}" --agent.num_agents "${NUM_AGENTS}" \
  --seed "${SEED}" --run.steps "${STEPS}" \
  --logdir "${TRAIN_DIR}" --logger.outputs "${OUTPUTS[@]}" "$@"

EVAL_DIR="${LOG_ROOT}/final100" EVAL_RUN_NAME="${RUN_NAME}-final100" \
  PYTHON="${PYTHON}" bash "${ROOT}/scripts/evaluate.sh" "${TRAIN_DIR}"
