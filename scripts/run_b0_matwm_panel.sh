#!/usr/bin/env bash

set -uo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU SEED" >&2
  exit 2
fi

GPU="$1"
SEED="$2"
CODE_ROOT="/root/dreamarl_replay_5bdaf16"
INFRA_ROOT="/root/dreamarl_replay_5bdaf16/external/dreamerv3"
PYTHON="/workspace/dreamarl/.venv/bin/python"
OUTPUT_ROOT="/workspace/dreamarl_b0_matwm_panel_20260821"
QUEUE_LOG="${OUTPUT_ROOT}/queue_gpu${GPU}_seed${SEED}.log"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${CODE_ROOT}/src:${INFRA_ROOT}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=3

mkdir -p "${OUTPUT_ROOT}"

TASKS=(
  "coop_mining:6"
  "gift_refinements:6"
  "chicken_in_the_matrix__arena:8"
  "stag_hunt_in_the_matrix__arena:8"
  "pure_coordination_in_the_matrix__repeated:2"
  "rationalizable_coordination_in_the_matrix__repeated:2"
  "externality_mushrooms__dense:5"
)

printf 'B0 MATWM panel started: gpu=%s seed=%s time=%s\n' \
  "${GPU}" "${SEED}" "$(date --iso-8601=seconds)" >>"${QUEUE_LOG}"

for entry in "${TASKS[@]}"; do
  substrate="${entry%%:*}"
  agents="${entry##*:}"
  task="meltingpot_${substrate}"
  experiment="${OUTPUT_ROOT}/${substrate}/seed_${SEED}"
  mkdir -p "${experiment}"

  printf 'START task=%s agents=%s seed=%s time=%s\n' \
    "${task}" "${agents}" "${SEED}" "$(date --iso-8601=seconds)" \
    >>"${QUEUE_LOG}"

  if WANDB_NAME="b0-recent-${substrate}-seed${SEED}" \
    WANDB_RUN_GROUP="b0-matwm-panel-20260821" \
    "${PYTHON}" -m dreamarl.scripts.train_dreamarl \
    --task "${task}" \
    --num-agents "${agents}" \
    --seed "${SEED}" \
    --marl-stage b0 \
    --replay-sampling recent \
    --behavior-optimizer joint \
    --train-ratio 256 \
    --total-env-steps 50000 \
    --experiment-dir "${experiment}" \
    --platform cuda \
    --python "${PYTHON}" \
    --infrastructure-root "${INFRA_ROOT}" \
    --save-every-seconds 86400 \
    --curve-eval-interval 10000 \
    --curve-eval-episodes 20 \
    --curve-eval-seed-offset 10000 \
    --curve-eval-policy-mode deterministic \
    --wandb-project dreamarl \
    --wandb-entity osaze-obahor \
    >"${experiment}/driver.log" 2>&1; then
    status="DONE"
  else
    status="FAILED"
  fi

  printf '%s task=%s agents=%s seed=%s time=%s\n' \
    "${status}" "${task}" "${agents}" "${SEED}" \
    "$(date --iso-8601=seconds)" >>"${QUEUE_LOG}"
done

printf 'B0 MATWM panel finished: gpu=%s seed=%s time=%s\n' \
  "${GPU}" "${SEED}" "$(date --iso-8601=seconds)" >>"${QUEUE_LOG}"
