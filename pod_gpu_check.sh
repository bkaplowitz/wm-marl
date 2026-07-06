#!/usr/bin/env bash
set -uo pipefail
HOST=160.250.71.21
PORT=31509
KEY=/Users/bkaplowitz/.ssh/runpod_key
SSH="ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -p $PORT root@$HOST"
REMOTE='echo "=== GPU ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; echo "=== COMPUTE APPS ==="; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; echo "=== PYTHON PROCS ==="; ps -eo pid,etime,pcpu,pmem,cmd | grep -E "benchmark|train_e2e|[p]ython" | head; echo "=== OUTPUT DIR ==="; ls -ltr /workspace/outputs/wm_marl/benchmark-policy/*/ 2>/dev/null | tail -n 20'
$SSH "$REMOTE"
echo "SSH_EXIT=$?"
