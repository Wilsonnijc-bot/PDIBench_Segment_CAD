#!/usr/bin/env bash
set -uo pipefail

PDI_GPU_ROOT="${PDI_GPU_ROOT:-/root/autodl-tmp/pdi}"
BATCH_ROOT="${1:?batch root is required}"
WORKERS="${2:-2}"
CODE_ROOT="$PDI_GPU_ROOT/code/PDI-Bench-edited"

mkdir -p "$BATCH_ROOT"
exec >>"$BATCH_ROOT/batch.log" 2>&1
echo "[$(date --iso-8601=seconds)] detached batch launcher started"
PYTHONUNBUFFERED=1 "$PDI_GPU_ROOT/env/pdi-bench/bin/python" \
  "$CODE_ROOT/evaluation/run_remote_exact_group_batch.py" \
  --gpu-root "$PDI_GPU_ROOT" \
  --code-root "$CODE_ROOT" \
  --batch-root "$BATCH_ROOT" \
  --manifest "$BATCH_ROOT/manifest.json" \
  --workers "$WORKERS"
status=$?
printf '%s\n' "$status" >"$BATCH_ROOT/exit_code"
echo "[$(date --iso-8601=seconds)] detached batch launcher exited status=$status"
exit "$status"
