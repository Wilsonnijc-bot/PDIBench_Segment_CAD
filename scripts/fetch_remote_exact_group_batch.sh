#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ENV_FILE="${PDI_GPU_ENV_FILE:-$PROJECT_ROOT/configs/gpu.local.env}"
BATCH_ID="${PDI_BATCH_ID:-exact-group-links2-7-all-v1}"
LOCAL_BATCH="$PROJECT_ROOT/results/remote-exact-group-batch/$BATCH_ID"

source "$GPU_ENV_FILE"
IDENTITY_FILE="$PROJECT_ROOT/$PDI_GPU_IDENTITY_FILE"
RSYNC_SSH="ssh -i $IDENTITY_FILE -o BatchMode=yes -o StrictHostKeyChecking=yes -p $PDI_GPU_PORT"
REMOTE_BATCH="$PDI_GPU_ROOT/batches/$BATCH_ID"

mkdir -p "$LOCAL_BATCH"
rsync -az -e "$RSYNC_SSH" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_BATCH/metrics.csv" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_BATCH/manifest.json" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_BATCH/batch.log" \
  "$LOCAL_BATCH/"
echo "Fetched current batch CSV: $LOCAL_BATCH/metrics.csv"
