#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ENV_FILE="${PDI_GPU_ENV_FILE:-$PROJECT_ROOT/configs/gpu.local.env}"

source "$GPU_ENV_FILE"
IDENTITY_FILE="$PROJECT_ROOT/$PDI_GPU_IDENTITY_FILE"
test -f "$IDENTITY_FILE"
SSH=(ssh -i "$IDENTITY_FILE" -o BatchMode=yes -o StrictHostKeyChecking=yes -p "$PDI_GPU_PORT" "$PDI_GPU_USER@$PDI_GPU_HOST")
RSYNC_SSH="ssh -i $IDENTITY_FILE -o BatchMode=yes -o StrictHostKeyChecking=yes -p $PDI_GPU_PORT"

"${SSH[@]}" "mkdir -p '$PDI_GPU_ROOT/code'"
rsync -az -e "$RSYNC_SSH" "$PROJECT_ROOT/scripts/install_dinov2_gpu.sh" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$PDI_GPU_ROOT/code/"
"${SSH[@]}" "PDI_GPU_ROOT='$PDI_GPU_ROOT' HF_ENDPOINT='${HF_ENDPOINT:-https://hf-mirror.com}' bash '$PDI_GPU_ROOT/code/install_dinov2_gpu.sh'"
