#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDITED_ROOT="$PROJECT_ROOT/PDI-Bench-edited"
GPU_ENV_FILE="${PDI_GPU_ENV_FILE:-$PROJECT_ROOT/configs/gpu.local.env}"
BATCH_ID="${PDI_BATCH_ID:-exact-group-links2-7-all-v1}"
SESSION_NAME="${PDI_TMUX_SESSION:-pdi-links2-7-all}"
WORKERS="${PDI_BATCH_WORKERS:-2}"
LOCAL_BATCH="$PROJECT_ROOT/results/remote-exact-group-batch/$BATCH_ID"

source "$GPU_ENV_FILE"
IDENTITY_FILE="$PROJECT_ROOT/$PDI_GPU_IDENTITY_FILE"
SSH=(ssh -i "$IDENTITY_FILE" -o BatchMode=yes -o StrictHostKeyChecking=yes -p "$PDI_GPU_PORT" "$PDI_GPU_USER@$PDI_GPU_HOST")
RSYNC_SSH="ssh -i $IDENTITY_FILE -o BatchMode=yes -o StrictHostKeyChecking=yes -p $PDI_GPU_PORT"
REMOTE_CODE="$PDI_GPU_ROOT/code/PDI-Bench-edited"
REMOTE_BATCH="$PDI_GPU_ROOT/batches/$BATCH_ID"

mkdir -p "$LOCAL_BATCH"
python "$EDITED_ROOT/evaluation/build_video_batch_manifest.py" \
  --dataset "COSMOS2.5=$PROJECT_ROOT/.tmp/COSMOS2.5_Videos" \
  --dataset "COSMOS3=$PROJECT_ROOT/.tmp/COSMOS3" \
  --dataset "LVP_ROBOWM=$PROJECT_ROOT/.tmp/LVP_ROBOWM" \
  --output "$LOCAL_BATCH/manifest.json"

read -r expected_count expected_bytes < <(python -c \
  "import json; d=json.load(open('$LOCAL_BATCH/manifest.json')); print(d['video_count'], d['total_size_bytes'])")
if [[ "$expected_count" -ne 405 ]]; then
  echo "Expected 405 videos, found $expected_count" >&2
  exit 1
fi

"${SSH[@]}" "mkdir -p '$REMOTE_CODE' '$REMOTE_BATCH/videos/COSMOS2.5' '$REMOTE_BATCH/videos/COSMOS3' '$REMOTE_BATCH/videos/LVP_ROBOWM' '$REMOTE_BATCH/references'"
rsync -az --delete \
  --exclude '.git/' --exclude '*.pth' --exclude '*.pt' --exclude '*.so' \
  --exclude 'build/' --exclude 'checkpoints/' \
  --exclude 'third_party/mega_sam/base/' \
  --exclude 'third_party/mega_sam/work_space/' \
  --exclude 'third_party/mega_sam/outputs/' \
  --exclude 'third_party/mega_sam/outputs_cvd/' \
  --exclude 'third_party/mega_sam/torchhub/' \
  -e "$RSYNC_SSH" "$EDITED_ROOT/" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_CODE/"
rsync -az -e "$RSYNC_SSH" "$PROJECT_ROOT/scripts/run_remote_exact_group_batch.sh" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_CODE/scripts/run_remote_exact_group_batch.sh"
rsync -az --delete -e "$RSYNC_SSH" "$PROJECT_ROOT/robot_link_first15/" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_BATCH/references/"

echo "Staging $expected_count videos ($expected_bytes bytes) to the GPU..."
rsync -az --delete -e "$RSYNC_SSH" "$PROJECT_ROOT/.tmp/COSMOS2.5_Videos/" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_BATCH/videos/COSMOS2.5/"
rsync -az --delete -e "$RSYNC_SSH" "$PROJECT_ROOT/.tmp/COSMOS3/" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_BATCH/videos/COSMOS3/"
rsync -az --delete -e "$RSYNC_SSH" "$PROJECT_ROOT/.tmp/LVP_ROBOWM/" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_BATCH/videos/LVP_ROBOWM/"
rsync -az -e "$RSYNC_SSH" "$LOCAL_BATCH/manifest.json" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_BATCH/manifest.json"

remote_count="$("${SSH[@]}" "find '$REMOTE_BATCH/videos' -type f -name '*.mp4' | wc -l")"
remote_bytes="$("${SSH[@]}" "find '$REMOTE_BATCH/videos' -type f -name '*.mp4' -printf '%s\\n' | awk '{s+=\$1} END {print s+0}'")"
if [[ "$remote_count" -ne "$expected_count" || "$remote_bytes" -ne "$expected_bytes" ]]; then
  echo "Remote staging verification failed: count=$remote_count bytes=$remote_bytes" >&2
  exit 1
fi

"${SSH[@]}" "test -x '$PDI_GPU_ROOT/env/pdi-bench/bin/python' && test -x '$PDI_GPU_ROOT/env/sam3/bin/python' && test -s '$PDI_GPU_ROOT/models/tracker/scaled_offline.pth' && test -s '$PDI_GPU_ROOT/models/sam3/sam3.pt'"
if ! "${SSH[@]}" "command -v tmux >/dev/null"; then
  "${SSH[@]}" "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tmux"
fi

if "${SSH[@]}" "tmux has-session -t '$SESSION_NAME' 2>/dev/null"; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi
"${SSH[@]}" "tmux new-session -d -s '$SESSION_NAME' \"PDI_GPU_ROOT='$PDI_GPU_ROOT' bash '$REMOTE_CODE/scripts/run_remote_exact_group_batch.sh' '$REMOTE_BATCH' '$WORKERS'\""
sleep 2
"${SSH[@]}" "tmux has-session -t '$SESSION_NAME' && pgrep -af run_remote_exact_group_batch.py && tail -n 8 '$REMOTE_BATCH/batch.log'"

echo "Remote batch detached successfully"
echo "tmux session: $SESSION_NAME"
echo "remote batch: $REMOTE_BATCH"
echo "remote CSV: $REMOTE_BATCH/metrics.csv"
