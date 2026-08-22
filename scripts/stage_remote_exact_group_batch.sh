#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDITED_ROOT="$PROJECT_ROOT/PDI-Bench-edited"
GPU_ENV_FILE="${PDI_GPU_ENV_FILE:-$PROJECT_ROOT/configs/gpu.local.env}"
BATCH_ID="${PDI_BATCH_ID:-exact-group-links2-7-balanced-v2}"
SESSION_NAME="${PDI_TMUX_SESSION:-pdi-links2-7-v2}"
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
  --replays-per-dataset 10 \
  --output "$LOCAL_BATCH/manifest.json"

read -r expected_count expected_bytes expected_replays < <(python -c \
  "import json; d=json.load(open('$LOCAL_BATCH/manifest.json')); print(d['video_count'], d['total_size_bytes'], d['replay_count'])")
if [[ "$expected_count" -lt 1 ]]; then
  echo "No videos were found in the active datasets" >&2
  exit 1
fi
if [[ "$expected_replays" -ne 30 ]]; then
  echo "Expected 30 replay videos, found $expected_replays" >&2
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
transfer_pids=()
for transfer in \
  "COSMOS2.5:$PROJECT_ROOT/.tmp/COSMOS2.5_Videos" \
  "COSMOS3:$PROJECT_ROOT/.tmp/COSMOS3" \
  "LVP_ROBOWM:$PROJECT_ROOT/.tmp/LVP_ROBOWM"; do
  dataset="${transfer%%:*}"
  source_dir="${transfer#*:}"
  rsync -a --delete --delete-excluded \
    --include='*/' --include='*.mp4' --exclude='*' \
    -e "$RSYNC_SSH" "$source_dir/" \
    "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_BATCH/videos/$dataset/" &
  transfer_pids+=("$!")
done
transfer_status=0
for transfer_pid in "${transfer_pids[@]}"; do
  wait "$transfer_pid" || transfer_status=1
done
if [[ "$transfer_status" -ne 0 ]]; then
  echo "One or more dataset transfers failed" >&2
  exit 1
fi
rsync -az -e "$RSYNC_SSH" "$LOCAL_BATCH/manifest.json" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_BATCH/manifest.json"

remote_count="$("${SSH[@]}" "find '$REMOTE_BATCH/videos' -type f -name '*.mp4' | wc -l")"
remote_bytes="$("${SSH[@]}" "find '$REMOTE_BATCH/videos' -type f -name '*.mp4' -printf '%s\\n' | awk '{s+=\$1} END {printf \"%.0f\\n\", s}'")"
if [[ "$remote_count" != "$expected_count" || "$remote_bytes" != "$expected_bytes" ]]; then
  echo "Remote staging verification failed: count=$remote_count bytes=$remote_bytes" >&2
  exit 1
fi
remote_non_video_count="$("${SSH[@]}" "find '$REMOTE_BATCH/videos' -type f ! -name '*.mp4' | wc -l")"
if [[ "$remote_non_video_count" != "0" ]]; then
  echo "Remote staging contains $remote_non_video_count unexpected non-video files" >&2
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
