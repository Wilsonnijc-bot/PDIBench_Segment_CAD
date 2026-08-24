#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ENV_FILE="${PDI_GPU_ENV_FILE:-$PROJECT_ROOT/configs/gpu.local.env}"
BATCH_ID="${PDI_BATCH_ID:-exact-group-links2-7-balanced-v2}"
LOCAL_BATCH="$PROJECT_ROOT/results/remote-exact-group-batch/$BATCH_ID"
LOCAL_OUTPUT_DIR="${PDI_LOCAL_OUTPUT_DIR:-$PROJECT_ROOT/outputs}"
LOCAL_REPLAY_DIR="${PDI_LOCAL_REPLAY_DIR:-$PROJECT_ROOT/replays}"

source "$GPU_ENV_FILE"
IDENTITY_FILE="$PROJECT_ROOT/$PDI_GPU_IDENTITY_FILE"
RSYNC_SSH="ssh -i $IDENTITY_FILE -o BatchMode=yes -o StrictHostKeyChecking=yes -p $PDI_GPU_PORT"
REMOTE_BATCH="$PDI_GPU_ROOT/batches/$BATCH_ID"

mkdir -p "$LOCAL_BATCH" "$LOCAL_OUTPUT_DIR" "$LOCAL_REPLAY_DIR"
rsync -az -e "$RSYNC_SSH" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_BATCH/metrics.csv" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_BATCH/manifest.json" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_BATCH/batch.log" \
  "$LOCAL_BATCH/"

rsync -az --prune-empty-dirs \
  --exclude='*/output/replay/combined_exact-group.mp4' \
  --include='*/' --include='*/output/***' --exclude='*' \
  -e "$RSYNC_SSH" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_BATCH/jobs/" \
  "$LOCAL_OUTPUT_DIR/"

fetched_replays=0
while IFS=$'\t' read -r job_id dataset relative_path; do
  video_number="$(basename "$relative_path" .mp4)"
  destination="$LOCAL_REPLAY_DIR/${dataset}_${video_number}.mp4"
  if [[ -s "$destination" ]]; then
    continue
  fi
  temporary="$destination.partial"
  rsync -az -e "$RSYNC_SSH" \
    "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_BATCH/jobs/$job_id/output/replay/combined_exact-group.mp4" \
    "$temporary"
  mv "$temporary" "$destination"
  fetched_replays=$((fetched_replays + 1))
done < <(python - "$LOCAL_BATCH/manifest.json" "$LOCAL_BATCH/metrics.csv" <<'PY'
import csv
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
with open(sys.argv[2], encoding="utf-8", newline="") as stream:
    complete = {
        (row["dataset"], row["relative_path"])
        for row in csv.DictReader(stream)
        if row["replay_status"] == "complete"
    }
for item in manifest["videos"]:
    key = (item["dataset"], item["relative_path"])
    if key in complete:
        print(item["job_id"], item["dataset"], item["relative_path"], sep="\t")
PY
)

echo "Fetched current outputs into: $LOCAL_OUTPUT_DIR"
echo "Fetched $fetched_replays new replay(s) into: $LOCAL_REPLAY_DIR"
