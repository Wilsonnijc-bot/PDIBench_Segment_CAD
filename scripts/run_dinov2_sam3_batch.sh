#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SINGLE_RUNNER="$PROJECT_ROOT/scripts/run_dinov2_sam3_video.sh"
RUN_VARIANT="${PDI_RUN_VARIANT:-refined-text}"
LOG_ROOT="${PDI_BATCH_LOG_DIR:-$PROJECT_ROOT/results/dinov2-sam3/batch-logs/$RUN_VARIANT}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 VIDEO_NAME [VIDEO_NAME ...]" >&2
  exit 2
fi

videos=("$@")
mkdir -p "$LOG_ROOT"

# Synchronize shared code and verify/install models once before concurrent jobs.
PDI_VIDEO_NAME="${videos[0]}" \
PDI_RUN_VARIANT="$RUN_VARIANT" \
PDI_PREPARE_ONLY=1 \
bash "$SINGLE_RUNNER"

run_video() {
  local video_name="$1"
  local video_stem="${video_name%.mp4}"
  local log_path="$LOG_ROOT/$video_stem.log"
  echo "Starting $video_name; log: $log_path"
  PDI_VIDEO_NAME="$video_name" \
  PDI_RUN_VARIANT="$RUN_VARIANT" \
  PDI_SKIP_CODE_SYNC=1 \
  PDI_SKIP_SAM3_INSTALL=1 \
  PDI_SKIP_DINOV2_INSTALL=1 \
  bash "$SINGLE_RUNNER" >"$log_path" 2>&1
}

failed=()
for ((offset = 0; offset < ${#videos[@]}; offset += 2)); do
  pair=("${videos[@]:offset:2}")
  pids=()
  for video_name in "${pair[@]}"; do
    run_video "$video_name" &
    pids+=("$!")
  done
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      failed+=("${pair[$index]}")
    else
      echo "Completed ${pair[$index]}"
    fi
  done
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed videos: ${failed[*]}" >&2
  exit 1
fi
echo "All ${#videos[@]} DINOv2-to-SAM3 runs completed"
