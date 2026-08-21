#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDITED_ROOT="$PROJECT_ROOT/PDI-Bench-edited"
ORIGINAL_ROOT="$PROJECT_ROOT/PDI-Bench-original"
GPU_ENV_FILE="${PDI_GPU_ENV_FILE:-$PROJECT_ROOT/configs/gpu.local.env}"
VIDEO_NAME="${1:-${PDI_VIDEO_NAME:-0000.mp4}}"
VIDEO_STEM="${VIDEO_NAME%.mp4}"
DEFAULT_VIDEO_PATH="$PROJECT_ROOT/.tmp/COSMOS_Videos/$VIDEO_NAME"
if [[ ! -s "$DEFAULT_VIDEO_PATH" && -s "$PROJECT_ROOT/.tmp/COSMOS2.5_Videos/$VIDEO_NAME" ]]; then
  DEFAULT_VIDEO_PATH="$PROJECT_ROOT/.tmp/COSMOS2.5_Videos/$VIDEO_NAME"
fi
VIDEO_PATH="${PDI_INPUT_VIDEO:-$DEFAULT_VIDEO_PATH}"

if [[ -n "${PDI_SEGMENTATION_DIR:-}" ]]; then
  SEGMENTATION_DIR="$PDI_SEGMENTATION_DIR"
else
  matches=("$PROJECT_ROOT"/results/dinov2-sam3/"$VIDEO_STEM"-*-refined-text)
  if [[ ${#matches[@]} -ne 1 || ! -d "${matches[0]}" ]]; then
    echo "Set PDI_SEGMENTATION_DIR to one completed DINOv2-SAM3 result" >&2
    exit 2
  fi
  SEGMENTATION_DIR="${matches[0]}"
fi

SEGMENTATION="$SEGMENTATION_DIR/segmentation.npz"
ORIGINAL_SEGMENTATION_DIR="${PDI_ORIGINAL_SEGMENTATION_DIR:-$SEGMENTATION_DIR}"
ORIGINAL_SEGMENTATION="$ORIGINAL_SEGMENTATION_DIR/segmentation.npz"
test -s "$VIDEO_PATH"
test -s "$SEGMENTATION"
test -s "$ORIGINAL_SEGMENTATION"

source "$GPU_ENV_FILE"
IDENTITY_FILE="$PROJECT_ROOT/$PDI_GPU_IDENTITY_FILE"
SSH=(ssh -i "$IDENTITY_FILE" -o BatchMode=yes -o StrictHostKeyChecking=yes -p "$PDI_GPU_PORT" "$PDI_GPU_USER@$PDI_GPU_HOST")
RSYNC_SSH="ssh -i $IDENTITY_FILE -o BatchMode=yes -o StrictHostKeyChecking=yes -p $PDI_GPU_PORT"
REMOTE_EDITED="$PDI_GPU_ROOT/code/PDI-Bench-edited"
REMOTE_ORIGINAL="$PDI_GPU_ROOT/code/PDI-Bench-original"
SEGMENTATION_SHA="$(shasum -a 256 "$SEGMENTATION" | awk '{print $1}')"
ORIGINAL_SEGMENTATION_SHA="$(shasum -a 256 "$ORIGINAL_SEGMENTATION" | awk '{print $1}')"
RUN_ID="$VIDEO_STEM-${SEGMENTATION_SHA:0:12}-${ORIGINAL_SEGMENTATION_SHA:0:12}-link7"
REMOTE_WORK="$PDI_GPU_ROOT/runs/pdi-link7-comparison-$RUN_ID"
REMOTE_VIDEO="$REMOTE_WORK/input/$VIDEO_NAME"
REMOTE_SEGMENTATION="$REMOTE_WORK/input/segmentation.npz"
REMOTE_ORIGINAL_SEGMENTATION="$REMOTE_WORK/input/original_gripper_segmentation.npz"
REMOTE_FROZEN="$REMOTE_WORK/frozen"
REMOTE_EDITED_OUTPUT="$REMOTE_WORK/edited"
REMOTE_ORIGINAL_CACHE="$REMOTE_WORK/original-cache"
REMOTE_ORIGINAL_OUTPUT="$REMOTE_WORK/original"
REMOTE_COMPARISON="$REMOTE_WORK/comparison.json"
LOCAL_OUTPUT="$PROJECT_ROOT/results/pdi-link7-comparison/$RUN_ID"

mkdir -p "$LOCAL_OUTPUT"
"${SSH[@]}" "mkdir -p '$REMOTE_EDITED' '$REMOTE_ORIGINAL' '$REMOTE_WORK/input' '$REMOTE_FROZEN' '$REMOTE_EDITED_OUTPUT' '$REMOTE_ORIGINAL_CACHE' '$REMOTE_ORIGINAL_OUTPUT'"
rsync -az --delete \
  --exclude '.git/' --exclude '*.pth' --exclude '*.pt' --exclude '*.so' \
  --exclude 'build/' --exclude 'checkpoints/' \
  --exclude 'third_party/mega_sam/base/' \
  --exclude 'third_party/mega_sam/work_space/' \
  --exclude 'third_party/mega_sam/outputs/' \
  --exclude 'third_party/mega_sam/outputs_cvd/' \
  --exclude 'third_party/mega_sam/torchhub/' \
  -e "$RSYNC_SSH" "$EDITED_ROOT/" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_EDITED/"
rsync -az --delete \
  --exclude '.git/' --exclude '*.pth' --exclude '*.pt' --exclude '*.so' \
  --exclude 'build/' --exclude 'checkpoints/' \
  --exclude 'third_party/mega_sam/work_space/' \
  --exclude 'third_party/mega_sam/outputs/' \
  --exclude 'third_party/mega_sam/outputs_cvd/' \
  -e "$RSYNC_SSH" "$ORIGINAL_ROOT/" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_ORIGINAL/"
rsync -az -e "$RSYNC_SSH" "$VIDEO_PATH" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_VIDEO"
rsync -az -e "$RSYNC_SSH" "$SEGMENTATION" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_SEGMENTATION"
rsync -az -e "$RSYNC_SSH" "$ORIGINAL_SEGMENTATION" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_ORIGINAL_SEGMENTATION"

LOCAL_VIDEO_SHA="$(shasum -a 256 "$VIDEO_PATH" | awk '{print $1}')"
REMOTE_VIDEO_SHA="$("${SSH[@]}" "sha256sum '$REMOTE_VIDEO' | cut -d' ' -f1")"
if [[ "$LOCAL_VIDEO_SHA" != "$REMOTE_VIDEO_SHA" ]]; then
  echo "Input transfer verification failed" >&2
  exit 1
fi

if [[ "${PDI_SKIP_BOOTSTRAP:-0}" != "1" ]]; then
  rsync -az -e "$RSYNC_SSH" "$PROJECT_ROOT/scripts/bootstrap_gpu.sh" \
    "$PDI_GPU_USER@$PDI_GPU_HOST:$PDI_GPU_ROOT/code/bootstrap_gpu.sh"
  "${SSH[@]}" "PDI_GPU_ROOT='$PDI_GPU_ROOT' bash '$PDI_GPU_ROOT/code/bootstrap_gpu.sh'"
fi

set +e
"${SSH[@]}" "source /root/miniconda3/etc/profile.d/conda.sh && conda activate '$PDI_GPU_ROOT/env/pdi-bench' && export PYTHONPATH='$REMOTE_EDITED/src' HF_ENDPOINT='${HF_ENDPOINT:-https://hf-mirror.com}' && cd '$REMOTE_EDITED' && python evaluation/prepare_abcd.py --video '$REMOTE_VIDEO' --segmentation '$REMOTE_SEGMENTATION' --output-dir '$REMOTE_FROZEN/edited' --selected-object link7 && python evaluation/run_multi_object.py --config '$REMOTE_EDITED/configs/default.yaml' --input '$REMOTE_VIDEO' --segmentation-npz '$REMOTE_FROZEN/edited/single_link_segmentation.npz' --output-dir '$REMOTE_EDITED_OUTPUT' --geometry-cache-dir '$PDI_GPU_ROOT/cache/megasam-video-geometry-v3' --tracker-checkpoint '$PDI_GPU_ROOT/models/tracker/scaled_offline.pth' --tracking-mode both && GEOMETRY_PATH=\$(python -c \"import json; print(json.load(open('$REMOTE_EDITED_OUTPUT/metrics.json'))['geometry']['cache_path'])\") && python evaluation/prepare_abcd.py --video '$REMOTE_VIDEO' --segmentation '$REMOTE_ORIGINAL_SEGMENTATION' --output-dir '$REMOTE_FROZEN/original-gripper' --selected-object link7 --geometry \"\$GEOMETRY_PATH\" --original-cache-dir '$REMOTE_ORIGINAL_CACHE' && python evaluation/run_original_frozen.py --original-root '$REMOTE_ORIGINAL' --video '$REMOTE_VIDEO' --cache-dir '$REMOTE_ORIGINAL_CACHE' --tracker-checkpoint '$PDI_GPU_ROOT/models/tracker/scaled_offline.pth' --output-dir '$REMOTE_ORIGINAL_OUTPUT' --text-query gripper && python evaluation/compare_single_link_baseline.py --edited-run '$REMOTE_EDITED_OUTPUT' --original-run '$REMOTE_ORIGINAL_OUTPUT' --object-name link7 --original-prompt gripper --output '$REMOTE_COMPARISON' --csv-output '$REMOTE_WORK/metrics.csv'" 2>&1 | tee "$LOCAL_OUTPUT/console.log"
REMOTE_STATUS=${PIPESTATUS[0]}
set -e

rsync -az -e "$RSYNC_SSH" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_WORK/" "$LOCAL_OUTPUT/"
if [[ $REMOTE_STATUS -ne 0 ]]; then
  echo "Remote link7 comparison failed with status $REMOTE_STATUS; partial artifacts were retrieved" >&2
  exit "$REMOTE_STATUS"
fi
test -s "$LOCAL_OUTPUT/edited/metrics.json"
test -s "$LOCAL_OUTPUT/original/metrics.json"
test -s "$LOCAL_OUTPUT/comparison.json"
test -s "$LOCAL_OUTPUT/metrics.csv"
echo "Link7 PDI comparison complete: $LOCAL_OUTPUT"
