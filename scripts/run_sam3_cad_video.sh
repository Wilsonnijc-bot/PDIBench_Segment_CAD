#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_ROOT="$PROJECT_ROOT/PDI-Bench-edited"
GPU_ENV_FILE="${PDI_GPU_ENV_FILE:-$PROJECT_ROOT/configs/gpu.local.env}"
MANIFEST="${PDI_SAM3_CAD_MANIFEST:-$BENCHMARK_ROOT/configs/sam3-cad-franka.yaml}"
VIDEO_NAME="${PDI_VIDEO_NAME:-0000.mp4}"
TRACKING_MODE="${PDI_TRACKING_MODE:-both}"
STAGED_INPUT="$PROJECT_ROOT/.tmp/drive-franka-round1/$VIDEO_NAME"
MANIFEST_SHA="$(shasum -a 256 "$MANIFEST" | awk '{print $1}')"
MANIFEST_PREFIX="${MANIFEST_SHA:0:12}"
LOCAL_RESULT="$PROJECT_ROOT/results/sam3-cad-multi/${VIDEO_NAME%.mp4}/$MANIFEST_PREFIX"

source "$GPU_ENV_FILE"
IDENTITY_FILE="$PROJECT_ROOT/$PDI_GPU_IDENTITY_FILE"
SSH=(ssh -i "$IDENTITY_FILE" -o BatchMode=yes -o StrictHostKeyChecking=yes -p "$PDI_GPU_PORT" "$PDI_GPU_USER@$PDI_GPU_HOST")
RSYNC_SSH="ssh -i $IDENTITY_FILE -o BatchMode=yes -o StrictHostKeyChecking=yes -p $PDI_GPU_PORT"
REMOTE_CODE="$PDI_GPU_ROOT/code/PDI-Bench-edited"
REMOTE_WORK="$PDI_GPU_ROOT/runs/sam3-cad-multi-${VIDEO_NAME%.mp4}-$MANIFEST_PREFIX"
REMOTE_VIDEO="$REMOTE_WORK/input/$VIDEO_NAME"
REMOTE_SEGMENTATION="$REMOTE_WORK/intermediate/segmentation.npz"
REMOTE_OUTPUT="$REMOTE_WORK/output"

if [[ ! -s "$STAGED_INPUT" ]]; then
  mkdir -p "$(dirname "$STAGED_INPUT")"
  rclone copyto --drive-root-folder-id 1rXdVXrwIjf9wMeCVxbmFWN7cZNcfdtb9 \
    "gdrive:$VIDEO_NAME" "$STAGED_INPUT"
fi
mkdir -p "$LOCAL_RESULT"

"${SSH[@]}" "mkdir -p '$REMOTE_CODE' '$REMOTE_WORK/input' '$REMOTE_WORK/intermediate' '$REMOTE_OUTPUT'"
rsync -az --delete \
  --exclude '.git/' \
  --exclude '*.pth' \
  --exclude '*.pt' \
  --exclude '*.so' \
  --exclude 'build/' \
  --exclude 'checkpoints/' \
  --exclude 'third_party/mega_sam/work_space/' \
  --exclude 'third_party/mega_sam/outputs/' \
  --exclude 'third_party/mega_sam/outputs_cvd/' \
  -e "$RSYNC_SSH" "$BENCHMARK_ROOT/" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_CODE/"
rsync -az -e "$RSYNC_SSH" \
  "$PROJECT_ROOT/scripts/bootstrap_gpu.sh" \
  "$PROJECT_ROOT/scripts/install_sam3_gpu.sh" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$PDI_GPU_ROOT/code/"
rsync -az -e "$RSYNC_SSH" "$STAGED_INPUT" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_VIDEO"

LOCAL_VIDEO_SHA="$(shasum -a 256 "$STAGED_INPUT" | awk '{print $1}')"
REMOTE_VIDEO_SHA="$("${SSH[@]}" "sha256sum '$REMOTE_VIDEO' | cut -d' ' -f1")"
if [[ "$LOCAL_VIDEO_SHA" != "$REMOTE_VIDEO_SHA" ]]; then
  echo "Input transfer verification failed" >&2
  exit 1
fi

if [[ "${PDI_SKIP_BOOTSTRAP:-0}" != "1" ]]; then
  "${SSH[@]}" "PDI_GPU_ROOT='$PDI_GPU_ROOT' bash '$PDI_GPU_ROOT/code/bootstrap_gpu.sh'"
fi

if [[ "${PDI_SKIP_SAM3_INSTALL:-0}" != "1" ]]; then
  "${SSH[@]}" "PDI_GPU_ROOT='$PDI_GPU_ROOT' bash '$PDI_GPU_ROOT/code/install_sam3_gpu.sh'"
fi

SAM3_CHECKPOINT="${PDI_SAM3_CHECKPOINT:-$PDI_GPU_ROOT/models/sam3/sam3.pt}"
SAM3_BPE="${PDI_SAM3_BPE:-$PDI_GPU_ROOT/models/sam3/bpe_simple_vocab_16e6.txt.gz}"
"${SSH[@]}" "source /root/miniconda3/etc/profile.d/conda.sh && conda activate '$PDI_GPU_ROOT/env/sam3' && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH='$REMOTE_CODE/src' && cd '$REMOTE_CODE' && python -m pdi_eval.perception.sam3_cad_segment --project-root '$REMOTE_CODE' --manifest '$REMOTE_CODE/configs/$(basename "$MANIFEST")' --input '$REMOTE_VIDEO' --output-npz '$REMOTE_SEGMENTATION' --checkpoint '$SAM3_CHECKPOINT' --bpe-path '$SAM3_BPE'"

if [[ "${PDI_SAM3_SEGMENT_ONLY:-0}" == "1" ]]; then
  rsync -az -e "$RSYNC_SSH" \
    "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_WORK/intermediate/" "$LOCAL_RESULT/intermediate/"
  echo "SAM3 CAD segmentation complete: $LOCAL_RESULT/intermediate"
  exit 0
fi

set +e
"${SSH[@]}" "source /root/miniconda3/etc/profile.d/conda.sh && conda activate '$PDI_GPU_ROOT/env/pdi-bench' && export PYTHONPATH='$REMOTE_CODE/src' && export HF_ENDPOINT='${HF_ENDPOINT:-https://hf-mirror.com}' && cd '$REMOTE_CODE' && python evaluation/run_multi_object.py --config '$REMOTE_CODE/configs/default.yaml' --input '$REMOTE_VIDEO' --segmentation-npz '$REMOTE_SEGMENTATION' --output-dir '$REMOTE_OUTPUT' --geometry-cache-dir '$PDI_GPU_ROOT/cache/megasam-video-geometry-v3' --tracker-checkpoint '$PDI_GPU_ROOT/models/tracker/scaled_offline.pth' --tracking-mode '$TRACKING_MODE'" 2>&1 | tee "$LOCAL_RESULT/console.log"
REMOTE_STATUS=${PIPESTATUS[0]}
set -e

rsync -az -e "$RSYNC_SSH" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_OUTPUT/" "$LOCAL_RESULT/"
if [[ $REMOTE_STATUS -ne 0 ]]; then
  echo "Remote multi-object PDI run failed with status $REMOTE_STATUS; partial artifacts were retrieved" >&2
  exit "$REMOTE_STATUS"
fi
test -s "$LOCAL_RESULT/metrics.json"
test -s "$LOCAL_RESULT/timing.json"
test -s "$LOCAL_RESULT/manifest.json"
if [[ "$TRACKING_MODE" == "both" || "$TRACKING_MODE" == "joint-query" ]]; then
  test -s "$LOCAL_RESULT/replay/combined_joint-query.mp4"
fi
if [[ "$TRACKING_MODE" == "both" || "$TRACKING_MODE" == "exact-group" ]]; then
  test -s "$LOCAL_RESULT/replay/combined_exact-group.mp4"
fi

if [[ "${PDI_CLEAN_REMOTE_RUN:-0}" == "1" ]]; then
  "${SSH[@]}" "test '$REMOTE_WORK' != '$PDI_GPU_ROOT/runs' && rm -rf -- '$REMOTE_WORK'"
fi
echo "Shared multi-object PDI run complete: $LOCAL_RESULT"
