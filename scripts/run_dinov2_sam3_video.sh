#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_ROOT="$PROJECT_ROOT/PDI-Bench-edited"
GPU_ENV_FILE="${PDI_GPU_ENV_FILE:-$PROJECT_ROOT/configs/gpu.local.env}"
VIDEO_NAME="${PDI_VIDEO_NAME:-0000.mp4}"
STAGED_INPUT="${PDI_INPUT_VIDEO:-$PROJECT_ROOT/.tmp/COSMOS_Videos/$VIDEO_NAME}"
REFERENCE_DIR="${PDI_REFERENCE_DIR:-$PROJECT_ROOT/robot_link_first15}"

test -s "$STAGED_INPUT"
test -d "$REFERENCE_DIR"

source "$GPU_ENV_FILE"
IDENTITY_FILE="$PROJECT_ROOT/$PDI_GPU_IDENTITY_FILE"
SSH=(ssh -i "$IDENTITY_FILE" -o BatchMode=yes -o StrictHostKeyChecking=yes -p "$PDI_GPU_PORT" "$PDI_GPU_USER@$PDI_GPU_HOST")
RSYNC_SSH="ssh -i $IDENTITY_FILE -o BatchMode=yes -o StrictHostKeyChecking=yes -p $PDI_GPU_PORT"
REMOTE_CODE="$PDI_GPU_ROOT/code/PDI-Bench-edited"
RUN_ID="$(basename "${VIDEO_NAME%.mp4}")-$(shasum -a 256 "$STAGED_INPUT" | awk '{print substr($1,1,12)}')"
REMOTE_WORK="$PDI_GPU_ROOT/runs/dinov2-sam3-$RUN_ID"
REMOTE_VIDEO="$REMOTE_WORK/input/$VIDEO_NAME"
REMOTE_REFERENCES="$REMOTE_WORK/references"
REMOTE_OUTPUT="$REMOTE_WORK/output"
LOCAL_OUTPUT="$PROJECT_ROOT/results/dinov2-sam3/$RUN_ID"

mkdir -p "$LOCAL_OUTPUT"
"${SSH[@]}" "mkdir -p '$REMOTE_CODE' '$REMOTE_WORK/input' '$REMOTE_REFERENCES' '$REMOTE_OUTPUT'"
rsync -az --delete \
  --exclude '.git/' --exclude '*.pth' --exclude '*.pt' --exclude '*.so' \
  --exclude 'build/' --exclude 'checkpoints/' --exclude 'third_party/mega_sam/work_space/' \
  -e "$RSYNC_SSH" "$BENCHMARK_ROOT/" "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_CODE/"
rsync -az -e "$RSYNC_SSH" \
  "$PROJECT_ROOT/scripts/install_sam3_gpu.sh" \
  "$PROJECT_ROOT/scripts/install_dinov2_gpu.sh" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$PDI_GPU_ROOT/code/"
rsync -az -e "$RSYNC_SSH" "$STAGED_INPUT" "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_VIDEO"
rsync -az --delete -e "$RSYNC_SSH" "$REFERENCE_DIR/" "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_REFERENCES/"

if [[ "${PDI_SKIP_SAM3_INSTALL:-0}" != "1" ]]; then
  "${SSH[@]}" "PDI_GPU_ROOT='$PDI_GPU_ROOT' bash '$PDI_GPU_ROOT/code/install_sam3_gpu.sh'"
fi
if [[ "${PDI_SKIP_DINOV2_INSTALL:-0}" != "1" ]]; then
  "${SSH[@]}" "PDI_GPU_ROOT='$PDI_GPU_ROOT' bash '$PDI_GPU_ROOT/code/install_dinov2_gpu.sh'"
fi

DINOV2_REVISION="f9e44c814b77203eaa57a6bdbbd535f21ede1415"
DINOV2_MODEL="${PDI_DINOV2_MODEL:-$PDI_GPU_ROOT/models/dinov2/base-$DINOV2_REVISION}"
SAM3_CHECKPOINT="${PDI_SAM3_CHECKPOINT:-$PDI_GPU_ROOT/models/sam3/sam3.pt}"
SAM3_BPE="${PDI_SAM3_BPE:-$PDI_GPU_ROOT/models/sam3/bpe_simple_vocab_16e6.txt.gz}"
set +e
"${SSH[@]}" "source /root/miniconda3/etc/profile.d/conda.sh && conda activate '$PDI_GPU_ROOT/env/sam3' && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH='$REMOTE_CODE/src' && python -m pdi_eval.perception.sam3_dinov2_segment --input '$REMOTE_VIDEO' --reference-dir '$REMOTE_REFERENCES' --output-npz '$REMOTE_OUTPUT/segmentation.npz' --dinov2-model '$DINOV2_MODEL' --sam3-checkpoint '$SAM3_CHECKPOINT' --sam3-bpe '$SAM3_BPE' --text-prompt '${PDI_SAM3_TEXT_PROMPT:-white robotic arm link}' --require-franka-links --reference-spatial-priors --padding-fraction '${PDI_DINOV2_BOX_PADDING:-0.10}'"
REMOTE_STATUS=$?
set -e

rsync -az -e "$RSYNC_SSH" "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_OUTPUT/" "$LOCAL_OUTPUT/"
if [[ $REMOTE_STATUS -ne 0 ]]; then
  echo "Remote DINOv2-to-SAM3 run failed with status $REMOTE_STATUS; partial artifacts were retrieved" >&2
  exit "$REMOTE_STATUS"
fi
test -s "$LOCAL_OUTPUT/dinov2_boxes.json"
test -s "$LOCAL_OUTPUT/dinov2_boxes.jpg"
test -s "$LOCAL_OUTPUT/first_frame_mask.png"
test -s "$LOCAL_OUTPUT/sam3_prompt_diagnostics.json"
test -s "$LOCAL_OUTPUT/segmentation.npz"
test -s "$LOCAL_OUTPUT/segmentation.json"
echo "DINOv2-to-SAM3 run complete: $LOCAL_OUTPUT"
