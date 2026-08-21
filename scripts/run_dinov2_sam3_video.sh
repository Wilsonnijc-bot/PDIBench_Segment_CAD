#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_ROOT="$PROJECT_ROOT/PDI-Bench-edited"
GPU_ENV_FILE="${PDI_GPU_ENV_FILE:-$PROJECT_ROOT/configs/gpu.local.env}"
VIDEO_NAME="${PDI_VIDEO_NAME:-0000.mp4}"
DEFAULT_INPUT="$PROJECT_ROOT/.tmp/COSMOS_Videos/$VIDEO_NAME"
if [[ ! -s "$DEFAULT_INPUT" && -s "$PROJECT_ROOT/.tmp/COSMOS2.5_Videos/$VIDEO_NAME" ]]; then
  DEFAULT_INPUT="$PROJECT_ROOT/.tmp/COSMOS2.5_Videos/$VIDEO_NAME"
fi
STAGED_INPUT="${PDI_INPUT_VIDEO:-$DEFAULT_INPUT}"
REFERENCE_DIR="${PDI_REFERENCE_DIR:-$PROJECT_ROOT/robot_link_first15}"

test -s "$STAGED_INPUT"
test -d "$REFERENCE_DIR"

source "$GPU_ENV_FILE"
IDENTITY_FILE="$PROJECT_ROOT/$PDI_GPU_IDENTITY_FILE"
SSH=(ssh -i "$IDENTITY_FILE" -o BatchMode=yes -o StrictHostKeyChecking=yes -p "$PDI_GPU_PORT" "$PDI_GPU_USER@$PDI_GPU_HOST")
RSYNC_SSH="ssh -i $IDENTITY_FILE -o BatchMode=yes -o StrictHostKeyChecking=yes -p $PDI_GPU_PORT"
REMOTE_CODE="$PDI_GPU_ROOT/code/PDI-Bench-edited"
RUN_ID="$(basename "${VIDEO_NAME%.mp4}")-$(shasum -a 256 "$STAGED_INPUT" | awk '{print substr($1,1,12)}')"
RUN_VARIANT="${PDI_RUN_VARIANT:-}"
if [[ -n "$RUN_VARIANT" ]]; then
  if [[ ! "$RUN_VARIANT" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "PDI_RUN_VARIANT may contain only letters, digits, dots, underscores, and hyphens" >&2
    exit 2
  fi
  RUN_ID="$RUN_ID-$RUN_VARIANT"
fi
REMOTE_WORK="$PDI_GPU_ROOT/runs/dinov2-sam3-$RUN_ID"
REMOTE_VIDEO="$REMOTE_WORK/input/$VIDEO_NAME"
REMOTE_REFERENCES="$REMOTE_WORK/references"
REMOTE_OUTPUT="$REMOTE_WORK/output"
LOCAL_OUTPUT="$PROJECT_ROOT/results/dinov2-sam3/$RUN_ID"

mkdir -p "$LOCAL_OUTPUT"
"${SSH[@]}" "mkdir -p '$REMOTE_CODE' '$REMOTE_WORK/input' '$REMOTE_REFERENCES' '$REMOTE_OUTPUT'"
if [[ "${PDI_SKIP_CODE_SYNC:-0}" != "1" ]]; then
  rsync -az --delete \
    --exclude '.git/' --exclude '*.pth' --exclude '*.pt' --exclude '*.so' \
    --exclude 'build/' --exclude 'checkpoints/' \
    --exclude 'third_party/mega_sam/base/' \
    --exclude 'third_party/mega_sam/torchhub/' \
    --exclude 'third_party/mega_sam/work_space/' \
    -e "$RSYNC_SSH" "$BENCHMARK_ROOT/" "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_CODE/"
  rsync -az -e "$RSYNC_SSH" \
    "$PROJECT_ROOT/scripts/install_sam3_gpu.sh" \
    "$PROJECT_ROOT/scripts/install_dinov2_gpu.sh" \
    "$PDI_GPU_USER@$PDI_GPU_HOST:$PDI_GPU_ROOT/code/"
fi
rsync -az -e "$RSYNC_SSH" "$STAGED_INPUT" "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_VIDEO"
rsync -az --delete -e "$RSYNC_SSH" "$REFERENCE_DIR/" "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_REFERENCES/"

if [[ "${PDI_SKIP_SAM3_INSTALL:-0}" != "1" ]]; then
  "${SSH[@]}" "PDI_GPU_ROOT='$PDI_GPU_ROOT' bash '$PDI_GPU_ROOT/code/install_sam3_gpu.sh'"
fi
if [[ "${PDI_SKIP_DINOV2_INSTALL:-0}" != "1" ]]; then
  "${SSH[@]}" "PDI_GPU_ROOT='$PDI_GPU_ROOT' bash '$PDI_GPU_ROOT/code/install_dinov2_gpu.sh'"
fi
if [[ "${PDI_PREPARE_ONLY:-0}" == "1" ]]; then
  echo "DINOv2-to-SAM3 GPU preparation complete"
  exit 0
fi

DINOV2_REVISION="f9e44c814b77203eaa57a6bdbbd535f21ede1415"
DINOV2_MODEL="${PDI_DINOV2_MODEL:-$PDI_GPU_ROOT/models/dinov2/base-$DINOV2_REVISION}"
SAM3_CHECKPOINT="${PDI_SAM3_CHECKPOINT:-$PDI_GPU_ROOT/models/sam3/sam3.pt}"
SAM3_BPE="${PDI_SAM3_BPE:-$PDI_GPU_ROOT/models/sam3/bpe_simple_vocab_16e6.txt.gz}"
SELECTED_TARGET_OPTION=""
if [[ -n "${PDI_SAM3_SELECTED_TARGET:-}" ]]; then
  if [[ ! "$PDI_SAM3_SELECTED_TARGET" =~ ^link[1-7]$ ]]; then
    echo "PDI_SAM3_SELECTED_TARGET must be link1 through link7" >&2
    exit 2
  fi
  SELECTED_TARGET_OPTION="--selected-target $PDI_SAM3_SELECTED_TARGET"
fi
set +e
"${SSH[@]}" "source /root/miniconda3/etc/profile.d/conda.sh && conda activate '$PDI_GPU_ROOT/env/sam3' && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH='$REMOTE_CODE/src' && python -m pdi_eval.perception.sam3_dinov2_segment --input '$REMOTE_VIDEO' --reference-dir '$REMOTE_REFERENCES' --output-npz '$REMOTE_OUTPUT/segmentation.npz' --dinov2-model '$DINOV2_MODEL' --sam3-checkpoint '$SAM3_CHECKPOINT' --sam3-bpe '$SAM3_BPE' --text-prompt '${PDI_SAM3_TEXT_PROMPT:-visual}' --link-text-prompt 'link4=${PDI_SAM3_LINK4_TEXT_PROMPT:-entire white oval on top of the black circle}' --link-text-prompt 'link5=${PDI_SAM3_LINK5_TEXT_PROMPT:-entire white elongated robot arm link surrounding and to the right of the black inset}' --link-text-prompt 'link7=${PDI_SAM3_LINK7_TEXT_PROMPT:-entire white quadrangular robot gripper}' --require-franka-links --reference-spatial-priors --padding-fraction '${PDI_DINOV2_BOX_PADDING:-0.10}' --minimum-tracked-fraction '${PDI_MINIMUM_TRACKED_FRACTION:-0.80}' $SELECTED_TARGET_OPTION"
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
