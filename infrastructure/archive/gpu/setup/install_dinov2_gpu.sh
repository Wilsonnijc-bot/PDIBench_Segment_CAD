#!/usr/bin/env bash
set -euo pipefail

PDI_GPU_ROOT="${PDI_GPU_ROOT:-/root/autodl-tmp/pdi}"
DINOV2_MODEL_ID="facebook/dinov2-base"
DINOV2_REVISION="f9e44c814b77203eaa57a6bdbbd535f21ede1415"
DINOV2_MODEL_DIR="${PDI_DINOV2_MODEL:-$PDI_GPU_ROOT/models/dinov2/base-$DINOV2_REVISION}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SAM3_ENV="$PDI_GPU_ROOT/env/sam3"

if [[ ! -x "$SAM3_ENV/bin/python" ]]; then
  echo "SAM3 environment is missing; run scripts/install_sam3_gpu.sh first" >&2
  exit 1
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$SAM3_ENV"
python -m pip install \
  "huggingface_hub>=0.30,<1.0" \
  "safetensors>=0.4,<1.0" \
  "transformers==4.41.2"

export HF_ENDPOINT
export HF_HOME="$PDI_GPU_ROOT/cache/huggingface"
python - "$DINOV2_MODEL_DIR" "$DINOV2_MODEL_ID" "$DINOV2_REVISION" <<'PY'
import sys

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[2],
    revision=sys.argv[3],
    local_dir=sys.argv[1],
    allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
)
PY

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - "$DINOV2_MODEL_DIR" <<'PY'
import sys
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(sys.argv[1], local_files_only=True, use_safetensors=True)
model.eval().cuda()
sample = torch.zeros(1, 3, 224, 224, device="cuda")
with torch.inference_mode():
    output = model(pixel_values=sample).last_hidden_state
assert output.shape[1:] == (257, 768), output.shape
print("DINOv2 ready", output.shape, torch.cuda.get_device_name(0))
PY

echo "DINOv2 model pinned at $DINOV2_REVISION"
echo "Set PDI_DINOV2_MODEL=$DINOV2_MODEL_DIR for pipeline runs."
