#!/usr/bin/env bash
set -euo pipefail

PDI_GPU_ROOT="${PDI_GPU_ROOT:-/root/autodl-tmp/pdi}"
SAM3_ENV="$PDI_GPU_ROOT/env/sam3"
SAM3_VERSION="0.1.4"
SAM3_MODELSCOPE_REVISION="96f3e1b404ba14f2cfac60ee6ae87c269a7b7923"
SAM3_CHECKPOINT="${PDI_SAM3_CHECKPOINT:-$PDI_GPU_ROOT/models/sam3/sam3.pt}"
SAM3_CHECKPOINT_SHA256="9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
SAM3_CHECKPOINT_URL="https://www.modelscope.cn/models/facebook/sam3/resolve/$SAM3_MODELSCOPE_REVISION/sam3.pt"
SAM3_MERGES="$PDI_GPU_ROOT/models/sam3/merges.txt"
SAM3_MERGES_SHA256="9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"
SAM3_MERGES_URL="https://www.modelscope.cn/models/facebook/sam3/resolve/$SAM3_MODELSCOPE_REVISION/merges.txt"
SAM3_BPE="$PDI_GPU_ROOT/models/sam3/bpe_simple_vocab_16e6.txt.gz"
ALIYUN_PYPI="http://mirrors.aliyun.com/pypi/simple"

source /root/miniconda3/etc/profile.d/conda.sh
if [[ ! -x "$SAM3_ENV/bin/python" ]]; then
  conda create -y -p "$SAM3_ENV" python=3.12 pip
fi
conda activate "$SAM3_ENV"

python -m pip install -i "$ALIYUN_PYPI" --trusted-host mirrors.aliyun.com --upgrade pip
python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -i "$ALIYUN_PYPI" --trusted-host mirrors.aliyun.com \
  "sam3==$SAM3_VERSION" \
  numpy==2.2.6 \
  opencv-python-headless==4.12.0.88 \
  pycollada==0.9.2 \
  psutil==7.2.2 \
  PyYAML==6.0.3 \
  trimesh==4.8.3

mkdir -p "$(dirname "$SAM3_CHECKPOINT")"
if [[ ! -f "$SAM3_CHECKPOINT" ]] || ! echo "$SAM3_CHECKPOINT_SHA256  $SAM3_CHECKPOINT" | sha256sum --check --status; then
  partial="$SAM3_CHECKPOINT.partial"
  curl --fail --location \
    --retry 8 --retry-delay 2 --retry-all-errors \
    --continue-at - --output "$partial" \
    "$SAM3_CHECKPOINT_URL"
  echo "$SAM3_CHECKPOINT_SHA256  $partial" | sha256sum --check
  mv "$partial" "$SAM3_CHECKPOINT"
fi
echo "$SAM3_CHECKPOINT_SHA256  $SAM3_CHECKPOINT" | sha256sum --check

if [[ ! -f "$SAM3_MERGES" ]] || ! echo "$SAM3_MERGES_SHA256  $SAM3_MERGES" | sha256sum --check --status; then
  curl --fail --location \
    --retry 8 --retry-delay 2 --retry-all-errors \
    --output "$SAM3_MERGES.partial" \
    "$SAM3_MERGES_URL"
  echo "$SAM3_MERGES_SHA256  $SAM3_MERGES.partial" | sha256sum --check
  mv "$SAM3_MERGES.partial" "$SAM3_MERGES"
fi
echo "$SAM3_MERGES_SHA256  $SAM3_MERGES" | sha256sum --check
gzip -n -c "$SAM3_MERGES" > "$SAM3_BPE.partial"
mv "$SAM3_BPE.partial" "$SAM3_BPE"

python - <<'PY'
import cv2
import torch
import trimesh
import sam3

if not torch.cuda.is_available():
    raise RuntimeError("SAM3 environment cannot see CUDA")
print("SAM3 CAD environment ready", torch.__version__, torch.cuda.get_device_name(0))
PY

echo "SAM3 package pinned at $SAM3_VERSION"
echo "SAM3 checkpoint downloaded from ModelScope revision $SAM3_MODELSCOPE_REVISION"
echo "Set PDI_SAM3_CHECKPOINT=$SAM3_CHECKPOINT for pipeline runs."
echo "Set PDI_SAM3_BPE=$SAM3_BPE for pipeline runs."
