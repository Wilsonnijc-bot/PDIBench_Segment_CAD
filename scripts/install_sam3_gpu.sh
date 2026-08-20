#!/usr/bin/env bash
set -euo pipefail

PDI_GPU_ROOT="${PDI_GPU_ROOT:-/root/autodl-tmp/pdi}"
SAM3_COMMIT="8f0b7f4d4e7eda2ed606ebde6702c93359ad01da"
SAM3_REPOSITORY="https://github.com/facebookresearch/sam3.git"
SAM3_CODE="$PDI_GPU_ROOT/code/sam3"
SAM3_ENV="$PDI_GPU_ROOT/env/sam3"

source /root/miniconda3/etc/profile.d/conda.sh
if [[ ! -x "$SAM3_ENV/bin/python" ]]; then
  conda create -y -p "$SAM3_ENV" python=3.12 pip
fi
conda activate "$SAM3_ENV"

python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

if [[ ! -d "$SAM3_CODE/.git" ]]; then
  if [[ -e "$SAM3_CODE" ]]; then
    echo "Refusing to replace non-git SAM3 path: $SAM3_CODE" >&2
    exit 1
  fi
  git clone --filter=blob:none --no-checkout "$SAM3_REPOSITORY" "$SAM3_CODE"
fi
git -C "$SAM3_CODE" fetch --depth 1 origin "$SAM3_COMMIT"
git -C "$SAM3_CODE" checkout --detach "$SAM3_COMMIT"
test "$(git -C "$SAM3_CODE" rev-parse HEAD)" = "$SAM3_COMMIT"

python -m pip install -e "$SAM3_CODE"
python -m pip install \
  numpy==2.2.6 \
  opencv-python-headless==4.12.0.88 \
  pycollada==0.9.2 \
  PyYAML==6.0.3 \
  trimesh==4.8.3

python - <<'PY'
import cv2
import torch
import trimesh
import sam3

if not torch.cuda.is_available():
    raise RuntimeError("SAM3 environment cannot see CUDA")
print("SAM3 CAD environment ready", torch.__version__, torch.cuda.get_device_name(0))
PY

echo "SAM3 source pinned at $SAM3_COMMIT"
echo "Checkpoint access still requires an authenticated Hugging Face account."
