#!/usr/bin/env bash
set -euo pipefail

PDI_GPU_ROOT="${PDI_GPU_ROOT:-/root/autodl-tmp/pdi}"
PDI_ENV="$PDI_GPU_ROOT/env/pdi-bench"
PDI_CODE="$PDI_GPU_ROOT/code/PDI-Bench-edited"
PDI_MODELS="$PDI_GPU_ROOT/models"
PYTORCH_WHEEL_BASE="${PDI_PYTORCH_WHEEL_BASE:-https://mirrors.aliyun.com/pytorch-wheels/cu118}"
CONDA_MAIN_CHANNEL="${PDI_CONDA_MAIN_CHANNEL:-https://mirror.sjtu.edu.cn/anaconda/pkgs/main}"
DEPTH_CHECKPOINT_URL="${PDI_DEPTH_CHECKPOINT_URL:-https://hf-mirror.com/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth}"
RAFT_CHECKPOINT_URL="${PDI_RAFT_CHECKPOINT_URL:-https://hf-mirror.com/sbalani/raft-things/resolve/main/raft-things.pth}"
DINO_V2_COMMIT="${PDI_DINO_V2_COMMIT:-7764ea0f912e53c92e82eb78a2a1631e92725fc8}"

if [[ ! -f "$PDI_CODE/requirements.txt" ]]; then
  echo "PDI source is missing at $PDI_CODE" >&2
  exit 1
fi

mkdir -p "$PDI_GPU_ROOT"/{env,models,cache,code,runs}
source /root/miniconda3/etc/profile.d/conda.sh
git config --global --add safe.directory "$PDI_CODE"

if [[ ! -x "$PDI_ENV/bin/python" ]]; then
  conda create --prefix "$PDI_ENV" --override-channels \
    -c "$CONDA_MAIN_CHANNEL" python=3.10 -y
fi
conda activate "$PDI_ENV"

if [[ -x "$PDI_ENV/bin/nvcc" ]]; then
  CUDA_HOME="$PDI_ENV"
elif [[ -x /usr/local/cuda/bin/nvcc ]] && /usr/local/cuda/bin/nvcc --version | grep -q 'release 11.8'; then
  CUDA_HOME="/usr/local/cuda"
else
  conda install -p "$PDI_ENV" --override-channels -c nvidia/label/cuda-11.8.0 -c defaults \
    cuda-nvcc=11.8 cuda-cccl=11.8 cuda-libraries-dev=11.8 \
    cuda-cudart-dev=11.8 libcublas-dev=11.11 -y
  CUDA_HOME="$PDI_ENV"
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$PDI_ENV/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"

python -m pip install --upgrade pip wheel 'setuptools==69.5.1'
if ! python -c 'import torch; assert torch.__version__.startswith("2.1.0") and torch.version.cuda == "11.8"' 2>/dev/null; then
  WHEEL_CACHE="$PDI_GPU_ROOT/cache/wheels"
  if [[ -s "$WHEEL_CACHE/torch-2.1.0+cu118-cp310-cp310-linux_x86_64.whl" ]]; then
    python -m pip install --no-deps \
      "$WHEEL_CACHE/torch-2.1.0+cu118-cp310-cp310-linux_x86_64.whl" \
      "$WHEEL_CACHE/torchvision-0.16.0+cu118-cp310-cp310-linux_x86_64.whl" \
      "$WHEEL_CACHE/torchaudio-2.1.0+cu118-cp310-cp310-linux_x86_64.whl"
  else
    python -m pip install --no-deps \
      "$PYTORCH_WHEEL_BASE/torch-2.1.0%2Bcu118-cp310-cp310-linux_x86_64.whl" \
      "$PYTORCH_WHEEL_BASE/torchvision-0.16.0%2Bcu118-cp310-cp310-linux_x86_64.whl" \
      "$PYTORCH_WHEEL_BASE/torchaudio-2.1.0%2Bcu118-cp310-cp310-linux_x86_64.whl"
  fi
fi

python -m pip install -r <(sed \
  -e '/^--extra-index-url /d' \
  -e '/^torch==/d' \
  -e '/^torchvision==/d' \
  -e '/^torchaudio==/d' \
  "$PDI_CODE/requirements.txt")
python -m pip install hydra-core iopath wandb yacs h5py safetensors tabulate gdown imageio-ffmpeg
if ! python -m xformers.info 2>/dev/null | grep -Eq 'memory_efficient_attention\.cutlassF:[[:space:]]+available$'; then
  python -m pip install --force-reinstall --no-cache-dir --no-deps xformers==0.0.22.post7+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
fi

if ! python -c 'from cotracker.predictor import CoTrackerPredictor' 2>/dev/null; then
  if [[ -f "$PDI_GPU_ROOT/cache/src/co-tracker/setup.py" ]]; then
    python -m pip install --no-deps --no-build-isolation \
      "$PDI_GPU_ROOT/cache/src/co-tracker"
  else
    python -m pip install --no-deps --no-build-isolation \
      git+https://github.com/facebookresearch/co-tracker.git@82e02e8029753ad4ef13cf06be7f4fc5facdda4d
  fi
fi
if ! python -c 'from torch_scatter import scatter_sum' 2>/dev/null; then
  python -m pip install torch-scatter --force-reinstall \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
fi

mkdir -p "$PDI_MODELS"/{tracker,depth_anything,raft}
if [[ ! -s "$PDI_MODELS/tracker/scaled_offline.pth" ]]; then
  wget -q --show-progress -O "$PDI_MODELS/tracker/scaled_offline.pth.part" \
    https://hf-mirror.com/facebook/cotracker3/resolve/main/scaled_offline.pth
  mv "$PDI_MODELS/tracker/scaled_offline.pth.part" \
    "$PDI_MODELS/tracker/scaled_offline.pth"
fi
if [[ ! -s "$PDI_MODELS/depth_anything/depth_anything_vitl14.pth" ]]; then
  wget -c -q --show-progress -O "$PDI_MODELS/depth_anything/depth_anything_vitl14.pth.part" \
    "$DEPTH_CHECKPOINT_URL"
  mv "$PDI_MODELS/depth_anything/depth_anything_vitl14.pth.part" \
    "$PDI_MODELS/depth_anything/depth_anything_vitl14.pth"
fi
if [[ ! -s "$PDI_MODELS/raft/raft-things.pth" ]]; then
  wget -c -q --show-progress -O "$PDI_MODELS/raft/raft-things.pth.part" \
    "$RAFT_CHECKPOINT_URL"
  mv "$PDI_MODELS/raft/raft-things.pth.part" \
    "$PDI_MODELS/raft/raft-things.pth"
fi

mkdir -p \
  "$PDI_CODE/third_party/mega_sam/Depth-Anything/checkpoints" \
  "$PDI_CODE/third_party/mega_sam/cvd_opt" \
  "$PDI_CODE/third_party/mega_sam/torchhub"
ln -sfn "$PDI_MODELS/depth_anything/depth_anything_vitl14.pth" \
  "$PDI_CODE/third_party/mega_sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth"
ln -sfn "$PDI_MODELS/raft/raft-things.pth" \
  "$PDI_CODE/third_party/mega_sam/cvd_opt/raft-things.pth"

DINO_V2_SOURCE="$PDI_CODE/third_party/mega_sam/torchhub/facebookresearch_dinov2_main"
if [[ ! -f "$DINO_V2_SOURCE/hubconf.py" ]]; then
  git clone https://github.com/facebookresearch/dinov2.git "$DINO_V2_SOURCE"
fi
git -C "$DINO_V2_SOURCE" fetch --depth 1 origin "$DINO_V2_COMMIT"
git -C "$DINO_V2_SOURCE" checkout --detach "$DINO_V2_COMMIT"

MEGA_SAM_BASE="$PDI_CODE/third_party/mega_sam/base"
ORIGINAL_MEGA_SAM_BASE="$PDI_GPU_ROOT/code/PDI-Bench-original/third_party/mega_sam/base"
if [[ ! -f "$MEGA_SAM_BASE/thirdparty/lietorch/lietorch/__init__.py" ]] && \
   [[ -f "$ORIGINAL_MEGA_SAM_BASE/thirdparty/lietorch/lietorch/__init__.py" ]]; then
  cp -a "$ORIGINAL_MEGA_SAM_BASE/." "$MEGA_SAM_BASE/"
fi
if [[ -d "$MEGA_SAM_BASE/.git" ]]; then
  git -C "$MEGA_SAM_BASE" submodule update --init --recursive
fi

if ! python -c 'import droid_backends; from lietorch import SE3' 2>/dev/null; then
  git config --global --add safe.directory \
    "$PDI_CODE/third_party/mega_sam/base"
  (cd "$PDI_CODE" && CONDA_PREFIX="$CUDA_HOME" bash scripts/build_mega_sam.sh)
fi

python - <<'PY'
import os
import torch
from torch_scatter import scatter_sum
from cotracker.predictor import CoTrackerPredictor
from xformers.ops.fmha.cutlass import FwOp

assert torch.__version__.startswith("2.1.0"), torch.__version__
assert torch.version.cuda == "11.8", torch.version.cuda
assert torch.cuda.is_available()
assert FwOp.is_available(), "xformers CUDA attention operator is unavailable"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
print("CoTracker / torch-scatter / xformers CUDA OK")
PY
