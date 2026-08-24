#!/usr/bin/env bash
set -euo pipefail

PDI_GPU_ROOT="${PDI_GPU_ROOT:-/root/autodl-tmp/pdi}"
FOUNDATIONPOSE_ENV="$PDI_GPU_ROOT/env/foundationpose"
FOUNDATIONPOSE_SOURCE="${PDI_FOUNDATIONPOSE_SOURCE:-$PDI_GPU_ROOT/cache/src/FoundationPose}"
FOUNDATIONPOSE_REVISION="a1b694b83e633c2cb6115b9063d940a687759392"
PYTORCH3D_REVISION="fdaf9bd6fed7977e4c2056e7c77c640781e58fcd"
NVDIFFRAST_REVISION="253ac4fcea7de5f396371124af597e6cc957bfae"
CONDA_MAIN_CHANNEL="${PDI_CONDA_MAIN_CHANNEL:-https://mirror.sjtu.edu.cn/anaconda/pkgs/main}"
ALIYUN_PYPI="${PDI_PYPI_INDEX:-http://mirrors.aliyun.com/pypi/simple}"
PYTORCH_WHEEL_BASE="${PDI_FOUNDATIONPOSE_TORCH_BASE:-https://mirrors.aliyun.com/pytorch-wheels/cu118}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"

mkdir -p "$PDI_GPU_ROOT"/{env,cache/src,models/foundationpose}
source /root/miniconda3/etc/profile.d/conda.sh

if [[ ! -d "$FOUNDATIONPOSE_SOURCE/.git" ]]; then
  git clone --filter=blob:none https://github.com/NVlabs/FoundationPose.git \
    "$FOUNDATIONPOSE_SOURCE"
fi
git -C "$FOUNDATIONPOSE_SOURCE" fetch --depth 1 origin "$FOUNDATIONPOSE_REVISION"
git -C "$FOUNDATIONPOSE_SOURCE" checkout --detach "$FOUNDATIONPOSE_REVISION"

if [[ ! -x "$FOUNDATIONPOSE_ENV/bin/python" ]]; then
  conda create -y -p "$FOUNDATIONPOSE_ENV" --override-channels \
    -c "$CONDA_MAIN_CHANNEL" python=3.11 pip
fi
conda activate "$FOUNDATIONPOSE_ENV"

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  build-essential cmake ninja-build libeigen3-dev libboost-all-dev pybind11-dev

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$FOUNDATIONPOSE_ENV/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$FOUNDATIONPOSE_ENV:${CMAKE_PREFIX_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="${MAX_JOBS:-8}"

python -m pip install -i "$ALIYUN_PYPI" --trusted-host mirrors.aliyun.com \
  --upgrade pip wheel 'setuptools==69.5.1' 'pybind11==2.13.6' cmake ninja
if ! python -c 'import torch; assert torch.__version__.startswith("2.1.0") and torch.version.cuda == "11.8"' 2>/dev/null; then
  python -m pip install --no-deps \
    "$PYTORCH_WHEEL_BASE/torch-2.1.0%2Bcu118-cp311-cp311-linux_x86_64.whl" \
    "$PYTORCH_WHEEL_BASE/torchvision-0.16.0%2Bcu118-cp311-cp311-linux_x86_64.whl" \
    "$PYTORCH_WHEEL_BASE/torchaudio-2.1.0%2Bcu118-cp311-cp311-linux_x86_64.whl"
fi

python -m pip install -i "$ALIYUN_PYPI" --trusted-host mirrors.aliyun.com \
  numpy==1.26.4 filelock typing-extensions sympy networkx jinja2 fsspec iopath
python -m pip install -i "$ALIYUN_PYPI" --trusted-host mirrors.aliyun.com \
  -r <(sed '/^numpy[<>=]/d' "$FOUNDATIONPOSE_SOURCE/requirements.txt")
python -m pip install -i "$ALIYUN_PYPI" --trusted-host mirrors.aliyun.com \
  --force-reinstall numpy==1.26.4 opencv-python==4.11.0.86
python -m pip install -i "$ALIYUN_PYPI" --trusted-host mirrors.aliyun.com \
  --force-reinstall --no-deps 'kornia==0.7.3'

# Ubuntu's pybind11 2.9 headers do not support Python 3.11. Put the pinned
# environment package first so FoundationPose's CMake build does not find it.
export CMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir):$CMAKE_PREFIX_PATH"

PYTORCH3D_SOURCE="$PDI_GPU_ROOT/cache/src/pytorch3d"
if [[ ! -d "$PYTORCH3D_SOURCE/.git" ]]; then
  git clone --filter=blob:none https://github.com/facebookresearch/pytorch3d.git \
    "$PYTORCH3D_SOURCE"
fi
git -C "$PYTORCH3D_SOURCE" fetch --depth 1 origin "$PYTORCH3D_REVISION"
git -C "$PYTORCH3D_SOURCE" checkout --detach "$PYTORCH3D_REVISION"
python -m pip install --no-build-isolation --no-deps "$PYTORCH3D_SOURCE"

NVDIFFRAST_SOURCE="$PDI_GPU_ROOT/cache/src/nvdiffrast"
if [[ ! -d "$NVDIFFRAST_SOURCE/.git" ]]; then
  git clone --filter=blob:none https://github.com/NVlabs/nvdiffrast.git \
    "$NVDIFFRAST_SOURCE"
fi
git -C "$NVDIFFRAST_SOURCE" fetch --depth 1 origin "$NVDIFFRAST_REVISION"
git -C "$NVDIFFRAST_SOURCE" checkout --detach "$NVDIFFRAST_REVISION"
python -m pip install --no-build-isolation --no-deps "$NVDIFFRAST_SOURCE"

(cd "$FOUNDATIONPOSE_SOURCE" && bash build_all_conda.sh)

python - "$FOUNDATIONPOSE_SOURCE" <<'PY'
import sys

source = sys.argv[1]
sys.path[:0] = [source, f"{source}/mycpp/build"]

import torch
import pytorch3d
import nvdiffrast.torch
import mycpp
import estimater

assert torch.cuda.is_available()
assert torch.version.cuda == "11.8", torch.version.cuda
print(
    "FoundationPose environment ready",
    torch.__version__,
    torch.cuda.get_device_name(0),
)
PY

echo "FoundationPose source pinned at $FOUNDATIONPOSE_REVISION"
echo "FoundationPose weights must be present under $FOUNDATIONPOSE_SOURCE/weights"
