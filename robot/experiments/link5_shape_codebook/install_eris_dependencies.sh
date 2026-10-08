#!/usr/bin/env bash
# Upstream FoundationPose installation, isolated under Wilson/dependency.
set -Eeuo pipefail
dependency=${1:?dependency}; workspace=${2:?workspace}
export PYTHONPATH="$workspace/infrastructure/deformation_detect/import_guard:$workspace" PYTHONNOUSERSITE=1
test -n "${SLURM_JOB_ID:-}" || { echo 'Slurm allocation required for native GPU builds' >&2; exit 2; }
mkdir -p "$dependency"/{source,models,cache,metadata,env}
trap 'rc=$?; printf "%s\n" "$rc" > "$dependency/metadata/link5-dependencies-${SLURM_JOB_ID}.exit"' EXIT
source /apps/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS="$dependency/cache/conda-link5" PIP_CACHE_DIR="$dependency/cache/pip"
export TMPDIR="$dependency/cache/tmp-link5";mkdir -p "$TMPDIR"
export PATH="$dependency/bin:$PATH"
if ! test -d "$dependency/source/FoundationPose/.git"; then
    git clone --recursive https://github.com/NVlabs/FoundationPose.git "$dependency/source/FoundationPose"
fi
git -C "$dependency/source/FoundationPose" checkout a1b694b83e633c2cb6115b9063d940a687759392
if ! test -d "$dependency/source/Shape-Anomaly-Codebook/.git"; then
    git clone https://github.com/alexandor91/Shape-Anomaly-Codebook.git "$dependency/source/Shape-Anomaly-Codebook"
fi
git -C "$dependency/source/Shape-Anomaly-Codebook" checkout 1b8de121de8e8801dd99b796127026c7dd279a10
for name in FoundationPose Shape-Anomaly-Codebook; do
    target="$workspace/infrastructure/vendor/$name"
    mkdir -p "$(dirname "$target")"
    if ! test -L "$target"; then
        if test -e "$target"; then
            mkdir -p "$workspace/results/link5_shape_codebook/metadata/vendor-source-snapshot"
            mv "$target" "$workspace/results/link5_shape_codebook/metadata/vendor-source-snapshot/$name"
        fi
        ln -s "$dependency/source/$name" "$target"
    fi
done
if ! test -d "$dependency/source/MinkowskiEngine/.git"; then
    git clone https://github.com/NVIDIA/MinkowskiEngine.git "$dependency/source/MinkowskiEngine"
fi
git -C "$dependency/source/MinkowskiEngine" checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
if ! test -d "$dependency/source/FCGF/.git"; then
    git clone https://github.com/chrischoy/FCGF.git "$dependency/source/FCGF"
fi
git -C "$dependency/source/FCGF" checkout 499813ea35f57d885cbc3458cd74fc57dae860c7
test -x "$dependency/env/foundationpose/bin/python" || conda create -y -p "$dependency/env/foundationpose" -c conda-forge python=3.11 pip cmake ninja eigen boost-cpp pybind11 'gxx_linux-64=13' 'gcc_linux-64=13'
if ! test -s "$dependency/metadata/foundationpose-packages.txt"; then
conda activate "$dependency/env/foundationpose"
py="$CONDA_PREFIX/bin/python"
"$py" -m pip install 'torch==2.8.0' 'torchvision==0.23.0' --index-url https://download.pytorch.org/whl/cu128
module load CUDA/12.9.0
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")" TORCH_CUDA_ARCH_LIST='9.0+PTX' MAX_JOBS=8
"$py" -m pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
"$py" -m pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
"$py" -m pip install -r "$dependency/source/FoundationPose/requirements.txt" gdown pycollada
bash "$dependency/source/FoundationPose/build_all_conda.sh"
"$py" -m gdown --folder 'https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i' -O "$dependency/source/FoundationPose/weights"
conda deactivate
module unload CUDA/12.9.0
fi
py="$dependency/env/foundationpose/bin/python"
export LD_LIBRARY_PATH="$dependency/env/foundationpose/lib:${LD_LIBRARY_PATH:-}"
for name in 2023-10-28-18-33-37 2024-01-11-20-02-45; do
    test -s "$dependency/source/FoundationPose/weights/$name/model_best.pth"
    test -s "$dependency/source/FoundationPose/weights/$name/config.yml"
done
"$py" -c "import torch; assert torch.cuda.is_available(); import sys;sys.path.insert(0,'$dependency/source/FoundationPose');from estimater import FoundationPose;import Utils;assert Utils.mycpp is not None;from learning.training.predict_score import ScorePredictor;from learning.training.predict_pose_refine import PoseRefinePredictor;ScorePredictor();PoseRefinePredictor();print('FOUNDATIONPOSE_IMPORT_AND_WEIGHTS_PASSED')"
"$py" -m pip freeze > "$dependency/metadata/foundationpose-packages.txt"
test -x "$dependency/env/shape-codebook/bin/python" || "$dependency/env/geometry/bin/python" -m venv --system-site-packages "$dependency/env/shape-codebook"
test -x "$dependency/env/shape-build/bin/x86_64-conda-linux-gnu-gcc" || conda create -y -p "$dependency/env/shape-build" -c conda-forge 'gcc_linux-64=11' 'gxx_linux-64=11' openblas
test -s "$dependency/env/shape-build/include/cblas.h"
test -s "$dependency/env/shape-build/lib/libopenblas.so"
export MAX_JOBS=8 CUDA_HOME="$dependency/env/geometry" TORCH_CUDA_ARCH_LIST='9.0+PTX'
export CC="$dependency/env/shape-build/bin/x86_64-conda-linux-gnu-gcc" CXX="$dependency/env/shape-build/bin/x86_64-conda-linux-gnu-g++"
export LD_LIBRARY_PATH="$dependency/env/shape-build/lib:$dependency/env/compiler/lib:$dependency/env/geometry/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$dependency/env/shape-build/lib:${LIBRARY_PATH:-}"
export PATH="$dependency/env/shape-codebook/bin:$dependency/env/shape-build/bin:$dependency/env/geometry/bin:$PATH"
py="$dependency/env/shape-codebook/bin/python"
"$py" -m pip install ninja 'setuptools<70' wheel 'numpy<2' future-fstrings
if ! test -d "$dependency/source/NVTX/.git"; then
    git clone --depth 1 --branch v3.1.0 https://github.com/NVIDIA/NVTX.git "$dependency/source/NVTX"
fi
test "$(git -C "$dependency/source/NVTX" rev-parse HEAD)" = e170594ac7cf1dac584da473d4ca9301087090c1
test -s "$dependency/source/NVTX/c/include/nvtx3/nvToolsExt.h"
export CPATH="$dependency/source/NVTX/c/include:${CPATH:-}"
cd "$dependency/source/MinkowskiEngine"
"$py" setup.py install --blas_include_dirs="$dependency/env/shape-build/include,$dependency/source/NVTX/c/include" --blas=openblas --force_cuda
"$py" -m pip install -r "$dependency/source/Shape-Anomaly-Codebook/requirements.txt" 'numpy<2' 'plyfile<1.1.4' easydict
mkdir -p "$dependency/models/shape_codebook"
curl --fail --location --retry 3 'https://huggingface.co/chrischoy/FCGF/resolve/main/2019-08-19_06-17-41.pth' -o "$dependency/models/shape_codebook/fcgf-3dmatch-32feat.pth"
"$py" -m robot.experiments.link5_shape_codebook.verify_backbone --config "$workspace/results/link5_shape_codebook/config.json"
"$py" -c 'import torch,MinkowskiEngine; assert torch.cuda.is_available();print("MINKOWSKIENGINE_GPU_IMPORT_PASSED")'
"$py" -m pip freeze > "$dependency/metadata/shape-codebook-packages.txt"
echo LINK5_DEPENDENCIES_COMPLETE
