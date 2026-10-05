#!/usr/bin/env bash
# Additional dependencies only; reuse the existing CUDA11.8 PDI/MegaSaM environment.
set -Eeuo pipefail
S3D_PYTHON="${S3D_PYTHON:-/root/autodl-tmp/pdi/env/pdi-bench/bin/python}"
S3D_DEPS="${S3D_DEPS:-/root/simple3d-deps}"
export CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2
mkdir -p "$S3D_DEPS"
test -d "$S3D_DEPS/Pointnet2_PyTorch/.git" || git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git "$S3D_DEPS/Pointnet2_PyTorch"
test -d "$S3D_DEPS/knn_cuda/.git" || git clone https://github.com/willxxy/knn_cuda.git "$S3D_DEPS/knn_cuda"
git -C "$S3D_DEPS/Pointnet2_PyTorch" checkout b5ceb6d9ca0467ea34beb81023f96ee82228f626
git -C "$S3D_DEPS/knn_cuda" checkout 8b21dbfc86988f56588a5ed8ca3cc354122c5656
"$S3D_PYTHON" -m pip install --no-cache-dir 'tifffile==2025.5.10' 'trimesh==5.1.1' 'pycollada==0.9.3' ninja
"$S3D_PYTHON" -m pip install --no-build-isolation --no-deps "$S3D_DEPS/Pointnet2_PyTorch/pointnet2_ops_lib"
"$S3D_PYTHON" -m pip install --no-deps "$S3D_DEPS/knn_cuda"
"$S3D_PYTHON" -c 'import torch,open3d; from knn_cuda import KNN; from pointnet2_ops import pointnet2_utils; assert torch.cuda.is_available()'
# If Open3D reports missing libEGL.so.1, install the OS dependency: apt-get install libegl1.
