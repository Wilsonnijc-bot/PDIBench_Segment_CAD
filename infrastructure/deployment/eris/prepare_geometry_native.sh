#!/usr/bin/env bash
# Resume native provisioning without reinstalling validated Python packages.
set -Eeuo pipefail
dependency=${1:?dependency directory required}
metadata="$dependency/metadata"
script_directory=$(cd "$(dirname "$0")" && pwd)
trap 'rc=$?; printf "%s\n" "$rc" > "$metadata/prepare-geometry-${SLURM_JOB_ID:-manual}.exit"; printf "%s\n" "$rc" > "$metadata/prepare-geometry.exit"' EXIT
source /apps/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS="$dependency/cache/conda-geometry"
export PIP_CACHE_DIR="$dependency/cache/pip"
export TMPDIR="/tmp/wilson-native-${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
if test -f "$dependency/cache/cuda-packages/manifest.json"; then
    # Archives are downloaded and hash-verified on the Mac, then relayed here.
    # CUDA shared libraries used by inference already come from torch+cu118.
    "$dependency/env/geometry/bin/python" "$script_directory/verify_cuda_packages.py" "$dependency/cache/cuda-packages"
    conda install -y --offline --no-deps -p "$dependency/env/geometry" "$dependency"/cache/cuda-packages/*.tar.bz2
else
    conda install -y -p "$dependency/env/geometry" --override-channels \
        -c https://conda.anaconda.org/nvidia/label/cuda-11.8.0 \
        -c https://conda.anaconda.org/conda-forge \
        cuda-toolkit=11.8 python=3.10
fi
export CUDA_HOME="$dependency/env/geometry" PATH="$dependency/env/geometry/bin:$PATH"
# Login-node compilers are not necessarily installed on the allocated CPU node.
if test -x /usr/bin/g++ && test "$(/usr/bin/g++ -dumpversion | cut -d. -f1)" -le 11; then
    export CC=/usr/bin/gcc CXX=/usr/bin/g++
else
    test -x "$dependency/env/compiler/bin/x86_64-conda-linux-gnu-c++" || \
        conda create -y -p "$dependency/env/compiler" -c conda-forge gcc_linux-64=11 gxx_linux-64=11
    export CC="$dependency/env/compiler/bin/x86_64-conda-linux-gnu-cc"
    export CXX="$dependency/env/compiler/bin/x86_64-conda-linux-gnu-c++"
fi
export LD_LIBRARY_PATH="$dependency/env/compiler/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST='9.0+PTX' MAX_JOBS=4
py="$dependency/env/geometry/bin/python"
cd "$dependency/source/mega_sam/base"
"$py" "$script_directory/prepare_native_build.py" "$PWD" "$metadata"
"$py" setup.py build_ext --inplace
"$py" -m pip install --no-build-isolation --no-deps .
"$py" -c 'import torch, droid_backends, lietorch_backends; print("NATIVE_IMPORTS_COMPLETE", droid_backends.__file__, lietorch_backends.__file__)' > "$metadata/native-imports-${SLURM_JOB_ID:-manual}.txt"
cp "$metadata/native-imports-${SLURM_JOB_ID:-manual}.txt" "$metadata/native-imports.txt"
"$py" -m pip freeze > "$metadata/geometry-packages.txt"
"$dependency/env/anomalydino/bin/python" -m pip freeze > "$metadata/anomalydino-packages.txt"
printf 'PREPARATION_COMPLETE geometry\n'
