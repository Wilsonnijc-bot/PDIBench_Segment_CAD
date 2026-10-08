#!/usr/bin/env bash
# Provision independent runtimes; never launch case inference here.
set -Eeuo pipefail
dependency=${1:?dependency directory required}
profile=${2:?sam3, geometry, or models required}
mkdir -p "$dependency"/{env,models,cache,metadata,source}
export CONDA_PKGS_DIRS="$dependency/cache/conda-$profile"
export PIP_CACHE_DIR="$dependency/cache/pip"
export TMPDIR="$dependency/cache/tmp-$profile"
mkdir -p "$TMPDIR"
source /apps/software/Miniforge3/24.11.3-0/etc/profile.d/conda.sh
trap 'rc=$?; printf "%s\n" "$rc" > "$dependency/metadata/prepare-'"$profile"'.exit"' EXIT
case "$profile" in
  sam3)
    test -x "$dependency/env/sam3/bin/python" || conda create -y -p "$dependency/env/sam3" python=3.12 pip
    py="$dependency/env/sam3/bin/python"
    "$py" -m pip install 'torch==2.10.0' 'torchvision==0.25.0' --index-url https://download.pytorch.org/whl/cu128
    "$py" -m pip install 'numpy==2.2.6' 'opencv-python-headless==4.12.0.88' 'transformers==4.41.2' 'huggingface-hub==0.36.2' 'sam3==0.1.4' 'timm==1.0.28' 'einops==0.8.2' 'pycollada==0.9.2' 'psutil==7.2.2' 'PyYAML==6.0.3' 'trimesh==4.8.3' scipy scikit-image Pillow matplotlib openpyxl imageio imageio-ffmpeg rich requests
    test -x "$dependency/env/qwen/bin/python" || "$py" -m venv --system-site-packages "$dependency/env/qwen"
    "$dependency/env/qwen/bin/python" -m pip install 'transformers==5.5.0' 'huggingface-hub==1.32.0' 'accelerate==1.15.0' 'typing-extensions==4.15.0'
    "$py" -m pip freeze > "$dependency/metadata/sam3-packages.txt"
    "$dependency/env/qwen/bin/python" -m pip freeze > "$dependency/metadata/qwen-packages.txt"
    mkdir -p "$dependency/bin"
    install -m 755 "$(dirname "$0")/git.sh" "$dependency/bin/git"
    ln -sfn "$("$py" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$dependency/bin/ffmpeg"
    ;;
  geometry)
    test -x "$dependency/env/geometry/bin/python" || conda create -y -p "$dependency/env/geometry" python=3.10 pip
    py="$dependency/env/geometry/bin/python"
    "$py" -m pip install 'torch==2.1.0' 'torchvision==0.16.0' 'torchaudio==2.1.0' --index-url https://download.pytorch.org/whl/cu118
    "$py" -m pip install 'numpy==1.26.4' 'transformers==4.41.2' 'timm==0.9.16' 'einops==0.8.0' 'opencv-python-headless==4.11.0.86' scipy scikit-learn matplotlib Pillow PyYAML rich imageio imageio-ffmpeg pandas tqdm h5py hydra-core omegaconf gdown ninja 'setuptools<81' wheel open3d
    "$py" -m pip install --no-deps 'xformers==0.0.22.post7+cu118' --index-url https://download.pytorch.org/whl/cu118
    "$py" -m pip install 'torch-scatter==2.1.2+pt21cu118' -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
    "$py" -m pip install --no-deps "$dependency/source/cotracker"
    test -x "$dependency/env/anomalydino/bin/python" || "$py" -m venv --system-site-packages "$dependency/env/anomalydino"
    "$dependency/env/anomalydino/bin/python" -m pip install 'faiss-cpu==1.8.0.post1' tifffile
    bash "$(dirname "$(realpath "$0")")/prepare_geometry_native.sh" "$dependency"
    ;;
  models)
    test -x "$dependency/env/download/bin/python" || conda create -y -p "$dependency/env/download" python=3.12 pip
    py="$dependency/env/download/bin/python"
    "$py" -m pip install huggingface-hub
    export HF_HOME="$dependency/cache/huggingface" HF_HUB_DISABLE_TELEMETRY=1
    "$py" "$(dirname "$0")/download_models.py" "$dependency"
    ;;
  *) echo "Unknown profile: $profile" >&2; exit 2;;
esac
printf 'PREPARATION_COMPLETE %s\n' "$profile"
