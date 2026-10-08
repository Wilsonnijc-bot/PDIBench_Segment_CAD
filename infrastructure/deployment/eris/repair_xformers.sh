#!/usr/bin/env bash
# Repair the CUDA-wheel mismatch observed in the first H200 preflight.
set -Eeuo pipefail
dependency=${1:?dependency directory required}
metadata="$dependency/metadata"
trap 'rc=$?; printf "%s\n" "$rc" > "$metadata/repair-xformers-${SLURM_JOB_ID:-manual}.exit"' EXIT
export PIP_CACHE_DIR="$dependency/cache/pip"
export TMPDIR="/tmp/wilson-xformers-${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
py="$dependency/env/geometry/bin/python"
"$py" -m pip freeze > "$metadata/geometry-packages-before-cu118-repair.txt"
"$py" -m pip install --force-reinstall --no-deps 'xformers==0.0.22.post7+cu118' --index-url https://download.pytorch.org/whl/cu118
"$py" -c 'import torch; from xformers import _cpp_lib; assert _cpp_lib._cpp_library_load_exception is None; print("XFORMERS_EXTENSIONS_LOADED", torch.__version__)'
"$py" -m xformers.info > "$metadata/xformers-cu118-info.txt"
"$py" -m pip check
"$py" -m pip freeze > "$metadata/geometry-packages.txt"
printf 'XFORMERS_REPAIR_COMPLETE\n'
