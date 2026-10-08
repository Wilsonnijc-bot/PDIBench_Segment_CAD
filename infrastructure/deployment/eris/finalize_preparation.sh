#!/usr/bin/env bash
# Required runtime imports found in the pinned MegaSAM source and video adapters.
set -Eeuo pipefail
dependency=${1:?dependency directory required}
trap 'rc=$?; printf "%s\n" "$rc" > "$dependency/metadata/finalize-preparation-${SLURM_JOB_ID:-manual}.exit"' EXIT
export PIP_CACHE_DIR="$dependency/cache/pip"
export TMPDIR="$dependency/cache/tmp-finalize"
mkdir -p "$TMPDIR" "$dependency/bin"
"$dependency/env/geometry/bin/python" -m pip install 'kornia==0.7.4' wandb
"$dependency/env/qwen/bin/python" -m pip install 'typing-extensions==4.15.0' 'anyio==4.14.2'
for profile in sam3 qwen geometry anomalydino; do
    "$dependency/env/$profile/bin/python" -m pip check
done
ln -sfn "$("$dependency/env/sam3/bin/python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$dependency/bin/ffmpeg"
"$dependency/env/geometry/bin/python" -m pip freeze > "$dependency/metadata/geometry-packages.txt"
"$dependency/env/anomalydino/bin/python" -m pip freeze > "$dependency/metadata/anomalydino-packages.txt"
"$dependency/env/qwen/bin/python" -m pip freeze > "$dependency/metadata/qwen-packages.txt"
printf 'FINALIZE_PREPARATION_COMPLETE\n'
