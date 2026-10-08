#!/usr/bin/env bash
set -Eeuo pipefail
workspace=${1:?workspace}; dependency=${2:?dependency}; config=${3:?config}
test -n "${SLURM_JOB_ID:-}"
cd "$workspace"
metadata="$workspace/results/link5_shape_codebook/full_video/metadata";mkdir -p "$metadata"
trap 'rc=$?; printf "%s\n" "$rc" > "$metadata/full-video-${SLURM_JOB_ID}.exit"' EXIT
export PYTHONPATH="$workspace/infrastructure/deformation_detect/import_guard:$workspace" PYTHONNOUSERSITE=1
export PATH="$dependency/bin:$dependency/env/geometry/bin:$PATH"
export LD_LIBRARY_PATH="$dependency/env/foundationpose/lib:$dependency/env/shape-build/lib:$dependency/env/compiler/lib:$dependency/env/geometry/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="$dependency/cache/huggingface" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
hostname > "$metadata/host-${SLURM_JOB_ID}.txt"
nvidia-smi > "$metadata/gpu-${SLURM_JOB_ID}.txt"
"$dependency/env/geometry/bin/python" -u -m robot.experiments.link5_shape_codebook.full_video --config "$config" --phase batch
echo LINK5_FULL_VIDEO_JOB_COMPLETE
