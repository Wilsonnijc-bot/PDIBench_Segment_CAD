#!/usr/bin/env bash
set -Eeuo pipefail
workspace=${1:?workspace}; dependency=${2:?dependency}; config=${3:?config}
kind=${4:-prepare}
exitrole=preparation
options=()
if test "$kind" = references; then exitrole=references;options=(--reference-only);fi
test -n "${SLURM_JOB_ID:-}" || { echo 'Slurm GPU allocation required' >&2; exit 2; }
cd "$workspace"
metadata="$workspace/results/link5_shape_codebook/metadata"
mkdir -p "$metadata"
trap 'rc=$?; printf "%s\n" "$rc" > "$metadata/${exitrole}-${SLURM_JOB_ID}.exit"' EXIT
export PYTHONPATH="$workspace/infrastructure/deformation_detect/import_guard:$workspace" PYTHONNOUSERSITE=1
export PATH="$dependency/bin:$dependency/env/geometry/bin:$PATH"
export LD_LIBRARY_PATH="$dependency/env/compiler/lib:$dependency/env/geometry/lib:$dependency/env/geometry/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME="$dependency/cache/huggingface" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
hostname > "$metadata/preparation-host-${SLURM_JOB_ID}.txt"
nvidia-smi > "$metadata/preparation-gpu-${SLURM_JOB_ID}.txt"
"$dependency/env/geometry/bin/python" -m robot.experiments.link5_shape_codebook.preflight --config "$config" --prepare-only
"$dependency/env/geometry/bin/python" -u -m robot.experiments.link5_shape_codebook.prepare_batch --config "$config" "${options[@]}"
echo LINK5_PREPARATION_JOB_COMPLETE
