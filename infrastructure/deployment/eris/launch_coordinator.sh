#!/usr/bin/env bash
# Launch only inside a Slurm H200 allocation, using the permanent coordinator.
set -Eeuo pipefail
workspace=${1:?workspace required}
dependency=${2:?dependency directory required}
manifest=${3:?manifest required}
metadata=${4:?launch metadata directory required}
test -n "${SLURM_JOB_ID:-}" || { echo 'A Slurm GPU allocation is required' >&2; exit 2; }
mkdir -p "$metadata"
trap 'rc=$?; printf "%s\n" "$rc" > "$metadata/coordinator-${SLURM_JOB_ID}.exit"' EXIT
cd "$workspace"
export PYTHONPATH="$workspace/infrastructure/deformation_detect/import_guard:$workspace" PYTHONNOUSERSITE=1
export PDI_IMPORT_ORIGINS_DIR="$metadata/runtime-import-origins-${SLURM_JOB_ID}"
export LD_LIBRARY_PATH="$dependency/env/compiler/lib:$dependency/env/geometry/lib:$dependency/env/geometry/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME="$dependency/cache/huggingface" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PATH="$dependency/bin:$dependency/env/geometry/bin:$PATH"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
test -f "$metadata/assets-verified.json"
for profile in sam3 geometry models; do
    test "$(cat "$dependency/metadata/prepare-$profile.exit")" = 0
done
hostname > "$metadata/compute-host-${SLURM_JOB_ID}.txt"
nvidia-smi > "$metadata/gpu-${SLURM_JOB_ID}.txt"
for profile in sam3 vlm1 geometry anomaly; do
    case "$profile" in
        sam3) interpreter="$dependency/env/sam3/bin/python";;
        vlm1) interpreter="$dependency/env/qwen/bin/python";;
        geometry) interpreter="$dependency/env/geometry/bin/python";;
        anomaly) interpreter="$dependency/env/anomalydino/bin/python";;
    esac
    "$interpreter" infrastructure/deployment/eris/validate_runtime.py --profile "$profile" --deployment-root "$(dirname "$dependency")" > "$metadata/runtime-$profile-${SLURM_JOB_ID}.json"
done
python="$dependency/env/sam3/bin/python"
"$python" -m deformation_detect coordinate --manifest "$manifest" --preflight-only
"$python" -u -m deformation_detect coordinate --manifest "$manifest" "${@:5}"
