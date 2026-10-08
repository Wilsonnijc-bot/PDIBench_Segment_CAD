#!/usr/bin/env bash
set -Eeuo pipefail
base=/PHShome/zy992/Wilson/deformationdetection
workspace=$base/workspace-cosmos3-0019-masking-20261007
dependency=/PHShome/zy992/Wilson/dependency
metadata=$base/metadata/cosmos3-0019-seed101-masking-20261007
cd "$workspace"
export PYTHONPATH="$workspace/infrastructure/deformation_detect/import_guard:$workspace" PYTHONNOUSERSITE=1
export PDI_IMPORT_ORIGINS_DIR="$metadata/runtime-import-origins-$SLURM_JOB_ID"
export LD_LIBRARY_PATH="$dependency/env/compiler/lib:$dependency/env/geometry/lib:$dependency/env/geometry/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME="$dependency/cache/huggingface" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PATH="$dependency/bin:$dependency/env/geometry/bin:$PATH"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
child_pid=
finish() {
    rc=$?
    trap - EXIT TERM HUP INT
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" || true
    fi
    printf '%s\n' "$rc" > "$metadata/job.exit"
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
run_python() {
    "$dependency/env/sam3/bin/python" -u "$@" &
    child_pid=$!
    wait "$child_pid"
    child_pid=
}
hostname > "$metadata/compute-host.txt"
nvidia-smi > "$metadata/gpu.txt"
run_python infrastructure/deployment/eris/validate_runtime.py --profile sam3 --deployment-root /PHShome/zy992/Wilson
run_python -m infrastructure.deformation_detect coordinate --manifest "$metadata/manifest.json" --preflight-only
run_python -m infrastructure.deformation_detect coordinate --manifest "$metadata/manifest.json"
run_python "$metadata/render.py" --manifest "$metadata/manifest.json"
printf 'MASKING_REPLAY_COMPLETE\n'
