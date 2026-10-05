#!/usr/bin/env bash
# BEGIN PDI SOURCE LOCATION
pdi_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pdi_checkout="$pdi_source_dir"
while [[ "$pdi_checkout" != / && ! -f "$pdi_checkout/documentation/architecture/layout.json" ]]; do
  pdi_checkout="$(dirname "$pdi_checkout")"
done
if [[ -f "$pdi_checkout/documentation/architecture/layout.json" ]]; then
  pdi_source_dir="$pdi_checkout/infrastructure/compat/experiments"
fi
# END PDI SOURCE LOCATION
# Run on the hosted instance, from the staged project. Outputs must be new or --resume.
set -Eeuo pipefail
cd "${pdi_source_dir}/.."
S3D_ROOT="$PWD"
S3D_PYTHON="${S3D_PYTHON:-/root/autodl-tmp/pdi/env/pdi-bench/bin/python}"
S3D_RUN_ID="${S3D_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
S3D_SESSION="${S3D_SESSION:-simple3d-$S3D_RUN_ID}"
S3D_PHASE="${S3D_PHASE:-all}"
S3D_WORKERS="${S3D_WORKERS:-2}"
S3D_RUN_ROOT="$S3D_ROOT/results/simple3d-$S3D_RUN_ID"
S3D_INPUTS="${S3D_INPUTS:-$S3D_RUN_ROOT/inputs}"
export S3D_ROOT S3D_PYTHON S3D_RUN_ID S3D_SESSION S3D_PHASE S3D_WORKERS S3D_RUN_ROOT S3D_INPUTS
export CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PYTHONPATH="$S3D_ROOT/PDI-Bench-edited/src:$S3D_ROOT"
command -v tmux >/dev/null
test -x "$S3D_PYTHON"
test -f "$S3D_INPUTS/object_manifest.json"
test -f "$S3D_INPUTS/link5_manifest.json"
test -f "$S3D_ROOT/assets/link5.dae"
test -f "$S3D_RUN_ROOT/metadata/execution/preflight.json"
"$S3D_PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "passed"' "$S3D_RUN_ROOT/metadata/execution/preflight.json"
if tmux has-session -t "$S3D_SESSION" 2>/dev/null; then
    echo "Session already exists: $S3D_SESSION" >&2
    exit 1
fi
mkdir -p "$S3D_RUN_ROOT/metadata/execution"
S3D_LOG="$S3D_RUN_ROOT/metadata/execution/$S3D_SESSION.log"
export S3D_LOG
tmux new-session -d -s "$S3D_SESSION" "bash '$S3D_ROOT/experiments/simple3d_gpu_job.sh'"
echo "Started tmux $S3D_SESSION; log $S3D_LOG"
