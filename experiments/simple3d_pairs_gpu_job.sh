#!/usr/bin/env bash
set -uo pipefail
cd /root/autodl-tmp/pdi/simple3d-evaluation
export CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/PDI-Bench-edited/src:$PWD"
pair_root="${S3D_PAIR_ROOT:-$PWD/results/simple3d-20261003-run2/link5_pair_visible_anchor}"
workers="${S3D_PAIR_WORKERS:-2}"
mkdir -p "$pair_root/metadata/execution"
rm -f "$pair_root/metadata/execution/run.exit"
scope_args=()
if [[ -n "${S3D_PAIR_LIMIT:-}" ]]; then scope_args=(--limit "$S3D_PAIR_LIMIT"); fi
/root/autodl-tmp/pdi/env/pdi-bench/bin/python experiments/run_simple3d_link5_pairs.py --root "$pair_root" --workers "$workers" --resume "${scope_args[@]}" 2>&1 | tee -a "$pair_root/metadata/execution/run.log"
rc=${PIPESTATUS[0]}
printf '\nEXIT_STATUS=%s\n' "$rc" >> "$pair_root/metadata/execution/run.log"
printf '%s\n' "$rc" > "$pair_root/metadata/execution/run.exit"
exit "$rc"
