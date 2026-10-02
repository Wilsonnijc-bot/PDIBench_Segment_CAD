#!/usr/bin/env bash
set -euo pipefail

gpu_root="${PDI_GPU_ROOT:-/root/autodl-tmp/pdi}"
code_root="$gpu_root/code/PDI-Bench-edited"
run_root="${PDI_OBJECT_RUN_ROOT:-$gpu_root/experiments/object-deformation-selected45-20260929}"
selected_root="$gpu_root/experiments/v1-v2-persistent-10-20260923/selected-45-v1"

cd "$code_root"
export PYTHONPATH="$code_root/src:$code_root"

arguments=(
  --output-root "$run_root"
  --video-root /root/autodl-tmp/motionsmoothness-robot/videos
  --sam-python "$gpu_root/env/sam3/bin/python"
  --pdi-python "$gpu_root/env/pdi-bench/bin/python"
  --sam3-checkpoint "$gpu_root/models/sam3/sam3.pt"
  --sam3-bpe "$gpu_root/models/sam3/bpe_simple_vocab_16e6.txt.gz"
  --tracker-checkpoint "$gpu_root/models/tracker/scaled_offline.pth"
  --gpu-lock "$selected_root/gpu.lock"
)
if [[ $# -gt 0 ]]; then
  arguments+=(--case "$1")
fi
exec "$gpu_root/env/pdi-bench/bin/python" -u -m pdi_eval.object_deformation_wrapper run "${arguments[@]}"
