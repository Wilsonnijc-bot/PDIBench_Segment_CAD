#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PDI_GPU_ROOT="${PDI_GPU_ROOT:-/root/autodl-tmp/pdi}"
export PDI_RUN_ROOT="${PDI_RUN_ROOT:-$PDI_GPU_ROOT/experiments/v1-persistent-10}"
export PYTHONPATH="$project_root/src:$project_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_root"
exec "${PDI_SAM_PYTHON:-$PDI_GPU_ROOT/env/qwen/bin/python}" \
  -m pdi_eval.experiment resume \
  --spec "$project_root/configs/experiment_v1_persistent_10.json"
