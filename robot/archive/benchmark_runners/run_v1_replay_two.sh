#!/usr/bin/env bash
# BEGIN PDI SOURCE LOCATION
pdi_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pdi_checkout="$pdi_source_dir"
while [[ "$pdi_checkout" != / && ! -f "$pdi_checkout/documentation/architecture/layout.json" ]]; do
  pdi_checkout="$(dirname "$pdi_checkout")"
done
if [[ -f "$pdi_checkout/documentation/architecture/layout.json" ]]; then
  pdi_source_dir="$pdi_checkout/infrastructure/compat/PDI-Bench-edited/scripts"
fi
# END PDI SOURCE LOCATION
set -euo pipefail

project_root="$(cd "${pdi_source_dir}/.." && pwd)"
export PDI_GPU_ROOT="${PDI_GPU_ROOT:-/root/autodl-tmp/pdi}"
export PDI_RUN_ROOT="${PDI_RUN_ROOT:-$PDI_GPU_ROOT/experiments/v1-persistent-10}"
export PYTHONPATH="$project_root/src:$project_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_root"
exec "${PDI_SAM_PYTHON:-$PDI_GPU_ROOT/env/qwen/bin/python}" \
  -m robot.workflows resume \
  --spec "$project_root/configs/experiment_v1_persistent_10.json"
