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
set -uo pipefail

project_root="$(cd "${pdi_source_dir}/.." && pwd)"
gpu_root="${PDI_GPU_ROOT:-/root/autodl-tmp/pdi}"
run_root="${PDI_RUN_ROOT:-$gpu_root/experiments/v1-persistent-10}"
log="$run_root/two_workers.log"
mkdir -p "$run_root"
cd "$project_root" || exit 1

{
  echo "Checking the V1 scoring environment."
  test -s "$gpu_root/models/tracker/scaled_offline.pth" || exit 31
  test -s "$gpu_root/models/depth_anything/depth_anything_vitl14.pth" || exit 32
  test -s "$gpu_root/models/raft/raft-things.pth" || exit 33
  "$gpu_root/env/pdi-bench/bin/python" - <<'PY' || exit 34
import torch
from cotracker.predictor import CoTrackerPredictor
from torch_scatter import scatter_sum
from lietorch import SE3
import droid_backends
assert torch.cuda.is_available()
assert (torch.ones(1, device='cuda') + 1).item() == 2
print('PDI preflight OK:', torch.__version__, torch.cuda.get_device_name(0), flush=True)
PY
  PYTHONPATH="$project_root/src" "$gpu_root/env/pdi-bench/bin/python" -m pdi_eval.experiment score --help >/dev/null || exit 35
  "$project_root/scripts/run_v1_replay_two.sh"
} 2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
printf 'EXIT_STATUS=%s\n' "$status" | tee -a "$log"
exit "$status"
