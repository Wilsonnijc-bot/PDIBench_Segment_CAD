#!/usr/bin/env bash
# Queue with --dependency=afterany:<current allocation>, keeping one active GPU.
set -Eeuo pipefail
workspace=${1:?workspace}; dependency=${2:?dependency}; cache=${3:?mask cache}
sources=${4:?source videos}; queries=${5:?saved initialized queries}
test -n "${SLURM_JOB_ID:-}"
cd "$workspace"
root="$workspace/results/link5_pair_selection_ablation_v1"
# Preserve interrupted raw writes and old allocation receipts before resuming.
/usr/bin/python3 - "$root" "$SLURM_JOB_ID" <<'PY'
import json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]);job=sys.argv[2]
archive=root/'metadata'/('interrupted-before-'+job);archive.mkdir(exist_ok=True)
for name in ('preflight.json','preparation-progress.json','preparation-exits.json'):
    path=root/'metadata'/name
    if path.exists():shutil.copy2(path,archive/name)
for entry in json.loads((root/'manifest.json').read_text())['entries']:
    case=entry['video_id'];receipt=root/'shared_inputs'/(case+'.json')
    for raw in (root/'shared_inputs'/(case+'.partial.npz'),root/'shared_inputs'/(case+'.npz')):
        if raw.exists() and (raw.name.endswith('.partial.npz') or not receipt.exists()):
            target=archive/raw.name
            if target.exists():raise RuntimeError('Preserved raw destination already exists')
            raw.rename(target)
PY
bash "$workspace/robot/experiments/link5_pair_selection_ablation/eris_job_v3.sh" "$workspace" "$dependency" \
  --manifest "$root/manifest.json" --cache "$cache" --sources "$sources" --queries "$queries" \
  --output "$root/shared_inputs" --scratch "$root/scratch-continuation-$SLURM_JOB_ID" \
  --mega-root "$dependency/source/infrastructure/vendor/mega_sam" \
  --tracker-checkpoint "$dependency/models/tracker_checkpoint/scaled_offline.pth" --workers 8
