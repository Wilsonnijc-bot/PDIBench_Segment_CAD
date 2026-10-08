#!/usr/bin/env bash
set -Eeuo pipefail
workspace=${1:?workspace}; dependency=${2:?dependency}; cache=${3:?cache}; sources=${4:?sources}; gpu_job=${5:?GPU job}
cd "$workspace"
root="$workspace/results/link5_pair_selection_ablation_v1"
metadata="$root/metadata/depth-score-report-${SLURM_JOB_ID:?Slurm required}"
/usr/bin/mkdir -p "$metadata"
trap 'rc=$?; printf "%s\n" "$rc" > "$metadata/job.exit"' EXIT
export PYTHONPATH="$workspace" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$dependency/env/compiler/lib:$dependency/env/geometry/lib:$dependency/env/geometry/lib64:${LD_LIBRARY_PATH:-}"
python="$dependency/env/geometry/bin/python"
"$python" - "$metadata/runtime.json" <<'PY'
import json,platform,sys,numpy,scipy,cv2
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,opencv=cv2.__version__,hostname=platform.node()),indent=2))
PY
"$python" -u -m robot.experiments.link5_pair_selection_ablation.stream_scores_depth --root "$root" --cache "$cache" --sources "$sources" --gpu-job "$gpu_job" --workers 5
