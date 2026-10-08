#!/usr/bin/env bash
# CPU-only dependent job; inference outputs and original masks are read-only inputs.
set -Euo pipefail
workspace=${1:?workspace}; dependency=${2:?dependency}; cache=${3:?cache}; sources=${4:?source videos}
cd "$workspace"
root="$workspace/results/link5_pair_selection_ablation_v1"
metadata="$root/metadata/depth-score-report-${SLURM_JOB_ID:?Slurm allocation required}"
/usr/bin/mkdir -p "$metadata"
trap 'rc=$?; printf "%s\n" "$rc" > "$metadata/job.exit"' EXIT
export PYTHONPATH="$workspace" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$dependency/env/compiler/lib:$dependency/env/geometry/lib:$dependency/env/geometry/lib64:${LD_LIBRARY_PATH:-}"
python="$dependency/env/geometry/bin/python"
"$python" - "$metadata/runtime.json" <<'PY'
import json,platform,sys,numpy,scipy,cv2
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,opencv=cv2.__version__,hostname=platform.node()),indent=2))
PY
"$python" -u -m robot.experiments.link5_pair_selection_ablation.filter_video_depth --root "$root" --cache "$cache" --workers 5
filter_rc=$?
"$python" -u -m robot.experiments.link5_pair_selection_ablation.run_scores_depth_v2 --manifest "$root/manifest.json" --inputs "$root/shared_inputs" --support "$root/depth_support" --output "$root/rigidity" --workers 5
score_rc=$?
"$python" -u -m robot.experiments.link5_pair_selection_ablation.build_report_v2 --root "$root" --sources "$sources"
report_rc=$?
printf 'filter=%s score=%s report=%s\n' "$filter_rc" "$score_rc" "$report_rc" > "$metadata/stages.exit"
if test "$filter_rc" = 0 && test "$score_rc" = 0 && test "$report_rc" = 0; then
    printf '%s\n' LINK5_DEPTH_FILTERED_REPORT_COMPLETE
    exit 0
fi
exit 1
