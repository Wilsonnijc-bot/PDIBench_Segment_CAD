#!/usr/bin/env bash
# Submit to Slurm with --partition=gpu-h200 --account=wang --gres=gpu:1.
# Worker count is five; GPU allocation is one. All paths must be remote paths.
set -Eeuo pipefail
workspace=${1:?isolated workspace}; dependency=${2:?dependency root}; shift 2
test -n "${SLURM_JOB_ID:-}" || { echo 'Slurm allocation required' >&2; exit 2; }
cd "$workspace"
export PYTHONPATH="$workspace/infrastructure/deformation_detect/import_guard:$workspace" PYTHONNOUSERSITE=1
export PATH="$dependency/bin:$dependency/env/geometry/bin:$PATH"
export LD_LIBRARY_PATH="$dependency/env/compiler/lib:$dependency/env/geometry/lib:$dependency/env/geometry/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME="$dependency/cache/huggingface" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
metadata="$workspace/results/link5_pair_selection_ablation_v1/metadata/allocation-$SLURM_JOB_ID-short"
mkdir -p "$metadata"
python="$dependency/env/geometry/bin/python"
if test "${1:-}" = --inside; then
    shift
    trap 'rc=$?; printf "%s\n" "$rc" > "$metadata/job.exit"' EXIT
    "$python" -u -m robot.experiments.link5_pair_selection_ablation.run_short_batch "$@" --preflight-only
    "$python" -u -m robot.experiments.link5_pair_selection_ablation.run_short_batch "$@" &
    worker=$!
    printf '%s\n' "$worker" > "$metadata/worker.pid"
    trap 'kill -TERM "$worker" 2>/dev/null || true; wait "$worker" || true; exit 130' TERM INT HUP
    wait "$worker"
    exit 0
fi
module load tmux/3.5a-GCCcore-13.3.0
socket="link5-pairs-short-$SLURM_JOB_ID"; session="$socket"
test ! -e "$metadata/job.exit"
tmux -L "$socket" has-session -t "$session" 2>/dev/null && exit 2
cp "$workspace/robot/experiments/link5_pair_selection_ablation/eris_short_job.sh" "$metadata/eris_job.sh"
chmod a-w "$metadata/eris_job.sh"
printf -v command 'bash %q %q %q --inside' "$metadata/eris_job.sh" "$workspace" "$dependency"
for arg in "$@"; do printf -v quoted ' %q' "$arg"; command+="$quoted"; done
printf -v quoted ' > %q 2>&1' "$metadata/job.log"; command+="$quoted"
tmux -L "$socket" new-session -d -s "$session" "$command"
cancel() {
    if test -f "$metadata/worker.pid"; then kill -TERM "$(cat "$metadata/worker.pid")" 2>/dev/null || true; fi
    exit 130
}
trap cancel TERM INT HUP
while tmux -L "$socket" has-session -t "$session" 2>/dev/null; do sleep 5; done
test -s "$metadata/job.exit"
rc=$(cat "$metadata/job.exit")
if test "$rc" = 0; then
    grep -Fxq LINK5_SHORT_VIDEO_STAGE_COMPLETE "$metadata/job.log"
fi
exit "$rc"
