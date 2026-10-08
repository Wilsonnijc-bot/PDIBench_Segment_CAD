#!/usr/bin/env bash
# An allocation-local tmux server; Slurm owns resources until its session exits.
set -Eeuo pipefail
workspace=${1:?workspace}; dependency=${2:?dependency}; config=${3:?config}; kind=${4:?prepare, dependencies, pose, train, validate, score or detector}
test -n "${SLURM_JOB_ID:-}"
module load tmux/3.5a-GCCcore-13.3.0
metadata="$workspace/results/link5_shape_codebook/metadata";mkdir -p "$metadata"
snapshot="$metadata/job-scripts/${kind}-${SLURM_JOB_ID}";mkdir -p "$snapshot"
socket="link5-${kind}-${SLURM_JOB_ID}";session="link5-${kind}-${SLURM_JOB_ID}"
test ! -f "$metadata/${kind}-${SLURM_JOB_ID}.exit"
if test "$kind" = prepare || test "$kind" = references; then
    script=eris_prepare_job.sh
    marker=LINK5_PREPARATION_COMPLETE
    exitfile="$metadata/preparation-${SLURM_JOB_ID}.exit"
    if test "$kind" = references; then marker=LINK5_FOUR_FRAME0_REFERENCE_COMPLETE;exitfile="$metadata/references-${SLURM_JOB_ID}.exit";fi
elif test "$kind" = round; then
    script=run_refined_round.sh
    marker=LINK5_REFINED_ROUND_COMPLETE
    exitfile="$metadata/round-${SLURM_JOB_ID}.exit"
elif test "$kind" = dependencies; then
    script=install_eris_dependencies.sh
    marker=LINK5_DEPENDENCIES_COMPLETE
    exitfile="$dependency/metadata/link5-dependencies-${SLURM_JOB_ID}.exit"
elif [[ "$kind" =~ ^(detector|pose_iteration_probe|pose_preflight|pose|train|validate|score)(_robot_structural)?$ ]]; then
    script=run_detector_job.sh
    marker=LINK5_STAGE_${kind}_FINISHED
    exitfile="$metadata/${kind}-${SLURM_JOB_ID}.exit"
else exit 2; fi
cp "$workspace/robot/experiments/link5_shape_codebook/$script" "$snapshot/$script"
chmod a-w "$snapshot/$script"
if test "$kind" = dependencies; then
    printf -v command 'bash %q %q %q' "$snapshot/$script" "$dependency" "$workspace"
else
    printf -v command 'bash %q %q %q %q %q' "$snapshot/$script" "$workspace" "$dependency" "$config" "$kind"
fi
printf -v wrapped '%s > %q 2>&1' "$command" "$metadata/${kind}-${SLURM_JOB_ID}.log"
tmux -L "$socket" new-session -d -s "$session" "$wrapped"
printf '%s\n' "$session" > "$metadata/${kind}-${SLURM_JOB_ID}.tmux"
while tmux -L "$socket" has-session -t "$session" 2>/dev/null; do sleep 10; done
test -s "$exitfile"
rc="$(cat "$exitfile")"
if test "$rc" = 0; then
    grep -Fxq "$marker" "$metadata/${kind}-${SLURM_JOB_ID}.log" || { echo 'Incomplete job: exit 0 without completion marker' >&2;exit 3; }
fi
exit "$rc"
