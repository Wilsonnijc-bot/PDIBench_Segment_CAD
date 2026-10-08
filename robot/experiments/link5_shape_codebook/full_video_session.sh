#!/usr/bin/env bash
set -Eeuo pipefail
workspace=${1:?workspace}; dependency=${2:?dependency}; config=${3:?config}
test -n "${SLURM_JOB_ID:-}"
module load tmux/3.5a-GCCcore-13.3.0
metadata="$workspace/results/link5_shape_codebook/full_video/metadata";mkdir -p "$metadata"
snapshot="$metadata/job-scripts/${SLURM_JOB_ID}";mkdir -p "$snapshot"
cp "$workspace/robot/experiments/link5_shape_codebook/run_full_video_job.sh" "$snapshot/run_full_video_job.sh"
chmod a-w "$snapshot/run_full_video_job.sh"
session="link5-full-video-${SLURM_JOB_ID}"
printf -v command 'bash %q %q %q %q > %q 2>&1' "$snapshot/run_full_video_job.sh" "$workspace" "$dependency" "$config" "$metadata/full-video-${SLURM_JOB_ID}.log"
tmux -L "$session" new-session -d -s "$session" "$command"
printf '%s\n' "$session" > "$metadata/session-${SLURM_JOB_ID}.txt"
while tmux -L "$session" has-session -t "$session" 2>/dev/null; do sleep 10; done
test -s "$metadata/full-video-${SLURM_JOB_ID}.exit"
rc="$(cat "$metadata/full-video-${SLURM_JOB_ID}.exit")"
if test "$rc" = 0; then
    grep -Fxq LINK5_FULL_VIDEO_JOB_COMPLETE "$metadata/full-video-${SLURM_JOB_ID}.log" || exit 3
fi
exit "$rc"
