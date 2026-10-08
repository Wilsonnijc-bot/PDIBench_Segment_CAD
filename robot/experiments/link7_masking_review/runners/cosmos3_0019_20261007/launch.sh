#!/usr/bin/env bash
# Immutable single-video masking allocation; workers and GPU count are separate.
#SBATCH --account=wang
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --job-name=cosmos3-0019-mask-replay
#SBATCH --output=/PHShome/zy992/Wilson/deformationdetection/metadata/cosmos3-0019-seed101-masking-20261007/slurm-%j.log
set -Eeuo pipefail
metadata=/PHShome/zy992/Wilson/deformationdetection/metadata/cosmos3-0019-seed101-masking-20261007
socket=cosmos3-mask-$SLURM_JOB_ID
session=cosmos3-0019-masking-$SLURM_JOB_ID
telemetry_pid=
cleanup() {
    rc=$?
    trap - EXIT TERM INT HUP
    tmux -L "$socket" kill-session -t "$session" 2>/dev/null || true
    if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; fi
    if [[ ! -f "$metadata/job.exit" ]]; then printf '%s\n' "$rc" > "$metadata/job.exit"; fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM HUP
trap 'exit 130' INT
command -v tmux
tmux -L "$socket" has-session -t "$session" 2>/dev/null && exit 2
printf 'GPUS=1 VIDEO_WORKERS=1 GPU_STAGE_SLOTS=1 CASES=1\n'
printf '%s\n' "$session" > "$metadata/tmux-session.txt"
printf '%s\n' "$socket" > "$metadata/tmux-socket.txt"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv -l 2 > "$metadata/gpu-telemetry.csv" &
telemetry_pid=$!
tmux -L "$socket" new-session -d -s "$session" "bash '$metadata/worker.sh' > '$metadata/job.log' 2>&1"
while tmux -L "$socket" has-session -t "$session" 2>/dev/null; do sleep 5; done
test -s "$metadata/job.exit"
exit "$(cat "$metadata/job.exit")"
