#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/pdi/simple3d-evaluation
export S3D_PAIR_ROOT="$PWD/results/simple3d-20261003-run2/link5_pair_visible_anchor"
continuation_log="$S3D_PAIR_ROOT/metadata/execution/continuation.log"
printf 'Full 45-video resume queued; waiting for the three-video producer.\n' >> "$continuation_log"
while tmux has-session -t simple3d-link5-anchor-preview-20261003 2>/dev/null; do
    sleep 15
done
if [[ "$(cat "$S3D_PAIR_ROOT/metadata/execution/run.exit")" != "0" ]]; then
    printf 'Preview failed; full campaign not started.\n' >> "$continuation_log"
    exit 1
fi
printf 'Preview succeeded. Resuming the remaining videos with two workers.\n' >> "$continuation_log"
unset S3D_PAIR_LIMIT
bash experiments/simple3d_pairs_gpu_job.sh
