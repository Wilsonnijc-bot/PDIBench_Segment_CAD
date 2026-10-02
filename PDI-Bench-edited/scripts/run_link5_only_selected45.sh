#!/usr/bin/env bash
# One native selected-45 run with two case workers in a detached tmux session.
set -Eeuo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root=/root/autodl-tmp/pdi/experiments/link5-only-selected45-updated-mask-20260929
log="$run_root/run.log"
exit_marker="$run_root/run.exit"
if [[ -e "$exit_marker" ]]; then
  echo "run exit marker already exists: $exit_marker" >&2
  exit 2
fi
cd "$project_root"
printf 'STARTING_LINK5_WITH_TWO_WORKERS\n' | tee -a "$log"
set +e
PDI_RUN_ROOT="$run_root" PYTHONPATH=src:. \
  /root/autodl-tmp/pdi/env/pdi-bench/bin/python -u -m pdi_eval.experiment run \
  --spec configs/experiment_link5_only_selected45.json 2>&1 | tee -a "$log"
rc="${PIPESTATUS[0]}"
set -e
printf 'EXIT_STATUS=%s\n' "$rc" | tee -a "$log"
printf '%s\n' "$rc" > "$exit_marker"
exit "$rc"
