#!/usr/bin/env bash
# A single resumable selected-45 batch, launched inside a unique remote tmux session.
set -Eeuo pipefail
if [[ $# -lt 2 ]]; then
  echo 'usage: run_link5_link7_batch.sh BATCH_ID SAMPLE [SAMPLE...]' >&2
  exit 2
fi
batch_id="$1"
shift
case "$batch_id" in
  *[!A-Za-z0-9_-]*|'') echo 'invalid batch ID' >&2; exit 2;;
esac
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root=/root/autodl-tmp/pdi/experiments/link5-link7-four-way-selected45-20260927
log_dir="$run_root/batches"
mkdir -p "$log_dir"
log="$log_dir/$batch_id.log"
exit_marker="$log_dir/$batch_id.exit"
if [[ -e "$exit_marker" ]]; then
  echo "batch exit marker exists: $exit_marker" >&2
  exit 2
fi
cd "$project_root"
sample_args=()
for sample in "$@"; do
  sample_args+=(--sample "$sample")
done
set +e
PDI_RUN_ROOT="$run_root" PYTHONPATH=src:. \
  /root/autodl-tmp/pdi/env/pdi-bench/bin/python -u -m pdi_eval.experiment resume \
  --spec configs/experiment_link5_link7_four_way_45.json "${sample_args[@]}" \
  2>&1 | tee "$log"
rc="${PIPESTATUS[0]}"
set -e
printf 'EXIT_STATUS=%s\n' "$rc" | tee -a "$log"
printf '%s\n' "$rc" > "$exit_marker"
exit "$rc"
