#!/usr/bin/env bash
# BEGIN PDI SOURCE LOCATION
pdi_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pdi_checkout="$pdi_source_dir"
while [[ "$pdi_checkout" != / && ! -f "$pdi_checkout/documentation/architecture/layout.json" ]]; do
  pdi_checkout="$(dirname "$pdi_checkout")"
done
if [[ -f "$pdi_checkout/documentation/architecture/layout.json" ]]; then
  pdi_source_dir="$pdi_checkout/infrastructure/compat/PDI-Bench-edited/scripts"
fi
# END PDI SOURCE LOCATION
# Run the remaining selected-45 groups in one tmux session, one group at a time.
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'usage: run_link5_link7_remaining.sh GROUP_MANIFEST [RUN_LABEL]' >&2
  exit 2
fi
manifest="$1"
run_label="${2:-remaining-v1}"
[[ "$run_label" =~ ^remaining-v[12]$ ]] || {
  echo "invalid run label: $run_label" >&2; exit 2;
}
project_root="$(cd "${pdi_source_dir}/.." && pwd)"
run_root=/root/autodl-tmp/pdi/experiments/link5-link7-four-way-selected45-20260927
control="$run_root/control"
batches="$run_root/batches"
overall_exit="$batches/$run_label.exit"
mkdir -p "$control" "$batches"
if [[ -e "$overall_exit" ]]; then
  echo "overall exit marker already exists: $overall_exit" >&2
  exit 2
fi
on_exit() {
  rc=$?
  printf '%s\n' "$rc" > "$overall_exit"
  printf 'EXIT_STATUS=%s\n' "$rc"
}
trap on_exit EXIT
cd "$project_root"
test -s "$manifest"

wait_for_signal() {
  local signal="$1"
  while [[ ! -f "$signal" ]]; do
    sleep 15
  done
}

mapfile -t group_lines < "$manifest"
for line in "${group_lines[@]}"; do
  read -r -a fields <<< "$line"
  [[ ${#fields[@]} -ge 2 && ${#fields[@]} -le 5 ]] || {
    echo "invalid group manifest line: $line" >&2; exit 2;
  }
  batch_id="${fields[0]}"
  [[ "$batch_id" =~ ^b(20|24|28|32|36|40|44|45)$ ]] || {
    echo "invalid group ID: $batch_id" >&2; exit 2;
  }
  samples=("${fields[@]:1}")
  sample_args=()
  for sample in "${samples[@]}"; do
    [[ "$sample" =~ ^(COSMOS2\.5|COSMOS3|LVP_ROBOWM)_[0-9]{4}$ ]] || {
      echo "invalid sample ID: $sample" >&2; exit 2;
    }
    sample_args+=(--sample "$sample")
  done
  marker="$batches/$batch_id.exit"
  log="$batches/$batch_id.log"
  if [[ -e "$marker" ]]; then
    if [[ "$(cat "$marker")" == 0 ]]; then
      echo "already completed $batch_id; continuing"
      continue
    fi
    echo "failed group marker exists: $marker" >&2
    exit 2
  fi
  echo "waiting for staged $batch_id: ${samples[*]}"
  wait_for_signal "$control/$batch_id.ready"
  echo "running $batch_id: ${samples[*]}"
  set +e
  PDI_RUN_ROOT="$run_root" PYTHONPATH=src:. \
    /root/autodl-tmp/pdi/env/pdi-bench/bin/python -u -m pdi_eval.experiment resume \
      --spec configs/experiment_link5_link7_four_way_45.json "${sample_args[@]}" \
      2>&1 | tee "$log"
  rc="${PIPESTATUS[0]}"
  set -e
  printf 'EXIT_STATUS=%s\n' "$rc" >> "$log"
  printf '%s\n' "$rc" > "$marker"
  if [[ "$rc" -ne 0 ]]; then
    echo "$batch_id exited $rc" >&2
    exit "$rc"
  fi
done
echo 'all remaining groups complete on GPU'
