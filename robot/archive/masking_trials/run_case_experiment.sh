#!/usr/bin/env bash
# BEGIN PDI SOURCE LOCATION
pdi_source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pdi_checkout="$pdi_source_dir"
while [[ "$pdi_checkout" != / && ! -f "$pdi_checkout/documentation/architecture/layout.json" ]]; do
  pdi_checkout="$(dirname "$pdi_checkout")"
done
if [[ -f "$pdi_checkout/documentation/architecture/layout.json" ]]; then
  pdi_source_dir="$pdi_checkout/infrastructure/compat/PDI-Bench-edited/persistent_masking"
fi
# END PDI SOURCE LOCATION
# One-case local experiment producing the six files shown in sep22 run/gemini/Cosmos3_0048.
set -euo pipefail

project_root="$(cd "${pdi_source_dir}/.." && pwd)"
case_name="${1:-Cosmos3_0048}"
[[ "$case_name" =~ ^[A-Za-z0-9_]+$ ]] || { echo 'Invalid case name' >&2; exit 2; }
run_name="${2:?Usage: run_case_experiment.sh CASE RUN_NAME NEW_REVIEW_DIR}"
review_root="${3:?Usage: run_case_experiment.sh CASE RUN_NAME NEW_REVIEW_DIR}"
[[ "$run_name" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'RUN_NAME must contain only letters, digits, _ or -' >&2; exit 2; }
run_root="$project_root/lasteset_qwen_results/experiments/local/$run_name/$case_name"
ffmpeg_binary="${FFMPEG_BINARY:?Set FFMPEG_BINARY to an ffmpeg executable}"
[[ "$review_root" = /* ]] || { echo 'Use an absolute review path' >&2; exit 2; }
[[ ! -e "$run_root" && ! -e "$review_root" ]] || { echo 'Work and review paths must both be new' >&2; exit 2; }
[[ -x "$ffmpeg_binary" ]] || { echo "FFmpeg is not executable: $ffmpeg_binary" >&2; exit 2; }

if [[ -f "$project_root/.env.vlm" ]]; then
  set -a
  source "$project_root/.env.vlm"
  set +a
fi
cd "$project_root"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
example_root="${VLM2_EXAMPLE_ROOT:-$project_root/persistent_masking/vlm_interface/images}"
examples=("$example_root/reference_1.png" "$example_root/reference_2.png" "$example_root/reference_3.png")
for path in "${examples[@]}"; do [[ -f "$path" ]] || { echo "Missing example: $path" >&2; exit 2; }; done

"$project_root/env-sam/bin/python" -u -m persistent_masking.naive_sam3 \
  --work "$run_root/naive" --cases "$case_name" --ffmpeg "$ffmpeg_binary"
"$project_root/env-sam/bin/python" -u -m persistent_masking.pipeline continue \
  --level local --work "$run_root" --cases "$case_name" \
  --examples "${examples[@]}" --ffmpeg "$ffmpeg_binary"

# The pipeline can exit zero while recording a failed case; never export that as a success.
"$project_root/env-sam/bin/python" - "$run_root/provenance.json" "$case_name" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1]))
case = sys.argv[2]
status = record['results'][case]['status']
if status != 'completed_checks':
    raise SystemExit(f'{case}: no validated replay (status={status}); see {sys.argv[1]}')
PY
"$project_root/env-sam/bin/python" -m persistent_masking.export_selected_replay \
  --work "$run_root" --destination "$review_root"
echo "Review files: $review_root/$case_name"
