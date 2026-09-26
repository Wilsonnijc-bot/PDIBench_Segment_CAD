#!/usr/bin/env bash
# One-case end-to-end GPU smoke test; never promotes diagnostic results.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="${1:?Supply a new absolute run directory}"
case="${2:-Cosmos25_0023}"
ffmpeg_binary="${FFMPEG_BINARY:?Set FFMPEG_BINARY}"
if [[ -e "$run_root" ]]; then
  echo "Preserve existing run: $run_root" >&2
  exit 1
fi
if [[ -f "$project_root/.env.vlm" ]]; then
  set -a
  source "$project_root/.env.vlm"
  set +a
fi
cd "$project_root"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
example_root="$project_root/persistent_masking/vlm_interface/images"
"$project_root/env-sam/bin/python" -u -m persistent_masking.naive_sam3 \
  --work "$run_root/naive" --cases "$case" --ffmpeg "$ffmpeg_binary"
"$project_root/env-sam/bin/python" -u -m persistent_masking.pipeline continue \
  --work "$run_root" --level local --cases "$case" --ffmpeg "$ffmpeg_binary" \
  --examples "$example_root/reference_1.png" \
  "$example_root/reference_2.png" \
  "$example_root/reference_3.png"
"$project_root/env-sam/bin/python" - "$run_root/provenance.json" "$case" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1]))
case = sys.argv[2]
result = record['results'][case]
if record['level'] != 'local' or result['status'] != 'completed_checks':
    raise SystemExit(f"SMOKE_FAIL {case}: {result['status']} {result.get('error', '')}")
print(f"SMOKE_PASS {case}: {result['status']}")
PY
