#!/usr/bin/env bash
# Run from scratch. No automatic promotion or previous-frame/point substitution.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$project_root/.env.vlm" ]]; then set -a; source "$project_root/.env.vlm"; set +a; fi
run_root="${1:?Supply a NEW absolute run directory}"
ffmpeg_binary="${FFMPEG_BINARY:?Set FFMPEG_BINARY}"
if [[ -e "$run_root" ]]; then
  echo "Preserve existing run: $run_root; supply a new path" >&2
  exit 1
fi
cd "$project_root"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
cases=(Cosmos3_0048 Cosmos25_0003 Cosmos25_0023 Cosmos25_0053)
example_root="${VLM2_EXAMPLE_ROOT:-$project_root/persistent_masking/vlm_interface/images}"
"$project_root/env-sam/bin/python" -u -m persistent_masking.naive_sam3 \
  --work "$run_root/naive" --cases "${cases[@]}" --ffmpeg "$ffmpeg_binary"
"$project_root/env-sam/bin/python" -u -m persistent_masking.pipeline continue \
  --work "$run_root" --cases "${cases[@]}" --ffmpeg "$ffmpeg_binary" \
  --examples "$example_root/reference_1.png" \
  "$example_root/reference_2.png" \
  "$example_root/reference_3.png"
