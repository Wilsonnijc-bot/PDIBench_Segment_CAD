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
# Complete one case at a time so frame lineage and failure state stay local.
set -euo pipefail
project_root="$(cd "${pdi_source_dir}/.." && pwd)"
if [[ -f "$project_root/.env.vlm" ]]; then set -a; source "$project_root/.env.vlm"; set +a; fi
run_root="${1:?Supply a new run directory}"
ffmpeg_binary="${FFMPEG_BINARY:?Set FFMPEG_BINARY}"
if [[ -e "$run_root" ]]; then echo "Preserve existing run: $run_root" >&2; exit 1; fi
cd "$project_root"; export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
example_root="${VLM2_EXAMPLE_ROOT:-$project_root/persistent_masking/vlm_interface/images}"
examples=("$example_root/reference_1.png" "$example_root/reference_2.png" "$example_root/reference_3.png")
for example in "${examples[@]}"; do
  [[ -f "$example" ]] || { echo "Missing VLM2 example: $example" >&2; exit 2; }
done
cases=(Cosmos3_0006 Cosmos3_0014 Cosmos3_0028 Cosmos3_0048 Cosmos3_0049 Cosmos25_0027 Cosmos25_0032 Cosmos25_0047 LVP_0042 LVP_0051)
for case in "${cases[@]}"; do
  case_root="$run_root/$case"
  mkdir -p "$case_root"
  "$project_root/env-sam/bin/python" -u -m persistent_masking.naive_sam3 --work "$case_root/naive" --cases "$case" --ffmpeg "$ffmpeg_binary"
  "$project_root/env-sam/bin/python" -u -m persistent_masking.pipeline continue \
    --work "$case_root" --cases "$case" \
    --reference-root "$project_root/lasteset_qwen_results/experiments/global/global_combined_lower_dark_20260912" \
    --examples "${examples[@]}" --ffmpeg "$ffmpeg_binary"
done
