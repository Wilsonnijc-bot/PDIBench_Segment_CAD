#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDITED_ROOT="$PROJECT_ROOT/PDI-Bench-edited"
GPU_ENV_FILE="${PDI_GPU_ENV_FILE:-$PROJECT_ROOT/configs/gpu.local.env}"
if [[ $# -eq 0 ]]; then
  videos=(0000.mp4 0001.mp4)
else
  videos=("$@")
fi

if [[ ${#videos[@]} -ne 2 ]]; then
  echo "Usage: $0 VIDEO_1.mp4 VIDEO_2.mp4" >&2
  exit 2
fi

source "$GPU_ENV_FILE"
IDENTITY_FILE="$PROJECT_ROOT/$PDI_GPU_IDENTITY_FILE"
SSH=(ssh -i "$IDENTITY_FILE" -o BatchMode=yes -o StrictHostKeyChecking=yes -p "$PDI_GPU_PORT" "$PDI_GPU_USER@$PDI_GPU_HOST")
RSYNC_SSH="ssh -i $IDENTITY_FILE -o BatchMode=yes -o StrictHostKeyChecking=yes -p $PDI_GPU_PORT"
REMOTE_CODE="$PDI_GPU_ROOT/code/PDI-Bench-edited"
REMOTE_GEOMETRY_CACHE="$PDI_GPU_ROOT/cache/megasam-video-geometry-v3"

video_path() {
  local name="$1"
  local path="$PROJECT_ROOT/.tmp/COSMOS_Videos/$name"
  if [[ ! -s "$path" ]]; then
    path="$PROJECT_ROOT/.tmp/COSMOS2.5_Videos/$name"
  fi
  test -s "$path"
  printf '%s\n' "$path"
}

segmentation_path() {
  local stem="${1%.mp4}"
  local matches=("$PROJECT_ROOT"/results/dinov2-sam3/"$stem"-*-refined-text)
  if [[ ${#matches[@]} -ne 1 || ! -s "${matches[0]}/segmentation.npz" ]]; then
    echo "Expected one refined-text segmentation for $1" >&2
    return 2
  fi
  printf '%s\n' "${matches[0]}/segmentation.npz"
}

"${SSH[@]}" "mkdir -p '$REMOTE_CODE' '$PDI_GPU_ROOT/runs' '$REMOTE_GEOMETRY_CACHE'"
rsync -az --delete \
  --exclude '.git/' --exclude '*.pth' --exclude '*.pt' --exclude '*.so' \
  --exclude 'build/' --exclude 'checkpoints/' \
  --exclude 'third_party/mega_sam/base/' \
  --exclude 'third_party/mega_sam/work_space/' \
  --exclude 'third_party/mega_sam/outputs/' \
  --exclude 'third_party/mega_sam/outputs_cvd/' \
  --exclude 'third_party/mega_sam/torchhub/' \
  -e "$RSYNC_SSH" "$EDITED_ROOT/" \
  "$PDI_GPU_USER@$PDI_GPU_HOST:$REMOTE_CODE/"

if [[ "${PDI_SKIP_BOOTSTRAP:-1}" != "1" ]]; then
  rsync -az -e "$RSYNC_SSH" "$PROJECT_ROOT/scripts/bootstrap_gpu.sh" \
    "$PDI_GPU_USER@$PDI_GPU_HOST:$PDI_GPU_ROOT/code/bootstrap_gpu.sh"
  "${SSH[@]}" "PDI_GPU_ROOT='$PDI_GPU_ROOT' bash '$PDI_GPU_ROOT/code/bootstrap_gpu.sh'"
fi

declare -a remote_works local_outputs
for index in "${!videos[@]}"; do
  name="${videos[$index]}"
  stem="${name%.mp4}"
  local_video="$(video_path "$name")"
  local_segmentation="$(segmentation_path "$name")"
  segmentation_sha="$(shasum -a 256 "$local_segmentation" | awk '{print $1}')"
  run_id="$stem-${segmentation_sha:0:12}-link7-exact-group"
  remote_work="$PDI_GPU_ROOT/runs/pdi-$run_id"
  local_output="$PROJECT_ROOT/results/pdi-link7-exact-group-parallel/$run_id"
  remote_works[$index]="$remote_work"
  local_outputs[$index]="$local_output"

  mkdir -p "$local_output"
  "${SSH[@]}" "mkdir -p '$remote_work/input' '$remote_work/frozen' '$remote_work/output'"
  rsync -az -e "$RSYNC_SSH" "$local_video" \
    "$PDI_GPU_USER@$PDI_GPU_HOST:$remote_work/input/$name"
  rsync -az -e "$RSYNC_SSH" "$local_segmentation" \
    "$PDI_GPU_USER@$PDI_GPU_HOST:$remote_work/input/segmentation.npz"

  local_sha="$(shasum -a 256 "$local_video" | awk '{print $1}')"
  remote_sha="$("${SSH[@]}" "sha256sum '$remote_work/input/$name' | cut -d' ' -f1")"
  if [[ "$local_sha" != "$remote_sha" ]]; then
    echo "Input transfer verification failed for $name" >&2
    exit 1
  fi
done

run_one() {
  local index="$1"
  local name="${videos[$index]}"
  local remote_work="${remote_works[$index]}"
  local local_output="${local_outputs[$index]}"
  local log_path="$local_output/console.log"

  set +e
  "${SSH[@]}" "source /root/miniconda3/etc/profile.d/conda.sh && conda activate '$PDI_GPU_ROOT/env/pdi-bench' && export PYTHONPATH='$REMOTE_CODE/src' HF_ENDPOINT='${HF_ENDPOINT:-https://hf-mirror.com}' && cd '$REMOTE_CODE' && python evaluation/prepare_abcd.py --video '$remote_work/input/$name' --segmentation '$remote_work/input/segmentation.npz' --output-dir '$remote_work/frozen' --selected-object link7 && python evaluation/run_multi_object.py --config '$REMOTE_CODE/configs/default.yaml' --input '$remote_work/input/$name' --segmentation-npz '$remote_work/frozen/single_link_segmentation.npz' --output-dir '$remote_work/output' --geometry-cache-dir '$REMOTE_GEOMETRY_CACHE' --tracker-checkpoint '$PDI_GPU_ROOT/models/tracker/scaled_offline.pth' --tracking-mode exact-group" 2>&1 | tee "$log_path"
  local remote_status=${PIPESTATUS[0]}
  set -e

  rsync -az -e "$RSYNC_SSH" \
    "$PDI_GPU_USER@$PDI_GPU_HOST:$remote_work/output/" "$local_output/"
  if [[ $remote_status -ne 0 ]]; then
    return "$remote_status"
  fi
  test -s "$local_output/metrics.json"
  test -s "$local_output/cotracker_exact-group.npz"
  test -s "$local_output/replay/combined_exact-group.mp4"
}

pids=()
for index in "${!videos[@]}"; do
  echo "Starting exact-group for ${videos[$index]}"
  run_one "$index" &
  pids+=("$!")
done

failed=()
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Completed ${videos[$index]}: ${local_outputs[$index]}"
  else
    failed+=("${videos[$index]}")
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed exact-group videos: ${failed[*]}" >&2
  exit 1
fi
echo "Both exact-group video runs completed in parallel"
