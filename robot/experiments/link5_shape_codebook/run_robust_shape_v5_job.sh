#!/usr/bin/env bash
set -Eeuo pipefail
task_workspace=/PHShome/zy992/Wilson/deformationdetection/workspace-link5-shape-codebook-four-frame0-20261006
task_dependency=/PHShome/zy992/Wilson/dependency
export HOSTNAME="$(hostname)"
module load tmux/3.5a-GCCcore-13.3.0
task_metadata="$task_workspace/results/link5_shape_codebook/metadata/robust-shape-v5-$SLURM_JOB_ID"
mkdir -p "$task_metadata"
cp "$task_workspace/robot/experiments/link5_shape_codebook/run_robust_shape_v5_job.sh" "$task_metadata/launcher.sh"
for task_source in fine_tune_structural_v5.py benign_variation_v4.py structural_invariant_head.py structural_geometry_head.py evaluate_structural_guides.py evaluate_calibrated_structural.py; do
  cp "$task_workspace/robot/experiments/link5_shape_codebook/$task_source" "$task_metadata/"
done
cat > "$task_metadata/worker.sh" <<'RUN'
#!/usr/bin/env bash
set -Eeuo pipefail
task_metadata=${1:?};task_workspace=${2:?};task_dependency=${3:?};task_variant=${4:?}
trap 'rc=$?; printf "%s\n" "$rc" > "$task_metadata/$task_variant.exit"' EXIT
cd "$task_workspace"
export PYTHONPATH="$task_workspace/infrastructure/deformation_detect/import_guard:$task_workspace" PYTHONNOUSERSITE=1
export PATH="$task_dependency/bin:$task_dependency/env/geometry/bin:$PATH"
export LD_LIBRARY_PATH="$task_dependency/env/foundationpose/lib:$task_dependency/env/shape-build/lib:$task_dependency/env/compiler/lib:$task_dependency/env/geometry/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
"$task_dependency/env/shape-codebook/bin/python" -u "$task_metadata/fine_tune_structural_v5.py" --config "$task_workspace/results/link5_shape_codebook/config.json" --warm-checkpoint "$task_workspace/results/link5_shape_codebook/structural_localization/benign_finetune_v4/structural_geometry_head/final.pt" --epochs 3000 --variant "$task_variant" --resume
"$task_dependency/env/shape-codebook/bin/python" -u "$task_metadata/evaluate_structural_guides.py" --config "$task_workspace/results/link5_shape_codebook/config.json" --trial robust_shape_v5 --variant "$task_variant" --epoch 3000
"$task_dependency/env/shape-codebook/bin/python" -u "$task_metadata/evaluate_calibrated_structural.py" --config "$task_workspace/results/link5_shape_codebook/config.json" --trial robust_shape_v5 --variant "$task_variant" --epoch 3000 --copies 100
RUN
chmod a-w "$task_metadata"/*.py "$task_metadata/worker.sh" "$task_metadata/launcher.sh"
task_socket="link5-robust-shape-v5-$SLURM_JOB_ID"
nvidia-smi --query-gpu=timestamp,uuid,name,memory.used,memory.total,utilization.gpu --format=csv -l 5 > "$task_metadata/gpu-telemetry.csv" &
task_monitor=$!
trap 'rc=$?; kill "$task_monitor" 2>/dev/null || true; printf "%s\n" "$rc" > "$task_metadata/completion.exit"' EXIT
for task_variant in structural_geometry_head structural_invariant_geometry_head; do
  printf -v task_command 'bash %q %q %q %q %q > %q 2>&1' "$task_metadata/worker.sh" "$task_metadata" "$task_workspace" "$task_dependency" "$task_variant" "$task_metadata/$task_variant.log"
  tmux -L "$task_socket" new-session -d -s "$task_variant" "$task_command"
done
while tmux -L "$task_socket" list-sessions >/dev/null 2>&1; do sleep 2; done
for task_variant in structural_geometry_head structural_invariant_geometry_head; do
  test -s "$task_metadata/$task_variant.exit"
  test "$(cat "$task_metadata/$task_variant.exit")" = 0
done
