#!/usr/bin/env bash
set -Eeuo pipefail
task_workspace=/PHShome/zy992/Wilson/deformationdetection/workspace-link5-shape-codebook-four-frame0-20261006
task_dependency=/PHShome/zy992/Wilson/dependency
export HOSTNAME="$(hostname)"
module load tmux/3.5a-GCCcore-13.3.0
task_metadata="$task_workspace/results/link5_shape_codebook/metadata/benign-finetune-v4-$SLURM_JOB_ID"
mkdir -p "$task_metadata"
cp "$task_workspace/robot/experiments/link5_shape_codebook/fine_tune_structural_v4.py" "$task_metadata/fine_tune_structural_v4.py"
cp "$task_workspace/robot/experiments/link5_shape_codebook/benign_variation_v4.py" "$task_metadata/benign_variation_v4.py"
cp "$task_workspace/robot/experiments/link5_shape_codebook/evaluate_structural_guides.py" "$task_metadata/"
cp "$task_workspace/robot/experiments/link5_shape_codebook/evaluate_calibrated_structural.py" "$task_metadata/"
cat > "$task_metadata/worker.sh" <<'RUN'
#!/usr/bin/env bash
set -Eeuo pipefail
task_metadata=${1:?}; task_workspace=${2:?}; task_dependency=${3:?}
trap 'rc=$?; kill "${task_monitor:-}" 2>/dev/null || true; printf "%s\n" "$rc" > "$task_metadata/completion.exit"' EXIT
cd "$task_workspace"
export PYTHONPATH="$task_workspace/infrastructure/deformation_detect/import_guard:$task_workspace" PYTHONNOUSERSITE=1
export PATH="$task_dependency/bin:$task_dependency/env/geometry/bin:$PATH"
export LD_LIBRARY_PATH="$task_dependency/env/foundationpose/lib:$task_dependency/env/shape-build/lib:$task_dependency/env/compiler/lib:$task_dependency/env/geometry/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nvidia-smi > "$task_metadata/gpu.txt"
nvidia-smi --query-gpu=timestamp,uuid,name,memory.used,memory.total,utilization.gpu --format=csv -l 5 > "$task_metadata/gpu-telemetry.csv" &
task_monitor=$!
function validate_trial() {
  local trial=$1 epoch=$2
  "$task_dependency/env/shape-codebook/bin/python" -u "$task_metadata/evaluate_structural_guides.py" --config "$task_workspace/results/link5_shape_codebook/config.json" --trial "$trial" --variant structural_geometry_head --epoch "$epoch"
  "$task_dependency/env/shape-codebook/bin/python" -u "$task_metadata/evaluate_calibrated_structural.py" --config "$task_workspace/results/link5_shape_codebook/config.json" --trial "$trial" --epoch "$epoch" --copies 100
}
# Two useful read-only comparisons share the same H200 with training. Keep the
# allocation alive until every owned validator has completed.
validate_trial report_fix_v2 1500 > "$task_metadata/baseline-v2.log" 2>&1 &
task_baseline=$!
validate_trial benign_finetune_v3 100 > "$task_metadata/initial-v3.log" 2>&1 &
task_initial=$!
"$task_dependency/env/shape-codebook/bin/python" -u "$task_metadata/fine_tune_structural_v4.py" --config "$task_workspace/results/link5_shape_codebook/config.json" --warm-checkpoint "$task_workspace/results/link5_shape_codebook/structural_localization/benign_finetune_v3/structural_geometry_head/latest.pt" --epochs 600 --resume
validate_trial benign_finetune_v4 300 > "$task_metadata/finetune-e300.log" 2>&1 &
task_e300=$!
validate_trial benign_finetune_v4 600 > "$task_metadata/finetune-e600.log" 2>&1 &
task_e600=$!
wait "$task_baseline"; printf '0\n' > "$task_metadata/baseline-v2.exit"
wait "$task_initial"; printf '0\n' > "$task_metadata/initial-v3.exit"
wait "$task_e300"; printf '0\n' > "$task_metadata/finetune-e300.exit"
wait "$task_e600"; printf '0\n' > "$task_metadata/finetune-e600.exit"
RUN
chmod a-w "$task_metadata/worker.sh" "$task_metadata/fine_tune_structural_v4.py" "$task_metadata/benign_variation_v4.py" "$task_metadata/evaluate_structural_guides.py" "$task_metadata/evaluate_calibrated_structural.py"
task_socket="link5-benign-finetune-v4-$SLURM_JOB_ID"
printf -v task_command 'bash %q %q %q %q > %q 2>&1' "$task_metadata/worker.sh" "$task_metadata" "$task_workspace" "$task_dependency" "$task_metadata/session.log"
tmux -L "$task_socket" new-session -d -s "$task_socket" "$task_command"
while tmux -L "$task_socket" has-session -t "$task_socket" 2>/dev/null; do sleep 2; done
test -s "$task_metadata/completion.exit"
exit "$(cat "$task_metadata/completion.exit")"
