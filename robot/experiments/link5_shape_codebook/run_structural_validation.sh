#!/usr/bin/env bash
# Run as an overlapping step inside the already-owned single-H200 allocation.
set -Eeuo pipefail
task_trial=${1:?trial}; task_epoch=${2:?checkpoint epoch}; task_variant=${3:-structural_geometry_head}
case "$task_trial" in report_fix_v2|benign_finetune_v3|benign_finetune_v4|robust_shape_v5|profile_shape_v6) ;; *) exit 2;; esac
case "$task_variant" in structural_geometry_head|structural_invariant_geometry_head|structural_profile_decoder) ;; *) exit 2;; esac
case "$task_epoch" in *[!0-9]*|'') exit 2;; esac
task_workspace=/PHShome/zy992/Wilson/deformationdetection/workspace-link5-shape-codebook-four-frame0-20261006
task_dependency=/PHShome/zy992/Wilson/dependency
export HOSTNAME="$(hostname)"
module load tmux/3.5a-GCCcore-13.3.0
task_name="link5-validation-${task_trial}-${task_variant}-e${task_epoch}-$SLURM_JOB_ID-a3"
case "$task_variant" in structural_invariant_geometry_head) task_code=i;; structural_profile_decoder) task_code=p;; *) task_code=g;; esac
# Unix-domain sockets have a much shorter path limit than regular files.
task_socket="l5v-$SLURM_JOB_ID-${task_epoch}-$task_code"
task_metadata="$task_workspace/results/link5_shape_codebook/metadata/$task_name"
mkdir -p "$task_metadata"
cp "$task_workspace/robot/experiments/link5_shape_codebook/evaluate_structural_guides.py" "$task_metadata/"
cp "$task_workspace/robot/experiments/link5_shape_codebook/evaluate_calibrated_structural.py" "$task_metadata/"
cat > "$task_metadata/worker.sh" <<'RUN'
#!/usr/bin/env bash
set -Eeuo pipefail
task_metadata=${1:?};task_workspace=${2:?};task_dependency=${3:?};task_trial=${4:?};task_epoch=${5:?};task_variant=${6:?}
cd "$task_workspace"
export PYTHONPATH="$task_workspace/infrastructure/deformation_detect/import_guard:$task_workspace" PYTHONNOUSERSITE=1
export PATH="$task_dependency/bin:$task_dependency/env/geometry/bin:$PATH"
export LD_LIBRARY_PATH="$task_dependency/env/foundationpose/lib:$task_dependency/env/shape-build/lib:$task_dependency/env/compiler/lib:$task_dependency/env/geometry/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nvidia-smi --query-gpu=timestamp,uuid,name,memory.used,memory.total,utilization.gpu --format=csv -l 5 > "$task_metadata/gpu-telemetry.csv" &
task_monitor=$!
trap 'rc=$?; kill "$task_monitor" 2>/dev/null || true; printf "%s\n" "$rc" > "$task_metadata/completion.exit"' EXIT
"$task_dependency/env/shape-codebook/bin/python" -u "$task_metadata/evaluate_structural_guides.py" --config "$task_workspace/results/link5_shape_codebook/config.json" --trial "$task_trial" --variant "$task_variant" --epoch "$task_epoch"
"$task_dependency/env/shape-codebook/bin/python" -u "$task_metadata/evaluate_calibrated_structural.py" --config "$task_workspace/results/link5_shape_codebook/config.json" --trial "$task_trial" --variant "$task_variant" --epoch "$task_epoch" --copies 100
RUN
chmod a-w "$task_metadata/worker.sh" "$task_metadata/evaluate_structural_guides.py" "$task_metadata/evaluate_calibrated_structural.py"
printf -v task_command 'bash %q %q %q %q %q %q %q > %q 2>&1' "$task_metadata/worker.sh" "$task_metadata" "$task_workspace" "$task_dependency" "$task_trial" "$task_epoch" "$task_variant" "$task_metadata/session.log"
tmux -L "$task_socket" new-session -d -s "$task_name" "$task_command"
while tmux -L "$task_socket" has-session -t "$task_name" 2>/dev/null; do sleep 2; done
test -s "$task_metadata/completion.exit"
exit "$(cat "$task_metadata/completion.exit")"
