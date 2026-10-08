#!/usr/bin/env bash
# This step shares the existing owned H200 with the two v5 workers.
set -Eeuo pipefail
task_workspace=/PHShome/zy992/Wilson/deformationdetection/workspace-link5-shape-codebook-four-frame0-20261006
task_dependency=/PHShome/zy992/Wilson/dependency
export HOSTNAME="$(hostname)"
module load tmux/3.5a-GCCcore-13.3.0
task_name="link5-profile-v6-$SLURM_JOB_ID"
task_metadata="$task_workspace/results/link5_shape_codebook/metadata/$task_name"
mkdir -p "$task_metadata"
for task_source in train_profile_v6.py profile_decoder.py evaluate_structural_guides.py evaluate_calibrated_structural.py; do
  cp "$task_workspace/robot/experiments/link5_shape_codebook/$task_source" "$task_metadata/"
done
cat > "$task_metadata/worker.sh" <<'RUN'
#!/usr/bin/env bash
set -Eeuo pipefail
task_metadata=${1:?};task_workspace=${2:?};task_dependency=${3:?}
task_monitor=''
trap 'rc=$?; if test -n "$task_monitor"; then kill "$task_monitor" 2>/dev/null || true; fi; printf "%s\n" "$rc" > "$task_metadata/completion.exit"' EXIT
cd "$task_workspace"
export PYTHONPATH="$task_workspace/infrastructure/deformation_detect/import_guard:$task_workspace" PYTHONNOUSERSITE=1
export PATH="$task_dependency/bin:$task_dependency/env/geometry/bin:$PATH"
export LD_LIBRARY_PATH="$task_dependency/env/foundationpose/lib:$task_dependency/env/shape-build/lib:$task_dependency/env/compiler/lib:$task_dependency/env/geometry/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nvidia-smi --query-gpu=timestamp,uuid,name,memory.used,memory.total,utilization.gpu --format=csv -l 5 > "$task_metadata/gpu-telemetry.csv" &
task_monitor=$!
"$task_dependency/env/shape-codebook/bin/python" -u "$task_metadata/train_profile_v6.py" --config "$task_workspace/results/link5_shape_codebook/config.json" --warm-checkpoint "$task_workspace/results/link5_shape_codebook/structural_localization/benign_finetune_v4/structural_geometry_head/final.pt" --epochs 2000 --variant structural_profile_decoder --resume
"$task_dependency/env/shape-codebook/bin/python" -u "$task_metadata/evaluate_structural_guides.py" --config "$task_workspace/results/link5_shape_codebook/config.json" --trial profile_shape_v6 --variant structural_profile_decoder --epoch 2000
"$task_dependency/env/shape-codebook/bin/python" -u "$task_metadata/evaluate_calibrated_structural.py" --config "$task_workspace/results/link5_shape_codebook/config.json" --trial profile_shape_v6 --variant structural_profile_decoder --epoch 2000 --copies 100
RUN
chmod a-w "$task_metadata"/*.py "$task_metadata/worker.sh"
printf -v task_command 'bash %q %q %q %q > %q 2>&1' "$task_metadata/worker.sh" "$task_metadata" "$task_workspace" "$task_dependency" "$task_metadata/session.log"
tmux -L "$task_name" new-session -d -s "$task_name" "$task_command"
while tmux -L "$task_name" has-session -t "$task_name" 2>/dev/null; do sleep 2; done
test -s "$task_metadata/completion.exit"
exit "$(cat "$task_metadata/completion.exit")"
