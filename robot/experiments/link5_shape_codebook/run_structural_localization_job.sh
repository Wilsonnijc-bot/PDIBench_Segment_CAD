#!/usr/bin/env bash
set -Eeuo pipefail
task_workspace=${1:?workspace}; task_dependency=${2:?dependency}
export HOSTNAME="$(hostname)"
module load tmux/3.5a-GCCcore-13.3.0
task_root="$task_workspace/results/link5_shape_codebook"
task_metadata="$task_root/metadata/structural-localization-$SLURM_JOB_ID"
mkdir -p "$task_metadata"
for task_source in train_structural_localization.py structural_geometry_head.py localization_objective.py robot_structural.py train_localization.py; do
  cp "$task_workspace/robot/experiments/link5_shape_codebook/$task_source" "$task_metadata/$task_source"
done
cp "$task_root/guides/manifest.json" "$task_metadata/guide-manifest.json"
cp "$task_root/guides/structural_distribution.json" "$task_metadata/distribution.json"
cat > "$task_metadata/worker.sh" <<'RUN'
#!/usr/bin/env bash
set -Eeuo pipefail
task_metadata=${1:?}; task_workspace=${2:?}; task_dependency=${3:?}
task_monitor_pid=''
trap 'rc=$?; if test -n "$task_monitor_pid"; then kill "$task_monitor_pid" 2>/dev/null || true; fi; printf "%s\n" "$rc" > "$task_metadata/completion.exit"' EXIT
cd "$task_workspace"
export PYTHONPATH="$task_workspace/infrastructure/deformation_detect/import_guard:$task_workspace" PYTHONNOUSERSITE=1
export PATH="$task_dependency/bin:$task_dependency/env/geometry/bin:$PATH"
export LD_LIBRARY_PATH="$task_dependency/env/foundationpose/lib:$task_dependency/env/shape-build/lib:$task_dependency/env/compiler/lib:$task_dependency/env/geometry/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
nvidia-smi > "$task_metadata/gpu.txt"
for task_source in train_structural_localization.py structural_geometry_head.py localization_objective.py robot_structural.py train_localization.py; do
  cmp "$task_metadata/$task_source" "$task_workspace/robot/experiments/link5_shape_codebook/$task_source"
done
"$task_dependency/env/shape-codebook/bin/python" -c 'import torch; from robot.experiments.link5_shape_codebook.structural_geometry_head import StructuralGeometryDetector; assert torch.cuda.device_count()==1; assert float(torch.ones(4,device="cuda").sum())==4; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name())' > "$task_metadata/runtime-preflight.log" 2>&1
nvidia-smi --query-gpu=timestamp,uuid,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv -l 5 > "$task_metadata/gpu-telemetry.csv" &
task_monitor_pid=$!
task_variants=(structural_point_residual structural_geometry_head)
task_pids=()
for task_variant in "${task_variants[@]}"; do
  "$task_dependency/env/shape-codebook/bin/python" -u "$task_metadata/train_structural_localization.py" --config "$task_workspace/results/link5_shape_codebook/config.json" --guide-manifest "$task_metadata/guide-manifest.json" --distribution "$task_metadata/distribution.json" --variant "$task_variant" > "$task_metadata/$task_variant.log" 2>&1 &
  task_pids+=("$!")
done
task_failed=0
for task_index in "${!task_pids[@]}"; do
  set +e
  wait "${task_pids[$task_index]}"; task_rc=$?
  set -e
  printf "%s\n" "$task_rc" > "$task_metadata/${task_variants[$task_index]}.exit"
  if test "$task_rc" != 0; then task_failed=1; fi
done
if test "$task_failed" != 0; then exit 1; fi
echo LINK5_STRUCTURAL_LOCALIZATION_EXPERIMENT_COMPLETE
RUN
chmod a-w "$task_metadata"/*.py "$task_metadata/worker.sh" "$task_metadata/guide-manifest.json" "$task_metadata/distribution.json"
task_socket="link5-structural-localization-$SLURM_JOB_ID"
printf -v task_command 'bash %q %q %q %q > %q 2>&1' "$task_metadata/worker.sh" "$task_metadata" "$task_workspace" "$task_dependency" "$task_metadata/session.log"
tmux -L "$task_socket" new-session -d -s "$task_socket" "$task_command"
while tmux -L "$task_socket" has-session -t "$task_socket" 2>/dev/null; do sleep 2; done
test -s "$task_metadata/completion.exit"
exit "$(cat "$task_metadata/completion.exit")"
