#!/usr/bin/env bash
set -Eeuo pipefail
task_workspace=${1:?workspace}; task_dependency=${2:?dependency}
export HOSTNAME="$(hostname)"
module load tmux/3.5a-GCCcore-13.3.0
task_root="$task_workspace/results/link5_shape_codebook"
task_metadata="$task_root/metadata/structural-guides-$SLURM_JOB_ID"
mkdir -p "$task_metadata"
cp "$task_workspace/robot/experiments/link5_shape_codebook/prepare_guides.py" "$task_metadata/prepare_guides.py"
cp "$task_root/guides/manifest.json" "$task_metadata/manifest.json"
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
export HF_HOME="$task_dependency/cache/huggingface" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nvidia-smi > "$task_metadata/gpu.txt"
cmp "$task_metadata/prepare_guides.py" "$task_workspace/robot/experiments/link5_shape_codebook/prepare_guides.py"
"$task_dependency/env/geometry/bin/python" -c 'import torch; from robot.experiments.link5_shape_codebook.prepare_guides import run; assert torch.cuda.device_count()==1; assert float(torch.ones(4,device="cuda").sum())==4; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name())' > "$task_metadata/runtime-preflight.log" 2>&1
nvidia-smi --query-gpu=timestamp,uuid,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv -l 5 > "$task_metadata/gpu-telemetry.csv" &
task_monitor_pid=$!
task_ids=(COSMOS2.5_0018 COSMOS2.5_0030 LVP_ROBOWM_0005)
task_pids=()
for task_id in "${task_ids[@]}"; do
  "$task_dependency/env/geometry/bin/python" -u "$task_metadata/prepare_guides.py" --config "$task_workspace/results/link5_shape_codebook/config.json" --manifest "$task_metadata/manifest.json" --video-id "$task_id" > "$task_metadata/$task_id.log" 2>&1 &
  task_pids+=("$!")
done
task_failed=0
for task_index in "${!task_pids[@]}"; do
  set +e
  wait "${task_pids[$task_index]}"; task_rc=$?
  set -e
  printf "%s\n" "$task_rc" > "$task_metadata/${task_ids[$task_index]}.exit"
  if test "$task_rc" != 0; then task_failed=1; fi
done
if test "$task_failed" != 0; then exit 1; fi
echo LINK5_STRUCTURAL_GUIDES_COMPLETE
RUN
chmod a-w "$task_metadata/worker.sh" "$task_metadata/prepare_guides.py" "$task_metadata/manifest.json"
task_socket="link5-structural-guides-$SLURM_JOB_ID"
printf -v task_command 'bash %q %q %q %q > %q 2>&1' "$task_metadata/worker.sh" "$task_metadata" "$task_workspace" "$task_dependency" "$task_metadata/session.log"
tmux -L "$task_socket" new-session -d -s "$task_socket" "$task_command"
while tmux -L "$task_socket" has-session -t "$task_socket" 2>/dev/null; do sleep 2; done
test -s "$task_metadata/completion.exit"
exit "$(cat "$task_metadata/completion.exit")"
