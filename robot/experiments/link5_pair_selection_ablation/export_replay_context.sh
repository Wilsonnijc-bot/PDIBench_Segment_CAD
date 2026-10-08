#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --job-name=link5-replay-context
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
set -Eeuo pipefail
workspace=/PHShome/zy992/Wilson/deformationdetection/workspace-link5-pair-selection-20261007-v2
dependency=/PHShome/zy992/Wilson/dependency
cd "$workspace"
root="$workspace/results/link5_pair_selection_ablation_v1"
metadata="$root/metadata/replay-context-${SLURM_JOB_ID:?}"
/usr/bin/mkdir -p "$metadata"
trap 'rc=$?; printf "%s\n" "$rc" > "$metadata/job.exit"' EXIT
export PYTHONPATH="$workspace" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$dependency/env/compiler/lib:$dependency/env/geometry/lib:$dependency/env/geometry/lib64:${LD_LIBRARY_PATH:-}"
"$dependency/env/geometry/bin/python" -u -m robot.experiments.link5_pair_selection_ablation.export_replay_context --root "$root" --workers 4
