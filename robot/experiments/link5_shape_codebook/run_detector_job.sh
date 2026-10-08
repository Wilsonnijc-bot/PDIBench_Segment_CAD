#!/usr/bin/env bash
# Runs the one detector only after fresh preparation and dependency jobs succeed.
set -Eeuo pipefail
workspace=${1:?workspace}; dependency=${2:?dependency}; config=${3:?config}
phase=${4:-detector}
kind=$phase
mode=paper_original
if [[ "$phase" == *_robot_structural ]]; then mode=robot_structural;phase=${phase%_robot_structural};fi
case "$phase" in pose_iteration_probe|pose_preflight|pose|train|validate|score|detector) ;; *) echo "Unknown stage: $phase" >&2;exit 2 ;; esac
test -n "${SLURM_JOB_ID:-}" || exit 2
cd "$workspace"
metadata="$workspace/results/link5_shape_codebook/metadata";mkdir -p "$metadata"
trap 'rc=$?; printf "%s\n" "$rc" > "$metadata/${kind}-${SLURM_JOB_ID}.exit"' EXIT
export PYTHONPATH="$workspace/infrastructure/deformation_detect/import_guard:$workspace" PYTHONNOUSERSITE=1
export PATH="$dependency/bin:$dependency/env/geometry/bin:$PATH"
export LD_LIBRARY_PATH="$dependency/env/foundationpose/lib:$dependency/env/shape-build/lib:$dependency/env/compiler/lib:$dependency/env/geometry/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
hostname > "$metadata/detector-host-${SLURM_JOB_ID}.txt"
nvidia-smi > "$metadata/detector-gpu-${SLURM_JOB_ID}.txt"
if test "$phase" = pose_iteration_probe; then
    "$dependency/env/foundationpose/bin/python" -u -m robot.experiments.link5_shape_codebook.pose_iteration_probe --config "$config"
fi
if test "$phase" = pose_preflight || test "$phase" = pose || test "$phase" = detector; then
    "$dependency/env/foundationpose/bin/python" -u -m robot.experiments.link5_shape_codebook.verify_pose_runtime --config "$config"
fi
if test "$phase" = pose || test "$phase" = detector; then
    "$dependency/env/shape-codebook/bin/python" -u -m robot.experiments.link5_shape_codebook.verify_backbone --config "$config"
    "$dependency/env/foundationpose/bin/python" -u -m robot.experiments.link5_shape_codebook.pose --config "$config" --reference
    "$dependency/env/geometry/bin/python" -u -m robot.experiments.link5_shape_codebook.structural_validation --config "$config"
    "$dependency/env/foundationpose/bin/python" -u -m robot.experiments.link5_shape_codebook.pose_sanity --config "$config"
fi
if test "$phase" = train || test "$phase" = detector; then
    "$dependency/env/shape-codebook/bin/python" -u -m robot.experiments.link5_shape_codebook.train_stage --config "$config" --augmentation-mode "$mode"
fi
if test "$phase" = validate || test "$phase" = detector; then
    "$dependency/env/shape-codebook/bin/python" -u -m robot.experiments.link5_shape_codebook.sanity --config "$config" --augmentation-mode "$mode"
fi
if test "$phase" = score || test "$phase" = detector; then
    "$dependency/env/foundationpose/bin/python" -u -m robot.experiments.link5_shape_codebook.pose --config "$config" --runtime --augmentation-mode "$mode"
    "$dependency/env/shape-codebook/bin/python" -u -m robot.experiments.link5_shape_codebook.score --config "$config" --augmentation-mode "$mode"
fi
echo "LINK5_STAGE_${kind}_FINISHED"
