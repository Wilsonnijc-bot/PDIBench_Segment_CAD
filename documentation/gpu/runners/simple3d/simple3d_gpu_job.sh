#!/usr/bin/env bash
set -uo pipefail
cd "$S3D_ROOT"
run_job() {
    local phase_rc=0
    local resume_args=()
    if [[ "${S3D_RESUME:-0}" == 1 ]]; then resume_args=(--resume); fi
    if [[ "$S3D_PHASE" == all ]]; then
        # Two independent workloads on the same GPU, with total workers bounded by
        # the requested count. Each child keeps its own durable log and exit marker.
        local child_workers=$((S3D_WORKERS / 2))
        if ((child_workers < 1)); then child_workers=1; fi
        S3D_PHASE=object S3D_WORKERS="$child_workers" S3D_LOG="$S3D_RUN_ROOT/metadata/execution/$S3D_SESSION-object.log" S3D_SESSION="$S3D_SESSION-object" bash "$S3D_ROOT/experiments/simple3d_gpu_job.sh" &
        local object_pid=$!
        S3D_PHASE=link5 S3D_WORKERS="$child_workers" S3D_LOG="$S3D_RUN_ROOT/metadata/execution/$S3D_SESSION-link5.log" S3D_SESSION="$S3D_SESSION-link5" bash "$S3D_ROOT/experiments/simple3d_gpu_job.sh" &
        local link5_pid=$!
        wait "$object_pid"; local object_rc=$?
        wait "$link5_pid"; local link5_rc=$?
        if ((object_rc != 0)); then return "$object_rc"; fi
        return "$link5_rc"
    fi
    if [[ "$S3D_PHASE" == object || "$S3D_PHASE" == all ]]; then
        "$S3D_PYTHON" experiments/run_simple3d_object_deformation.py --inputs "$S3D_INPUTS" --run-id "$S3D_RUN_ID" --workers "$S3D_WORKERS" "${resume_args[@]}"
        phase_rc=$?
        if ((phase_rc != 0)); then return "$phase_rc"; fi
    fi
    if [[ "$S3D_PHASE" == link5 || "$S3D_PHASE" == all ]]; then
        "$S3D_PYTHON" experiments/run_simple3d_link5.py --inputs "$S3D_INPUTS" --run-id "$S3D_RUN_ID" --workers "$S3D_WORKERS" --reference both --cad assets/link5.dae "${resume_args[@]}"
        phase_rc=$?
    fi
    return "$phase_rc"
}
run_job 2>&1 | tee "$S3D_LOG"
rc=${PIPESTATUS[0]}
printf '\nEXIT_STATUS=%s\n' "$rc" >> "$S3D_LOG"
printf '%s\n' "$rc" > "$S3D_RUN_ROOT/metadata/execution/$S3D_SESSION.exit"
exit "$rc"
