#!/usr/bin/env bash
set -Eeuo pipefail
workspace=${1:?workspace}
config=${2:?config}
metadata="$workspace/results/link5_shape_codebook/full_video/metadata"
snapshot="$metadata/postprocessing-source.py"
cp "$workspace/robot/experiments/link5_shape_codebook/postprocess_completed.py" "$snapshot"
trap 'rc=$?; printf "%s\n" "$rc" > "$metadata/postprocess-5587413.exit"' EXIT
export PYTHONPATH="$workspace" OMP_NUM_THREADS=2
/PHShome/zy992/Wilson/dependency/env/geometry/bin/python -u -c 'import importlib.util,pathlib,sys; spec=importlib.util.spec_from_file_location("robot.experiments.link5_shape_codebook.postprocessing_frozen",sys.argv[1]); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.run(pathlib.Path(sys.argv[2]))' "$snapshot" "$config"
