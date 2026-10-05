#!/usr/bin/env bash
# Native scorer orchestration only; invoke this script inside a unique tmux session.
set -Eeuo pipefail
if [[ $# != 4 ]]; then
  echo 'Usage: run_anomalydino_gpu.sh RUN_ROOT PYTHON DINO_REPO CONVERTED_CHECKPOINT' >&2
  exit 2
fi
anomaly_run_root=$1
anomaly_python=$2
anomaly_dino_repo=$3
anomaly_checkpoint=$4
mkdir -p "$anomaly_run_root/metadata" "$anomaly_run_root/outputs"
exec > "$anomaly_run_root/metadata/run.log" 2>&1
trap 'anomaly_rc=$?; printf "%s\n" "$anomaly_rc" > "$anomaly_run_root/metadata/run.exit"; exit "$anomaly_rc"' EXIT
cd "$anomaly_run_root/code/PDI-Bench-edited"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
"$anomaly_python" - "$anomaly_run_root" "$anomaly_dino_repo" "$anomaly_checkpoint" <<'PY'
import hashlib, json, pathlib, socket, subprocess, sys
import torch
from object.scoring.anomalydino.anomalydino import load_upstream
from object.scoring.anomalydino.runner import read_pairs, write_json
root, dino_repo, checkpoint = map(pathlib.Path, sys.argv[1:])
assert root.is_dir() and dino_repo.is_dir() and checkpoint.is_file()
assert (dino_repo / 'hubconf.py').is_file()
load_upstream()
assert torch.cuda.is_available()
assert (torch.ones(2, device='cuda:0') + 1).sum().item() == 4
manifest, pairs = read_pairs(root / 'metadata/pair_manifest.json')
assert pairs and len(pairs) == manifest['pair_count']
for pair in pairs:
    for role in ('reference', 'query'):
        path = root / 'inputs' / pair[role + '_crop']
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pair[role + '_sha256']
source = root / 'code/PDI-Bench-edited'
write_json(root / 'metadata/preflight.json', {
    'status': 'passed', 'hostname': socket.gethostname(),
    'gpu': torch.cuda.get_device_name(0), 'python': sys.version, 'interpreter': sys.executable,
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'dino_repo_commit': subprocess.check_output(['git', '-C', str(dino_repo), 'rev-parse', 'HEAD'], text=True).strip(),
    'code_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in source.rglob('*.py') if '__pycache__' not in p.parts},
    'pair_count': len(pairs), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
print('PREFLIGHT PASSED', flush=True)
PY
"$anomaly_python" -m pip freeze > "$anomaly_run_root/metadata/environment.txt"
"$anomaly_python" -m object.scoring.anomalydino run \
  --pairs "$anomaly_run_root/metadata/pair_manifest.json" \
  --input-root "$anomaly_run_root/inputs" \
  --output-root "$anomaly_run_root/outputs" \
  --model-name dinov2_vitb14 --device cuda:0 --resolution 448 \
  --rotation --no-masking --faiss-on-cpu \
  --dino-repo "$anomaly_dino_repo" --checkpoint "$anomaly_checkpoint"
