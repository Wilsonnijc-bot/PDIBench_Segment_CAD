"""Preflight and atomically restart the owned Simple3D campaign at new concurrency."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time


def write(path, value):
    temp = path.with_name(path.name + '.resize.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--session', required=True)
    args = parser.parse_args()
    assert args.workers in (2, 4)
    root = args.root.resolve()
    project = root.parents[2]
    execution = root/'metadata/execution'
    python = '/root/autodl-tmp/pdi/env/pdi-bench/bin/python'
    assert root.name == 'link5_pair_visible_anchor' and (project/'experiments/simple3d_pairs_gpu_job.sh').is_file()
    manifest_sha = digest(root/'inputs/manifest.json')
    preflight = json.loads((root/'metadata/preflight.json').read_text())
    assert preflight['status'] == 'passed' and preflight['manifest_sha256'] == manifest_sha
    subprocess.run(['bash', '-n', str(project/'experiments/simple3d_pairs_gpu_job.sh')], check=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               CUDA_HOME='/usr/local/cuda-11.8', TORCH_CUDA_ARCH_LIST='8.0')
    probe = "import torch; assert torch.cuda.is_available(); assert torch.ones(2,device='cuda').sum().item()==2; print('CONCURRENT_CUDA_PROBE_PASSED',flush=True)"
    # All probes pass before touching the working producer.
    probes = [subprocess.Popen([python, '-c', probe], cwd=project, env=env) for _ in range(args.workers)]
    assert all(p.wait() == 0 for p in probes), 'concurrent CUDA probe failed; producer unchanged'
    previous_run = json.loads((root/'metadata/run.json').read_text())
    pane = int(subprocess.check_output(['tmux', 'list-panes', '-t', args.session, '-F', '#{pane_pid}'], text=True).strip())
    command = subprocess.check_output(['ps', '-p', str(pane), '-o', 'args='], text=True)
    assert 'simple3d_pairs_gpu_job.sh' in command and os.getpgid(pane) == pane, command
    preserved = {str(p.relative_to(root)): digest(p) for p in root.glob('cases/*/bin_*/comparison.json')
                 if json.loads(p.read_text())['simple3d_status'] == 'complete'}
    record = dict(status='switching', requested_workers=args.workers,
                  at=datetime.now(timezone.utc).isoformat(), previous_run=previous_run,
                  preserved_comparison_sha256=preserved)
    write(execution/'concurrency_switch.json', record)
    watcher = 'simple3d-hibernate-after-completion-20261004'
    subprocess.run(['tmux', 'kill-session', '-t', watcher], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.killpg(pane, signal.SIGTERM)
    for _ in range(40):
        if subprocess.run(['pgrep', '-g', str(pane)], capture_output=True).returncode != 0:
            break
        time.sleep(.25)
    else:
        os.killpg(pane, signal.SIGKILL)
    subprocess.run(['tmux', 'kill-session', '-t', args.session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    scenes = []
    for path in root.glob('cases/*/bin_*/s3dp_*.mp4'):
        evidence = path.parent/'comparison.json'
        if evidence.exists() and json.loads(evidence.read_text())['simple3d_status'] == 'complete':
            continue
        scene = path.stem
        assert scene.startswith('s3dp_') and all(c.isalnum() or c == '_' for c in scene)
        scenes.append(scene)
        mega = project/'PDI-Bench-edited/third_party/mega_sam'
        for directory in ('work_space', 'cache_flow', 'reconstructions'):
            scratch = mega/directory/scene
            if scratch.is_dir() and not scratch.is_symlink():
                shutil.rmtree(scratch)
        for directory, suffix in [('outputs', '_droid.npz'), ('outputs_cvd', '_sgd_cvd_hr.npz')]:
            (mega/directory/(scene+suffix)).unlink(missing_ok=True)
        path.unlink()
    for name, sha in preserved.items():
        assert digest(root/name) == sha
    launch = shlex.join(['env', 'S3D_PAIR_WORKERS='+str(args.workers), 'S3D_PAIR_ROOT='+str(root),
                         'bash', str(project/'experiments/simple3d_pairs_gpu_job.sh')])
    subprocess.run(['tmux', 'new-session', '-d', '-s', args.session, '-c', str(project), launch], check=True)
    for _ in range(120):
        run = json.loads((root/'metadata/run.json').read_text())
        if run['started'] != previous_run['started'] and run['command'][run['command'].index('--workers')+1] == str(args.workers):
            break
        time.sleep(.5)
    else:
        raise RuntimeError('new producer did not write its runtime configuration')
    schedule = json.loads((execution/'shutdown_schedule.json').read_text())
    schedule.update(run_sha256=digest(root/'metadata/run.json'), campaign_tmux=args.session,
                    authorized_at=datetime.now(timezone.utc).isoformat(), worker_count=args.workers)
    write(execution/'shutdown_schedule.json', schedule)
    watch = shlex.join(['python3', '-u', str(project/'experiments/simple3d_shutdown_after_completion.py'), '--root', str(root)])
    watch += ' >> '+shlex.quote(str(execution/'shutdown_watcher.log'))+' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', watcher, '-c', str(project), watch], check=True)
    record.update(status='running', new_run=run, interrupted_scenes_cleaned=scenes,
                  preserved_successful_pairs=len(preserved), shutdown_rearmed=True)
    write(execution/'concurrency_switch.json', record)
    print(json.dumps(dict(status='RUNNING', workers=args.workers,
                          preserved_scores=len(preserved), shutdown_rearmed=True)), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
