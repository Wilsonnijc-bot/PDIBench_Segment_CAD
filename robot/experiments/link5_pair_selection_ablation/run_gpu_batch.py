"""Five video processes on one allocated H200, with telemetry and cancellation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .prepare_gpu import sha, validate_sources, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'cache', 'sources', 'queries', 'output', 'scratch', 'mega-root', 'tracker-checkpoint'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.workers != 5:
        raise ValueError('This authorized run requires five total video workers on one GPU')
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Slurm allocation required')
    import torch
    import cotracker.predictor
    from infrastructure.shared.inference.mega_sam_wrapper import MegaSamWrapper
    if torch.cuda.device_count() != 1 or 'H200' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Exactly one visible H200 required')
    if (torch.ones(2, device='cuda') + 1).sum().item() != 4:
        raise RuntimeError('CUDA startup operation failed')
    entries = json.loads(args.manifest.read_text())['entries']
    for entry in entries:
        validate_sources(entry, args.cache, args.sources, args.queries)
    for path in (args.tracker_checkpoint, args.mega_root / 'checkpoints/megasam_final.pth',
                 args.mega_root / 'Depth-Anything/checkpoints/depth_anything_vitl14.pth',
                 args.mega_root / 'cvd_opt/raft-things.pth'):
        if not path.is_file():
            raise FileNotFoundError(path)
    for relative in ('Depth-Anything/run_videos.py', 'UniDepth/scripts/demo_mega-sam.py',
                     'camera_tracking_scripts/test_demo.py', 'cvd_opt/preprocess_flow.py', 'cvd_opt/cvd_opt.py'):
        if not (args.mega_root / relative).is_file():
            raise FileNotFoundError(relative)
    metadata = args.output.parent / 'metadata'
    metadata.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total', '--format=csv,noheader'], text=True).strip()
    write(metadata / 'preflight.json', dict(status='passed', allocation_id=os.environ['SLURM_JOB_ID'], gpu=gpu,
        gpu_count=1, video_workers=5, cases=len(entries), interpreter=sys.executable, manifest_sha256=sha(args.manifest)))
    print('RIGIDITY_GPU_PREFLIGHT_PASSED', flush=True)
    if args.preflight_only:
        return
    stop = threading.Event()
    active = {}
    lock = threading.RLock()
    results = []
    started = time.monotonic()

    def cancel(signum, frame):
        stop.set()
        with lock:
            for process in active.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)

    def monitor():
        path = metadata / ('telemetry-' + os.environ['SLURM_JOB_ID'] + '.jsonl')
        with path.open('a') as stream:
            while not stop.is_set():
                row = dict(unix_time=time.time(), elapsed_seconds=time.monotonic() - started)
                try:
                    row['gpu_csv'] = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.total,utilization.gpu,utilization.memory', '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
                    row['host_memory'] = Path('/proc/meminfo').read_text()
                    row['load_average'] = os.getloadavg()
                    row['pressure'] = {name: Path('/proc/pressure/' + name).read_text() for name in ('cpu', 'memory', 'io')}
                    with lock:
                        row['active_videos'] = list(active)
                    row['completed_videos'] = sum(r['exit_status'] == 0 for r in results)
                    row['videos_per_hour'] = row['completed_videos'] * 3600 / max(row['elapsed_seconds'], 1)
                except Exception as error:
                    row['monitor_error'] = str(error)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                stop.wait(5)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    def video(entry):
        case = entry['video_id']
        if stop.is_set():
            return dict(video_id=case, exit_status=130, status='cancelled_before_launch')
        logs = metadata / 'video_logs'
        logs.mkdir(exist_ok=True)
        stamp = time.time_ns()
        log = logs / f'{case}-{stamp}.log'
        command = [sys.executable, '-u', '-m', 'robot.experiments.link5_pair_selection_ablation.prepare_gpu']
        for name in ('manifest', 'cache', 'sources', 'queries', 'output', 'scratch', 'mega_root', 'tracker_checkpoint'):
            command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
        command.extend(['--video-id', case])
        t0 = time.monotonic()
        with log.open('x') as stream:
            with lock:
                if stop.is_set():
                    return dict(video_id=case, exit_status=130, status='cancelled_before_launch')
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active[case] = process
            while process.poll() is None:
                if stop.wait(1):
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
            rc = process.wait()
        with lock:
            active.pop(case, None)
        result = dict(video_id=case, exit_status=rc, seconds=time.monotonic() - t0, log=str(log))
        print('VIDEO_EXIT', case, rc, flush=True)
        return result

    try:
        # Measure one real video, then a five-video capacity wave before the rest.
        # Longest clips lead the capacity wave to expose temporal memory peaks.
        ordered = sorted(entries, key=lambda e: e['frame_count'], reverse=True)
        waves = [ordered[:1], ordered[1:6], ordered[6:]]
        for stage, wave in enumerate(waves):
            if stop.is_set():
                break
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = [pool.submit(video, entry) for entry in wave]
                for future in as_completed(futures):
                    results.append(future.result())
                    write(metadata / 'preparation-progress.json', dict(gpu_count=1, workers=5, stage=stage,
                        elapsed_seconds=time.monotonic() - started, total=len(entries), results=results))
            if stage < 2 and any(r['exit_status'] for r in results):
                raise RuntimeError('Measured capacity wave failed; diagnose logs before scheduling remainder')
        if stop.is_set():
            raise SystemExit(130)
        write(metadata / 'preparation-exits.json', results)
        # Failures remain explicit rows for both methods; still create the table.
        with lock:
            if stop.is_set():
                raise SystemExit(130)
            scoring = subprocess.Popen([sys.executable, '-u', '-m', 'robot.experiments.link5_pair_selection_ablation.run_scores',
                '--manifest', str(args.manifest), '--inputs', str(args.output), '--output', str(args.output.parent / 'rigidity'), '--workers', '5'], start_new_session=True)
            active['dependent_cpu_scoring'] = scoring
        while scoring.poll() is None:
            if stop.wait(1):
                try:
                    os.killpg(scoring.pid, signal.SIGTERM)
                    scoring.wait(timeout=20)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(scoring.pid, signal.SIGKILL)
                break
        score_rc = scoring.wait()
        with lock:
            active.pop('dependent_cpu_scoring', None)
        if stop.is_set():
            raise SystemExit(130)
        if score_rc or any(r['exit_status'] for r in results):
            raise SystemExit(1)
        print('LINK5_RIGIDITY_EXPERIMENT_COMPLETE', flush=True)
    finally:
        stop.set()
        monitor_thread.join(timeout=15)


if __name__ == '__main__':
    main()
