"""Measured eight-worker capacity trial, then preparation on one H200."""
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
    if args.workers not in (5, 8):
        raise ValueError('Validated baseline five or requested trial eight workers required')
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
    write(metadata / ('preflight-capacity-' + os.environ['SLURM_JOB_ID'] + '.json'), dict(status='passed', allocation_id=os.environ['SLURM_JOB_ID'], gpu=gpu,
        gpu_count=1, video_workers=args.workers, cases=len(entries), interpreter=sys.executable, manifest_sha256=sha(args.manifest)))
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
                    row['pressure'] = {name: (Path('/proc/pressure/' + name).read_text() if Path('/proc/pressure/' + name).exists() else 'unavailable') for name in ('cpu', 'memory', 'io')}
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
        complete = {p.stem for p in args.output.glob('*.json') if p.with_suffix('.npz').exists()}
        # prepare_gpu verifies checksums before reuse, including all prior receipts.
        for entry in entries:
            if entry['video_id'] in complete:
                result = video(entry)
                if result['exit_status']:
                    raise RuntimeError('Previously completed input failed reuse validation')
                result['reused'] = True
                results.append(result)
        ordered = sorted([e for e in entries if e['video_id'] not in complete], key=lambda e: e['frame_count'], reverse=True)
        baseline = json.loads((metadata / 'allocation-5660284/preparation-progress-capacity.json').read_text())
        long_rows = [r for r in baseline['results'] if r['video_id'] != 'COSMOS3_0001' and r['exit_status'] == 0]
        baseline_seconds = max(r['seconds'] for r in long_rows)
        baseline_frames_per_second = 5 * 189 / baseline_seconds
        wave = ordered[:args.workers]
        def run_wave(wave, workers, stage):
            before = time.monotonic()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(video, entry) for entry in wave]
                wave_results = []
                for future in as_completed(futures):
                    result = future.result(); results.append(result); wave_results.append(result)
                    write(metadata / 'preparation-progress.json', dict(gpu_count=1, workers=workers, stage=stage,
                        elapsed_seconds=time.monotonic()-started, total=len(entries), results=results))
            return wave_results, time.monotonic()-before
        measured, seconds = run_wave(wave, args.workers, 'capacity_trial') if wave else ([], 0)
        if stop.is_set(): raise SystemExit(130)
        rows = [json.loads(line) for line in (metadata / ('telemetry-' + os.environ['SLURM_JOB_ID'] + '.jsonl')).read_text().splitlines()]
        peak = max((int(r['gpu_csv'].split(',')[1]) for r in rows if r.get('gpu_csv')), default=0)
        total = int(rows[-1]['gpu_csv'].split(',')[2]) if rows else 143771
        rate = sum(e['frame_count'] for e in wave) / max(seconds, 1)
        safe = all(r['exit_status'] == 0 for r in measured) and peak < .90 * total
        workers = args.workers if safe and rate > baseline_frames_per_second else 5
        write(metadata / ('capacity-decision-' + os.environ['SLURM_JOB_ID'] + '.json'), dict(
            trial_workers=args.workers, retained_workers=workers, gpu_count=1, peak_mib=peak, total_mib=total,
            seconds=seconds, source_frames=sum(e['frame_count'] for e in wave), frames_per_second=rate,
            five_worker_frames_per_second=baseline_frames_per_second, safe=safe,
            comparison_limit='Actual remaining clips; matched pipeline but different clip identities/lengths.'))
        remaining = ordered[len(wave):]
        failed = [e for e in wave if any(r['video_id'] == e['video_id'] and r['exit_status'] for r in measured)]
        if failed:
            # Fresh scratch preserves every failed native output, never substitutes it.
            args.scratch = args.scratch.with_name(args.scratch.name + '-five-worker-retry')
            remaining = failed + remaining
        run_wave(remaining, workers, 'remaining')
        if stop.is_set(): raise SystemExit(130)
        # Last status per video includes any conservative retry after capacity failure.
        final = {r['video_id']:r for r in results}
        write(metadata / 'preparation-exits.json', list(final.values()))
        if any(r['exit_status'] for r in final.values()): raise SystemExit(1)
        print('LINK5_SHARED_INPUTS_COMPLETE', flush=True)
    finally:
        stop.set()
        monitor_thread.join(timeout=15)


if __name__ == '__main__':
    main()
