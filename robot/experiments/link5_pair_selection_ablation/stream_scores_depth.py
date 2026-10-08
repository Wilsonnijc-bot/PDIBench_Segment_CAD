"""Finish each video with shared depth filtering and both scores as inputs arrive."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
import json
import os
from pathlib import Path
import subprocess
import time

from .filter_video_depth import process as filter_video
from .run_scores_depth_v2 import process as score_video
from .build_report_v2 import build
from .prepare_gpu import write
from .selectors import METHODS


def finish_video(job):
    entry, root, cache = job
    root = Path(root)
    depth = filter_video((entry, str(root), cache))
    if depth['status'] != 'complete':
        rows = [dict(case=entry['video_id'], cohort=entry['cohort'], method=method,
                     status='failed', error='Depth filtering: ' + depth['error']) for method in METHODS]
    else:
        rows = score_video((entry, str(root / 'shared_inputs'), str(root / 'rigidity'), str(root / 'depth_support')))
    return dict(video_id=entry['video_id'], depth=depth, scores=rows)


def save_progress(root, entries, finished, active):
    rows = []
    depths = []
    for entry in entries:
        case = entry['video_id']
        if case in finished:
            rows.extend(finished[case]['scores']); depths.append(finished[case]['depth'])
        else:
            status = 'running' if case in active else 'pending'
            rows.extend(dict(case=case, cohort=entry['cohort'], method=method, status=status) for method in METHODS)
            depths.append(dict(video_id=case, status=status))
    folder = root / 'rigidity'; folder.mkdir(exist_ok=True)
    write(folder / 'summary.json', rows)
    keys = sorted({key for row in rows for key in row})
    temporary = folder / 'rigidity_scores.csv.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    temporary.replace(folder / 'rigidity_scores.csv')
    write(root / 'depth_support/summary.json', depths)
    write(root / 'metadata/stream_progress.json', dict(
        unix_time=time.time(), finished_videos=len(finished), total_videos=len(entries), active_videos=list(active),
        successful_scores=sum(row['status'] == 'complete' for row in rows),
        failed_scores=sum(row['status'] == 'failed' for row in rows), expected_scores=len(rows)))


def gpu_finished(root, job):
    if (root / 'metadata' / ('allocation-' + job) / 'job.exit').exists():
        return True
    text = subprocess.check_output(['sacct', '-j', job, '--format=JobIDRaw,State', '--noheader', '--parsable2'], text=True)
    row = next((line for line in text.splitlines() if line.startswith(job + '|')), '')
    return any(state in row for state in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cache', 'sources'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--gpu-job', required=True); p.add_argument('--workers', type=int, default=5)
    a = p.parse_args()
    if not a.gpu_job.isdigit() or not os.getenv('SLURM_JOB_ID'):
        raise ValueError('Owned GPU job and allocated CPU job required')
    entries = json.loads((a.root / 'manifest.json').read_text())['entries']
    remaining = {e['video_id']: e for e in entries}; finished = {}; active = {}
    save_progress(a.root, entries, finished, {})
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        while remaining or active:
            for case, entry in list(remaining.items()):
                if len(active) >= a.workers:
                    break
                receipt = a.root / 'shared_inputs' / (case + '.json')
                if receipt.exists() and receipt.with_suffix('.npz').exists():
                    future = pool.submit(finish_video, (entry, str(a.root), str(a.cache)))
                    active[future] = entry; del remaining[case]
            save_progress(a.root, entries, finished, {e['video_id']: True for e in active.values()})
            if active:
                done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    entry = active.pop(future); case = entry['video_id']
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = dict(video_id=case, depth=dict(video_id=case, status='failed', error=str(exc)),
                            scores=[dict(case=case, cohort=entry['cohort'], method=method, status='failed', error=str(exc)) for method in METHODS])
                    finished[case] = result
                    for row in result['scores']:
                        write(a.root / 'rigidity' / case / row['method'] / 'result.json', row)
                    print('VIDEO_END_TO_END_COMPLETE', case, [(r['method'], r['status']) for r in result['scores']], flush=True)
            elif remaining:
                if gpu_finished(a.root, a.gpu_job):
                    for case, entry in remaining.items():
                        error = 'GPU preparation ended without complete receipted raw inputs'
                        finished[case] = dict(video_id=case, depth=dict(video_id=case, status='failed', error=error),
                            scores=[dict(case=case, cohort=entry['cohort'], method=method, status='failed', error=error) for method in METHODS])
                    remaining.clear()
                else:
                    time.sleep(10)
    save_progress(a.root, entries, finished, {})
    build(a.root, a.sources)
    failures = sum(row['status'] != 'complete' for result in finished.values() for row in result['scores'])
    print('STREAMED_DEPTH_FILTERED_SCORES', 2 * len(entries) - failures, 2 * len(entries), flush=True)
    if failures:
        raise SystemExit(1)
    print('LINK5_DEPTH_FILTERED_REPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
