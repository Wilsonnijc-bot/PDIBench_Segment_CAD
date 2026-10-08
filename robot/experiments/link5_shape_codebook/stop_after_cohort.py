"""Stop the owned allocation after the user-selected matched cohort is complete.

This CPU monitor does not alter inference, poses, checkpoints, or the original
launch config. Slurm cancellation is recorded honestly as a user-requested stop.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run(root, job):
    scope_path = root/'metadata/evaluation_scope.json'
    scope = json.loads(scope_path.read_text())
    ids = scope['selected_video_ids']
    assert len(ids) == 30 and len(set(ids)) == 30
    previous = -1
    while True:
        complete = []
        total = 0
        for video in ids:
            folder = root/'cases'/video
            if not (folder/'completion.json').exists():
                continue
            summaries = []
            for mode in ('paper_original', 'robot_structural'):
                summary = json.loads((folder/f'video_{mode}.json').read_text())
                rows = json.loads((folder/f'frames_{mode}.json').read_text())
                n = summary['expected_frame_count']
                if sorted(r['frame_id'] for r in rows) != list(range(n)):
                    raise ValueError(f'{video}: incomplete source-frame coverage')
                if not summary['sum_is_complete'] or any(r['status'] != 'complete' for r in rows):
                    raise ValueError(f'{video}: unavailable frame score; cannot declare cohort complete')
                if not math.isclose(sum(r['raw_mean_top80'] for r in rows), summary['sum_all_frame_raw_mean_top80'], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f'{video}: sum mismatch')
                summaries.append(summary)
            assert summaries[0]['expected_frame_count'] == summaries[1]['expected_frame_count']
            complete.append(video)
            total += summaries[0]['expected_frame_count']
        if len(complete) != previous:
            print('LINK5_SELECTED_COHORT_PROGRESS', len(complete), '/30', total, 'frames per checkpoint', flush=True)
            previous = len(complete)
        if len(complete) == 30:
            break
        state = subprocess.check_output(['squeue', '-h', '-j', str(job), '-o', '%T'], text=True).strip()
        if not state:
            raise RuntimeError('allocation ended before the requested 30 full videos completed')
        time.sleep(10)
    # Do not claim an ordinary successful exit for an intentionally interrupted
    # 45-video launch. Preserve its logs and scheduler cancellation state.
    subprocess.run(['scancel', str(job)], check=True)
    while subprocess.check_output(['squeue', '-h', '-j', str(job)], text=True).strip():
        time.sleep(5)
    scope.update(status='complete', completed_video_count=30, completed_frame_count=total,
                 allocation_stop='user_requested_scancel_after_selected_cohort', job_id=job)
    save(scope_path, scope)
    save(root/'completion.json', dict(status='complete_selected_cohort', video_count=30,
         expected_frame_count=total, selected_video_ids=ids, workers=10, gpus=2,
         primary_video_metric='sum_all_frame_raw_mean_top80', include_frame0=True,
         exploratory=True, failed_sensitivity_checks_preserved=True,
         allocation_stop='user_requested_scancel_after_selected_cohort', job_id=job,
         scope_file='metadata/evaluation_scope.json'))
    print('LINK5_SELECTED_30_FULL_VIDEOS_COMPLETE', total, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job', type=int, required=True)
    args = parser.parse_args()
    run(args.root, args.job)
