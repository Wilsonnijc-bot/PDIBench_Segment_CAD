"""Check real rigid-pose alignment before paying for detector training."""
import argparse
from pathlib import Path

import numpy as np

from .audit_alignment import pair_distance
from .common import observed_sample, read, reference_align, sha, write, split_path, reference_ids
from .score import estimate_pose


def run(config_path, config):
    output = Path(config['output'])
    normalization = read(output/'link5_normalization.json')
    settings = config['sanity']
    split = read(split_path(output))
    with np.load(split['train'][0]['input_path'], allow_pickle=False) as archive:
        reference = observed_sample(archive['xyz_camera'], 2000)[0]
    diagonal = float(np.linalg.norm(np.ptp(reference, axis=0)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    entries = []
    ids=reference_ids(config)
    videos=ids if ids else settings['normal_video_ids']
    for video_id in videos:
        observations = read(output/'cases'/video_id/'observations.json')
        first = next(row for row in observations if row['frame_id'] == 0)
        with np.load(first['input_path'], allow_pickle=False) as archive:
            first_cloud = observed_sample(archive['xyz_camera'], 2000)[0]
        later = [row for row in observations if row['frame_id'] > 0]
        samples=[first] if ids else later[::max(1,len(later)//3)][:3]
        for row in samples:
            pose = estimate_pose(config_path, config, row)
            entry = dict(video_id=video_id, frame_id=row['frame_id'], pose_status=pose['status'],
                         input_sha256=sha(row['input_path']), passed=False)
            if pose['status'] == 'ok':
                with np.load(row['input_path'], allow_pickle=False) as archive:
                    aligned = reference_align(archive['xyz_camera'], normalization['T_ref'], pose['T_camera_from_link5'])
                median, p95 = pair_distance(first_cloud, observed_sample(aligned, 2000)[0])
                entry.update(median_distance=median, p95_distance=p95, cad_mask_iou=pose['cad_mask_iou'],
                             passed=median <= settings['pose_alignment_median_diagonal']*diagonal
                             and p95 <= settings['pose_alignment_p95_diagonal']*diagonal)
                for axis, (x, y) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
                    axis.scatter(aligned[::10, x], aligned[::10, y], s=1, alpha=.25)
            entries.append(entry)
            print('LINK5_NORMAL_POSE_CHECK', video_id, row['frame_id'], entry['passed'], flush=True)
    for axis, (x, y) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        axis.scatter(reference[:, x], reference[:, y], s=1, color='black', alpha=.5)
        axis.set_aspect('equal', adjustable='datalim')
    fig.suptitle('Independent FoundationPose: four normal frame0 registration checks; reference in black' if ids else 'Independent FoundationPose: normal rigid poses returned to frame0; reference in black')
    fig.tight_layout()
    destination = output/'sanity'
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination/'runtime_pose_overlay.png', dpi=180)
    plt.close(fig)
    passed = len(entries) == (len(ids) if ids else 3*len(videos)) and all(row['passed'] for row in entries)
    receipt = dict(status='passed' if passed else 'failed', normalization_sha256=sha(output/'link5_normalization.json'),
                   scope='four known-normal frame0 native registration QC; later test frames checked individually at inference' if ids else 'provisional normal later-frame rigid alignment',
                   changes_training_coordinates=False,
                   thresholds=settings, reference_bbox_diagonal=diagonal, pose_checks=entries)
    write(destination/'pose_checks.json', receipt)
    if not passed:
        raise RuntimeError('normal runtime rigid-pose alignment failed; detector training is disabled')
    print('LINK5_NORMAL_POSE_CHECKS_PASSED', flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, read(args.config))


if __name__ == '__main__':
    main()
