"""Rigidly align the three deformation guides for a later real-frame diagnostic."""
import argparse
from pathlib import Path
import os

import numpy as np
import torch

from robot.experiments.link5_shape_codebook.common import normalize, read, reference_align, sha, write
from robot.experiments.link5_shape_codebook.pose import Link5Pose


def run(config_path):
    config = read(config_path); root = Path(config['output'])
    normalization = read(root / 'link5_normalization.json'); manifest = read(root / 'guides/manifest.json')
    norm_hash = sha(root / 'link5_normalization.json')
    if torch.cuda.device_count()!=1 or float(torch.ones(4,device='cuda').sum())!=4:raise ValueError('one allocated GPU required')
    adapter = Link5Pose({**config['pose'], 'debug_directory':str(root / 'guides/pose-debug')})
    rows = []
    for guide in manifest['guides']:
        receipt = read(root / 'guides/cases' / guide['video_id'] / 'status.json')
        if receipt['status']!='complete' or receipt['guide']!=guide:raise ValueError('exact guide cloud required')
        for observation in receipt['observations']:
            path = Path(observation['input_path']); output = path.parent / 'pose'
            if sha(path)!=observation['input_sha256']:raise ValueError('guide input identity changed')
            if (output / 'pose.json').exists():
                pose = read(output / 'pose.json')
                if pose['input_sha256']!=sha(path) or pose['provenance']!=adapter.provenance:raise ValueError('pose receipt identity changed')
            else:pose = adapter.estimate(path, output)
            row = dict(video_id=guide['video_id'], frame_id=observation['frame_id'], input_sha256=sha(path),
                       pose_status=pose['status'], cad_mask_iou=pose.get('cad_mask_iou'), pose_sha256=sha(output / 'pose.json'))
            if pose['status']=='ok':
                with np.load(path,allow_pickle=False) as a:points = a['xyz_camera'].copy()
                aligned = reference_align(points, normalization['T_ref'], pose['T_camera_from_link5'])
                arrays = dict(xyz_reference=aligned, points_normalized=normalize(aligned, normalization))
                destination = path.parent / 'guide_aligned.npz'
                if destination.exists():
                    with np.load(destination) as a:
                        for name,x in arrays.items():
                            if not np.array_equal(a[name],x):raise ValueError('existing aligned guide changed')
                else:np.savez_compressed(destination,**arrays)
                row.update(aligned_path=str(destination), aligned_sha256=sha(destination))
            rows.append(row);print('LINK5_GUIDE_POSE',row,flush=True)
    if sha(root / 'link5_normalization.json')!=norm_hash:raise ValueError('fixed normalization changed')
    write(root / 'guides/pose_summary.json',dict(status='complete',rows=rows,normalization_sha256=norm_hash,
          guide_manifest_sha256=sha(root / 'guides/manifest.json'),adapter_sources=dict(prepare_guide_poses=sha(__file__)),
          allocation_id=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('HOSTNAME'),
          coordinate_policy='independent native rigid registration, T_ref @ inverse(T_t); fixed normal center/scale; no scale adjustment/ICP',
          labels='Guide families only. No pointwise defect ground truth and no training/codebook updates.'))
    print('LINK5_GUIDE_POSES_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4);run(args.config)
