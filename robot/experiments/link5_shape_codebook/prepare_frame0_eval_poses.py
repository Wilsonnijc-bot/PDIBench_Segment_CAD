"""Diagnose heldout frame0 camera variation with the pipeline's rigid alignment."""
import argparse
from pathlib import Path

import numpy as np
import torch

from robot.experiments.link5_shape_codebook.common import normalize, read, reference_align, sha, write
from robot.experiments.link5_shape_codebook.pose import Link5Pose
from robot.experiments.link5_shape_codebook.train_localization import frozen_inputs


def run(config_path):
    config=read(config_path);root,split,norm,heldout=frozen_inputs(config)
    heldout=[r for r in heldout if r['video_id'] not in ['COSMOS2.5_0030','LVP_ROBOWM_0005']]
    destination=root/'structural_localization/pose_alignment';destination.mkdir(exist_ok=True)
    adapter=Link5Pose({**config['pose'],'debug_directory':str(destination/'debug')})
    rows=[]
    for row in heldout:
        folder=destination/row['video_id'];pose_path=folder/'pose/pose.json'
        if pose_path.exists():
            pose=read(pose_path)
            if pose['input_sha256']!=row['input_sha256'] or pose['provenance']!=adapter.provenance:raise ValueError('pose context identity changed')
        else:pose=adapter.estimate(row['input_path'],folder/'pose')
        result={**row,'pose_status':pose['status'],'cad_mask_iou':pose.get('cad_mask_iou'),'pose_sha256':sha(pose_path)}
        if pose['status']=='ok':
            with np.load(row['input_path'],allow_pickle=False) as a:points=a['xyz_camera'].copy()
            aligned=reference_align(points,norm['T_ref'],pose['T_camera_from_link5'])
            path=folder/'aligned.npz'
            arrays=dict(xyz_reference=aligned,points_normalized=normalize(aligned,norm))
            if path.exists():
                with np.load(path) as a:
                    for k,v in arrays.items():np.testing.assert_array_equal(a[k],v)
            else:np.savez_compressed(path,**arrays)
            result.update(aligned_path=str(path),aligned_sha256=sha(path))
        rows.append(result);write(destination/'progress.json',dict(rows=rows))
        print('HELDOUT_FRAME0_POSE',row['video_id'],pose['status'],pose.get('cad_mask_iou'),flush=True)
    frozen_inputs(config)
    write(destination/'summary.json',dict(status='complete',rows=rows,frame0_count=len(rows),training_updates=0,normal_codebook_updates=0,
        normalization_sha256=sha(root/'link5_normalization.json'),source_sha256=sha(__file__),
        interpretation='Additional pose-aligned validation scenario; raw-frame0 diagnostics are retained. Native rigid alignment preserves cloud size/shape. Reconstruction scale errors are not corrected.'))
    print('LINK5_HELDOUT_POSE_CONTEXT_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4);run(args.config)
