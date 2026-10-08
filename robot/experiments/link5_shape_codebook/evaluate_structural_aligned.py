"""Read-only, pose-aligned heldout validation in the real pipeline's frame."""
import argparse
from pathlib import Path

import numpy as np
import torch

from robot.experiments.link5_shape_codebook.common import observed_sample, read, sha, write
from robot.experiments.link5_shape_codebook.robot_structural import RobotStructuralAugmentation
from robot.experiments.link5_shape_codebook.structural_checkpoint import load_checkpoint
from robot.experiments.link5_shape_codebook.train_localization import frozen_inputs
import robot.experiments.link5_shape_codebook.train_structural_localization_v2 as training


def run(config_path, variant, epoch, trial='report_fix_v2', save_predictions=False):
    config=read(config_path);root,split,norm,heldout=frozen_inputs(config)
    pose_path=root/'structural_localization/pose_alignment/summary.json';poses=read(pose_path)
    if poses['status']!='complete' or poses['normalization_sha256']!=sha(root/'link5_normalization.json'):
        raise ValueError('complete frozen-normalization pose context required')
    expected=[r for r in heldout if r['video_id'] not in ['COSMOS2.5_0030','LVP_ROBOWM_0005']]
    if len(expected)!=39 or {r['video_id'] for r in expected}!={r['video_id'] for r in poses['rows']}:
        raise ValueError('exact39 heldout pose coverage required')
    failed=[r for r in poses['rows'] if r['pose_status']!='ok']
    aligned={r['video_id']:r for r in poses['rows'] if r['pose_status']=='ok'}
    for row in aligned.values():
        if sha(row['aligned_path'])!=row['aligned_sha256'] or sha(row['input_path'])!=row['input_sha256']:
            raise ValueError('pose-aligned source identity changed')
    folder=root/'structural_localization'/trial/variant
    checkpoint=folder/(f'epoch_{epoch:04d}.pt' if epoch else 'final.pt')
    model,payload=load_checkpoint(config,checkpoint);epoch=payload['epoch']
    distribution=payload['policy']['augmentation_distribution']
    generator=RobotStructuralAugmentation(read(root/'augmentation/robot_structural_geometry.json'),distribution['settings'])
    destination=folder/f'pose_validation_epoch_{epoch:04d}';destination.mkdir(exist_ok=True)
    if save_predictions:(destination/'predictions').mkdir(exist_ok=True)
    original_loader=training.load_points
    def loader(row, normalization):
        if row['video_id'] not in aligned:return original_loader(row,normalization)
        with np.load(aligned[row['video_id']]['aligned_path'],allow_pickle=False) as a:
            points,indices=observed_sample(a['points_normalized'],10000,seed=0)
        return torch.from_numpy(points).cuda(),indices
    # Substitute only this experiment adapter's observation loader in this
    # separate read-only process. Numerical detector/generator/loss are unchanged.
    training.load_points=loader
    try:
        receipt=training.evaluate(model,split,norm,[r for r in expected if r['video_id'] in aligned],generator,distribution,
                                  destination,epoch,final=save_predictions)
    finally:training.load_points=original_loader
    receipt.update(scenario='native rigid pose-aligned normal frame0, then corresponding synthetic structural deformation',
                   original_raw_frame0_report_preserved=True,checkpoint_sha256=sha(checkpoint),pose_summary_sha256=sha(pose_path),
                   pose_failed_inputs=failed,source_sha256=sha(__file__),training_updates=0,normal_codebook_updates=0,
                   interpretation='Additional validation scenario matching test-time pose alignment; fixed center/scale, no reconstruction scale correction. Raw camera-frame validation remains available.')
    if failed:
        receipt['acceptance']['status']='incomplete';receipt['acceptance']['checks']['pose_coverage']=False
    write(destination/f'evaluation_epoch_{epoch:04d}.json',receipt)
    print('LINK5_POSE_ALIGNED_STRUCTURAL_EVALUATION_COMPLETE',variant,epoch,receipt['summary'],receipt['acceptance']['status'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--variant',choices=['structural_point_residual','structural_geometry_head','structural_invariant_geometry_head'],required=True)
    parser.add_argument('--epoch',type=int,default=0)
    parser.add_argument('--trial',choices=['report_fix_v2','benign_finetune_v3','benign_finetune_v4','robust_shape_v5'],default='report_fix_v2')
    parser.add_argument('--save-predictions',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4);run(args.config,args.variant,args.epoch,args.trial,args.save_predictions)
