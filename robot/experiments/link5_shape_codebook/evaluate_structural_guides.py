"""Read-only checkpoint predictions on the exact three observed shape guides."""
import argparse
from pathlib import Path

import numpy as np
import torch

from robot.experiments.link5_shape_codebook.common import observed_sample, raw_metrics, read, sha, write
from robot.experiments.link5_shape_codebook.structural_checkpoint import load_checkpoint
from robot.experiments.link5_shape_codebook.train_localization import prediction


def run(config_path, variant, epoch, trial='report_fix_v2'):
    config=read(config_path);root=Path(config['output']);folder=root/'structural_localization'/trial/variant
    checkpoint=folder/(f'epoch_{epoch:04d}.pt' if epoch else 'final.pt')
    model,payload=load_checkpoint(config,checkpoint);epoch=payload['epoch']
    pose=read(root/'guides/pose_summary.json');manifest=read(root/'guides/manifest.json')
    if pose['normalization_sha256']!=sha(root/'link5_normalization.json') or pose['guide_manifest_sha256']!=sha(root/'guides/manifest.json'):
        raise ValueError('guide alignment provenance mismatch')
    destination=folder/f'guide_predictions_epoch_{epoch:04d}';destination.mkdir(exist_ok=True)
    result=[]
    for guide in manifest['guides']:
        metrics=[]
        for row in [r for r in pose['rows'] if r['video_id']==guide['video_id']]:
            if row['pose_status']!='ok':
                metrics.append(dict(frame_id=row['frame_id'],status='unavailable',pose_status=row['pose_status']));continue
            if sha(row['aligned_path'])!=row['aligned_sha256']:raise ValueError('aligned guide identity changed')
            with np.load(row['aligned_path'],allow_pickle=False) as a:points,indices=observed_sample(a['points_normalized'],10000,seed=0)
            arrays=prediction(model,torch.from_numpy(points).cuda())
            archive=destination/f"{guide['video_id']}_{row['frame_id']:05d}.npz"
            if archive.exists():
                with np.load(archive) as a:
                    for name,x in arrays.items():np.testing.assert_array_equal(a[name],x)
            else:np.savez_compressed(archive,**arrays,sampled_input_indices=indices)
            metrics.append(dict(frame_id=row['frame_id'],status='complete',cad_mask_iou=row['cad_mask_iou'],
                                pose_sha256=row['pose_sha256'],input_sha256=row['input_sha256'],aligned_sha256=row['aligned_sha256'],
                                archive=str(archive),archive_sha256=sha(archive),**raw_metrics(arrays['raw_point_scores'])))
        by_id={r['frame_id']:r for r in metrics}
        good=all(by_id[i]['status']=='complete' for i in [0,guide['frame_id']])
        ratio=by_id[guide['frame_id']]['raw_mean_top80']/max(by_id[0]['raw_mean_top80'],1e-8) if good else None
        result.append(dict(video_id=guide['video_id'],family=guide['guide_family'],requested_frame_id=guide['frame_id'],
                           selected_frame_vs_frame0_raw_top80_ratio=ratio,frames=metrics))
        print('STRUCTURAL_REAL_GUIDE',variant,guide['video_id'],ratio,flush=True)
    write(destination/'result.json',dict(status='complete',variant=variant,epoch=epoch,checkpoint_sha256=sha(checkpoint),guides=result,
          training_updates=0,normal_codebook_updates=0,dense_defect_ground_truth=False,
          interpretation='Three user-selected qualitative shape guides, not independent labeled validation. Different visibility, rigid pose fitting and native reconstruction scale can affect observed-frame scores.'))
    print('LINK5_STRUCTURAL_GUIDE_EVALUATION_COMPLETE',variant,epoch,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--variant',choices=['structural_native','structural_point_residual','structural_geometry_head','structural_invariant_geometry_head','structural_profile_decoder'],required=True)
    parser.add_argument('--epoch',type=int,default=0)
    parser.add_argument('--trial',choices=['report_fix_v2','benign_finetune_v3','benign_finetune_v4','robust_shape_v5','profile_shape_v6'],default='report_fix_v2')
    args=parser.parse_args();torch.set_num_threads(4);run(args.config,args.variant,args.epoch,args.trial)
