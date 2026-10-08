"""Required alignment, rigid invariance, synthetic anomaly and heldout gates."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .common import MODES,mode_folder,read, sha, transform_points, write, split_path
from .score import load_detector, predict


def run(config_path,config,mode):
    output=Path(config['output']);settings=config['sanity'];split=read(split_path(output))
    heldout_path=output/'splits/heldout_frame0.json'
    if split.get('reference_only'):
        heldout_receipt=read(heldout_path)
        if heldout_receipt['status']!='complete' or heldout_receipt['training_split_sha256']!=sha(split_path(output)):
            raise ValueError('heldout manifest belongs to a different normal reference')
        split={**split,'test':heldout_receipt['test']}
        if {r['video_id'] for r in split['test']}!=set(split['declared_heldout_video_ids']):raise ValueError('heldout frame0 coverage mismatch')
        for row in split['test']:
            if row['frame_id']!=0 or row['input_sha256']!=sha(row['input_path']):raise ValueError('heldout input changed')
    model,modules,normalization=load_detector(output,mode)
    training=mode_folder(output,'training',mode)
    checks=dict(status='failed',augmentation_mode=mode,checkpoint_sha256=sha(training/'link5/final.pt'),
                normalization_sha256=sha(output/'link5_normalization.json'),split_sha256=sha(split_path(output)),
                alignment=read(output/'alignment/audit.json')['status'],thresholds=settings,pose_checks=[],synthetic_deformations=[])
    destination=mode_folder(output,'sanity',mode);destination.mkdir(parents=True,exist_ok=True)
    if split.get('reference_only'):checks['heldout_manifest_sha256']=sha(heldout_path)
    baseline=[];heldout=[]
    for kind,target in [('train',baseline),('test',heldout)]:
        for row in split[kind]:
            with np.load(row['input_path'],allow_pickle=False) as archive:points=archive['xyz_camera'].copy()
            arrays,metrics=predict(model,modules,points,normalization)
            target.append(dict(video_id=row['video_id'],**metrics))
            if kind=='test':
                np.savez_compressed(destination/(row['video_id']+'_heldout_frame0.npz'),**arrays)
    train_p95=float(np.quantile([r['raw_mean_top80'] for r in baseline],.95))
    heldout_p95=float(np.quantile([r['raw_mean_top80'] for r in heldout],.95))
    checks.update(training_normals=baseline,heldout_normals=heldout,train_score_p95=train_p95,heldout_score_p95=heldout_p95,
        heldout_normals_passed=heldout_p95<=settings['heldout_normal_p95_vs_train_p95_max_ratio']*max(train_p95,1e-8))
    row=split['test'][0]
    with np.load(row['input_path'],allow_pickle=False) as archive:normal=archive['xyz_camera'].copy()
    original,original_metrics=predict(model,modules,normal,normalization)
    angle=.7;c,s=np.cos(angle),np.sin(angle)
    motion=np.eye(4);motion[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];motion[:3,3]=[.13,-.21,.19]
    moved=transform_points(normal,motion);restored=transform_points(moved,np.linalg.inv(motion))
    arrays,metrics=predict(model,modules,restored,normalization)
    delta=float(abs(metrics['raw_mean_top80']-original_metrics['raw_mean_top80'])/max(original_metrics['raw_mean_top80'],1e-8))
    checks['synthetic_rigid_invariance']=dict(relative_score_change=delta,maximum_coordinate_error=float(abs(restored-normal).max()),
        passed=delta<=settings['rigid_score_relative_tolerance'])
    xyz=torch.from_numpy(original['points_normalized']).cuda()
    if mode=='paper_original':
        augment=modules['augmentation'].NegativeAugmentation(modules['augmentation'].AugConfig(severities=(.01,.1)))
        normals=modules['augmentation'].estimate_normals(xyz)
        tests=[(kind,str(severity),lambda kind=kind,severity=severity:augment(xyz,normals,atype=kind,severity=severity))
               for severity in (.01,.1) for kind in ('sink','concavity','bulges')]
    else:
        from .robot_structural import RobotStructuralAugmentation
        effective=read(training/'effective_config.json')
        augment=RobotStructuralAugmentation(effective['augmentation']['structural_geometry'],effective['augmentation']['robot_structural'])
        tests=[('shortening',str(alpha),lambda alpha=alpha:augment.axial(xyz,alpha,.8)) for alpha in (.75,.95)]
        tests += [('lengthening',str(alpha),lambda alpha=alpha:augment.axial(xyz,alpha,.8)) for alpha in (1.05,1.25)]
        tests += [('bending',str(angle),lambda angle=angle:augment.bend(xyz,angle,.2,.5)) for angle in (10,35)]
    for kind,parameter,generate in tests:
        torch.manual_seed(123);torch.cuda.manual_seed_all(123)
        anomalous,offset,mask=generate()
        torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        with torch.no_grad():
            pred,logits,_=model(anomalous[None])
            raw=(pred.abs().sum(-1)*logits.sigmoid())[0].cpu().numpy()
        from .common import raw_metrics
        score=raw_metrics(raw)['raw_mean_top80'];ratio=score/max(original_metrics['raw_mean_top80'],1e-8)
        checks['synthetic_deformations'].append(dict(type=kind,parameter=parameter,score=score,ratio=ratio,
                                                    actual_max_displacement=float(offset.norm(dim=-1).max()),
                                                    passed=ratio>=settings['synthetic_deformation_min_ratio']))
        np.savez_compressed(destination/f'synthetic_{kind}_{parameter}.npz',normal=xyz.cpu().numpy(),
            anomalous=anomalous.cpu().numpy(),gt_offset=offset.cpu().numpy(),gt_mask=mask.cpu().numpy(),raw_scores=raw)
    checks['synthetic_deformation_passed']=all(r['passed'] for r in checks['synthetic_deformations'])
    # The real rigid-pose gate ran before training. Revalidate its provenance;
    # never accept a receipt belonging to different normalization or observations.
    pose_receipt=read(output/'sanity/pose_checks.json')
    if pose_receipt['normalization_sha256']!=sha(output/'link5_normalization.json'):
        raise ValueError('pose gate normalization changed')
    if pose_receipt['thresholds']!=settings:raise ValueError('pose gate settings changed')
    for row in pose_receipt['pose_checks']:
        source=output/'cases'/row['video_id']/'observations'/f"frame_{row['frame_id']:05d}"/'input.npz'
        if sha(source)!=row['input_sha256']:raise ValueError('pose gate input changed')
    checks['pose_checks']=pose_receipt['pose_checks']
    required=[checks['alignment']=='passed',checks['heldout_normals_passed'],checks['synthetic_rigid_invariance']['passed'],
              checks['synthetic_deformation_passed'],bool(checks['pose_checks']) and all(r['passed'] for r in checks['pose_checks'])]
    checks['status']='passed' if all(required) else 'failed'
    write(destination/'checks.json',checks)
    if checks['status']!='passed':raise RuntimeError('basic Link5 sanity checks failed; complete evaluation remains disabled')
    print('LINK5_SANITY_CHECKS_PASSED',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--augmentation-mode',choices=MODES,default='paper_original')
    args=parser.parse_args();run(args.config,read(args.config),args.augmentation_mode)


if __name__=='__main__':main()
