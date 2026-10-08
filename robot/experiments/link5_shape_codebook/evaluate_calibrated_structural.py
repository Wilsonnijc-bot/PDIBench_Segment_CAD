"""Normal-only calibration and controlled structural localization evaluation.

All threshold inputs are independent benign draws from the four fixed references.
No guide or heldout observation is used to fit the model or choose the threshold.
Raw-camera and native-FoundationPose-aligned scenarios are both retained.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from robot.experiments.link5_shape_codebook.benign_variation import DEFAULTS, paired_variation
from robot.experiments.link5_shape_codebook.common import observed_sample, raw_metrics, read, sha, write
from robot.experiments.link5_shape_codebook.robot_structural import RobotStructuralAugmentation
from robot.experiments.link5_shape_codebook.structural_checkpoint import load_checkpoint
from robot.experiments.link5_shape_codebook.train_localization import frozen_inputs, load_points, localization_metrics, prediction
from robot.experiments.link5_shape_codebook.train_structural_localization_v2 import acceptance, evaluation_grid


def summarize(rows, clean):
    local=[r for r in rows if r['positive_count']>=32 and r['negative_count']>=32]
    if len(rows)!=390 or len(local)!=390 or len(clean)!=39:
        raise ValueError('complete 39 clean / 390 spatially labeled synthetic cases required')
    summary=dict(synthetic_cases=len(rows),localization_cases=len(local),
                 median_point_auroc=float(np.median([r['point_auroc'] for r in local])),
                 median_point_average_precision=float(np.median([r['point_average_precision'] for r in local])),
                 median_region_vs_outside_ratio=float(np.median([r['region_vs_outside_ratio'] for r in local])),
                 median_matched_region_score_ratio=float(np.median([r['matched_region_score_ratio'] for r in local])),
                 median_raw_top80_ratio=float(np.median([r['raw_top80_ratio'] for r in rows])),
                 median_heldout_clean_false_positive_fraction=float(np.median([r['false_positive_fraction'] for r in clean])),
                 median_defect_point_recall_at_normal_threshold=float(np.median([r['recall_at_frozen_clean_threshold'] for r in local])),
                 median_synthetic_outside_false_positive_fraction=float(np.median([r['outside_false_positive_fraction'] for r in local])))
    strong={}
    for kind in ('shortening','lengthening','bending'):
        cases=[r for r in rows if r['type']==kind and (r['example'] in ('shortening_25pct','lengthening_25pct') or
               (kind=='bending' and r['parameters']['angle_degrees']>=35))]
        strong[kind]=dict(cases=len(cases),median_auroc=float(np.median([r['point_auroc'] for r in cases])),
                         median_defect_point_recall=float(np.median([r['recall_at_frozen_clean_threshold'] for r in cases])),
                         median_region_outside_ratio=float(np.median([r['region_vs_outside_ratio'] for r in cases])),
                         median_matched_clean_ratio=float(np.median([r['matched_region_score_ratio'] for r in cases])),
                         median_raw_top80_ratio=float(np.median([r['raw_top80_ratio'] for r in cases])))
    summary['strong_family_metrics']=strong
    return summary


@torch.no_grad()
def run(config_path, trial, epoch, copies, variant='structural_geometry_head'):
    config=read(config_path);root,split,norm,heldout=frozen_inputs(config)
    heldout=[r for r in heldout if r['video_id'] not in ['COSMOS2.5_0030','LVP_ROBOWM_0005']]
    if len(heldout)!=39:raise ValueError('exact development-validation coverage required')
    folder=root/'structural_localization'/trial/variant
    checkpoint=folder/(f'epoch_{epoch:04d}.pt' if epoch else 'final.pt')
    model,payload=load_checkpoint(config,checkpoint);epoch=payload['epoch'];model.eval()
    destination=folder/f'normal_calibrated_epoch_{epoch:04d}'
    if (destination/'completion.json').exists():
        old=read(destination/'completion.json')
        if old['checkpoint_sha256']==sha(checkpoint) and old['copies_per_reference']==copies and old['source_sha256']==sha(__file__):
            print('LINK5_CALIBRATED_STRUCTURAL_ALREADY_COMPLETE',trial,epoch,flush=True);return
        raise ValueError('preserve an evaluation with different identity/settings')
    destination.mkdir(exist_ok=True);(destination/'predictions').mkdir(exist_ok=True)
    samples=[(r,*load_points(r,norm)) for r in split['train']]
    calibration=[];calibration_rows=[];cursor=0
    for i,(row,xyz,indices) in enumerate(samples):
        model.reference_exclusion=i
        for j in range(copies+1):
            seed=900000+i*10000+j
            if j==0:clean=xyz;parameters=dict(identity=True,exact_observation=True)
            else:
                clean,_,_,_,parameters=paired_variation(xyz,xyz,xyz.new_zeros(len(xyz)),seed)
            scores=prediction(model,clean)['raw_point_scores'];calibration.append(scores)
            calibration_rows.append(dict(video_id=row['video_id'],copy=j,seed=seed,excluded_geometry_source=i,
                                         source_input_sha256=row['input_sha256'],parameters=parameters,
                                         score_begin=cursor,score_end=cursor+len(scores),**raw_metrics(scores)))
            cursor+=len(scores)
    values=np.concatenate(calibration);threshold=float(np.quantile(values,.995));model.reference_exclusion=None
    np.savez_compressed(destination/'normal_calibration.npz',raw_scores=values)
    write(destination/'threshold.json',dict(status='complete',threshold=threshold,quantile=.995,
          copies_per_reference=copies,exact_reference_copies=4,benign_derived_copies=4*copies,
          independent_seed_range='900000 + source_index*10000 + draw_index',benign_settings=DEFAULTS,
          calibration_rows=calibration_rows,score_archive_sha256=sha(destination/'normal_calibration.npz'),
          checkpoint_sha256=sha(checkpoint),source_sha256=sha(__file__),
          guide_threshold_inputs=0,heldout_threshold_inputs=0,normal_codebook_updates=0))
    print('STRUCTURAL_NORMAL_CALIBRATION',trial,epoch,threshold,flush=True)
    pose_path=root/'structural_localization/pose_alignment/summary.json';contexts=read(pose_path)
    aligned={r['video_id']:r for r in contexts['rows'] if r['pose_status']=='ok'}
    if contexts['status']!='complete' or contexts['normalization_sha256']!=sha(root/'link5_normalization.json') or set(aligned)!={r['video_id'] for r in heldout}:
        raise ValueError('exact native pose alignment coverage/provenance required')
    distribution=payload['policy']['augmentation_distribution']
    generator=RobotStructuralAugmentation(read(root/'augmentation/robot_structural_geometry.json'),distribution['settings'])
    reports=[]
    for scenario in ('raw','aligned'):
        clean_receipts=[];rows=[]
        for row in heldout:
            if scenario=='raw':xyz,indices=load_points(row,norm)
            else:
                context=aligned[row['video_id']]
                if sha(context['aligned_path'])!=context['aligned_sha256'] or sha(context['input_path'])!=context['input_sha256']:
                    raise ValueError('native aligned input identity changed')
                with np.load(context['aligned_path'],allow_pickle=False) as a:
                    points,indices=observed_sample(a['points_normalized'],10000,seed=0)
                xyz=torch.from_numpy(points).cuda()
            clean=prediction(model,xyz);scores=clean['raw_point_scores']
            clean_receipts.append(dict(video_id=row['video_id'],split='heldout',
                                  false_positive_fraction=float((scores>threshold).mean()),**raw_metrics(scores)))
            archive=destination/'predictions'/f"{scenario}_clean_{row['video_id']}.npz"
            np.savez_compressed(archive,**clean,sampled_input_indices=indices)
            for name,kind,parameters in evaluation_grid(distribution):
                anomalous,offset,mask=generator.bend(xyz,**parameters) if kind=='bending' else generator.axial(xyz,**parameters)
                torch.testing.assert_close(xyz-anomalous,offset,rtol=0,atol=1e-7)
                arrays=prediction(model,anomalous)
                metrics=localization_metrics(scores,arrays['raw_point_scores'],mask.cpu().numpy(),offset.cpu().numpy(),threshold)
                rows.append(dict(video_id=row['video_id'],frame_id=0,type=kind,example=name,parameters=parameters,**metrics))
                if row['video_id']=='LVP_ROBOWM_0001':
                    np.savez_compressed(destination/'predictions'/f"{scenario}_{name}_{row['video_id']}.npz",**arrays,
                                        normal=xyz.cpu().numpy(),anomalous=anomalous.cpu().numpy(),
                                        gt_offset=offset.cpu().numpy(),gt_mask=mask.cpu().numpy(),sampled_input_indices=indices)
        summary=summarize(rows,clean_receipts);gate=acceptance(rows,clean_receipts,summary)
        # Retain the fixed original gate and additionally disclose threshold
        # sensitivity, so a high threshold cannot hide an ineffective detector.
        strong_recall={k:bool(v['median_defect_point_recall']>=.5) for k,v in summary['strong_family_metrics'].items()}
        report=dict(status='complete',epoch=epoch,trial=trial,scenario=scenario,training_inputs=4,
                    independent_optimizer_input_heldout_frame0s=39,synthetic_cases=rows,clean_clouds=clean_receipts,
                    frozen_clean_threshold=threshold,summary=summary,acceptance=gate,
                    additional_strong_threshold_sensitivity=strong_recall,
                    effective_research_check=bool(gate['status']=='passed' and all(strong_recall.values())),
                    checkpoint_sha256=sha(checkpoint),source_sha256=sha(__file__),pose_context_sha256=sha(pose_path),
                    threshold_recipe='99.5% quantile of four exact source-excluded normals and independent benign copies from those same four normals',
                    threshold_input_sources=4,guide_calibration_sources=0,heldout_calibration_sources=0,
                    training_updates=0,normal_codebook_updates=0,production_gate_changed=False,
                    interpretation='Development validation used to diagnose/select models, not a blind final benchmark. Real guides have no dense defect GT. Aligned synthetic examples are generated after rigid pose removal; raw-camera results remain available.')
        write(destination/(scenario+'.json'),report)
        reports.append(dict(scenario=scenario,summary=summary,acceptance=gate,
                            effective_research_check=report['effective_research_check']))
        print('STRUCTURAL_NORMAL_CALIBRATED_EVALUATION',trial,epoch,scenario,summary,gate,flush=True)
    write(destination/'completion.json',dict(status='complete',checkpoint_sha256=sha(checkpoint),epoch=epoch,trial=trial,
          copies_per_reference=copies,source_sha256=sha(__file__),reports=reports,training_updates=0,normal_codebook_updates=0))
    print('LINK5_CALIBRATED_STRUCTURAL_EVALUATION_COMPLETE',trial,epoch,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--trial',choices=['report_fix_v2','benign_finetune_v3','benign_finetune_v4','robust_shape_v5','profile_shape_v6'],required=True)
    parser.add_argument('--variant',choices=['structural_geometry_head','structural_invariant_geometry_head','structural_profile_decoder'],default='structural_geometry_head')
    parser.add_argument('--epoch',type=int,default=0);parser.add_argument('--copies',type=int,default=100)
    args=parser.parse_args();torch.set_num_threads(4)
    if not 1<=args.copies<10000:raise ValueError('positive bounded calibration-copy count required')
    run(args.config,args.trial,args.epoch,args.copies,args.variant)
