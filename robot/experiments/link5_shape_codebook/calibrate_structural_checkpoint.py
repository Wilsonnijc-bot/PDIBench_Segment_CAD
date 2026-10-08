"""Normal-only leave-one-reference-out threshold calibration, without training."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from robot.experiments.link5_shape_codebook.common import observed_sample, raw_metrics, read, sha, write
from robot.experiments.link5_shape_codebook.structural_checkpoint import load_checkpoint
from robot.experiments.link5_shape_codebook.train_localization import frozen_inputs, load_points, prediction
from robot.experiments.link5_shape_codebook.train_structural_localization_v2 import acceptance


def run(config_path, variant, epoch):
    config=read(config_path);root,split,norm,heldout=frozen_inputs(config)
    heldout=[r for r in heldout if r['video_id'] not in ['COSMOS2.5_0030','LVP_ROBOWM_0005']]
    folder=root/'structural_localization/report_fix_v2'/variant
    checkpoint=folder/(f'epoch_{epoch:04d}.pt' if epoch else 'final.pt')
    model,payload=load_checkpoint(config,checkpoint);epoch=payload['epoch']
    if not hasattr(model,'reference_exclusion'):raise ValueError('leave-reference-out calibration applies to the geometry bank head')
    calibration=[];normal_rows=[]
    for i,row in enumerate(split['train']):
        model.reference_exclusion=i;xyz,indices=load_points(row,norm);arrays=prediction(model,xyz)
        calibration.append(arrays['raw_point_scores'])
        normal_rows.append(dict(video_id=row['video_id'],excluded_geometry_source=i,input_sha256=row['input_sha256'],**raw_metrics(arrays['raw_point_scores'])))
    threshold=float(np.quantile(np.concatenate(calibration),.995));model.reference_exclusion=None
    destination=folder/f'calibration_epoch_{epoch:04d}';destination.mkdir(exist_ok=True)
    contexts=read(root/'structural_localization/pose_alignment/summary.json')
    aligned={r['video_id']:r for r in contexts['rows'] if r['pose_status']=='ok'}
    clean_scores={}
    for row in heldout:
        xyz,indices=load_points(row,norm)
        clean_scores[('raw',row['video_id'])]=prediction(model,xyz)['raw_point_scores']
        if row['video_id'] in aligned:
            context=aligned[row['video_id']]
            if sha(context['aligned_path'])!=context['aligned_sha256']:raise ValueError('pose-aligned input changed')
            with np.load(context['aligned_path'],allow_pickle=False) as a:points,indices=observed_sample(a['points_normalized'],10000,seed=0)
            clean_scores[('aligned',row['video_id'])]=prediction(model,torch.from_numpy(points).cuda())['raw_point_scores']
    reports=[]
    paths=[('raw',folder/f'evaluation_epoch_{epoch:04d}.json'),
           ('aligned',folder/f'pose_validation_epoch_{epoch:04d}/evaluation_epoch_{epoch:04d}.json')]
    for scenario,path in paths:
        if not path.exists():continue
        report=copy.deepcopy(read(path));original_threshold=report['frozen_clean_threshold'];clean=[]
        for row in report['clean_clouds']:
            if row['split']=='train':continue
            scores=clean_scores[(scenario,row['video_id'])]
            row.update(false_positive_fraction=float((scores>threshold).mean()));clean.append(row)
        report['clean_clouds']=clean
        report['summary']['median_heldout_clean_false_positive_fraction']=float(np.median([r['false_positive_fraction'] for r in clean]))
        # Rank/separation metrics do not depend on the threshold. Preserve the
        # old threshold-based synthetic recalls under explicit historical names.
        for row in report['synthetic_cases']:
            for name in ['recall_at_frozen_clean_threshold','outside_false_positive_fraction']:
                row['previous_selfmatch_threshold_'+name]=row.pop(name)
        report['frozen_clean_threshold']=threshold
        report['acceptance']=acceptance(report['synthetic_cases'],clean,report['summary'])
        report.update(calibration='99.5%quantile of four normal source-excluded geometry-context predictions; no heldout/guide labels used',
                      calibration_normal_rows=normal_rows,previous_selfmatch_threshold=original_threshold,
                      source_report_sha256=sha(path),checkpoint_sha256=sha(checkpoint),source_sha256=sha(__file__),
                      synthetic_recall_at_new_threshold='not recomputed in this calibration diagnostic',
                      pose_context_complete=len(aligned)==39,training_updates=0,normal_codebook_updates=0)
        if scenario=='aligned' and len(aligned)!=39:
            report['acceptance']['status']='incomplete'
        write(destination/(scenario+'.json'),report)
        reports.append(dict(scenario=scenario,acceptance=report['acceptance'],summary=report['summary']))
        print('STRUCTURAL_CALIBRATED_REPORT',scenario,epoch,report['summary'],report['acceptance'],flush=True)
    write(destination/'threshold.json',dict(status='complete',threshold=threshold,quantile=.995,normal_sources=normal_rows,
          checkpoint_sha256=sha(checkpoint),heldout_sources_used_for_threshold=0,guide_sources_used_for_threshold=0,
          inference_geometry_bank='all four normals; source exclusion only for normal calibration',reports=reports))
    print('LINK5_NORMAL_ONLY_CALIBRATION_COMPLETE',epoch,threshold,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--variant',default='structural_geometry_head');parser.add_argument('--epoch',type=int,default=0)
    args=parser.parse_args();torch.set_num_threads(4);run(args.config,args.variant,args.epoch)
