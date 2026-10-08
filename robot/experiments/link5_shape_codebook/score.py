"""Original model/map inference plus requested raw scores and rigid alignment."""
import argparse
import csv
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from .common import ROOT, MODES, mode_folder, normalize, observed_sample, raw_metrics, read, reference_align, sha, write, split_path
from .upstream import build_model, restore_hash_keys


def load_detector(output,mode='paper_original'):
    training=mode_folder(output,'training',mode)
    checkpoint=training/'link5/final.pt'
    if read(training/'completion.json')['checkpoint_sha256']!=sha(checkpoint):raise ValueError('trained checkpoint changed')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload.get('augmentation_mode','paper_original')!=mode:raise ValueError('requested augmentation mode differs from checkpoint')
    normalization=read(output/'link5_normalization.json')
    if sha(output/'link5_normalization.json')!=payload['normalization_sha256']:raise ValueError('normalization changed after training')
    if sha(split_path(output))!=payload['split_sha256']:raise ValueError('training split changed')
    model,modules,_=build_model(payload['config'])
    model.load_state_dict(payload['model'],strict=True);restore_hash_keys(model,payload['codebook_hash_state'])
    model.cuda().eval()
    return model,modules,normalization


@torch.no_grad()
def predict(model,modules,points_reference,normalization):
    normalized=normalize(points_reference,normalization)
    points,ids=observed_sample(normalized,10000,seed=0)
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    offset,logits,aux=model(torch.from_numpy(points).cuda()[None])
    raw=(offset.abs().sum(-1)*torch.sigmoid(logits))[0].cpu().numpy()
    paper=modules['losses'].offset_to_score(offset,logits)[0].cpu().numpy()
    arrays=dict(raw_point_scores=raw,paper_point_scores=paper,predicted_offset=offset[0].cpu().numpy(),
                validity_logits=logits[0].cpu().numpy(),sampled_input_indices=ids,points_normalized=points,
                points_reference=np.asarray(points_reference)[ids],selected_scale=aux['best_scale'].cpu().numpy())
    return arrays,raw_metrics(raw)


def estimate_pose(config_path,config,row):
    input_path=Path(row['input_path']);folder=input_path.parent/'pose'
    if not (folder/'pose.json').exists():
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/'run.log').open('w') as stream:
            subprocess.run([config['environments']['foundationpose'],'-u','-m','robot.experiments.link5_shape_codebook.pose',
                '--config',str(config_path),'--input',str(input_path),'--output',str(folder)],cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT,check=False)
    record=read(folder/'pose.json')
    if record['input_sha256']!=sha(input_path):raise ValueError('pose input identity mismatch')
    return record


def score_row(config_path,config,row,model,modules,normalization,train_ids,mode):
    input_path=Path(row['input_path']);folder=input_path.parent
    suffix='' if mode=='paper_original' else '_robot_structural'
    result=dict(video_id=row['video_id'],frame_id=row['frame_id'],status='failed',
                reference_split='train' if row['video_id'] in train_ids else 'heldout',
                primary_evaluation_case=row['video_id'] not in train_ids,
                augmentation_mode=mode,primary_metric='raw_mean_top80',raw_score_units='fixed-normalized coordinates',visibility={k:row[k] for k in ('valid_point_count','mask_area','mask_bbox_coverage','image_boundary_contact')})
    try:
        with np.load(input_path,allow_pickle=False) as archive:points=archive['xyz_camera'].copy()
        if row['frame_id']==0:
            aligned=points;result['pose_policy']='unmodified frame0 camera reference; no FoundationPose'
        else:
            pose=estimate_pose(config_path,config,row)
            result['pose_status']=pose['status'];result['cad_mask_iou']=pose.get('cad_mask_iou')
            if pose['status']!='ok':
                raise ValueError('obvious pose failure; score unavailable, frame retained as failure')
            aligned=reference_align(points,normalization['T_ref'],pose['T_camera_from_link5'])
            result['pose_policy']='T_ref @ inv(T_camera_from_link5), independent SE3 registration'
        arrays,metrics=predict(model,modules,aligned,normalization)
        np.savez_compressed(folder/('anomaly'+suffix+'.npz'),**arrays)
        result.update(status='complete',**metrics,anomaly_path=str(folder/('anomaly'+suffix+'.npz')),
                      normalization_sha256=sha(Path(config['output'])/'link5_normalization.json'))
    except Exception as exc:result.update(error_type=type(exc).__name__,error=str(exc))
    write(folder/('score'+suffix+'.json'),result)
    return result


def summarize(output,rows,mode='paper_original'):
    scores=mode_folder(output,'scores',mode)
    videos=[]
    for video_id in sorted({r['video_id'] for r in rows}):
        selected=[r for r in rows if r['video_id']==video_id and r['frame_id']>0]
        valid=[r['raw_mean_top80'] for r in selected if r['status']=='complete']
        membership=next(r['reference_split'] for r in rows if r['video_id']==video_id)
        videos.append(dict(video_id=video_id,reference_split=membership,primary_evaluation_case=membership=='heldout',
                      sampled_count=len(selected),scored_count=len(valid),failed_count=len(selected)-len(valid),
                      max_sampled_frame_score=float(max(valid)) if valid else None,
                      median_sampled_frame_score=float(np.median(valid)) if valid else None,
                      mean_top3_sampled_frame_scores=float(np.mean(sorted(valid,reverse=True)[:3])) if valid else None))
    write(scores/'frames.json',rows);write(scores/'videos.json',videos)
    for name,items in [('videos',videos),('frames',[{k:v for k,v in r.items() if k!='visibility'}|r['visibility'] for r in rows])]:
        keys=list(dict.fromkeys(k for row in items for k in row))
        with (scores/f'{name}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(items)
    from scipy.stats import spearmanr
    good=[r for r in rows if r['status']=='complete' and r['frame_id']>0]
    correlations={}
    for key in ('valid_point_count','mask_area','mask_bbox_coverage','image_boundary_contact'):
        x=np.asarray([r['visibility'][key] for r in good],float);y=np.asarray([r['raw_mean_top80'] for r in good])
        rho,p=spearmanr(x,y) if len(x)>2 and np.ptp(x)>0 and np.ptp(y)>0 else (np.nan,np.nan)
        correlations[key]=dict(spearman_rho=float(rho) if np.isfinite(rho) else None,p_value=float(p) if np.isfinite(p) else None,count=len(x))
    write(scores/'visibility_correlations.json',correlations)
    return videos


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--augmentation-mode',choices=MODES,default='paper_original')
    args=parser.parse_args();config=read(args.config);output=Path(config['output']);mode=args.augmentation_mode
    training=mode_folder(output,'training',mode)
    receipt=read(mode_folder(output,'sanity',mode)/'checks.json')
    if receipt['status']!='passed':raise ValueError('basic sanity gate failed; complete45-video scoring forbidden')
    if receipt['checkpoint_sha256']!=sha(training/'link5/final.pt'):raise ValueError('sanity receipt belongs to a different detector')
    model,modules,normalization=load_detector(output,mode)
    train_ids={row['video_id'] for row in read(split_path(output))['train']}
    results=[]
    for case in config['cases']:
        for row in read(output/'cases'/case['id']/'observations.json'):
            result=score_row(args.config,config,row,model,modules,normalization,train_ids,mode);results.append(result)
            print('LINK5_SCORED',case['id'],row['frame_id'],result['status'],result.get('raw_mean_top80'),flush=True)
        summarize(output,results,mode)
    videos=summarize(output,results,mode)
    write(output/('completion.json' if mode=='paper_original' else 'completion_robot_structural.json'),dict(status='complete' if all(v['failed_count']==0 for v in videos) else 'complete_with_failures',
        video_count=len(videos),frame_count=len(results),augmentation_mode=mode,primary_metric='raw_mean_top80',paper_exact=False,
        checkpoint_sha256=sha(training/'link5/final.pt')))
    print('LINK5_EVALUATION_FINISHED',flush=True)


if __name__=='__main__':main()
