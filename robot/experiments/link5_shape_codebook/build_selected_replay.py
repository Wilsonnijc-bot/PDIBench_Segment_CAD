"""Publish only the exact four-cloud reference and observed test predictions.

No inference, new geometry, score normalization or historical diagnostic tabs.
The single shared display ceiling is three times the saved normal threshold;
the hover values and detector decisions retain the original raw scores.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import shutil

import numpy as np

from .common import FOUR_REFERENCE_IDS, ROOT, normalize, read, sha, write


def packed(array):
    values=np.ascontiguousarray(array,dtype='<f4')
    encoded=base64.b64encode(values.tobytes()).decode('ascii')
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(encoded),dtype='<f4').reshape(values.shape),values)
    return encoded


def run(root,destination,trial,variant,epoch):
    root=Path(root).resolve();destination=Path(destination).resolve()
    folder=root/'structural_localization'/trial/variant
    calibrated=folder/f'normal_calibrated_epoch_{epoch:04d}'
    report=read(calibrated/'aligned.json');done=read(calibrated/'completion.json')
    checkpoint=folder/f'epoch_{epoch:04d}.pt'
    if not checkpoint.exists():checkpoint=folder/'final.pt'
    checkpoint_sha=sha(checkpoint)
    if report['checkpoint_sha256']!=checkpoint_sha or done['checkpoint_sha256']!=checkpoint_sha or done['status']!='complete':
        raise ValueError('complete predictions and exact checkpoint required')
    policy=read(folder/'training_policy.json')
    if {r['video_id'] for r in policy['normal_manifest']}!=set(FOUR_REFERENCE_IDS) or any(r['frame_id']!=0 for r in policy['normal_manifest']):
        raise ValueError('only the four normal frame0 sources are allowed')
    construction=read(root/'normal_reference/construction.json')
    if construction['normal_video_ids']!=list(FOUR_REFERENCE_IDS) or construction['normal_frame_ids']!=[0]*4 or construction['later_frames_in_reference']:
        raise ValueError('four exact normal frame0 clouds required')
    pooled=root/'normal_reference/pooled_frame0.npz'
    if sha(pooled)!=construction['pooled_sha256']:raise ValueError('normal reference changed')
    norm=read(root/'link5_normalization.json')
    sources=[]
    for row in construction['observations']:
        path=root/'cases'/row['video_id']/'observations/frame_00000/input.npz'
        if sha(path)!=row['input_sha256']:raise ValueError('normal source changed')
        with np.load(path,allow_pickle=False) as a:sources.append(a['xyz_camera'].copy())
    with np.load(pooled,allow_pickle=False) as a:
        np.testing.assert_array_equal(a['xyz_camera'],np.concatenate(sources))
        normal=normalize(a['xyz_camera'],norm)
    threshold=float(report['frozen_clean_threshold'])
    if not np.isfinite(threshold) or threshold<=0:raise ValueError('finite positive saved threshold required')
    destination.mkdir(parents=True,exist_ok=True);clouds=destination/'clouds';clouds.mkdir(exist_ok=True)
    href=lambda path:os.path.relpath(path,destination)
    reference_packet=clouds/('normal-'+sha(pooled)[:12]+'.js')
    reference_packet.write_text('window.LINK5_SELECTED_NORMAL('+json.dumps(dict(points=packed(normal)),separators=(',',':'))+');\n')
    payload=dict(checkpoint=href(checkpoint),threshold=threshold,color_max=3*threshold,
                 effective=bool(report['effective_research_check']),normal=dict(packet=href(reference_packet),point_count=len(normal)),tests=[])
    records=[]

    def append(path,video,frame,label,weak_pose=False):
        with np.load(path,allow_pickle=False) as a:
            points=a['points_normalized'].copy();scores=a['raw_point_scores'].copy()
            offset=a['predicted_offset'];logits=a['validity_logits']
            np.testing.assert_allclose(np.abs(offset).sum(-1)/(1+np.exp(-logits)),scores,rtol=2e-6,atol=1e-7)
        if points.shape!=(len(scores),3) or not np.isfinite(points).all() or not np.isfinite(scores).all():
            raise ValueError('finite saved point/score correspondence required')
        key=f'{video}-f{frame}';packet=clouds/(key+'-'+checkpoint_sha[:12]+'.js')
        packet.write_text('window.LINK5_SELECTED_TEST('+json.dumps(dict(key=key,points=packed(points),scores=packed(scores)),separators=(',',':'))+');\n')
        payload['tests'].append(dict(key=key,label=label,packet=href(packet),point_count=len(points),
                                    flagged_fraction=float((scores>threshold).mean()),weak_pose=bool(weak_pose)))
        records.append(dict(video_id=video,frame_id=frame,archive=href(path),archive_sha256=sha(path),
                            packet_sha256=sha(packet),point_count=len(points),raw_min=float(scores.min()),raw_max=float(scores.max())))

    guide_folder=folder/f'guide_predictions_epoch_{epoch:04d}'
    guides=read(guide_folder/'result.json')
    if guides['checkpoint_sha256']!=checkpoint_sha:raise ValueError('guide checkpoint differs')
    labels={'LVP_ROBOWM_0005':'LVP 005 · last frame (3s)','COSMOS2.5_0018':'COSMOS2.5 018 · 3s','COSMOS2.5_0030':'COSMOS2.5 030 · 5s'}
    for video in labels:
        guide=next(g for g in guides['guides'] if g['video_id']==video)
        item=next(f for f in guide['frames'] if f['frame_id']!=0)
        path=guide_folder/Path(item['archive']).name
        if item['status']!='complete' or sha(path)!=item['archive_sha256']:raise ValueError('real guide prediction changed')
        append(path,video,item['frame_id'],labels[video],item['cad_mask_iou']<.3)
    for item in sorted(report['clean_clouds'],key=lambda r:r['video_id']):
        video=item['video_id'];path=calibrated/'predictions'/f'aligned_clean_{video}.npz'
        family,number=video.rsplit('_',1);family='LVP' if family=='LVP_ROBOWM' else family
        append(path,video,0,f'{family} {int(number):03d} · frame0')
    if len(payload['tests'])!=42 or len({r['key'] for r in payload['tests']})!=42:
        raise ValueError('exact three real guide frames and39 observed test frame0s required')
    payload['default_test']=payload['tests'][0]['key']
    (destination/'data.js').write_text('window.LINK5_SELECTED='+json.dumps(payload,separators=(',',':'))+';\n')
    source=Path(__file__).parent
    for filename,target in [('selected_replay.html','index.html'),('selected_replay_app.js','app.js'),('selected_replay.css','style.css')]:
        shutil.copy2(source/filename,destination/target)
    page=(destination/'index.html').read_text()
    for target in ('data.js','app.js','style.css'):
        page=page.replace('"'+target+'"','"'+target+'?v='+sha(destination/target)[:12]+'"')
    (destination/'index.html').write_text(page)
    asset=ROOT/'infrastructure/shared/replay/assets/plotly.min.js'
    if not (destination/'plotly.min.js').exists():shutil.copy2(asset,destination/'plotly.min.js')
    keep={'index.html','app.js','style.css','data.js','plotly.min.js','clouds','provenance.json'}
    removed=[]
    for path in list(destination.iterdir()):
        if path.name in keep:continue
        if path.is_symlink():raise ValueError('unexpected replay symlink: '+str(path))
        if path.is_dir():
            files=[p for p in path.rglob('*') if p.is_file()]
            removed.append(dict(name=path.name,files=len(files),bytes=sum(p.stat().st_size for p in files)))
            shutil.rmtree(path)
        else:
            removed.append(dict(name=path.name,files=1,bytes=path.stat().st_size));path.unlink()
    required={reference_packet.name}|{Path(t['packet']).name for t in payload['tests']}
    for path in list(clouds.iterdir()):
        if path.name not in required:path.unlink()
    provenance=dict(status='verified',trial=trial,variant=variant,epoch=epoch,checkpoint=href(checkpoint),checkpoint_sha256=checkpoint_sha,
                    normal_archive=href(pooled),normal_sha256=sha(pooled),normal_frame_ids=[0]*4,normal_video_ids=list(FOUR_REFERENCE_IDS),
                    normal_point_count=len(normal),normal_exact_concatenation=True,normalization_sha256=sha(root/'link5_normalization.json'),
                    validation=href(calibrated/'aligned.json'),validation_sha256=sha(calibrated/'aligned.json'),
                    effective_research_check=payload['effective'],production_promotion=False,
                    display_only_candidate=True,threshold=threshold,color_max=payload['color_max'],color_ceiling_multiple=3,
                    score_formula_verified=True,inference_rerun=False,observed_test_clouds=records,
                    synthetic_clouds_in_replay=0,guide_frame0_companions_in_replay=0,removed_replay_artifacts=removed,
                    removed_replay_views=['VLM prompts and masks','synthetic deformation examples','default generator examples',
                                          'optimizer pairs','training comparisons','historical model predictions'])
    write(destination/'provenance.json',provenance)
    print('LINK5_SELECTED_REPLAY_READY',len(normal),len(records),'observed tests; effective:',payload['effective'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--trial',required=True);p.add_argument('--variant',required=True);p.add_argument('--epoch',type=int,required=True)
    a=p.parse_args();run(a.root,a.destination,a.trial,a.variant,a.epoch)
