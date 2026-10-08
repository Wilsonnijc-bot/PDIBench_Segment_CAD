"""Export exact saved observations/predictions to an offline interactive replay.

No inference. One frame script is loaded at a time; original point correspondence
and raw scores are preserved as float32 arrays, with shared display color scales.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np

from .common import ROOT, MODES, normalize, raw_metrics, read, reference_align, sha, write
from .full_video import location, validate_contract
from .replay_framework import render as render_published_framework


def packed(array,dtype):
    values=np.ascontiguousarray(array,dtype=dtype)
    encoded=base64.b64encode(values.tobytes()).decode('ascii')
    if not np.array_equal(np.frombuffer(base64.b64decode(encoded),dtype=dtype).reshape(values.shape),values):
        raise ValueError('replay binary roundtrip failed')
    return encoded


def run(config, destination=None):
    validate_contract(config);root=Path(config['output']);full=location(config)
    destination=Path(destination) if destination else full/'replay';destination.mkdir(parents=True,exist_ok=True)
    previous=read(destination/'export_state.json') if (destination/'export_state.json').exists() else {}
    scope_path=full/'metadata/evaluation_scope.json'
    scope=read(scope_path) if scope_path.exists() else None
    selected=set(scope['selected_video_ids']) if scope else {c['id'] for c in config['cases']}
    selected_cases=[c for c in config['cases'] if c['id'] in selected]
    state={};cases=[];maxima={m:0. for m in MODES};counts=dict(videos=len(selected_cases),full_video_scored_frames={m:0 for m in MODES},display_frames=0)
    normalization=read(root/'link5_normalization.json')
    preflight=read(full/'metadata/preflight.json') if (full/'metadata/preflight.json').exists() else None
    expected={x['video_id']:x['frame_count'] for x in preflight['cases']} if preflight else {}
    def href(path):return os.path.relpath(path,destination)
    for case in selected_cases:
        folder=full/'cases'/case['id'];original=root/'cases'/case['id'];rows=[]
        if (folder/'preparation.json').exists():
            info=read(folder/'preparation.json');n=info['frame_count'];fps=info['fps'];status='prepared'
        else:
            cap=cv2.VideoCapture(case['video']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));cap.release();status='queued'
        if case['id'] in expected and expected[case['id']]!=n:raise ValueError('source frame coverage changed')
        if (folder/'worker.json').exists():status=read(folder/'worker.json')['status']
        summaries={m:read(folder/f'video_{m}.json') if (folder/f'video_{m}.json').exists() else None for m in MODES}
        for fid in range(n):
            observation=folder/'observations'/f'frame_{fid:05d}'
            # Until dense preparation exists, only exact existing frame0 evidence
            # can be shown. It is explicitly labeled as a sanity prediction.
            if not (observation/'input.npz').exists() and fid==0:observation=original/'observations/frame_00000'
            input_path=observation/'input.npz';packet_path=destination/'data'/case['id']/f'frame_{fid:05d}.js'
            frame=dict(frame_id=fid,images={},models={},status='awaiting full-video observation',packet=None,valid_point_count=None,pose=None)
            if not input_path.exists() or not (observation/'observation.json').exists():rows.append(frame);continue
            info=read(observation/'observation.json');frame.update(status=info.get('status','complete'),valid_point_count=info['valid_point_count'])
            frame['images']={name:href(observation/file) for name,file in (('rgb','rgb.png'),('mask','mask.png')) if (observation/file).exists()}
            pose_path=observation/'pose/pose.json';pose=read(pose_path) if pose_path.exists() else None
            if pose:
                frame['pose']={k:pose.get(k) for k in ('status','cad_mask_iou','pose_rank_score','mask_association_fallback_used')}
                if (pose_path.parent/'cad_projection.png').exists():frame['images']['cad']=href(pose_path.parent/'cad_projection.png')
            paths={};fingerprints=[input_path,observation/'observation.json']
            if pose_path.exists():fingerprints.append(pose_path)
            for mode in MODES:
                suffix='' if mode=='paper_original' else '_robot_structural';score_path=observation/f'score{suffix}.json';prediction=observation/f'anomaly{suffix}.npz'
                if score_path.exists():
                    score=read(score_path);origin='full_video';fingerprints.append(score_path)
                elif fid==0 and (root/('sanity' if mode=='paper_original' else 'sanity_robot_structural')/(case['id']+'_heldout_frame0.npz')).exists():
                    receipt=root/('sanity' if mode=='paper_original' else 'sanity_robot_structural')/'checks.json'
                    score=next(x for x in read(receipt)['heldout_normals'] if x['video_id']==case['id'])
                    score={**score,'status':'complete'};origin='frame0_sanity'
                    prediction=receipt.parent/(case['id']+'_heldout_frame0.npz');fingerprints.append(receipt)
                else:score=dict(status='pending');origin=None
                item={k:score.get(k) for k in ('status','raw_mean_top80','raw_mean','raw_p95','raw_max','error')};item['origin']=origin
                frame['models'][mode]=item
                if score['status']=='complete' and prediction.exists():
                    paths[mode]=(prediction,score);fingerprints.append(prediction)
                    if origin=='full_video':counts['full_video_scored_frames'][mode]+=1
            signature='|'.join(str(p)+':'+str(p.stat().st_size)+':'+str(p.stat().st_mtime_ns) for p in fingerprints)
            key=case['id']+'/'+str(fid);state[key]=signature
            # Shared raw color range comes from saved frame p95, never per-frame
            # minmax scores. Scores themselves are not changed for display.
            for mode,item in frame['models'].items():
                if item.get('raw_p95') is not None:maxima[mode]=max(maxima[mode],item['raw_p95'])
            if previous.get(key)!=signature or not packet_path.exists():
                with np.load(input_path,allow_pickle=False) as a:
                    camera=a['xyz_camera'].copy();pixels=a['pixels_yx'].copy();rgb=a['rgb'].copy()
                arrays={}
                for mode,(path,score) in paths.items():
                    with np.load(path,allow_pickle=False) as a:arrays[mode]={k:a[k].copy() for k in ('raw_point_scores','sampled_input_indices','points_normalized')}
                    metrics=raw_metrics(arrays[mode]['raw_point_scores'])
                    if not np.isclose(metrics['raw_mean_top80'],score['raw_mean_top80'],rtol=2e-6,atol=1e-7):raise ValueError('replay raw score differs from saved metric')
                if arrays:
                    first=next(iter(arrays.values()));ids=first['sampled_input_indices'];aligned=first['points_normalized']
                    for a in arrays.values():
                        if not np.array_equal(a['sampled_input_indices'],ids) or not np.array_equal(a['points_normalized'],aligned):
                            raise ValueError('paired checkpoints did not score identical observations')
                else:
                    ids=np.arange(len(camera));aligned=normalize(camera,normalization) if fid==0 else None
                    if fid and pose and pose['status']=='ok':aligned=normalize(reference_align(camera,normalization['T_ref'],pose['T_camera_from_link5']),normalization)
                selected=pixels[ids];colors=rgb[selected[:,0],selected[:,1]] if len(selected) else np.zeros((0,3),np.uint8)
                packet=dict(video_id=case['id'],frame_id=fid,count=len(ids),camera=packed(normalize(camera[ids],normalization),'<f4'),
                    aligned=packed(aligned,'<f4') if aligned is not None else None,pixels=packed(selected,'<u2'),rgb=packed(colors,'u1'),
                    observed_indices=packed(ids,'<u4'),scores={m:packed(a['raw_point_scores'],'<f4') for m,a in arrays.items()},
                    input_source=href(input_path),input_sha256=pose['input_sha256'] if pose else sha(input_path))
                packet_path.parent.mkdir(parents=True,exist_ok=True)
                temp=packet_path.with_suffix('.tmp');temp.write_text('window.LINK5_FRAME('+json.dumps(packet,separators=(',',':'))+');\n');temp.replace(packet_path)
            frame['packet']=href(packet_path);counts['display_frames']+=1;rows.append(frame)
        video_path=destination/'videos'/(case['id']+'.mp4')
        if not video_path.exists():
            if sha(case['video'])!=case['video_sha256']:raise ValueError('replay source video changed')
            video_path.parent.mkdir(exist_ok=True);shutil.copy2(case['video'],video_path)
        cases.append(dict(video_id=case['id'],generator=case['id'].rsplit('_',1)[0],number=case['id'].rsplit('_',1)[1],source_video=href(video_path),source_video_sha256=case['video_sha256'],
            frame_count=n,fps=fps,status=status,summaries=summaries,frames=rows))
    manifest=dict(title='Link5 full-video replay',cases=cases,numbers=sorted({c['number'] for c in cases}),
        modes=list(MODES),counts=counts,expected_frames=sum(c['frame_count'] for c in cases),
        color_max={m:max(v,1e-6) for m,v in maxima.items()},primary_video_metric='sum_all_frame_raw_mean_top80',include_frame0=True,
        camera_coordinates_display='one fixed training center/scale for both camera and canonicalized clouds',
        sensitivity_status='Both refined detectors passed validation; four frame0 normal references.' if config.get('normal_reference') else 'Both detectors failed synthetic sensitivity; full-video evaluation explicitly requested as exploratory.',
        checkpoints=config['full_video']['checkpoint_sha256'])
    temp=destination/'manifest.tmp';temp.write_text('window.LINK5_REPLAY='+json.dumps(manifest,separators=(',',':'))+';\n');temp.replace(destination/'manifest.js')
    (destination/'index.html').write_text(render_published_framework(ROOT))
    shutil.copy2(Path(__file__).with_name('replay_app.js'),destination/'app.js')
    asset=destination/'plotly.min.js'
    if not asset.exists():shutil.copy2(ROOT/'infrastructure/shared/replay/assets/plotly.min.js',asset)
    if not (destination/'labels.js').exists():(destination/'labels.js').write_text('window.LINK5_LABELS=null;\n')
    write(destination/'export_state.json',state);write(full/'metadata/replay_export.json',dict(status='exported',**counts,expected_frames=manifest['expected_frames']))
    print('LINK5_FULL_VIDEO_REPLAY_EXPORTED',counts,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();run(read(args.config))


if __name__=='__main__':main()
