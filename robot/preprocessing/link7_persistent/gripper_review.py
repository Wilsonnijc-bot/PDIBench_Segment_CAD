"""Gripper positive refinement and mask-aware temporal correction experiments."""

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import argparse
import hashlib
import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from robot.preprocessing.link7_persistent.gripper_negatives import video_for, dump
from robot.preprocessing.link7_persistent.palm_recovery import json_object
from robot.preprocessing.link7_persistent.naive_sam3 import render

CASES=['Cosmos25_0023','Cosmos25_0053','LVP_0049','LVP_0059','Cosmos3_0048','Cosmos25_0003']
ROOT=(_workspace_root())


def point(raw,box):
    d=json_object(raw);v=d.get('point_2d',d.get('point'))
    a=np.asarray(v,dtype=float)
    if a.shape!=(2,) or not np.isfinite(a).all() or (a<0).any() or (a>1000).any():raise ValueError('Invalid or absent point')
    return (a/1000*(np.array(box[2:])-box[:2])+box[:2]).tolist()


def load_frame(video,f):
    cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();cap.release()
    if not ok:raise ValueError('Missing frame')
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)


def preview(rgb,points,labels,path):
    im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
    for i,(xy,label) in enumerate(zip(points,labels)):
        x,y=xy;col='lime' if label else 'red';d.ellipse((x-4,y-4,x+4,y+4),fill=col);d.text((x+5,y),str(i+1),fill=col)
    im.save(path)


def ground(a):
    from infrastructure.shared.inference.generation.link_crop_wrapper.run_segment_vlm import load_model, generate
    proc,model,arch=load_model((_workspace_root() / 'infrastructure/models/Qwen3.5-9B'))
    original=json.loads((a.results/'provenance/manifest.json').read_text())
    old=json.loads((a.results/'provenance/arm_wrist.json').read_text())
    for case in a.cases:
        out=a.work/case;inp=out/'inputs';inp.mkdir(parents=True,exist_ok=True)
        r=next(x for x in original if x['case']==case);oldr=next(x for x in old if x['case']==case)
        f=oldr['frame'];rgb=load_frame(video_for(case),f);h,w=rgb.shape[:2]
        call=next(x for x in r['crop_boxes'] if x['frame']==f);box=call.get('box_xyxy',call.get('box'))
        # Use real, unpadded pixels and enough anatomical context for every case.
        box=[max(0,box[0]),max(0,box[1]),min(w,box[2]),min(h,box[3])]
        ref=Image.open(a.results/case/'inputs/reference.png').convert('RGB');ref.save(inp/'positive_reference.png')
        rec=dict(case=case,frame=f,video=str(video_for(case)),source_hw=[h,w],calls=[],architecture=arch)
        full=Image.fromarray(rgb);full.save(inp/'localization_context.png')
        locate='Image 1 is a reference crop of the robot gripper. In Image 2 locate the entire gripper: the white hand/palm housing and black fingers BELOW the metallic collar. Exclude the long forearm and upper white wrist housing. Return one tight bbox_2d [x1,y1,x2,y2] normalized 0 to 1000 relative to Image 2; null if invisible.'
        raw=generate(model,proc,[dict(role='user',content=[dict(type='image',image=ref),dict(type='image',image=full),dict(type='text',text=locate)])],180)
        entry=dict(target='gripper_box',prompt=locate,raw=raw)
        try:
            try: value=json_object(raw)['bbox_2d']
            except ValueError: value=json.loads(raw.strip())
            b=np.asarray(value,float)
            if b.shape!=(4,) or not np.isfinite(b).all() or (b<0).any() or (b>1000).any():raise ValueError('Invalid gripper box')
            b=b/1000*[w,h,w,h]
            if (b[2:]-b[:2]<8).any():raise ValueError('Tiny or reversed box')
            margin=np.maximum((b[2:]-b[:2])*.15,8)
            box=np.r_[np.maximum(0,b[:2]-margin),np.minimum([w,h],b[2:]+margin)].round().astype(int).tolist()
            entry['box']=box
        except (ValueError,KeyError,TypeError) as e:
            rec.update(status='failed',error=str(e));rec['calls'].append(entry);dump(out/'seed.json',rec);continue
        rec['calls'].append(entry)
        crop=full.crop(tuple(box));crop.save(inp/'positive_candidate.png')
        points=[]
        targets=['black_left','black_right','white_center']
        descriptions={
          'black_left':'Point inside the left BLACK gripper finger or black deformed gripper material.',
          'black_right':'Point inside the right BLACK gripper finger or a different black deformed gripper part.',
          'white_center':'Point at the CENTER of the WHITE gripper palm housing below the metal collar. Stay in the middle of the broad white surface, away from the lower tip.'}
        for target in targets:
            prior=((np.asarray(points)-box[:2])/(np.array(box[2:])-box[:2])*1000).round().astype(int).tolist() if points else []
            prompt=descriptions[target]+f' Previously selected positive coordinates: {prior}. Choose a distinct point. Return only {{"point_2d":[x,y]}} with 0 to 1000 coordinates relative to this image; null if invisible.'
            messages=[dict(role='system',content=[dict(type='text',text='Locate gripper parts precisely. Output JSON only, no explanation. The gripper includes its white hand housing and black fingers, excludes the long arm and upper wrist.')]),dict(role='user',content=[dict(type='image',image=crop),dict(type='text',text=prompt)])]
            raw=generate(model,proc,messages,300);entry=dict(target=target,prompt=prompt,raw=raw,box=box)
            try:
                xy=point(raw,box)
                if points and np.linalg.norm(np.asarray(points)-xy,axis=1).min()<3:raise ValueError('Duplicate positive')
                points.append(xy);entry['source_xy']=xy
            except ValueError as e:entry['error']=str(e)
            rec['calls'].append(entry)
        neg=oldr['negatives'].copy()
        rec.update(points=points+list(neg.values()),labels=[1]*len(points)+[0]*len(neg),positive_count=len(points),negative_source='retained previous Qwen; checked with current mask at common checkpoints',status='ready' if len(points)==3 else 'failed')
        dump(out/'seed.json',rec);preview(rgb,rec['points'],rec['labels'],out/'points.png');print(case,rec['points'],flush=True)


def sam(a):
    from robot.preprocessing.link7_persistent.v1_mask import predictor
    for case in a.cases:
        out=a.work/case;seed=json.loads((out/'seed.json').read_text())
        if seed['status']!='ready' and not a.checkpoint:continue
        f=seed['frame'];points=seed.get('points',[]);labels=seed.get('labels',[]);h,w=seed['source_hw']
        if a.checkpoint:
            correction=json.loads((out/f'check_{a.checkpoint}.json').read_text())
            if correction['status']!='correct':continue
            f=correction['frame'];points=correction['points'];labels=correction['labels']
        previous=out/('seed_masks.npz' if a.checkpoint==1 else 'review_1_masks.npz')
        if a.checkpoint and not previous.exists():previous=out/'seed_masks.npz'
        if a.checkpoint and not previous.exists():previous=(_workspace_root() / 'negative_work')/case/'both.npz'
        cap=cv2.VideoCapture(seed['video']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        masks=np.load(previous)['masks'].copy() if a.checkpoint else np.zeros((n,h,w),bool)
        p=predictor();sid=p.handle_request(dict(type='start_session',resource_path=seed['video'],offload_video_to_cpu=True))['session_id']
        try:
            d=p.handle_request(dict(type='add_prompt',session_id=sid,frame_index=f,obj_id=0,points=(np.asarray(points)/[w,h]).tolist(),point_labels=labels))['outputs']
            masks[f]=d['out_binary_masks'][list(d['out_obj_ids']).index(0)]
            xy=np.floor(points).astype(int);member=masks[f,xy[:,1],xy[:,0]].tolist()
            for direction,limit in [('forward',n-f-1)]+([] if a.checkpoint else [('backward',f)]):
                if not limit:continue
                for response in p.handle_stream_request(dict(type='propagate_in_video',session_id=sid,propagation_direction=direction,start_frame_index=f,max_frame_num_to_track=limit)):
                    t=int(response['frame_index']);data=response['outputs'];ids=list(data['out_obj_ids'])
                    if 0 in ids:masks[t]=data['out_binary_masks'][ids.index(0)]
            name=f'review_{a.checkpoint}' if a.checkpoint else 'seed'
            np.savez_compressed(out/f'{name}_masks.npz',masks=masks)
            dump(out/f'{name}_metrics.json',dict(frame=f,points=points,labels=labels,membership=member,membership_ok=member==[bool(x) for x in labels],empty_frames=np.flatnonzero(~masks.any((1,2))).tolist()))
            render(seed['video'],masks,out/'masking.mp4',a.ffmpeg);print(case,name,member,flush=True)
        finally:p.handle_request(dict(type='close_session',session_id=sid));p.shutdown()


def check(a):
    from infrastructure.shared.inference.generation.link_crop_wrapper.run_segment_vlm import load_model, generate
    proc,model,arch=load_model((_workspace_root() / 'infrastructure/models/Qwen3.5-9B'))
    for case in a.cases:
        out=a.work/case;seed=json.loads((out/'seed.json').read_text())
        path=out/('seed_masks.npz' if a.checkpoint==1 else 'review_1_masks.npz')
        if not path.exists():path=out/'seed_masks.npz'
        if not path.exists():path=(_workspace_root() / 'negative_work')/case/'both.npz'
        if not path.exists():continue
        masks=np.load(path)['masks'];f=int((len(masks)-1)*(.5 if a.checkpoint==1 else .75));rgb=load_frame(seed['video'],f);h,w=rgb.shape[:2]
        # Broad context from current support; retain surrounding unmasked gripper material.
        ys,xs=np.where(masks[f]);box=[0,0,w,h]
        if len(xs):
            cx,cy=(xs.min()+xs.max())/2,(ys.min()+ys.max())/2
            sx,sy=max(160,(xs.max()-xs.min())*2.5),max(180,(ys.max()-ys.min())*2.5)
            box=[max(0,int(cx-sx/2)),max(0,int(cy-sy/2)),min(w,int(cx+sx/2)),min(h,int(cy+sy/2))]
        clean=Image.fromarray(rgb).crop(box);over=rgb.copy();over[masks[f]]=(.55*over[masks[f]]+.45*np.array([0,190,255])).astype(np.uint8);overlay=Image.fromarray(over).crop(box)
        ref=Image.open(a.results/case/'inputs/reference.png').convert('RGB')
        clean.save(out/'inputs'/f'check_{a.checkpoint}_rgb.png');overlay.save(out/'inputs'/f'check_{a.checkpoint}_mask.png')
        prompt='Image 1: original gripper reference. Image 2: current clean RGB. Image 3: SAME crop with current SAM mask in cyan. The target is the ENTIRE gripper including WHITE palm housing and BLACK fingers/deformed parts; exclude upper wrist, forearm and held objects. Does cyan omit visible white or black gripper material or include unrelated material? Return JSON {"action":"keep" or "correct" or "unclear","reason":"visible evidence","positive_points":[[x,y],...],"negative_points":[[x,y],...]}. If correct, give THREE positive points: center of white gripper housing and two distinct black gripper interiors; optionally up to two negatives strictly inside leaked non-gripper material. Coordinates 0 to 1000 relative to Image 2. If keep or unclear return empty lists. Do not mistake cyan color for actual material: use Image 2 to place points. Do not force a point onto invisible material.'
        raw=generate(model,proc,[dict(role='user',content=[dict(type='image',image=ref),dict(type='image',image=clean),dict(type='image',image=overlay),dict(type='text',text=prompt)])],500)
        rec=dict(frame=f,box=box,prompt=prompt,raw=raw,source_masks=str(path),status='failed',points=[],labels=[])
        try:
            d=json_object(raw);action=d['action'];rec['reason']=d['reason']
            if action not in ['keep','correct','unclear']:raise ValueError('Invalid action')
            for key,label in [('positive_points',1),('negative_points',0)]:
                for xy in d[key]:rec['points'].append(point(json.dumps({'point_2d':xy}),box));rec['labels'].append(label)
            if action=='correct' and rec['labels'].count(1)!=3:raise ValueError('Need three visible positives')
            if action!='correct' and rec['points']:raise ValueError('No correction points allowed')
            rec['status']=action
        except (ValueError,KeyError,TypeError) as e:rec['error']=str(e)
        dump(out/f'check_{a.checkpoint}.json',rec)
        preview(rgb,rec['points'],rec['labels'],out/f'check_{a.checkpoint}_points.png');print(case,rec['status'],rec.get('reason'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['ground','sam','check']);p.add_argument('--work',type=Path,required=True);p.add_argument('--results',type=Path,default=(_workspace_root() / 'robot/results/persistent_masking'));p.add_argument('--cases',nargs='+',default=CASES);p.add_argument('--checkpoint',type=int,default=0);p.add_argument('--ffmpeg',default=os.environ.get('FFMPEG_BINARY'));a=p.parse_args();globals()[a.stage](a)
