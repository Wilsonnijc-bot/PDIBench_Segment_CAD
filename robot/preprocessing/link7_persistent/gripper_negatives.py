"""Qwen arm/wrist grounding and full-video SAM propagation from retained positives."""

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = (_workspace_root())
TARGETS = {
    'arm': 'the long forearm link with a black rim and white inset, between the elbow and the terminal wrist housing. Choose its elbow-side half, comfortably inside the link',
    'wrist': 'the white upright end-cap at the gripper-side end of the black-rimmed forearm. Point on its upper white surface near its dark top cap',
}


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def parse(raw, key, size):
    from robot.preprocessing.link7_persistent.palm_recovery import json_object
    data = json_object(raw)
    value = data.get(key, data.get('bbox_2d') if key == 'box' else None)
    xy = np.asarray(value, dtype=float)
    if xy.shape != (size,) or not np.isfinite(xy).all() or (xy < 0).any() or (xy > 1000).any():
        raise ValueError(f'Invalid {key}: {value}')
    return xy


def cases():
    records = json.loads(((_workspace_root() / 'qwen_results/provenance/manifest.json')).read_text())
    for r in records:
        if '/' in r['case']:
            continue
        c = r.get('raw_point_provenance', {}).get('correction.json', {})
        yield r['case'], c


def video_for(case):
    explicit = json.loads(os.environ.get('PDI_CASE_VIDEOS', '{}'))
    if case in explicit:
        return Path(explicit[case])
    folders = {'Cosmos3': '15sRTmHwkBQO0FmDRnp0BePAArCad2_Gj', 'Cosmos25': '1rXdVXrwIjf9wMeCVxbmFWN7cZNcfdtb9', 'LVP': '1irS6zoWSykw64DuaiwxyUVHdIffEo3Oa'}
    prefix, number = case.split('_')
    root = Path(os.environ.get('PERSISTENT_MASKING_VIDEO_ROOT',
                               '/root/autodl-tmp/hierarchical-deformation-gdrive'))
    return root/folders[prefix]/(number+'.mp4')


def ground(args):
    from infrastructure.shared.inference.generation.link_crop_wrapper.run_segment_vlm import load_model, generate
    processor, model, architecture = load_model(args.model)
    for case, correction in cases():
        if args.cases and case not in args.cases:
            continue
        out = args.work/case
        saved = out/'grounding.json'
        if saved.exists() and not args.overwrite:
            continue
        pos = [p for p,l in zip(correction.get('proposal',{}).get('points_xy',[]), correction.get('proposal',{}).get('labels',[])) if l == 1]
        record = dict(case=case, architecture=architecture, calls=[], positive_source='retained_Qwen_provenance', positives=pos)
        if correction.get('status') != 'ready_for_sam' or len(pos) != 3:
            record.update(status='skipped_no_valid_three_positive_seed'); dump(saved,record); continue
        video = video_for(case); frame = correction['frame']
        cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES,frame); ok,bgr=cap.read(); cap.release()
        if not ok:
            raise ValueError(f'Cannot read {video}:{frame}')
        im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)); w,h=im.size
        inputs=out/'inputs'; inputs.mkdir(parents=True,exist_ok=True); im.save(inputs/'context.png')
        record.update(frame=frame,video=str(video),source_hw=[h,w],negatives={})
        def ask(images, prompt, target, stage):
            messages=[dict(role='user',content=[*[dict(type='image',image=i) for i in images],dict(type='text',text=prompt)])]
            raw=generate(model,processor,messages,180)
            record['calls'].append(dict(target=target,stage=stage,prompt=prompt,raw=raw))
            dump(saved,record)
            return raw
        if saved.exists() and args.targets != ['arm', 'wrist']:
            previous=json.loads(saved.read_text())
            record['negatives'].update(previous.get('negatives',{}))
            record['calls'].extend(c for c in previous.get('calls',[]) if c['target'] not in args.targets)
            record['retained_target_provenance'] = {t: 'previous_Qwen_run' for t in record['negatives'] if t not in args.targets}
        for target in args.targets:
            description=TARGETS[target]
            if args.full_context and target == 'wrist':
                description='the upright white terminal wrist housing with a dark top cap, directly ABOVE the shiny metal collar and gripper. Choose the broad white upper housing, not the metal collar, long forearm, gripper or fingers'
            try:
                center=np.mean(pos,axis=0)
                extent=np.array([w*.28,h*.38])
                context_box=np.r_[np.maximum(0,center-extent),np.minimum([w,h],center+extent)].round().astype(int)
                if args.full_context: context_box=np.array([0,0,w,h])
                context=im.crop(tuple(context_box)); context.save(inputs/'local_context.png')
                box_prompt=f'Locate {description}. Return one bounding box using bbox_2d with coordinates from 0 to 1000 relative to this image. Return null if hidden.'
                if args.full_context:
                    box_prompt=f'The gripper is centered near normalized coordinates {np.round(np.mean(pos,axis=0)/[w,h]*1000).astype(int).tolist()}. Locate {description}. The requested wrist is the small terminal housing immediately attached above THIS gripper, not the larger elbow or robot base. Return its tight bounding box as {{"box": [x1,y1,x2,y2]}} in 0 to 1000 coordinates of this image. If hidden or uncertain return {{"box":null}}.'
                raw=ask([context],box_prompt,target,'box')
                box=parse(raw,'box',4)/1000*np.tile(context_box[2:]-context_box[:2],2)+np.tile(context_box[:2],2)
                record['calls'][-1]['context_box']=context_box.tolist()
                if (box[2:] - box[:2] < 8).any(): raise ValueError('Empty or tiny target box')
                margin=np.maximum((box[2:]-box[:2])*.25,16)
                box=np.r_[np.maximum(0,box[:2]-margin),np.minimum([w,h],box[2:]+margin)].round().astype(int)
                crop=im.crop(tuple(box)); crop.save(inputs/f'{target}.png')
                point_prompt=f'In this image place one point well inside {description}. This negative excludes this housing from the gripper mask. Stay away from outlines and background. Return {{"point_2d":[x,y]}} using 0 to 1000 coordinates relative to this image. If not identifiable return {{"point_2d":null}}.'
                if args.full_context:
                    point_prompt=f'Image 1 is context. Image 2 is a crop. In IMAGE 2 place one point well inside {description}. This negative excludes this housing from the gripper mask. Stay away from outlines and background. Return {{"point_2d":[x,y]}} using 0 to 1000 coordinates relative ONLY to Image 2. If not identifiable return {{"point_2d":null}}.'
                raw=ask([im,crop] if args.full_context else [crop],point_prompt,target,'point')
                xy=parse(raw,'point_2d',2)/1000*(box[2:]-box[:2])+box[:2]
                if (xy >= [w,h]).any() or np.linalg.norm(np.asarray(pos)-xy,axis=1).min()<8: raise ValueError('Point outside image or conflicts with positive')
                record['negatives'][target]=xy.tolist()
                record['calls'][-1].update(crop_box=box.tolist(),source_xy=xy.tolist())
            except (ValueError,KeyError,TypeError) as e:
                record.setdefault('errors',{})[target]=str(e)
        record['status']='grounded_pending_visual_review' if len(record['negatives'])==2 else 'negative_localization_failed'
        record['input_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.glob('*.png')}
        dump(saved,record)
        preview=im.copy();draw=ImageDraw.Draw(preview)
        for label,xy in [(f'P{i+1}',xy) for i,xy in enumerate(pos)]+list(record['negatives'].items()):
            x,y=xy; color='lime' if label.startswith('P') else 'red'
            draw.ellipse((x-5,y-5,x+5,y+5),fill=color);draw.text((x+7,y+5),label,fill=color)
        preview.save(out/'points.png');print(case,record['status'],record['negatives'],flush=True)


def segment(args):
    from robot.preprocessing.link7_persistent.v1_mask import predictor
    for case,_ in cases():
        if args.cases and case not in args.cases: continue
        out=args.work/case; record=json.loads((out/'grounding.json').read_text())
        if record['status']!='grounded_pending_visual_review': continue
        result=out/f'{args.mode}.npz'
        if result.exists() and not args.overwrite: continue
        points=record['positives'][:]; labels=[1]*3
        for name in ([] if args.mode=='positive' else ['arm'] if args.mode=='arm' else ['arm','wrist']):
            points.append(record['negatives'][name]); labels.append(0)
        h,w=record['source_hw'];frame=record['frame'];cap=cv2.VideoCapture(record['video']);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        p=predictor();sid=p.handle_request(dict(type='start_session',resource_path=record['video'],offload_video_to_cpu=True))['session_id']
        try:
            data=p.handle_request(dict(type='add_prompt',session_id=sid,frame_index=frame,obj_id=0,points=(np.asarray(points)/[w,h]).tolist(),point_labels=labels))['outputs']
            mask=np.asarray(data['out_binary_masks'][list(data['out_obj_ids']).index(0)],bool)
            xy=np.floor(points).astype(int);membership=mask[xy[:,1],xy[:,0]].tolist()
            masks=np.zeros((count,h,w),bool);masks[frame]=mask;seen={frame}
            for direction,limit in [('forward',count-1-frame),('backward',frame)]:
                if not limit: continue
                for response in p.handle_stream_request(dict(type='propagate_in_video',session_id=sid,propagation_direction=direction,start_frame_index=frame,max_frame_num_to_track=limit)):
                    t=int(response['frame_index']);d=response['outputs'];ids=list(d['out_obj_ids'])
                    if 0 in ids: masks[t]=d['out_binary_masks'][ids.index(0)];seen.add(t)
            np.savez_compressed(result,masks=masks)
            dump(out/f'{args.mode}_metrics.json',dict(points_xy=points,labels=labels,grounding_sha256=hashlib.sha256((out/'grounding.json').read_bytes()).hexdigest(),prompt_membership=membership,prompt_membership_ok=membership==[bool(x) for x in labels],frame_count=count,updated_frames=len(seen),empty_frames=np.flatnonzero(~masks.any((1,2))).tolist(),areas=masks.sum((1,2)).tolist()))
            print(case,args.mode,'completed',flush=True)
        finally:
            p.handle_request(dict(type='close_session',session_id=sid));p.shutdown()


def replay(args):
    for case,_ in cases():
        if args.cases and case not in args.cases:continue
        out=args.work/case; path=out/f'{args.mode}.npz'
        if not path.exists():continue
        record=json.loads((out/'grounding.json').read_text());masks=np.load(path)['masks'];h,w=record['source_hw']
        cap=cv2.VideoCapture(record['video']);fps=cap.get(cv2.CAP_PROP_FPS)
        raw=out/'render.mp4';writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(2*w,h))
        for mask in masks:
            ok,bgr=cap.read()
            if not ok: raise ValueError('Video shorter than masks')
            overlay=bgr.copy();overlay[mask]=(.55*overlay[mask]+.45*np.array([255,190,0])).astype(np.uint8)
            writer.write(np.concatenate([bgr,overlay],axis=1))
        cap.release();writer.release()
        ffmpeg=os.environ.get('FFMPEG_BINARY') or shutil.which('ffmpeg')
        if ffmpeg is None:
            import imageio_ffmpeg
            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg,'-y','-loglevel','error','-i',str(raw),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'masking.mp4')],check=True);raw.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['ground','segment','replay']);p.add_argument('--work',type=Path,required=True);p.add_argument('--model',type=Path,default=(_workspace_root() / 'infrastructure/models/Qwen3.5-9B'));p.add_argument('--cases',nargs='+');p.add_argument('--mode',choices=['positive','arm','both'],default='both');p.add_argument('--overwrite',action='store_true');p.add_argument('--full-context',action='store_true');p.add_argument('--targets',nargs='+',choices=['arm','wrist'],default=['arm','wrist']);a=p.parse_args()
    {'ground':ground,'segment':segment,'replay':replay}[a.stage](a)
