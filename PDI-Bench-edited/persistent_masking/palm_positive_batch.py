"""Three-positive-only palm test on three frozen gripper-deformation cases."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_palm_positive'
VIDEO_DIR=Path('/root/autodl-tmp/hierarchical-deformation-gdrive/1irS6zoWSykw64DuaiwxyUVHdIffEo3Oa')
SELECTION=ROOT/'persistent_masking/palm_positive_selection.json'


def selection():return json.loads(SELECTION.read_text())['videos']


def video_for(case):
    v=next(v for v in selection() if v['id']==case)
    return Path(v.get('directory',str(VIDEO_DIR)))/v['file']


def initialize(case):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from persistent_masking.v1_mask import predictor,spread
    from generation.link_crop_wrapper.dinov2_reference_boxes import Dinov2DenseEncoder,localize_reference_groups
    from generation.link_crop_wrapper.sam3_link_tracker import select_prompt_result
    video=video_for(case);out=OUT/case;out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video));ok,bgr=cap.read();count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    if not ok or count<2:raise ValueError('Expected a readable video with at least two frames')
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    refs=sorted((ROOT/'results_v1/references/by_link/palm').glob('*.png'))
    enc=Dinov2DenseEncoder(ROOT/'models/dinov2','cuda')
    boxes,_=localize_reference_groups(Image.fromarray(rgb),{'palm':refs},enc,scene_side=840,reference_side=448,
        top_fraction=.12,padding_fraction=.10,minimum_contrast=.02,reference_spatial_priors=True)
    del enc;torch.cuda.empty_cache();box=boxes[0]
    p=predictor();sid=p.handle_request(dict(type='start_session',resource_path=str(video),offload_video_to_cpu=True))['session_id']
    try:
        d=p.handle_request(dict(type='add_prompt',session_id=sid,frame_index=0,text='visual',
            bounding_boxes=[list(box.box_xywh_normalized)],bounding_box_labels=[1]))['outputs']
        obj,mask,score=select_prompt_result(np.asarray(d['out_obj_ids']),np.asarray(d['out_binary_masks']),np.asarray(d['out_probs']),box.box_xyxy)
        interior=cv2.erode(mask.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        pos=spread(interior & (rgb.mean(2)>90),3)
        if len(pos)!=3:raise ValueError('Insufficient interior seed candidates')
        np.savez_compressed(out/'seeds.npz',xy=pos,labels=np.ones(3,dtype=int),initial_mask=mask)
        (out/'initialization.json').write_text(json.dumps(dict(box=list(box.box_xyxy),initial_area=int(mask.sum()),
            sam_probability=score,positive_count=3,negative_count=0,positive_xy=pos.tolist()),indent=2))
        # Retain a SAM propagation control for the same palm box initialization.
        control=np.zeros((count,*mask.shape),bool);control[0]=mask
        for response in p.handle_stream_request(dict(type='propagate_in_video',session_id=sid,propagation_direction='forward',start_frame_index=0)):
            data=response['outputs'];ids=list(data['out_obj_ids'])
            if obj in ids:control[int(response['frame_index'])]=data['out_binary_masks'][ids.index(obj)]
        np.savez_compressed(out/'sam_only_masks.npz',masks=control)
    finally:p.handle_request(dict(type='close_session',session_id=sid));p.shutdown()


def stage(case,name):
    from reg2invbaselinecode import v1_mask
    v1_mask.OUT=OUT/case;v1_mask.VIDEO=video_for(case)
    if name=='initialize':initialize(case)
    elif name=='track':track_chunks(case)
    else:
        # Keep the existing conservative >=3-visible-positive rule. Missing
        # support is explicit; do not fabricate extra or invisible points.
        v1_mask.guided()


def track_chunks(case):
    import numpy as np
    import torch
    from reg2invbaselinecode import v1_mask
    sys.path.insert(0,str(ROOT/'third_party/co-tracker-82e02e8029753ad4ef13cf06be7f4fc5facdda4d'))
    from cotracker.predictor import CoTrackerPredictor
    rgb=v1_mask.frames();seeds=np.load(OUT/case/'seeds.npz');n=len(seeds['xy'])
    model=CoTrackerPredictor(checkpoint=str(ROOT/'models/cotracker3-offline.pth'),offline=True,v2=False).cuda().eval()
    tracks=np.zeros((len(rgb),n,2),np.float32);visible=np.zeros((len(rgb),n),bool)
    anchor=seeds['xy'];anchor_visible=np.ones(n,bool);chunks=[]
    with torch.inference_mode():
        for start in range(0,len(rgb)-1,48):
            end=min(len(rgb),start+49)
            clip=torch.from_numpy(rgb[start:end]).permute(0,3,1,2)[None].float().cuda()
            queries=torch.from_numpy(np.column_stack([np.zeros(n),anchor]).astype(np.float32))[None].cuda()
            xy,vis=model(clip,queries=queries)
            xy=xy[0].cpu().numpy();vis=vis[0].cpu().numpy().astype(bool)
            # Do not mark an invisible boundary anchor as recovered just because
            # CoTracker treats supplied points as visible in its first frame.
            vis[0]&=anchor_visible
            offset=0 if start==0 else 1
            tracks[start+offset:end]=xy[offset:];visible[start+offset:end]=vis[offset:]
            anchor=xy[-1];anchor_visible=vis[-1];chunks.append([start,end-1])
            del clip,queries
            print('CoTracker chunk',start,end-1,flush=True)
    np.savez_compressed(OUT/case/'tracks.npz',xy=tracks,visible=visible,labels=seeds['labels'])
    (OUT/case/'tracking_chunks.json').write_text(json.dumps(dict(chunks=chunks,policy='49-frame offline chunks with one shared anchor frame; coordinates chained, no additional user points'),indent=2))


def call(env,module,*args,log):
    cmd=[str(ROOT/env/'bin/python'),'-m','persistent_masking.palm_shared_gpu']
    if env=='env-qwen':cmd+=['--qwen']
    cmd+=[module,*map(str,args)]
    with log.open('w') as f:
        subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,
                       env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1'})


def all_cases():
    OUT.mkdir(exist_ok=True);summary=[]
    for v in selection():
        case=v['id'];out=OUT/case;out.mkdir(exist_ok=True);rec=dict(case=case,status='running')
        try:
            for name,file in [('initialize','sam_only_masks.npz'),('track','tracks.npz'),('guided','guided_masks.npz')]:
                if not (out/file).exists():call('env-sam','persistent_masking.palm_positive_batch',name,'--case',case,'--selection',SELECTION,'--output',OUT,log=out/f'{name}.log')
            recovery=out/'recovery'
            # Preparation is CPU-only and does not need a GPU admission slot.
            subprocess.run([str(ROOT/'env-sam/bin/python'),'-m','persistent_masking.palm_recovery','prepare','--video',str(video_for(case)),
                '--masks',str(out/'guided_masks.npz'),'--output',str(recovery)],cwd=ROOT,check=True)
            m=json.loads((recovery/'manifest.json').read_text());rec['candidate_frames']=[c['frame'] for c in m['calls']]
            if not m['calls']:rec['status']='no_area_event_not_reviewed'
            else:
                call('env-qwen','persistent_masking.palm_recovery','review','--model',ROOT/'models/Qwen3.5-9B','--output',recovery,'--classify-only',log=out/'review.log')
                ds=json.loads((recovery/'diagnoses.json').read_text());rec['diagnoses']={s:sum(d['state']==s for d in ds) for s in ['normal','deformed','unclear']}
                rec['skipped']=sum(bool(d.get('skipped')) for d in ds)
                if not any(d['state']=='deformed' and not d['parse_error'] for d in ds):
                    rec['status']='all_review_frames_skipped' if rec['skipped']==len(ds) else 'no_confirmed_palm_deformation'
                else:
                    call('env-qwen','persistent_masking.palm_grounding','--output',recovery,'--model',ROOT/'models/Qwen3.5-9B',
                         '--black-edges','--three-positive-only',log=out/'grounding.log')
                    c=json.loads((recovery/'correction.json').read_text());rec['correction_frame']=c['frame']
                    if c['status']!='ready_for_sam':rec['status']='grounding_failed'
                    else:
                        if c['proposal']['labels']!=[1,1,1]:raise ValueError('Three positive-only contract violated')
                        call('env-sam','persistent_masking.palm_recovery','repair','--output',recovery,'--positive-only',log=out/'repair.log')
                        rec['status']='proposal_completed'
        except Exception as e:rec.update(status='failed',error=str(e))
        summary.append(rec);(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(rec,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['all','initialize','track','guided']);p.add_argument('--case')
    p.add_argument('--selection',type=Path,default=SELECTION);p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args();SELECTION=args.selection.resolve();OUT=args.output.resolve()
    all_cases() if args.stage=='all' else stage(args.case,args.stage)
