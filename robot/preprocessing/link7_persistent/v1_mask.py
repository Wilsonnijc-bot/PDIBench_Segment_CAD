"""Palm initialization, CoTracker3 landmarks, and SAM3 point-guided refinement."""

from pathlib import Path as _LayoutPath
from infrastructure.pdibench.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import argparse
import gc
import os
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image

ROOT=(_workspace_root())
OUT=(_workspace_root() / 'reg2inv_results_v1/LVP_0049')
ORIGINAL_CASE=(_workspace_root() / 'reg2inv_results/LVP_0049')
PALM_REFERENCES=(_workspace_root() / 'reg2inv_results_v1/references/by_link/palm')
VIDEO=(_workspace_root() / 'reg2inv_results/diagnosis_LVP_0049/source.mp4')
if not VIDEO.exists():
    VIDEO=Path('/root/autodl-tmp/hierarchical-deformation-gdrive/1irS6zoWSykw64DuaiwxyUVHdIffEo3Oa/0049.mp4')


def frames():
    cap=cv2.VideoCapture(str(VIDEO)); rgb=[]
    while True:
        ok,bgr=cap.read()
        if not ok:break
        rgb.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release()
    if not rgb:raise ValueError('Video contains no frames')
    return np.stack(rgb)


def predictor():
    from sam3.model_builder import build_sam3_video_predictor
    p=build_sam3_video_predictor(checkpoint_path=os.environ.get('PDI_SAM3_CHECKPOINT', str((_workspace_root() / 'infrastructure/models/sam3/sam3.pt'))),
        bpe_path=os.environ.get('PDI_SAM3_BPE', str((_workspace_root() / 'infrastructure/models/sam3/bpe_simple_vocab_16e6.txt.gz'))),apply_temporal_disambiguation=False)
    p.model.use_prev_mem_frame=True
    # SAM3's point-prompt path can build a valid first-object mask before any
    # propagation, but recent releases require a per-frame output cache when
    # assembling that response. An empty entry lets the new mask populate it.
    build_output=p.model._build_tracker_output
    def build_output_with_first_prompt(inference_state,frame_idx,refined_obj_id_to_mask=None):
        if refined_obj_id_to_mask is not None:
            inference_state.setdefault('cached_frame_outputs',{}).setdefault(frame_idx,{})
        return build_output(inference_state,frame_idx,refined_obj_id_to_mask)
    p.model._build_tracker_output=build_output_with_first_prompt
    return p


def spread(mask,count):
    y,x=np.where(mask); pts=np.column_stack([x,y])
    if len(pts)==0:raise ValueError('No eligible landmark seeds')
    chosen=[int(np.argmin(((pts-pts.mean(0))**2).sum(1)))];distance=np.full(len(pts),np.inf)
    for _ in range(min(count,len(pts))-1):
        distance=np.minimum(distance,((pts-pts[chosen[-1]])**2).sum(1))
        chosen.append(int(distance.argmax()))
    return pts[chosen].astype(np.float32)


def initialize():
    import torch
    from infrastructure.shared.inference.generation.link_crop_wrapper.dinov2_reference_boxes import Dinov2DenseEncoder, localize_reference_groups
    from infrastructure.shared.inference.generation.link_crop_wrapper.sam3_link_tracker import select_prompt_result
    rgb=frames();image=Image.fromarray(rgb[0])
    refs=sorted(PALM_REFERENCES.glob('*.png'))
    enc=Dinov2DenseEncoder((_workspace_root() / 'infrastructure/models/dinov2'),'cuda')
    boxes,_=localize_reference_groups(image,{'palm':refs},enc,scene_side=840,reference_side=448,
        top_fraction=.12,padding_fraction=.10,minimum_contrast=.02,reference_spatial_priors=True)
    box=boxes[0];del enc;gc.collect();torch.cuda.empty_cache()
    p=predictor();sid=p.handle_request(dict(type='start_session',resource_path=str(VIDEO),offload_video_to_cpu=True))['session_id']
    result=p.handle_request(dict(type='add_prompt',session_id=sid,frame_index=0,text='visual',
        bounding_boxes=[list(box.box_xywh_normalized)],bounding_box_labels=[1]))['outputs']
    obj,mask,score=select_prompt_result(np.asarray(result['out_obj_ids']),np.asarray(result['out_binary_masks']),
        np.asarray(result['out_probs']),box.box_xyxy)
    original=np.load((_workspace_root() / 'reg2inv_results/LVP_0049/masks/links.npz'))['object_masks'][0]
    # White appearance only chooses interior seeds at t=0, never clips future masks.
    interior=cv2.erode(mask.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
    bright=rgb[0].mean(2)>90
    pos=spread(interior & bright,12)
    yy,xx=np.indices(mask.shape)
    my,mx=np.where(mask)
    finger=original[5] & ~cv2.dilate(mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool) & (yy>np.quantile(my,.8))
    neg_parts=[]
    if finger.sum()>8:neg_parts.append(spread(finger,4))
    wrist=cv2.erode(original[4].astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
    if wrist.sum()>8:neg_parts.append(spread(wrist,3))
    negative=np.concatenate(neg_parts) if neg_parts else np.empty((0,2),np.float32)
    np.savez_compressed((_workspace_root() / 'reg2inv_results_v1/LVP_0049/seeds.npz'),xy=np.concatenate([pos,negative]),labels=np.r_[np.ones(len(pos)),np.zeros(len(negative))].astype(int),initial_mask=mask)
    metadata=dict(box=list(box.box_xyxy),sam_object_id=int(obj),sam_probability=score,
                  positive_count=len(pos),negative_count=len(negative),initial_area=int(mask.sum()))
    ((_workspace_root() / 'reg2inv_results_v1/LVP_0049/initialization.json')).write_text(json.dumps(metadata,indent=2))
    control=np.zeros((49,*mask.shape),bool);control[0]=mask
    for response in p.handle_stream_request(dict(type='propagate_in_video',session_id=sid,propagation_direction='forward',start_frame_index=0)):
        data=response['outputs'];ids=list(data['out_obj_ids'])
        if obj in ids:control[int(response['frame_index'])]=data['out_binary_masks'][ids.index(obj)]
    np.savez_compressed((_workspace_root() / 'reg2inv_results_v1/LVP_0049/sam_only_masks.npz'),masks=control)
    p.handle_request(dict(type='close_session',session_id=sid));p.shutdown()
    print(metadata,flush=True)


def track():
    import torch
    sys.path.insert(0,str((_workspace_root() / 'infrastructure/vendor/co-tracker-82e02e8029753ad4ef13cf06be7f4fc5facdda4d')))
    from cotracker.predictor import CoTrackerPredictor
    rgb=frames();seeds=np.load((_workspace_root() / 'reg2inv_results_v1/LVP_0049/seeds.npz'))
    queries=np.column_stack([np.zeros(len(seeds['xy'])),seeds['xy']]).astype(np.float32)
    model=CoTrackerPredictor(checkpoint=str((_workspace_root() / 'infrastructure/models/cotracker3-offline.pth')),offline=True,v2=False).cuda().eval()
    video=torch.from_numpy(rgb).permute(0,3,1,2)[None].float().cuda()
    with torch.inference_mode():xy,visible=model(video,queries=torch.from_numpy(queries)[None].cuda())
    np.savez_compressed((_workspace_root() / 'reg2inv_results_v1/LVP_0049/tracks.npz'),xy=xy[0].cpu().numpy(),visible=visible[0].cpu().numpy(),labels=seeds['labels'])
    print('CoTracker3 complete',xy.shape,flush=True)


def guided():
    tracks=np.load((_workspace_root() / 'reg2inv_results_v1/LVP_0049/tracks.npz'));xy=tracks['xy'];labels=tracks['labels'];visible=tracks['visible']
    rgb=frames();h,w=rgb.shape[1:3];masks=np.zeros((len(rgb),h,w),bool);records=[]
    p=predictor();sid=p.handle_request(dict(type='start_session',resource_path=str(VIDEO),offload_video_to_cpu=True))['session_id']
    for f in range(len(rgb)):
        good=visible[f].astype(bool)&np.isfinite(xy[f]).all(1)&(xy[f,:,0]>=0)&(xy[f,:,0]<w)&(xy[f,:,1]>=0)&(xy[f,:,1]<h)
        npos=int(((labels==1)&good).sum())
        if npos<3:
            records.append(dict(frame=f,status='insufficient_visible_positive_tracks',positive=npos));continue
        # Full-frame SAM sees positive and negative points. No white threshold,
        # convex-hull clipping, or intersection with the preceding mask.
        data=p.handle_request(dict(type='add_prompt',session_id=sid,frame_index=f,obj_id=0,
            points=(xy[f,good]/[w,h]).tolist(),point_labels=labels[good].tolist()))['outputs']
        ids=list(data['out_obj_ids'])
        if 0 in ids:masks[f]=data['out_binary_masks'][ids.index(0)]
        records.append(dict(frame=f,status='ok' if masks[f].any() else 'empty_sam_mask',positive=npos,
                            negative=int(((labels==0)&good).sum()),area=int(masks[f].sum())))
        print('guided',f,records[-1],flush=True)
    np.savez_compressed((_workspace_root() / 'reg2inv_results_v1/LVP_0049/guided_masks.npz'),masks=masks)
    ((_workspace_root() / 'reg2inv_results_v1/LVP_0049/tracking_quality.json')).write_text(json.dumps(records,indent=2))
    p.handle_request(dict(type='close_session',session_id=sid));p.shutdown()


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['initialize','track','guided'])
    args=parser.parse_args();globals()[args.stage]()
