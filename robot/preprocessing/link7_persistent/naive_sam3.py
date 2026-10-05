"""Prompt SAM3 once at frame zero, then track that object without corrections.

Default: existing DINO gripper-reference box. Text-only initialization is optional.
"""

from infrastructure.pdibench.layout import root as _workspace_root
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import gc
from PIL import Image

import cv2
import numpy as np

from robot.preprocessing.link7_persistent.gripper_negatives import video_for
from robot.preprocessing.link7_persistent.v1_mask import predictor, ROOT

CASES = ['Cosmos3_0048', 'Cosmos25_0003', 'Cosmos25_0023', 'Cosmos25_0053', 'LVP_0049']


def render(video, masks, dest, ffmpeg):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count, h, w = masks.shape
    raw = dest.parent/'naive_render.mp4'
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), fps, (2*w, h))
    if not writer.isOpened():
        raise RuntimeError('Video writer failed')
    try:
        for mask in masks:
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Source video shorter than mask sequence')
            overlay = frame.copy()
            overlay[mask] = (.55*overlay[mask] + .45*np.array([255,190,0])).astype(np.uint8)
            writer.write(np.concatenate([frame, overlay], axis=1))
    finally:
        cap.release()
        writer.release()
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-i', str(raw), '-c:v', 'libx264',
                    '-profile:v', 'main', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dest)], check=True)
    raw.unlink()
    return fps


def run(args):
    for case in args.cases:
        out = args.work/case
        out.mkdir(parents=True, exist_ok=True)
        video = video_for(case)
        cap = cv2.VideoCapture(str(video))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        cap.release()
        if not ok or count < 1:
            raise ValueError(f'Cannot read {video}')
        h,w = frame.shape[:2]
        masks = np.zeros((count,h,w), dtype=bool)
        metadata = dict(case=case, video=str(video), prompt=args.prompt, prompt_frame=0,
                        prompt_calls=1, point_prompts=0, box_prompts=0, later_corrections=0,
                        selection='highest SAM3 score at frame 0; keep that object ID',
                        source_frame_count=count, source_hw=[h,w])
        box = None
        if args.initializer == 'reference-box':
            import torch
            from infrastructure.shared.inference.generation.link_crop_wrapper.dinov2_reference_boxes import Dinov2DenseEncoder, localize_reference_groups
            refs=sorted(Path(os.environ.get('PDI_PALM_REFERENCES', str((_workspace_root() / 'documentation/data/references/palm')))).glob('*.png'))
            if not refs:
                raise ValueError('Missing original gripper reference images')
            encoder=Dinov2DenseEncoder(Path(os.environ.get('PDI_DINO_DIRECTORY', str((_workspace_root() / 'infrastructure/models/dinov2')))),'cuda')
            boxes,_=localize_reference_groups(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),
                {'palm':refs},encoder,scene_side=840,reference_side=448,top_fraction=.12,
                padding_fraction=.10,minimum_contrast=.02,reference_spatial_priors=True)
            box=boxes[0]
            metadata.update(prompt='visual',box_prompts=1,initializer='existing_DINO_reference_box',
                            box_xyxy=list(box.box_xyxy),box_xywh_normalized=list(box.box_xywh_normalized))
            del encoder
            gc.collect()
            torch.cuda.empty_cache()
        p = predictor()
        sid = p.handle_request(dict(type='start_session', resource_path=str(video),
                                    offload_video_to_cpu=True))['session_id']
        seen = {0}
        try:
            request=dict(type='add_prompt',session_id=sid,frame_index=0,text=metadata['prompt'])
            if box is not None:
                request.update(bounding_boxes=[list(box.box_xywh_normalized)],bounding_box_labels=[1])
            d = p.handle_request(request)['outputs']
            ids = list(d['out_obj_ids'])
            scores = np.asarray(d['out_probs'],dtype=float).reshape(-1)
            metadata['initial_candidates'] = [dict(object_id=int(i),score=float(s)) for i,s in zip(ids,scores)]
            if ids:
                selected = int(np.argmax(scores))
                obj = ids[selected]
                metadata['selected_object_id'] = int(obj)
                masks[0] = np.asarray(d['out_binary_masks'][selected],bool)
                for response in p.handle_stream_request(dict(type='propagate_in_video',session_id=sid,
                        propagation_direction='forward',start_frame_index=0,max_frame_num_to_track=count-1)):
                    t = int(response['frame_index'])
                    data = response['outputs']
                    if not 0 <= t < count:
                        raise ValueError('SAM returned out-of-range frame')
                    seen.add(t)
                    tracked_ids = list(data['out_obj_ids'])
                    if obj in tracked_ids:
                        masks[t] = np.asarray(data['out_binary_masks'][tracked_ids.index(obj)],bool)
                metadata['status'] = 'propagated'
            else:
                metadata['status'] = 'no_initial_detection'
                metadata['selected_object_id'] = None
        finally:
            p.handle_request(dict(type='close_session',session_id=sid))
            p.shutdown()
        np.savez_compressed(out/'masks.npz',masks=masks)
        metadata.update(returned_frames=len(seen),empty_frames=np.flatnonzero(~masks.any((1,2))).tolist(),
                        areas=masks.sum((1,2)).tolist())
        dest = out/'naive_masking.mp4'
        metadata['fps'] = render(video,masks,dest,args.ffmpeg)
        metadata['mp4_sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
        (out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(case,metadata['status'],'empty frames',len(metadata['empty_frames']),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',default=CASES)
    parser.add_argument('--prompt',default='robot gripper')
    parser.add_argument('--initializer',choices=['text','reference-box'],default='reference-box')
    parser.add_argument('--ffmpeg',default=os.environ.get('FFMPEG_BINARY') or shutil.which('ffmpeg'))
    args = parser.parse_args()
    if not args.ffmpeg:
        parser.error('Supply --ffmpeg or set FFMPEG_BINARY')
    run(args)
