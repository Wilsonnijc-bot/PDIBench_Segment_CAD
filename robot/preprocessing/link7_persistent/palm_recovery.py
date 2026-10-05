"""Area-triggered, palm-only Qwen review and SAM correction proposals.

prepare runs locally. review and repair explicitly load GPU models.
"""

from pathlib import Path as _LayoutPath
from infrastructure.pdibench.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = (_workspace_root())
PALM_VISIBILITY = """FIRST CHECK VISIBILITY, BEFORE JUDGING DEFORMATION.
The requested target is the gripper PALM, not the forearm, wrist, fingers alone,
or a nearby object. Check that the same palm is identifiable and sufficiently
visible in BOTH images. If it is absent from the crop, outside the frame,
hidden by another object, or too occluded to judge, SKIP this comparison:
state=unclear, issue=target_not_visible, severity=unclear, probability=null.
Do not infer deformation from crop truncation or missing visual evidence.
Do not substitute another visible robot part for the palm. A changed but
clearly visible palm can still be judged deformed; partial occlusion is allowed
only when enough palm structure remains visible to support the judgment."""
PALM_FOCUS = """Inspect ONLY the rigid gripper palm/hand housing seen in Image 1.
Exclude articulated fingers, wrist, and any held object from the target.
Palm material remains the target if deformation changes its white appearance to
dark material. Assess major structural deformation, not color change alone.
Normal jaw motion and finger-only defects must not trigger a palm diagnosis.
If the palm cannot be distinguished reliably, return unclear. Frame 0 is a
comparison reference, not guaranteed ground-truth normal geometry."""
PALM_RESPONSE = """Return exactly one JSON object with these fields:
segment: palm
state: normal, deformed, or unclear
issue: concise problem, none, or target_not_visible when visibility fails
evidence: one sentence describing what is actually visible
severity: none, mild, moderate, severe, or unclear
probability: deformation probability from 0 to 1 for a visible target; null for unclear
For an evaluable target use deformed when probability >= 0.5, otherwise normal.
Visibility failure takes precedence over classification: return unclear and null,
not normal or deformed. Return no text outside the JSON object."""
LOCALIZE = """Locate the SAME rigid gripper palm in this candidate RGB image.
Coordinates must be normalized x,y in [0,1] relative to this candidate crop.
Return strict JSON with only: status (ok or unclear), positive_points (2 to 8
points strictly inside visible palm material, including deformed/dark palm
material if confidently identifiable), negative_points (0 to 8 points strictly
inside identifiable fingers, wrist, held object or background), evidence (text).
Do not place points on uncertain boundaries or invisible/occluded material.
Do not treat all dark material as palm. Return unclear and empty point lists
when anatomical identity is uncertain. A deformation verdict is not permission
to guess coordinates."""


def json_object(raw):
    # Accept a fenced JSON object without changing its field values.
    start=raw.find('{')
    if start<0:raise ValueError('No JSON object in response')
    data,_=json.JSONDecoder().raw_decode(raw,start)
    if not isinstance(data,dict):raise ValueError('Expected JSON object')
    return data


def parse_palm_response(raw):
    from infrastructure.shared.inference.generation.link_crop_wrapper.run_segment_vlm import parse_response, RESPONSE_KEYS
    d=json_object(raw)
    if set(d)!=set(RESPONSE_KEYS):raise ValueError('Unexpected palm response fields')
    if d['segment']!='palm':raise ValueError('Wrong target')
    if d['state']=='unclear':
        if d['probability'] is not None or d['severity']!='unclear':raise ValueError('Skipped comparisons need null probability and unclear severity')
        if not all(isinstance(d[k],str) and d[k].strip() for k in ['issue','evidence']):raise ValueError('Missing skip explanation')
        return dict(d,skipped=True,skip_reason=d['issue'])
    if d['issue']=='target_not_visible':raise ValueError('Invisible target cannot receive a binary verdict')
    result=parse_response(raw,'palm')
    if float(d['probability'])!=result['probability']:raise ValueError('No silent probability inversion')
    return dict(result,skipped=False,skip_reason='')


def events(areas, threshold=.25, separation=8):
    areas = np.asarray(areas, dtype=float)
    if areas.ndim != 1 or len(areas)<2 or not np.isfinite(areas).all() or (areas<0).any():
        raise ValueError('Expected nonnegative finite per-frame areas')
    if threshold<=0 or separation<1:raise ValueError('Invalid event settings')
    changes = np.zeros(len(areas))
    changes[1:] = np.abs(np.log(np.maximum(areas[1:],1)/np.maximum(areas[:-1],1)))
    # Keep the first threshold crossing, not the largest error in the video.
    selected = []
    for f in np.flatnonzero(changes>=threshold):
        if not selected or f-selected[-1]>=separation:
            selected.append(int(f))
        if len(selected)==2:break
    return selected, changes


def window(frame, count, total):
    start = max(0,min(frame-4,total-count))
    return list(range(start,min(total,start+count)))


def post_surge_window(detected, count, total):
    # events() indexes the later frame of the A[t-1] -> A[t] transition.
    # User's surge frame is the earlier side: transition 11 -> 12 starts at 12.
    return list(range(detected[0],min(total,detected[0]+count))) if detected else []


def crop_box(mask, reference_size=None):
    y,x = np.where(mask)
    if not len(x):raise ValueError('Cannot locate crop from empty support')
    width,height = x.max()-x.min()+1,y.max()-y.min()+1
    if reference_size is not None:
        width,height = max(width,reference_size[0]),max(height,reference_size[1])
    # Context must not shrink with the failing mask. Minimum 160x160 original pixels.
    side_x,side_y = max(160,int(np.ceil(width*2))),max(160,int(np.ceil(height*2)))
    cx,cy = (x.min()+x.max())/2,(y.min()+y.max())/2
    return [int(cx-side_x/2),int(cy-side_y/2),int(cx-side_x/2)+side_x,int(cy-side_y/2)+side_y]


def prepare(args):
    masks = np.load(args.masks)['masks'].astype(bool)
    if masks.ndim!=3:raise ValueError('Expected palm masks shaped T,H,W')
    cap=cv2.VideoCapture(str(args.video));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
    while True:
        ok,bgr=cap.read()
        if not ok:break
        frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames)!=len(masks) or masks.shape[1:]!=frames[0].shape[:2]:raise ValueError('Video/mask mismatch')
    if fps<=0:raise ValueError('Invalid FPS')
    areas=masks.sum((1,2));detected,changes=events(areas,args.threshold,max(1,round(.5*fps)))
    selected=detected[:1]
    candidates=post_surge_window(detected,10,len(frames))
    windows=[candidates] if candidates else []
    dest=args.output;dest.mkdir(parents=True,exist_ok=True)
    ref_box=crop_box(masks[0]);ref_size=((ref_box[2]-ref_box[0])/2,(ref_box[3]-ref_box[1])/2)
    Image.fromarray(frames[0]).crop(ref_box).save(dest/'reference.png')
    records=[]
    for f in candidates:
        # A recent-mask union provides context; it is never applied to RGB pixels.
        support=np.any(masks[max(0,f-4):f+1],axis=0)
        if not support.any():support=masks[0]
        box=crop_box(support,ref_size)
        Image.fromarray(frames[f]).crop(box).save(dest/f'frame_{f:05d}.png')
        records.append(dict(frame=f,time=f/fps,crop=f'frame_{f:05d}.png',box_xyxy=box))
    manifest=dict(video=str(args.video.resolve()),masks=str(args.masks.resolve()),fps=fps,
        mask_sha256=hashlib.sha256(args.masks.read_bytes()).hexdigest(),
        video_sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(),
        threshold=args.threshold,detected_event_frames=detected,
        surge_transition=[selected[0]-1,selected[0]] if selected else None,
        surge_frame=selected[0]-1 if selected else None,event_frames=selected,event_windows=windows,
        area=areas.tolist(),absolute_log_area_change=changes.tolist(),
        reference_box=ref_box,calls=records,status='ready' if candidates else 'no_area_event',
        policy='First transition only; ten consecutive frames starting at its later frame; no earlier frames or second-event selection')
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (dest/'prompt.txt').write_text(PALM_FOCUS+'\n\n'+LOCALIZE+'\n')
    print(json.dumps(dict(events=selected,windows=windows,calls=len(records),status=manifest['status'])))


def parse_localization(raw, box, image_hw):
    data=json_object(raw)
    if set(data)!={'status','positive_points','negative_points','evidence'}:raise ValueError('Invalid localization fields')
    if data['status'] not in ['ok','unclear']:raise ValueError('Invalid localization status')
    if not isinstance(data['evidence'],str) or not data['evidence'].strip():raise ValueError('Missing localization evidence')
    coords=[];labels=[]
    for key,label in [('positive_points',1),('negative_points',0)]:
        points=np.asarray(data[key],dtype=float)
        if not points.size:continue
        if points.ndim!=2 or points.shape[1]!=2 or len(points)>8 or not np.isfinite(points).all() or (points<0).any() or (points>1).any():
            raise ValueError('Invalid normalized point coordinates')
        points=points*[box[2]-box[0],box[3]-box[1]]+box[:2]
        h,w=image_hw
        if (points<0).any() or (points[:,0]>=w).any() or (points[:,1]>=h).any():raise ValueError('Points outside source frame/padding')
        coords.extend(points.tolist());labels.extend([label]*len(points))
    if data['status']=='ok' and labels.count(1)<2:raise ValueError('Need at least two palm positives')
    if data['status']=='unclear' and coords:raise ValueError('Unclear localization must not supply guessed prompts')
    if len({tuple(p) for p in coords})!=len(coords):raise ValueError('Duplicate or contradictory points')
    return dict(status=data['status'],points_xy=coords,labels=labels,evidence=data['evidence'])


def review(args):
    from infrastructure.shared.inference.generation.link_crop_wrapper.run_segment_vlm import load_model, generate, LOOSE_V2_SYSTEM_PROMPT
    system_prompt=PALM_VISIBILITY+'\n\n'+LOOSE_V2_SYSTEM_PROMPT+'\n'+PALM_FOCUS
    manifest=json.loads((args.output/'manifest.json').read_text())
    if not manifest['calls']:
        print('No area events; no model loaded.');return
    previous={}
    if args.reuse_responses and (args.output/'diagnoses.json').exists():
        old_provenance=json.loads((args.output/'inference_provenance.json').read_text())
        if old_provenance['model_path']!=str(args.model.resolve()) or old_provenance['system_prompt']!=system_prompt:
            raise ValueError('Cannot reuse responses from different model or prompt')
        previous={d['frame']:d for d in json.loads((args.output/'diagnoses.json').read_text())}
    processor,model,architecture=load_model(args.model)
    import torch
    import transformers
    provenance=dict(model_path=str(args.model.resolve()),architecture=architecture,
        torch=torch.__version__,transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(0),do_sample=False,enable_thinking=False,
        system_prompt=system_prompt,response_request=PALM_RESPONSE,prompt_version='palm-visibility-first-v1',
        device_map=getattr(model,'hf_device_map',None),
        localization_prompt=LOCALIZE)
    download_provenance=args.model/'download_provenance.json'
    if download_provenance.exists():provenance['download']=json.loads(download_provenance.read_text())
    (args.output/'inference_provenance.json').write_text(json.dumps(provenance,indent=2))
    print(f'Loaded {architecture}; {len(manifest["calls"])} palm comparisons',flush=True)
    diagnoses=[]
    for call in manifest['calls']:
        with Image.open(args.output/'reference.png') as ref,Image.open(args.output/call['crop']) as candidate:
            messages=[dict(role='system',content=[dict(type='text',text=system_prompt)]),
                dict(role='user',content=[dict(type='text',text='Requested segment: palm. Image 1: frame 0. Black out-of-frame padding is unavailable evidence.'),
                    dict(type='image',image=ref.convert('RGB')),dict(type='text',text=f'Image 2: frame {call["frame"]}.'),
                    dict(type='image',image=candidate.convert('RGB')),dict(type='text',text=PALM_RESPONSE)])]
            prior=previous.get(call['frame'])
            if prior and prior['box_xyxy']==call['box_xyxy'] and prior['crop']==call['crop']:
                raw=prior['raw_response']
            else:raw=generate(model,processor,messages,220)
        try:
            parsed=parse_palm_response(raw)
            error=''
        except (ValueError,TypeError,KeyError) as e:parsed=dict(state='unclear',probability=None,skipped=True,skip_reason='parse_error');error=str(e)
        diagnoses.append(dict(**call,**parsed,parse_error=error,raw_response=raw))
        (args.output/'diagnoses.json').write_text(json.dumps(diagnoses,indent=2))
        print(f'Frame {call["frame"]}: {parsed["state"]} {parsed.get("probability", "")} {error}',flush=True)
    positives=[d for d in diagnoses if d['state']=='deformed' and not d['parse_error']]
    all_skipped=all(d['skipped'] for d in diagnoses)
    result=dict(architecture=architecture,status='localization_pending' if positives else ('all_frames_skipped' if all_skipped else 'no_confirmed_deformation'),binary_list=[
        dict(frame=d['frame'],deformed=None if d['state']=='unclear' else d['state']=='deformed',skipped=d['skipped'],skip_reason=d['skip_reason']) for d in diagnoses])
    if positives and not args.classify_only:
        first=min(positives,key=lambda d:d['frame'])
        with Image.open(args.output/'reference.png') as ref,Image.open(args.output/first['crop']) as candidate:
            messages=[dict(role='system',content=[dict(type='text',text=PALM_FOCUS)]),dict(role='user',content=[
                dict(type='image',image=ref.convert('RGB')),dict(type='image',image=candidate.convert('RGB')),dict(type='text',text=LOCALIZE)])]
            raw=generate(model,processor,messages,450)
        result.update(frame=first['frame'],raw_localization=raw,status='localization_failed')
        try:
            masks=np.load(manifest['masks'])['masks']
            proposal=parse_localization(raw,first['box_xyxy'],masks.shape[1:])
            result.update(proposal=proposal,status='ready_for_sam' if proposal['status']=='ok' else 'localization_unclear')
        except (ValueError,TypeError,KeyError) as e:result['localization_error']=str(e)
    (args.output/'correction.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'])


def repair(args):
    from robot.preprocessing.link7_persistent.v1_mask import predictor
    manifest=json.loads((args.output/'manifest.json').read_text());proposal=json.loads((args.output/'correction.json').read_text())
    if proposal['status']!='ready_for_sam':raise ValueError('No validated localization proposal')
    mask_path=Path(manifest['masks'])
    if hashlib.sha256(mask_path.read_bytes()).hexdigest()!=manifest['mask_sha256']:raise ValueError('Masks changed after preparation')
    original=np.load(mask_path)['masks'];h,w=original.shape[1:];f=proposal['frame']
    p=predictor();sid=p.handle_request(dict(type='start_session',resource_path=manifest['video'],offload_video_to_cpu=True))['session_id']
    try:
        points=np.array(proposal['proposal']['points_xy']);labels=proposal['proposal']['labels']
        if args.positive_only:
            points=points[np.asarray(labels)==1];labels=[1]*len(points)
        data=p.handle_request(dict(type='add_prompt',session_id=sid,frame_index=f,obj_id=0,
            points=(points/[w,h]).tolist(),point_labels=labels))['outputs']
        ids=list(data['out_obj_ids'])
        if 0 not in ids:raise ValueError('SAM returned no corrected palm')
        corrected=np.asarray(data['out_binary_masks'][ids.index(0)],bool)
        xy=np.floor(points).astype(int);membership=corrected[xy[:,1],xy[:,0]]
        if not np.array_equal(membership,np.asarray(labels,bool)):raise ValueError('SAM correction contradicts point prompts')
        if not (corrected&~original[f]).any():raise ValueError('Correction recovers no previously omitted pixels')
        # A bounded proposal, never overwrite trusted masks or claim segmentation truth.
        start=max(0,f-10);end=len(original)-1
        output=original.copy();output[f]=corrected;updated=[f]
        for direction,limit in [('forward',end-f),('backward',f-start)]:
            if limit==0:continue
            for response in p.handle_stream_request(dict(type='propagate_in_video',session_id=sid,
                propagation_direction=direction,start_frame_index=f,max_frame_num_to_track=limit)):
                t=int(response['frame_index']);d=response['outputs'];ids=list(d['out_obj_ids'])
                if start<=t<=end and 0 in ids:output[t]=d['out_binary_masks'][ids.index(0)];updated.append(t)
        np.savez_compressed(args.output/'proposed_masks.npz',masks=output)
        (args.output/'repair.json').write_text(json.dumps(dict(status='proposal_not_validated_ground_truth',
            supplied_label_mode='positive_only' if args.positive_only else 'positive_and_negative',
            correction_frame=f,window=[start,end],updated_frames=sorted(set(updated)),
            added_pixels=int((corrected&~original[f]).sum()),removed_pixels=int((original[f]&~corrected).sum())),indent=2))
    finally:
        p.handle_request(dict(type='close_session',session_id=sid));p.shutdown()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','review','repair'])
    parser.add_argument('--video',type=Path,default=(_workspace_root() / 'reg2inv_results/diagnosis_LVP_0049/source.mp4'))
    parser.add_argument('--masks',type=Path,default=(_workspace_root() / 'reg2inv_results_v1/LVP_0049/guided_masks.npz'))
    parser.add_argument('--output',type=Path,default=(_workspace_root() / 'results_palm_recovery/LVP_0049'))
    parser.add_argument('--threshold',type=float,default=.25)
    parser.add_argument('--classify-only',action='store_true',help='Save diagnoses; use separate concise point grounding next')
    parser.add_argument('--positive-only',action='store_true',help='Repair ablation: omit all negative prompts')
    parser.add_argument('--model',type=Path)
    parser.add_argument('--reuse-responses',action='store_true',help='Reparse saved responses for the same prepared inputs')
    args=parser.parse_args()
    if args.stage=='review' and args.model is None:parser.error('review requires --model')
    globals()[args.stage](args)
