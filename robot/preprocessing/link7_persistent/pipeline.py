"""Run VLM1 frame selection, VLM2 SAM prompting, propagation, and validation."""

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from robot.preprocessing.link7_persistent.vlm2_sam_prompting import run_case
from robot.preprocessing.link7_persistent.frame_lineage import read_original_frame, validate_lineage
from robot.preprocessing.link7_persistent.vlm_client import VLMClient
from robot.preprocessing.link7_persistent.interface.config import public_config, role_config
from robot.preprocessing.link7_persistent.vlm_prompts import VLM1_RESPONSE_REQUEST, VLM1_SYSTEM_PROMPT, VLM2_NEGATIVE_PROMPT, VLM2_POSITIVE_PROMPT, VLM2_SYSTEM_PROMPT
from robot.preprocessing.link7_persistent.gripper_negatives import video_for
from robot.preprocessing.link7_persistent.palm_recovery import crop_box

ROOT = (_workspace_root())
CASES = ['Cosmos3_0048','Cosmos25_0003','Cosmos25_0023','Cosmos25_0053']


def sha(path):
    result=hashlib.sha256()
    with open(path,'rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def validate_vlm2_call_images(seed, diagnoses, video, selected_png, example_hashes):
    """Check each point call against its own source frame, including fallbacks."""
    deformed = sorted(int(d['frame']) for d in diagnoses
                      if d.get('state') == 'deformed' and not d.get('parse_error'))
    positives = [call for call in seed['calls']
                 if call.get('role', '').startswith('positive_points')]
    negatives = [call for call in seed['calls']
                 if call.get('role', '').startswith('negative_points')]
    alternates = [call for call in seed['calls']
                  if call.get('role', '').endswith('_luna_fallback')]
    if (not 1 <= len(positives) <= 2 or not 1 <= len(negatives) <= 2
            or len(alternates) > 1 or len(seed['calls']) > 3
            or not deformed or seed['earliest_deformed_frame'] != deformed[0]):
        raise ValueError('VLM2 fallback call lineage mismatch')
    frame = read_original_frame(video, deformed[0] + 1)
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format='PNG')
    expected = hashlib.sha256(buffer.getvalue()).hexdigest()
    if any(call['image_sha256'][0] != expected for call in positives
           if 'image_sha256' in call):
        raise ValueError('Wrong VLM2 positive-call source frame')
    if positives[-1]['image_sha256'][0] != selected_png:
        raise ValueError('Selected VLM2 frame differs from final positive call')
    if any(call['image_sha256'][0] != selected_png for call in negatives
           if 'image_sha256' in call):
        raise ValueError('Wrong VLM2 negative-call source frame')
    if any(len(call['image_sha256']) != 4 or
           call['image_sha256'][1:] != example_hashes for call in seed['calls']
           if 'image_sha256' in call):
        raise ValueError('VLM2 reference images changed')


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False))
    temp.replace(path)


def load(a):return json.loads((a.work/'provenance.json').read_text())
def commit(a,record):save(a.work/'provenance.json',record)


def surge_windows(areas,fps,threshold=.25):
    areas=np.asarray(areas,float)
    if areas.ndim!=1 or len(areas)<2 or not np.isfinite(areas).all() or (areas<0).any():
        raise ValueError('Invalid mask area sequence')
    separation=max(1,round(fps*.5))
    events=[]
    for t in range(1,len(areas)):
        # A zero-area disappearance followed by positive area is also an upward event.
        ratio=max(areas[t],1)/max(areas[t-1],1)
        if math.log(ratio)>=threshold and (not events or t-events[-1]['event_frame']>=separation):
            events.append(dict(surge_frame=t-1,event_frame=t,area_before=int(areas[t-1]),
                               area_after=int(areas[t]),ratio=float(ratio)))
    if events:
        frames=sorted({f for event in events for f in range(max(0,event['event_frame']-4),min(len(areas),event['event_frame']+6))})
    else:
        # Human-label fallback: exactly one frame near 30%, two around 50%,
        # one near 60%, and one near 70% of the original video.
        last=len(areas)-1
        targets=[.30,.48,.52,.60,.70]
        frames=sorted({max(0,min(last,round(last*t))) for t in targets})
    return events,frames


def prepare(a):
    if (a.work/'provenance.json').exists():raise FileExistsError('Run already prepared; preserve frozen provenance')
    # Fail before creating partial run state when an example path is wrong.
    # The previous retry copied img_0179 and then stopped on a missing later
    # example, leaving an apparently live run with only naive masks.
    missing=[str(path) for path in a.examples if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing VLM2 example image(s): '+', '.join(missing))
    example_dir=a.work/'examples'
    example_dir.mkdir(parents=True,exist_ok=True)
    for path in a.examples:
        Image.open(path).convert('RGB').save(example_dir/path.name)
    examples=[str((example_dir/p.name).resolve()) for p in a.examples]
    record=dict(level=a.level,status='running',cases=a.cases,results={},
        config=dict(vlm1=public_config('vlm1'),vlm2=public_config('vlm2'),
        vlm2_malformed_fallback=public_config('vlm2_malformed_fallback'),
        vlm1_system_prompt=VLM1_SYSTEM_PROMPT,vlm1_response_request=VLM1_RESPONSE_REQUEST,
        vlm2_system_prompt=VLM2_SYSTEM_PROMPT,vlm2_positive_prompt=VLM2_POSITIVE_PROMPT,
        vlm2_negative_prompt=VLM2_NEGATIVE_PROMPT,
        vlm2_images='selected reseed frame + three annotated references',
        events='upward log area ratio >= 0.25; >=round(fps*0.5) separation; all events',
        windows='event_frame-4 through event_frame+5, clipped, deduplicated, sorted',
        vlm1_crop='recent 5 naive masks union, 2x extent, min160px, clamped source bounds',
        vlm1_reference='existing nondeformed reference crop; reference itself is not reclassified',
        reseed='earliest valid VLM1 deformed frame + 1',sam='new session, one five-point prompt, bidirectional full-video propagation',
        examples=examples,example_sha256=[sha(p) for p in examples],minimum_point_distance_px=5),
        environment=dict(host=os.uname().nodename,python=sys.version,implementation={str(p.relative_to(ROOT)):sha(p) for p in [_SOURCE_PATH,(_workspace_root() / 'robot/preprocessing/link7_persistent/interface/config.py'),(_workspace_root() / 'robot/preprocessing/link7_persistent/vlm_client.py'),(_workspace_root() / 'robot/preprocessing/link7_persistent/vlm_prompts.py'),(_workspace_root() / 'robot/preprocessing/link7_persistent/vlm2_sam_prompting.py'),(_workspace_root() / 'robot/preprocessing/link7_persistent/vlm2_interface.py'),(_workspace_root() / 'robot/preprocessing/link7_persistent/frame_lineage.py'),(_workspace_root() / 'robot/preprocessing/link7_persistent/naive_sam3.py'),(_workspace_root() / 'robot/preprocessing/link7_persistent/gripper_review.py'),(_workspace_root() / 'robot/preprocessing/link7_persistent/v1_mask.py'),(_workspace_root() / 'infrastructure/shared/inference/generation/link_crop_wrapper/run_segment_vlm.py')]},
                         sam_checkpoint_sha256=sha(Path(os.environ.get('PDI_SAM3_CHECKPOINT', str((_workspace_root() / 'infrastructure/models/sam3/sam3.pt')))))))
    commit(a,record)
    for case in a.cases:
        r=record['results'][case]=dict(status='preparing',case=case)
        try:
            video=video_for(case);naive=a.work/'naive'/case
            if not (naive/'provenance.json').exists():
                naive=a.work/case/'naive'/case
            metadata=json.loads((naive/'provenance.json').read_text())
            masks=np.load(naive/'masks.npz')['masks']
            if metadata['status']!='propagated' or metadata['returned_frames']!=len(masks):raise ValueError('Incomplete naive propagation')
            fps=metadata['fps'];events,frames=surge_windows(masks.sum((1,2)),fps)
            r.update(video=str(video),source_sha256=sha(video),source_hw=list(masks.shape[1:]),source_frame_count=len(masks),fps=fps,
                     naive=metadata,naive_masks_sha256=sha(naive/'masks.npz'),events=events,vlm1_frames=frames,calls=[])
            inp=a.work/'vlm1_inputs'/case;inp.mkdir(parents=True,exist_ok=True)
            h,w=masks.shape[1:]
            reference=a.reference_root/case/'inputs/reference.png'
            if reference.exists():
                shutil.copyfile(reference,inp/'reference.png')
                reference_source=str(reference)
            else:
                # New human-label cases have no prior VLM1 reference. Build a clean
                # frame-0 reference crop from the fresh naive SAM support.
                support=masks[0]
                ys,xs=np.where(support)
                if len(xs):
                    refbox=crop_box(support); refbox=[max(0,refbox[0]),max(0,refbox[1]),min(w,refbox[2]),min(h,refbox[3])]
                else: refbox=[0,0,w,h]
                Image.fromarray(read_original_frame(video,0)).crop(refbox).save(inp/'reference.png')
                reference_source='fresh naive frame-0 support crop'
            r['reference']=dict(path=str(inp/'reference.png'),source=reference_source,sha256=sha(inp/'reference.png'))
            if not frames:r['status']='no_naive_surge';commit(a,record);continue
            refsize=(masks[0].any(0).sum(),masks[0].any(1).sum())
            for f in frames:
                support=masks[max(0,f-4):f+1].any(0)
                if not support.any():support=masks[0]
                if support.any():
                    box=crop_box(support,refsize);box=[max(0,box[0]),max(0,box[1]),min(w,box[2]),min(h,box[3])]
                else:box=[0,0,w,h]
                rgb=read_original_frame(video,f);path=inp/f'frame_{f:05d}.png';Image.fromarray(rgb).crop(box).save(path)
                r['calls'].append(dict(frame=f,crop=str(path),box_xyxy=box,sha256=sha(path)))
            out=a.work/'artifacts'/case;out.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(r['calls'][0]['crop'],out/'vlm1_initial_frame.png')
            r['status']='vlm1_pending'
            print(case,'events',[e['event_frame'] for e in events],'VLM1 frames',frames,flush=True)
        except Exception as error:r.update(status='failed_prepare',error=str(error));traceback.print_exc()
        commit(a,record)


def select_frames(a):
    from robot.preprocessing.link7_persistent.palm_recovery import parse_palm_response
    record=load(a)
    if not any(r['status']=='vlm1_pending' for r in record['results'].values()):return
    client=VLMClient('vlm1_deformation_frame_selection',role_config('vlm1'))
    for case in a.cases:
        r=record['results'][case]
        if r['status']!='vlm1_pending':continue
        previous = list(r.get('diagnoses', []))
        r['diagnoses']=[]
        for index, call in enumerate(r['calls']):
            if (index < len(previous)
                    and previous[index].get('frame') == call['frame']
                    and previous[index].get('sha256') == call['sha256']
                    and ('raw_response' in previous[index] or previous[index].get('parse_error'))):
                r['diagnoses'].append(previous[index])
                continue
            ref=Image.open(r['reference']['path']).convert('RGB');candidate=Image.open(call['crop']).convert('RGB')
            user_prompt=f"Image 1 is the nondeformed reference. Image 2 is source frame {call['frame']}. "+record['config']['vlm1_response_request']
            entry=dict(call)
            try:
                response=client.ask([ref,candidate],user_prompt,system_prompt=record['config']['vlm1_system_prompt'])
                raw=response['answer'];entry['raw_response']=raw;entry['vlm_request']=response
                entry.update(parse_palm_response(raw));entry['parse_error']=''
            except Exception as error:entry.update(state='unclear',parse_error=str(error))
            r['diagnoses'].append(entry)
            commit(a,record)
            print(case,'VLM1',call['frame'],entry['state'],flush=True)
        valid=[d['frame'] for d in r['diagnoses'] if d['state']=='deformed' and not d.get('parse_error')]
        r['status']='vlm2_pending' if valid else 'no_confirmed_deformation'
        if valid:
            r['earliest_deformed_frame']=min(valid);r['reseeding_frame']=min(valid)+1
            if r['reseeding_frame']>=r['source_frame_count']:r['status']='reseed_frame_unavailable'
        commit(a,record)


def prompt_sam(a):
    record=load(a)
    for case in a.cases:
        r=record['results'][case]
        if r['status']!='vlm2_pending':continue
        try:
            def persist(value):r['vlm2_partial']=value;commit(a,record)
            seed=run_case(r['video'],r['diagnoses'],record['config']['examples'],a.work/'artifacts'/case,persist=persist,fallback_attempts=getattr(a, 'vlm2_fallback_attempts', 1))
            # First VLM1 input is the real diagnosis crop; preserve it rather than a different full-frame export.
            shutil.copyfile(r['calls'][0]['crop'],a.work/'artifacts'/case/'vlm1_initial_frame.png')
            r['seed']=seed;r.pop('vlm2_partial',None)
            r['selected_deformed_frame']=seed['earliest_deformed_frame']
            r['reseeding_frame']=seed['frame']
            r['status']='sam_pending'
            save(a.work/'sam'/case/'seed.json',seed)
            print(case,'VLM2 frame',seed['frame'],'points',seed['points'],flush=True)
        except Exception as error:r.update(status='failed_vlm2',error=str(error));traceback.print_exc()
        finally:
            latest=a.work/'artifacts'/case/'vlm2_interface'
            if latest.exists():
                shutil.copytree(latest,a.work/'vlm2_interface',dirs_exist_ok=True)
        commit(a,record)


def segment(a):
    from robot.preprocessing.link7_persistent.gripper_review import sam
    record=load(a)
    for case in a.cases:
        r=record['results'][case]
        if r['status']!='sam_pending':continue
        try:
            seed=r['seed'];validate_lineage(seed,seed['frame'],r['reseeding_frame'])
            sam(SimpleNamespace(cases=[case],work=a.work/'sam',checkpoint=0,ffmpeg=a.ffmpeg))
            output=a.work/'sam'/case
            r['metrics']=json.loads((output/'seed_metrics.json').read_text())
            r['masks_sha256']=sha(output/'seed_masks.npz')
            r['mask_source']=str(output/'seed_masks.npz')
            shutil.copyfile(output/'masking.mp4',a.work/'artifacts'/case/'masking.mp4')
            r['status']='validate_pending'
        except Exception as error:r.update(status='failed_sam',error=str(error));traceback.print_exc()
        commit(a,record)


def repair_overmask(a):
    """Optional stage; the original VLM1/VLM2 selection remains unchanged."""
    if a.object_mask_root is None:
        raise ValueError('VLM3 requires --object-mask-root with existing task-object masks')
    from robot.preprocessing.link7_persistent.vlm3_overmask import repair_persistent_run
    repair_persistent_run(a.work, a.cases, a.object_mask_root)


def mp4_faststart(path):
    atoms=[]
    with open(path,'rb') as handle:
        end=path.stat().st_size
        while handle.tell()+8<=end:
            pos=handle.tell();size=int.from_bytes(handle.read(4),'big');kind=handle.read(4).decode('ascii',errors='replace');header=8
            if size==1:size=int.from_bytes(handle.read(8),'big');header=16
            if size==0:size=end-pos
            if size<header:raise ValueError('Invalid MP4 atom')
            atoms.append(kind);handle.seek(pos+size)
    return 'moov' in atoms and 'mdat' in atoms and atoms.index('moov')<atoms.index('mdat')


def validate(a):
    record=load(a)
    for case in a.cases:
        r=record['results'][case]
        if r['status']!='validate_pending':continue
        try:
            out=a.work/'artifacts'/case;seed=r['seed'];masks=np.load(r['mask_source'])['masks'];h,w=r['source_hw']
            validate_lineage(seed,r['metrics']['frame'],r['reseeding_frame'])
            actual=read_original_frame(r['video'],r['reseeding_frame']);saved=np.asarray(Image.open(out/'vlm2_reseed_frame.png'))
            if not np.array_equal(actual,saved):raise ValueError('Reseeding PNG differs from original source pixels')
            if sha(out/'vlm1_initial_frame.png')!=r['calls'][0]['sha256']:raise ValueError('VLM1 PNG differs from actual first input')
            if masks.shape!=(r['source_frame_count'],h,w):raise ValueError('Mask sequence dimensions mismatch')
            expected_input_hash=sha(out/'vlm2_reseed_frame.png')
            validate_vlm2_call_images(seed,r['diagnoses'],r['video'],expected_input_hash,
                                      record['config']['example_sha256'])
            points=np.asarray(Image.open(out/'points.png'));green=(points[:,:,1]>220)&(points[:,:,0]<50)&(points[:,:,2]<50);red=(points[:,:,0]>220)&(points[:,:,1]<50)&(points[:,:,2]<50)
            if not green.any() or not red.any():raise ValueError('Blank point preview')
            replay=out/'masking.mp4';cap=cv2.VideoCapture(str(replay));n=0
            while True:
                ok,frame=cap.read()
                if not ok:break
                if frame.shape[:2]!=(h,2*w):raise ValueError('Replay is not source RGB beside mask')
                n+=1
            cap.release()
            if n!=len(masks) or not mp4_faststart(replay):raise ValueError('Incomplete replay or missing faststart')
            # Encoder explicitly fixes Main/yuv420p; FFmpeg inspection is retained as additional evidence.
            check=subprocess.run([a.ffmpeg,'-hide_banner','-i',str(replay)],capture_output=True,text=True)
            info=check.stderr
            if 'h264 (Main)' not in info or 'yuv420p' not in info:raise ValueError('Replay codec/profile mismatch')
            r['validation']=dict(frames_decoded=n,frame_lineage_ok=True,exact_inputs_ok=True,points_visible=True,
                                 membership_ok=r['metrics']['membership_ok'],empty_frames=r['metrics']['empty_frames'],codec='H.264 Main',pixel_format='yuv420p',faststart=True,
                                 visual_review='pending',artifact_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
            # Membership is a diagnostic because the predicted mask can be
            # valid while a point lies on a thin boundary after propagation.
            r['status']='completed_checks' if not r['metrics']['empty_frames'] else 'failed_mask_validation'
        except Exception as error:r.update(status='failed_validation',error=str(error));traceback.print_exc()
        commit(a,record)
    record['status']='completed_checks' if all(r['status']=='completed_checks' for r in record['results'].values()) else 'completed_with_case_failures'
    commit(a,record)
    print('FINAL',record['status'],{c:r['status'] for c,r in record['results'].items()},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','select_frames','prompt_sam','segment','validate','repair_overmask','continue','resume'])
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',default=CASES)
    parser.add_argument('--level',choices=['local','global'],default='global')
    parser.add_argument('--reference-root',type=Path,default=(_workspace_root() / 'robot/results/persistent_masking/experiments/global/global_combined_lower_dark_20260912'))
    parser.add_argument('--examples',nargs=3,type=Path,required=True)
    parser.add_argument('--ffmpeg',required=True)
    parser.add_argument('--enable-vlm3', action='store_true', help='Run optional object-overmask repair after validation')
    parser.add_argument('--object-mask-root', type=Path, help='Existing task-object segmentation root for optional VLM3')
    a=parser.parse_args();a.work=a.work.resolve()
    if a.stage in {'continue','resume'}:
        stages=['prepare','select_frames','prompt_sam','segment','validate']
        if a.stage=='resume':stages=stages[1:]
        if a.enable_vlm3:
            if a.object_mask_root is None:
                parser.error('--enable-vlm3 requires --object-mask-root')
            stages.append('repair_overmask')
        for stage in stages:
            role = 'vlm1' if stage == 'select_frames' else 'vlm2' if stage == 'prompt_sam' else None
            config = role_config(role) if role else None
            python = config['python'] if config and config['backend']=='local_gpu' else sys.executable
            arguments=[python,'-u','-m','robot.preprocessing.link7_persistent.pipeline',stage,*sys.argv[2:]]
            subprocess.run(arguments,check=True)
    else:globals()[a.stage](a)


if __name__=='__main__':main()
