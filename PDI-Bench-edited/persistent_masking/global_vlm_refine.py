"""Frozen global VLM grounding and mask-aware temporal correction experiment."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image
from persistent_masking.vlm_client import VisionRoute
from persistent_masking.gripper_review import load_frame, preview, point, sam
from persistent_masking.palm_recovery import json_object

WHITE = ('Image 1 is the reference. In Image 2, point at the center of the WHITE PART '
         'of the mixed white-and-dark hanging gripper in the lower part of the image. '
         'Place the positive point strictly inside the broad white gripper material itself, '
         'below the shiny metal collar, not at the center of the whole mixed-color object. '
         'Exclude the upper white wrist housing, forearm, collar, dark material, held object, and background. '
         'Follow the white gripper material even when deformed. Return only JSON '
         '{"point_2d":[x,y]}, coordinates 0 to 1000 relative to Image 2; use null if not visible.')
DARK = ('Image 1 is the original gripper reference. Image 2 is the current candidate crop. '
        'Place one positive point strictly inside the lower dark/black material of the hanging gripper '
        'below the white palm and shiny collar, including deformed black gripper material or fingers. '
        'Choose a broad visible interior, away from boundaries. Exclude the upper black rim, '
        'forearm trim, wrist, collar, white material, held objects, and background. '
        'Return only JSON {"point_2d":[x,y]}, coordinates 0 to 1000 relative to Image 2; use null if not visible.')
REVIEW = ('Image 1 is the reference gripper. Image 2 is the clean current RGB crop. Image 3 is the '
          'same crop with the current mask in cyan. The target is the entire mixed white-and-dark '
          'hanging gripper below the shiny collar, including deformed white housing and black gripper '
          'material/fingers. Exclude upper wrist, forearm, collar, held objects and background. '
          'Does cyan omit visible gripper material or include unrelated material? Use Image 2 to '
          'judge real material, not cyan color. Return only JSON {"action":"keep" or "correct" '
          'or "unclear","reason":"visible evidence","negative_points":[[x,y],...]}. '
          'For correct, optionally give up to two negative points strictly inside leaked non-gripper '
          'material (upper wrist and forearm preferred if leaked). Coordinates 0 to 1000 relative '
          'to Image 2. For keep or unclear return an empty list. Do not invent invisible parts.')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2))


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):h.update(block)
    return h.hexdigest()


def render(video,masks,dest,ffmpeg):
    from persistent_masking.naive_sam3 import render as legacy_render
    temp=dest.with_name('encoding.mp4')
    legacy_render(video,masks,temp,ffmpeg)
    subprocess.run([ffmpeg,'-y','-loglevel','error','-i',str(temp),'-c:v','libx264',
                    '-profile:v','main','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
    temp.unlink()


def ask(route, images, prompt, paths, calls, target):
    raw=route.ask(images,prompt)
    calls.append(dict(target=target,prompt=prompt,raw=raw,inputs=[str(p) for p in paths],api=route.records[-1]))
    return raw


def positives(route,ref,crop,box,paths,calls):
    pts=[]
    for target in ('dark_first','dark_second','white'):
        prompt=WHITE if target=='white' else DARK
        if target=='dark_second':
            if not pts: return pts
            prior=((np.array(pts[0])-box[:2])/(np.array(box[2:])-box[:2])*1000).round(2).tolist()
            prompt+=f' The first dark point has normalized Image 2 crop coordinates {prior}. Select a second spatially distinct dark interior point; do not repeat or place it immediately next to the first point.'
        raw=ask(route,[ref,crop],prompt,paths,calls,target)
        try:
            xy=point(raw,box)
            if not (np.array(xy)<np.array(box[2:])).all():raise ValueError('Point on excluded crop edge')
            if pts and np.linalg.norm(np.array(pts)-xy,axis=1).min()<5:raise ValueError('Duplicate or insufficiently distinct point (<5 source pixels)')
            pts.append(xy);calls[-1]['source_xy']=xy
        except (ValueError,TypeError,KeyError) as e:
            calls[-1]['error']=str(e)
    return pts


def run_experiment(a):
    import persistent_masking.gripper_review as review_module
    review_module.render=render
    a.work.mkdir(parents=True,exist_ok=False)
    route=VisionRoute(a.provider,a.model)
    run=dict(level='global',status='running',model=a.model,provider=a.provider,cases=a.cases,
             source_run=str(a.source),generation=dict(temperature=0,max_tokens=1800),
             implementation={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('vlm_client.py'))},
             sam_checkpoint=dict(path=str(Path(__file__).resolve().parents[1]/'models/sam3/sam3.pt'),
                 sha256=sha(Path(__file__).resolve().parents[1]/'models/sam3/sam3.pt')),
             source_inputs={case:{str(p.relative_to(a.source/case)):sha(p) for p in sorted((a.source/case/'inputs').glob('*.png'))} for case in a.cases},policy=dict(white=WHITE,dark=DARK,review=REVIEW,
             checkpoints=[.5,.75],crop='seed crop retained; temporal mask bbox expanded 2.5x, minimum 160x180 pixels',
             seed_negatives='retained previous arm/wrist points',minimum_positive_distance_source_pixels=5,
             seed_policy='requery all three positives for every case',sam='existing SAM3 predictor, independent corrected seed and forward-only replacement'),results=[])
    dump(a.work/'provenance.json',run)
    for case in a.cases:
        print('CASE',case,flush=True)
        out=a.work/case;inp=out/'inputs';inp.mkdir(parents=True)
        old=json.loads((a.source/case/'grounding.json').read_text())
        ref=Image.open(a.source/case/'inputs/reference.png').convert('RGB')
        crop=Image.open(a.source/case/'inputs/candidate.png').convert('RGB')
        ref.save(inp/'reference.png');crop.save(inp/'candidate.png')
        box=old['crop_box'];calls=[]
        pts=positives(route,ref,crop,box,[inp/'reference.png',inp/'candidate.png'],calls)
        neg=[p for p,l in zip(old['points'],old['labels']) if l==0]
        seed=dict(case=case,video=old['video'],frame=old['frame'],source_hw=old['source_hw'],
                  points=pts+neg,labels=[1]*len(pts)+[0]*len(neg),positive_count=len(pts),
                  status='ready' if len(pts)==3 else 'failed',calls=calls,crop_box=box,
                  negative_source=str(a.source/case/'grounding.json'))
        dump(out/'seed.json',seed)
        preview(load_frame(seed['video'],seed['frame']),seed['points'],seed['labels'],out/'points.png')
        result=dict(case=case,seed=seed,checks=[],status='incomplete')
        run['results'].append(result);dump(a.work/'provenance.json',run)
        if seed['status']!='ready':
            result['status']='failed_seed';dump(a.work/'provenance.json',run);continue
        args=SimpleNamespace(cases=[case],work=a.work,checkpoint=0,ffmpeg=a.ffmpeg)
        sam(args)
        current=out/'seed_masks.npz'
        for index,fraction in enumerate((.5,.75),1):
            masks=np.load(current)['masks'];f=int((len(masks)-1)*fraction)
            rgb=load_frame(seed['video'],f);h,w=rgb.shape[:2];ys,xs=np.where(masks[f]);box=[0,0,w,h]
            if len(xs):
                cx,cy=(xs.min()+xs.max())/2,(ys.min()+ys.max())/2
                sx,sy=max(160,(xs.max()-xs.min())*2.5),max(180,(ys.max()-ys.min())*2.5)
                box=[max(0,int(cx-sx/2)),max(0,int(cy-sy/2)),min(w,int(cx+sx/2)),min(h,int(cy+sy/2))]
            clean=Image.fromarray(rgb).crop(box);over=rgb.copy()
            over[masks[f]]=(.55*over[masks[f]]+.45*np.array([0,190,255])).astype(np.uint8)
            overlay=Image.fromarray(over).crop(box)
            cleanpath=inp/f'check_{index}_rgb.png';maskpath=inp/f'check_{index}_mask.png'
            clean.save(cleanpath);overlay.save(maskpath)
            calls=[];raw=ask(route,[ref,clean,overlay],REVIEW,[inp/'reference.png',cleanpath,maskpath],calls,'mask_review')
            rec=dict(frame=f,box=box,calls=calls,source_masks=str(current),status='failed',points=[],labels=[])
            try:
                response=json_object(raw);action=response['action']
                if action not in ('keep','correct','unclear'):raise ValueError('Invalid action')
                rec['reason']=response['reason'];rec['status']=action
                if action=='correct':
                    pts=positives(route,ref,clean,box,[inp/'reference.png',cleanpath],calls)
                    negatives=response['negative_points']
                    if not isinstance(negatives,list) or len(negatives)>2:raise ValueError('Invalid negatives')
                    neg=[point(json.dumps({'point_2d':p}),box) for p in negatives]
                    if len(pts)!=3:raise ValueError('Need three distinct visible positives')
                    allpts=np.array(pts+neg)
                    if (allpts[:,0]>=w).any() or (allpts[:,1]>=h).any():raise ValueError('Out of frame point')
                    if neg and (np.linalg.norm(np.array(pts)[:,None]-np.array(neg),axis=2)<5).any():raise ValueError('Conflicting positive/negative')
                    rec.update(points=pts+neg,labels=[1]*3+[0]*len(neg))
            except (ValueError,TypeError,KeyError) as e:rec.update(status='failed',error=str(e))
            dump(out/f'check_{index}.json',rec);result['checks'].append(rec)
            print(case,'CHECK',index,rec['status'],rec.get('reason'),flush=True)
            if rec['status']=='correct':
                args.checkpoint=index;sam(args);current=out/f'review_{index}_masks.npz'
                preview(rgb,rec['points'],rec['labels'],out/'points.png')
            dump(a.work/'provenance.json',run)
        masks=np.load(current)['masks']
        cap=cv2.VideoCapture(str(out/'masking.mp4'));decoded=0
        while True:
            ok,_=cap.read()
            if not ok:break
            decoded+=1
        cap.release()
        result.update(status='completed' if decoded==len(masks) else 'failed_replay',
                      validation=dict(decoded_frames=decoded,expected_frames=len(masks),empty_frames=np.flatnonzero(~masks.any((1,2))).tolist()),
                      replay_source_masks=str(current),masks_sha256=sha(current),replay_sha256=sha(out/'masking.mp4'),
                      seed_metrics=json.loads((out/'seed_metrics.json').read_text()),
                      correction_metrics=[json.loads(p.read_text()) for p in sorted(out.glob('review_*_metrics.json'))])
        result['inputs']={str(p.relative_to(out)):sha(p) for p in sorted(inp.glob('*.png'))}
        dump(a.work/'provenance.json',run)
    run['status']='completed' if all(r['status']=='completed' for r in run['results']) else 'incomplete'
    dump(a.work/'provenance.json',run)


def main(a):
    if a.work.exists():
        raise FileExistsError(f'Preserve existing experiment: {a.work}; choose a new run name')
    try:
        run_experiment(a)
    except Exception as error:
        provenance=a.work/'provenance.json'
        if provenance.exists():
            record=json.loads(provenance.read_text())
            record.update(status='failed',failure=dict(type=type(error).__name__,message=str(error)))
            dump(provenance,record)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--provider',choices=['qwen','openrouter'],default='openrouter');p.add_argument('--model',default='google/gemini-3.8-flash')
    p.add_argument('--cases',nargs='+',default=['Cosmos25_0023','Cosmos25_0053','LVP_0049','Cosmos3_0048']);p.add_argument('--ffmpeg',required=True)
    main(p.parse_args())
