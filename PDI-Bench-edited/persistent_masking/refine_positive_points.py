"""Experimental tight-crop prompt ablation followed by SAM3 propagation.

Not a shared production default: frozen-prompt regression failed on Cosmos3_0048.
See lasteset_qwen_results/provenance/frozen_prompt_controls.json.
"""
import argparse,json,hashlib,os,shutil,subprocess
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
from persistent_masking.gripper_negatives import video_for,dump
from persistent_masking.palm_recovery import json_object
ROOT=Path(__file__).resolve().parents[1]
CASES={'Cosmos25_0023':['white'],'Cosmos25_0053':['black_left','black_right','white'],'LVP_0049':['white'],'Cosmos3_0048':['black_left','black_right','white'],'Cosmos25_0003':['black_left','black_right','white']}
DESC={'black_left':'the lower dark/black gripper material below the white palm and below the shiny collar, toward the hanging fingers. Choose a broad interior pixel well inside the dark material; avoid the upper black rim, forearm trim, collar, wrist, white palm, fingers, and background','black_right':'a second spatially distinct point in the lower dark/black gripper material below the white palm and below the shiny collar, toward the hanging fingers. Choose a broad interior pixel well inside dark material; avoid the upper black rim, forearm trim, collar, wrist, white palm, fingers, and background','white':'the center of the hanging white gripper palm/block below the shiny collar. Choose white palm material, away from the collar, wrist, dark rim, fingers, and background'}

def parse(raw):
 d=json_object(raw)
 if 'point' in d: v=d['point']
 elif 'point_2d' in d:v=d['point_2d']
 else:
  if isinstance(d,list):v=d[0].get('point_2d',d[0].get('point'))
  else: raise ValueError('missing point')
 a=np.asarray(v,float)
 if a.shape!=(2,) or not np.isfinite(a).all() or (a<0).any() or (a>1000).any():raise ValueError('invalid point')
 return a
def run_ground(a):
 from generation.link_crop_wrapper.run_segment_vlm import load_model,generate
 proc,model,arch=load_model(a.model);manifest=json.loads((a.results/'provenance/manifest.json').read_text());old=json.loads((a.results/'provenance/arm_wrist.json').read_text())
 for case,targets in CASES.items():
  if a.cases and case not in a.cases:continue
  r=next(x for x in manifest if x['case']==case);o=next(x for x in old if x['case']==case);f=o['frame'];call=next(x for x in r['crop_boxes'] if x['frame']==f);box=np.array(call['box_xyxy'],float)
  cropfile=a.results/case/'inputs'/call['crop'];ref=Image.open(a.results/case/'inputs/reference.png').convert('RGB');crop=Image.open(cropfile).convert('RGB');out=a.work/case; (out/'inputs').mkdir(parents=True,exist_ok=True);crop.save(out/'inputs/candidate.png');ref.save(out/'inputs/reference.png')
  h,w=o['source_hw']; points=[];labels=[];records=[]
  oldpos=o['positives']
  if 'black_left' not in targets: points.extend(oldpos[:2]);labels.extend([1,1])
  for t in targets:
   prior=((np.asarray(points[-1:])-box[:2])/(box[2:]-box[:2])*1000).round().astype(int).tolist() if t=='black_right' and points else []
   prompt=f'Image 1 is the original gripper reference. Image 2 is the tight candidate crop. Locate ONE point strictly inside {DESC[t]}. Return only JSON {{"point_2d":[x,y]}} with integer coordinates 0 to 1000 relative to Image 2. If uncertain return {{"point_2d":null}}. No explanation.'
   if t=='white':prompt='Image 1 is the reference. In Image 2, point at the center of the hanging WHITE gripper block in the lower part of the image. Return one point_2d in JSON, coordinates 0 to 1000 relative to Image 2.'
   if t=='black_right' and prior:prompt+=f' The first dark point is {prior[0]}. Select a different location; do not repeat that coordinate.'
   raw=generate(model,proc,[dict(role='user',content=[dict(type='image',image=ref),dict(type='image',image=crop),dict(type='text',text=prompt)])],120);rec={'target':t,'prompt':prompt,'raw':raw}
   try:
    xy=parse(raw)/1000*(box[2:]-box[:2])+box[:2]
    if points and np.linalg.norm(np.asarray(points)-xy,axis=1).min()<5:raise ValueError('duplicate point')
    points.append(xy.tolist());labels.append(1);rec['source_xy']=xy.tolist()
   except (ValueError,TypeError,KeyError) as e:rec['error']=str(e)
   records.append(rec)
  status='ready' if len(points)==3 else 'failed';positive_count=len(points);points+=list(o['negatives'].values());labels+=[0]*len(o['negatives']);result=dict(case=case,frame=f,video=str(video_for(case)),source_hw=[h,w],crop_box=box.tolist(),points=points,labels=labels,status=status,positive_count=positive_count,negative_source='retained arm_wrist.json',calls=records,architecture=arch)
  dump(out/'grounding.json',result);im=crop.copy();d=ImageDraw.Draw(im)
  for i,(xy,label) in enumerate(zip(points,labels)):
   x,y=np.asarray(xy)-box[:2];color='lime' if label else 'red';d.ellipse((x-4,y-4,x+4,y+4),fill=color);d.text((x+5,y),f'{i+1}',fill=color)
  im.save(out/'points.png');print(case,status,points,flush=True)
def run_sam(a):
 from persistent_masking.v1_mask import predictor
 for case in CASES:
  if a.cases and case not in a.cases:continue
  o=json.loads((a.work/case/'grounding.json').read_text())
  if o['status']!='ready':continue
  cap=cv2.VideoCapture(o['video']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release();h,w=o['source_hw'];p=predictor();sid=p.handle_request(dict(type='start_session',resource_path=o['video'],offload_video_to_cpu=True))['session_id'];m=np.zeros((n,h,w),bool);f=o['frame'];pts=np.asarray(o['points']);labs=o['labels']
  try:
   d=p.handle_request(dict(type='add_prompt',session_id=sid,frame_index=f,obj_id=0,points=(pts/[w,h]).tolist(),point_labels=labs))['outputs'];ids=list(d['out_obj_ids']);m[f]=d['out_binary_masks'][ids.index(0)];xy=np.floor(pts).astype(int);membership=m[f,xy[:,1],xy[:,0]].tolist();dump(a.work/case/'validation.json',dict(points=pts.tolist(),labels=labs,membership=membership,membership_ok=membership==[bool(x) for x in labs]))
   for direction,lim in [('forward',n-f-1),('backward',f)]:
    for z in p.handle_stream_request(dict(type='propagate_in_video',session_id=sid,propagation_direction=direction,start_frame_index=f,max_frame_num_to_track=lim)):
     t=int(z['frame_index']);ids=list(z['outputs']['out_obj_ids']);
     if 0 in ids:m[t]=z['outputs']['out_binary_masks'][ids.index(0)]
   np.savez_compressed(a.work/case/'masks.npz',masks=m);ff=a.ffmpeg;raw=a.work/case/'raw.mp4';cap=cv2.VideoCapture(o['video']);fps=cap.get(cv2.CAP_PROP_FPS);wr=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(2*w,h))
   for mask in m:
    ok,bgr=cap.read(); over=bgr.copy();over[mask]=(.55*over[mask]+.45*np.array([255,190,0])).astype(np.uint8);wr.write(np.concatenate([bgr,over],1))
   cap.release();wr.release();subprocess.run([ff,'-y','-loglevel','error','-i',str(raw),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(a.work/case/'masking_refined.mp4')],check=True);raw.unlink();print(case,'mask complete',flush=True)
  finally:p.handle_request(dict(type='close_session',session_id=sid));p.shutdown()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['ground','sam']);p.add_argument('--work',type=Path,required=True);p.add_argument('--results',type=Path,default=ROOT/'lasteset_qwen_results');p.add_argument('--model',type=Path,default=ROOT/'models/Qwen3.5-9B');p.add_argument('--cases',nargs='+',choices=list(CASES));p.add_argument('--ffmpeg',required=True);a=p.parse_args();(run_ground if a.stage=='ground' else run_sam)(a)
