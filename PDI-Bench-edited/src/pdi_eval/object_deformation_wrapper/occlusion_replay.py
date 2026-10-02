"""Export separate frame-synchronized occlusion catch replays from saved audits."""
from __future__ import annotations
import argparse
import base64
import html
import json
from pathlib import Path
import cv2
import numpy as np
from .occlusion import align, atomic, crop, sample, sha256


def export(case: Path):
    detection_path = case/'occlusion/detection.json'
    d = json.loads(detection_path.read_text())
    for item in d['inputs'].values():
        if sha256(Path(item['path'])) != item['sha256']:
            raise ValueError('Detection input changed: '+item['path'])
    with np.load(d['inputs']['object']['path']) as z: objects=z['object_masks'][:,0]
    with np.load(d['inputs']['gripper']['path']) as z: grippers=z['object_masks'][:,5]
    with np.load(d['inputs']['tracks']['path']) as z:
        i=z['object_names'].tolist().index('task_object');a,b=z['object_offsets'][i:i+2]
        tracks=z['tracks'][:,a:b];visibility=z['visibility'][:,a:b]
    cap=cv2.VideoCapture(str(case/'replay/source.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    if fps<=0: raise ValueError('Invalid source video frame rate')
    overlays=[]; boxes=[]
    for t,row in enumerate(d['frames']):
        obj,grip=objects[t],grippers[t]; overlay=np.zeros((*obj.shape,4),np.uint8)
        expected=None
        if row['expected_area'] is not None:
            expected=align(crop(objects[row['reference_frame']]),obj,grip)
            if expected is None or int(expected.sum())!=row['expected_area']:
                raise ValueError('Cannot reproduce expected silhouette')
            replaced=expected & ~obj & grip
            if abs(replaced.sum()/expected.sum()-row['occluded_fraction'])>1e-8:
                raise ValueError('Replay replacement differs from detection')
            overlay[replaced]=(85,110,255,165) # BGRA -> coral
            cv2.drawContours(overlay,cv2.findContours(expected.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(240,215,90,255),1)
        for mask,color in [(grip,(225,110,225,220)),(obj,(80,220,255,255))]:
            cv2.drawContours(overlay,cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,color,1)
        ref=row['reference_frame']
        if ref is not None and row.get('mask_valid', True):
            eligible=sample(objects[ref]&~grippers[ref],tracks[ref])&visibility[ref]
            hits=sample(grip,tracks[t])
            for xy,keep,hit in zip(tracks[t],eligible,hits):
                if keep and np.isfinite(xy).all():
                    x,y=np.rint(xy).astype(int)
                    if 0<=x<obj.shape[1] and 0<=y<obj.shape[0]:
                        cv2.circle(overlay,(x,y),1,(85,110,255,255) if hit else (230,250,240,230),-1)
        mask=obj if expected is None else obj|expected
        yy,xx=np.where(mask)
        if len(xx):
            side=max(100,int(max(xx.max()-xx.min(),yy.max()-yy.min())*2.1))
            side=min(side,*obj.shape)
            cx,cy=(xx.min()+xx.max())/2,(yy.min()+yy.max())/2
            boxes.append([int(np.clip(cx-side/2,0,obj.shape[1]-side)),int(np.clip(cy-side/2,0,obj.shape[0]-side)),side,side])
        else: boxes.append([0,0,obj.shape[1],obj.shape[0]])
        ok,png=cv2.imencode('.png',overlay)
        if not ok: raise ValueError('PNG encoding failed')
        overlays.append(base64.b64encode(png).decode())
    comparison_path=case/'score/rigidity_occlusion_filtered.json'
    d['comparison']=json.loads(comparison_path.read_text()) if comparison_path.exists() else None
    d.update(fps=fps,width=objects.shape[2],height=objects.shape[1],overlays=overlays,boxes=boxes)
    # Keep rendering data in one HTML file; source video remains an unmodified sibling asset.
    template=Path(__file__).with_name('occlusion_replay.html').read_text()
    atomic(case/'occlusion/replay.html',template.replace('__DATA__',json.dumps(d,separators=(',',':')).replace('</','<\\/')))
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cases',nargs='+');args=p.parse_args()
    existing={c.name for c in (args.root/'cases').iterdir() if (c/'occlusion/replay.html').exists()}
    selected=args.cases or sorted(existing|{'COSMOS3_0025','COSMOS3_0056','COSMOS2.5_0065'})
    for name in selected:
        d=export(args.root/'cases'/name)
        print(name,d['flagged_intervals'],flush=True)
    # A subset export must not remove other existing viewers from the index.
    rows=[]
    for name in sorted(existing|set(selected)):
        d=json.loads((args.root/'cases'/name/'occlusion/detection.json').read_text())
        intervals=', '.join(f'{a}–{b}' for a,b in d['flagged_intervals'])
        rows.append(f'<tr><td><a href="../../cases/{html.escape(name)}/occlusion/replay.html">{html.escape(name)}</a></td><td>{d["flagged_frames"]} / {d["frame_count"]}</td><td>{intervals}</td><td>{len(d.get("failed_mask_frames", []))}</td></tr>')
    atomic(args.root/'occlusion/replay/index.html','''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Occlusion catches</title><style>body{background:#12181a;color:#edf2ed;font:16px system-ui;margin:40px auto;padding:0 24px;max-width:1000px}h1{font-size:28px}p{color:#bac9c2;line-height:1.6;max-width:72ch}a{color:#f3c77c}table{border-collapse:collapse;width:100%;margin:30px 0}th,td{text-align:left;padding:18px 12px;border-bottom:1px solid #3b4845}th{color:#bac9c2;font-weight:500}@media(max-width:600px){th,td{padding:12px 5px;font-size:13px}}</style><h1>Occlusion catches</h1><p>Inspect the detector’s catches alongside the original video, a magnified object view, mask evidence, and rigidity history. Frame numbers are zero based.</p><table><thead><tr><th>Open replay</th><th>Flagged frames</th><th>Caught intervals</th><th>Failed mask frames</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table><p>Coral timeline bands are flagged intervals. Use “Previous catch” and “Next catch” to jump between them. These are occlusion-risk annotations; original rigidity scores are unchanged.</p><p><a href="../README.md">Method and limitations</a></p></html>''')


if __name__=='__main__': main()
