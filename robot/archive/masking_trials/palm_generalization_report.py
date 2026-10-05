"""Render saved cross-video masks and summarize operational results."""

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'PDI-Bench-edited/persistent_masking/palm_generalization_report.py'
        break

import argparse
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=_SOURCE_PATH.parents[1];OUT=ROOT/'results_palm_generalization'


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--positive-only',action='store_true')
    parser.add_argument('--mask-replay',action='store_true',help='Explicitly export mask replays')
    parser.add_argument('--diagnostic-frames',action='store_true',help='Explicitly export contact sheets')
    args=parser.parse_args();OUT=args.output
    if not args.mask_replay:parser.error('Mask replay export requires --mask-replay; current Qwen deliverables are inputs and point previews only')
    records=json.loads((OUT/'summary.json').read_text());stats=[]
    for row in records:
        case=row['case'];dest=OUT/case
        if not (dest/'guided_masks.npz').exists():continue
        sam=np.load(dest/'sam_only_masks.npz')['masks'];guided=np.load(dest/'guided_masks.npz')['masks']
        proposal=dest/('recovery/proposed_masks.npz' if args.positive_only else 'recovery/lower_boundary_negative_trial/proposed_masks.npz')
        final=np.load(proposal)['masks'] if proposal.exists() else guided
        video=OUT/'source_videos'/f'{case.split("_")[1]}.mp4'
        cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
        while True:
            ok,bgr=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        cap.release();count=len(frames);assert count==len(sam)==len(guided)==len(final)
        h,w=frames[0].shape[:2];assert sam.shape==guided.shape==final.shape==(count,h,w)
        tmp=dest/'comparison_raw.mp4';writer=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w*2,h+36))
        sheet=Image.new('RGB',(960,7*310),'white');metrics=[]
        samples=np.linspace(0,count-1,7).round().astype(int).tolist()
        for f,rgb in enumerate(frames):
            views=[]
            for mask,color in [(sam[f],[255,155,0]),(final[f],[0,190,255])]:
                a=rgb.copy();a[mask]=(.55*a[mask]+.45*np.array(color)).astype(np.uint8);views.append(a)
            canvas=np.full((h+36,w*2,3),255,np.uint8);canvas[36:,:w]=views[0];canvas[36:,w:]=views[1]
            for x,title in [(10,'SAM propagation control'),(w+10,'SAM proposal; unverified target' if proposal.exists() else 'CoTracker-guided; no accepted correction')]:
                cv2.putText(canvas,f'{case} | {f}: {title}',(x,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
            writer.write(cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
            if f in samples:
                union=sam[f]|guided[f]|final[f];y,x=np.where(union)
                if len(x):
                    cx,cy=(x.min()+x.max())/2,(y.min()+y.max())/2;side=max(150,x.max()-x.min()+40,y.max()-y.min()+40)
                    box=(int(cx-side/2),int(cy-side/2),int(cx+side/2),int(cy+side/2))
                    for j,im in enumerate([rgb,*views]):
                        sheet.paste(Image.fromarray(im).crop(box).resize((295,295)),(j*320,samples.index(f)*310))
                        ImageDraw.Draw(sheet).text((j*320+3,samples.index(f)*310+297),f'{f}: '+['RGB','SAM control','Unverified proposal' if proposal.exists() else 'Tracking; no correction'][j],fill='black')
            metrics.append(dict(frame=f,sam_area=int(sam[f].sum()),guided_area=int(guided[f].sum()),final_area=int(final[f].sum())))
        writer.release()
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(tmp),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/'mask_comparison.mp4')],check=True);tmp.unlink()
        if args.diagnostic_frames:sheet.save(dest/'mask_contact_sheet.png')
        check=cv2.VideoCapture(str(dest/'mask_comparison.mp4'));assert int(check.get(cv2.CAP_PROP_FRAME_COUNT))==count;check.release()
        (dest/'mask_metrics.json').write_text(json.dumps(metrics,indent=2))
        area=final.sum((1,2));track=np.load(dest/'tracks.npz')
        stats.append(dict(case=case,status=row['status'],frames=count,nonempty=int(np.count_nonzero(area)),
            final_mask_source='corrected_proposal' if proposal.exists() else 'guided_masks',
            minimum_visible_positive=int((track['visible'].astype(bool)&(track['labels']==1)[None,:]).sum(1).min()),
            large_area_transitions=int((np.abs(np.diff(np.log(np.maximum(area,1))))>=.25).sum()),
            min_area=int(area.min()),max_area=int(area.max()),candidate_frames=row.get('candidate_frames',[]),diagnoses=row.get('diagnoses')))
    (OUT/'validation_summary.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
