"""Compare actual SAM correction outputs with saved v1 masks on source RGB."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image,ImageDraw


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--video',type=Path,required=True);parser.add_argument('--original-masks',type=Path,required=True)
    parser.add_argument('--mask-replay',action='store_true',help='Explicitly export mask video; current default deliverables are Qwen inputs and points only')
    parser.add_argument('--diagnostic-frames',action='store_true',help='Explicitly export a contact sheet')
    args=parser.parse_args();out=args.output
    if not args.mask_replay:parser.error('Mask replay export requires --mask-replay; keep normal Qwen outputs in qwen_results')
    original=np.load(args.original_masks)['masks'];new=np.load(out/'proposed_masks.npz')['masks']
    assert original.shape==new.shape
    cap=cv2.VideoCapture(str(args.video));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
    while True:
        ok,bgr=cap.read()
        if not ok:break
        frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release();assert len(frames)==len(new)
    h,w=frames[0].shape[:2]
    tmp=out/'comparison_raw.mp4'
    writer=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w*2,h+40))
    sheet=Image.new('RGB',(1000,7*300),'white');metrics=[]
    samples=np.linspace(0,len(frames)-1,7).round().astype(int).tolist()
    for f,rgb in enumerate(frames):
        views=[]
        for mask,color in [(original[f],[255,155,0]),(new[f],[0,190,255])]:
            im=rgb.copy();im[mask]=(.55*im[mask]+.45*np.array(color)).astype(np.uint8);views.append(im)
        canvas=np.full((h+40,w*2,3),255,np.uint8)
        canvas[40:,:w]=views[0];canvas[40:,w:]=views[1]
        for x,title in [(10,'Reference mask'),(w+10,'New SAM proposal')]:
            cv2.putText(canvas,f'{title} | frame {f} | {f/fps:.3f}s',(x,27),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
        writer.write(cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
        metrics.append(dict(frame=f,original_area=int(original[f].sum()),new_area=int(new[f].sum()),
            added_pixels=int((new[f]&~original[f]).sum()),removed_pixels=int((original[f]&~new[f]).sum())))
        if f in samples:
            row=samples.index(f);union=original[f]|new[f];y,x=np.where(union)
            if not len(x):continue
            cx,cy=(x.min()+x.max())/2,(y.min()+y.max())/2
            side=max(180,x.max()-x.min()+40,y.max()-y.min()+40)
            box=(int(cx-side/2),int(cy-side/2),int(cx+side/2),int(cy+side/2))
            for j,view in enumerate([rgb,*views]):
                im=Image.fromarray(view).crop(box).resize((280,280))
                sheet.paste(im,(j*330,row*300));ImageDraw.Draw(sheet).text((j*330+4,row*300+282),
                    f'{f}: '+['RGB','Original mask','SAM proposal'][j],fill='black')
    writer.release()
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(tmp),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(out/'sam_comparison.mp4')],check=True);tmp.unlink()
    if args.diagnostic_frames:sheet.save(out/'sam_comparison.png')
    (out/'mask_comparison_metrics.json').write_text(json.dumps(metrics,indent=2))
    print('Rendered',len(frames),'frames')


if __name__=='__main__':main()
