"""Render prompt-role and immediate-mask diagnostics from saved SAM responses."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--video',type=Path,required=True)
    args=parser.parse_args();out=args.output
    masks=np.load(out/'seed_masks.npz');f=int(masks['frame'])
    cap=cv2.VideoCapture(str(args.video));cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();cap.release()
    if not ok:raise ValueError('Missing source frame')
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    correction=json.loads((out/'correction.json').read_text());points=np.array(correction['proposal']['points_xy']);labels=correction['proposal']['labels']
    union=masks['original']|masks['positives_only']|masks['with_negatives']
    y,x=np.where(union);cx,cy=(x.min()+x.max())/2,(y.min()+y.max())/2
    side=max(150,x.max()-x.min()+50,y.max()-y.min()+50)
    box=(int(cx-side/2),int(cy-side/2),int(cx+side/2),int(cy+side/2))
    canvas=Image.new('RGB',(1280,365),'white');draw=ImageDraw.Draw(canvas)
    for j,key in enumerate(['rgb','original','positives_only','with_negatives']):
        a=rgb.copy()
        if key!='rgb':
            m=masks[key];a[m]=(.55*a[m]+.45*np.array([0,190,255])).astype(np.uint8)
        im=Image.fromarray(a);d=ImageDraw.Draw(im)
        if key in ['positives_only','with_negatives']:
            for i,((px,py),label) in enumerate(zip(points,labels)):
                if key=='positives_only' and not label:continue
                color='lime' if label else 'red';d.ellipse((px-2,py-2,px+2,py+2),fill=color)
        canvas.paste(im.crop(box).resize((310,310)),(j*320,30))
        draw.text((j*320+5,7),{'rgb':f'RGB frame {f}','original':'Original input mask','positives_only':'Three positives only','with_negatives':'With negative prompt(s)'}[key],fill='black')
    draw.text((5,347),'Cyan = actual SAM mask. Green = positive prompts. Red = negative prompts.',fill='black')
    canvas.save(out/'seed_comparison.png')


if __name__=='__main__':main()
