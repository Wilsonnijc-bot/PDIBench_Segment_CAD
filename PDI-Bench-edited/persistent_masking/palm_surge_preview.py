"""Show actual RGB and saved SAM masks around the two area triggers."""
import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_palm_recovery/LVP_0049'


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    masks=np.load(ROOT/'reg2inv_results_v1/LVP_0049/guided_masks.npz')['masks']
    cap=cv2.VideoCapture(str(ROOT/'reg2inv_results/diagnosis_LVP_0049/source.mp4'))
    frames=[]
    while True:
        ok,bgr=cap.read()
        if not ok:break
        frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release()
    fig,axes=plt.subplots(4,3,figsize=(12,12))
    for event_index,event in enumerate(manifest['event_frames']):
        ids=[event-1,event,event+1]
        union=masks[ids].any(0);y,x=np.where(union)
        cx,cy=(x.min()+x.max())/2,(y.min()+y.max())/2
        x0,x1=max(0,int(cx)-100),min(832,int(cx)+100)
        y0,y1=max(0,int(cy)-100),min(480,int(cy)+100)
        for col,f in enumerate(ids):
            rgb=frames[f];overlay=rgb.copy();mask=masks[f]
            overlay[mask]=(.55*overlay[mask]+.45*np.array([0,200,255])).astype(np.uint8)
            area=int(mask.sum());previous=int(masks[f-1].sum()) if f else area
            delta=(area/previous-1)*100 if previous else float('nan')
            for offset,image in enumerate([rgb,overlay]):
                ax=axes[event_index*2+offset,col]
                ax.imshow(image[y0:y1,x0:x1]);ax.axis('off')
                ax.set_title(f'Frame {f} | {f/16:.4f}s'+(' | TRIGGER' if f==event else '')+
                    (f'\nSAM area {area:,} px | change {delta:+.1f}%' if offset else '\nOriginal RGB'),fontsize=10)
    fig.suptitle('Palm mask area triggers: frame 12 shrinkage, frame 23 expansion\nCyan = actual saved CoTracker3-guided SAM mask',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'surge_frames.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
