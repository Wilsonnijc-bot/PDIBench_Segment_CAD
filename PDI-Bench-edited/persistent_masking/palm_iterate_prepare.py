"""Prepare a small fixed set of extra grounding frames for the same video."""
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image
from persistent_masking.palm_recovery import crop_box

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_palm_recovery/anatomical_edges'


def main():
    masks=np.load(ROOT/'reg2inv_results_v1/LVP_0049/guided_masks.npz')['masks']
    cap=cv2.VideoCapture(str(ROOT/'reg2inv_results/diagnosis_LVP_0049/source.mp4'))
    parent=ROOT/'results_palm_recovery/after_first_surge'
    reference=Image.open(parent/'reference.png')
    for frame in [13,20,28,36]:
        dest=OUT/f'frame_{frame:05d}';dest.mkdir(parents=True,exist_ok=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read()
        if not ok:raise ValueError('Missing frame')
        support=masks[max(0,frame-3):frame+1].any(0)
        box=crop_box(support,(80,80))
        Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop(box).save(dest/f'frame_{frame:05d}.png')
        shutil.copy2(parent/'reference.png',dest/'reference.png')
        shutil.copy2(parent/'diagnoses.json',dest/'diagnoses.json')
        manifest=json.loads((parent/'manifest.json').read_text())
        manifest['calls']=[dict(frame=frame,time=frame/16,crop=f'frame_{frame:05d}.png',box_xyxy=box)]
        manifest['purpose']='Explicit localization test frame; classification results remain the original 12-21 window'
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    cap.release();reference.close()


if __name__=='__main__':main()
