"""Keep the grounding image focused on palm material, without masking its pixels."""

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'PDI-Bench-edited/persistent_masking/palm_tight_crop.py'
        break

import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT=_SOURCE_PATH.parents[1]


def main():
    parent=ROOT/'results_palm_recovery/after_first_surge'
    masks=np.load(ROOT/'reg2inv_results_v1/LVP_0049/guided_masks.npz')['masks']
    cap=cv2.VideoCapture(str(ROOT/'reg2inv_results/diagnosis_LVP_0049/source.mp4'))
    out=ROOT/'results_palm_recovery/literal_tight';out.mkdir(exist_ok=True)
    manifest=json.loads((parent/'manifest.json').read_text())
    for f in [0,13]:
        _,labels,stats,_=cv2.connectedComponentsWithStats(masks[f].astype(np.uint8),8)
        component=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);y,x=np.where(labels==component)
        width=x.max()-x.min()+1;height=y.max()-y.min()+1
        box=[int(x.min()-.5*width),int(y.min()-.9*height),int(x.max()+.5*width+1),int(y.max()+.25*height+1)]
        cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read()
        if not ok:raise ValueError('Missing RGB')
        Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop(box).save(out/('reference.png' if f==0 else 'frame_00013.png'))
        if f==0:manifest['reference_box']=box
        else:manifest['calls']=[dict(frame=13,time=13/16,crop='frame_00013.png',box_xyxy=box)]
    cap.release()
    manifest['purpose']='Grounding-only retry with unmasked focused crop; no new deformation classification'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'diagnoses.json').write_text((parent/'diagnoses.json').read_text())


if __name__=='__main__':main()
