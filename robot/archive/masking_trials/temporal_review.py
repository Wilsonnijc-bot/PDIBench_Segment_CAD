
# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'PDI-Bench-edited/persistent_masking/temporal_review.py'
        break

import argparse,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from persistent_masking.palm_recovery import json_object
from persistent_masking.gripper_negatives import video_for
ROOT=_SOURCE_PATH.parents[1]
def main(a):
 from generation.link_crop_wrapper.run_segment_vlm import load_model,generate
 proc,model,arch=load_model(a.model)
 for case in a.cases:
  out=a.work/case;g=json.loads((out/'grounding.json').read_text());m=np.load(out/'masks.npz')['masks'];f=int(len(m)*.75);cap=cv2.VideoCapture(g['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();cap.release();rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);mask=m[f];over=rgb.copy();over[mask]=(.55*over[mask]+.45*np.array([0,190,255])).astype(np.uint8);Image.fromarray(rgb).save(out/'inputs/review_rgb.png');Image.fromarray(over).save(out/'inputs/review_mask.png');ref=Image.open(a.results/case/'inputs/reference.png').convert('RGB');prompt='Image 1 is the original gripper reference. Image 2 is the clean later RGB frame. Image 3 is the same frame with the current SAM mask shown cyan. Check only whether visible WHITE palm housing or BLACK gripper material is missing from the mask. Return only JSON: {"action":"keep" or "correct" or "unclear","reason":"short visible evidence"}. Do not suggest points yet.';raw=generate(model,proc,[dict(role='user',content=[dict(type='image',image=ref),dict(type='image',image=Image.fromarray(rgb)),dict(type='image',image=Image.fromarray(over)),dict(type='text',text=prompt)])],180);json.dump(dict(case=case,frame=f,prompt=prompt,raw=raw),open(out/'temporal_review.json','w'),indent=2);print(case,raw,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--results',type=Path,default=ROOT/'qwen_results');p.add_argument('--model',type=Path,default=ROOT/'models/Qwen3.5-9B');p.add_argument('--cases',nargs='+',required=True);main(p.parse_args())
