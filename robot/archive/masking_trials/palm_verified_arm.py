"""Prepare explicitly analyst-placed arm negatives for a diagnostic ablation."""

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'PDI-Bench-edited/persistent_masking/palm_verified_arm.py'
        break

import json
from pathlib import Path
import shutil
from PIL import Image,ImageDraw

ROOT=_SOURCE_PATH.parents[1]
POINTS={'Cosmos25_0003':[845.,135.],'Cosmos3_0048':[875.,70.]}


def main():
    for case,point in POINTS.items():
        base=ROOT/'results_palm_cosmos'/case/'recovery';out=base/'arm_negative_verified';out.mkdir(exist_ok=True)
        data=json.loads((base/'correction.json').read_text())
        assert data['proposal']['labels']==[1,1,1]
        data['proposal']['points_xy'].append(point);data['proposal']['labels'].append(0)
        data['proposal']['evidence']='Original three model-positive points plus one analyst-placed arm-interior negative; diagnostic only, not automatic grounding'
        data['negative_source']='analyst_visual_placement'
        (out/'correction.json').write_text(json.dumps(data,indent=2));shutil.copy2(base/'manifest.json',out/'manifest.json')
        image=Image.open(base/'arm_negative/full_candidate.png').convert('RGB');draw=ImageDraw.Draw(image)
        for i,(xy,label) in enumerate(zip(data['proposal']['points_xy'],data['proposal']['labels'])):
            x,y=xy;color='lime' if label else 'red';draw.ellipse((x-5,y-5,x+5,y+5),fill=color);draw.text((x+6,y),('P' if label else 'N')+str(i),fill=color)
        image.save(out/'point_preview.png')


if __name__=='__main__':main()
