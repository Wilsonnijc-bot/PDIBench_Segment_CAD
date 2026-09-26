"""One lower-boundary negative trial; preserve the accepted positive points."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image,ImageDraw
from persistent_masking.palm_recovery import json_object


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    args=p.parse_args();base=args.base;out=base/'lower_boundary_negative_trial';out.mkdir(exist_ok=True)
    from generation.link_crop_wrapper.run_segment_vlm import load_model,generate
    manifest=json.loads((base/'manifest.json').read_text());old=json.loads((base/'correction.json').read_text())
    f=old['frame'];call=next(c for c in manifest['calls'] if c['frame']==f);box=call['box_xyxy']
    request='In Image 2, place one negative point right outside the lower edge of the palm and fingers, on the non-robot side of their outer boundary. Do not place it on the boundary or inside the palm or any finger. If an object occludes the boundary, choose another clearly visible location along the lower edge. Return one point_2d in JSON, using coordinates from 0 to 1000. If no such location is clear, return {"point_2d":null}.'
    processor,model,architecture=load_model(args.model)
    with Image.open(base/'reference.png') as ref,Image.open(base/call['crop']) as candidate:
        messages=[dict(role='system',content=[dict(type='text',text='Image 1 shows the original gripper. Image 2 is the target. Locate the requested negative point precisely. Output JSON only.')]),
            dict(role='user',content=[dict(type='image',image=ref.convert('RGB')),dict(type='image',image=candidate.convert('RGB')),dict(type='text',text=request)])]
        raw=generate(model,processor,messages,100)
    record=dict(prompt=request,prompt_version='lower-boundary-v2',raw=raw,architecture=architecture,frame=f,positive_points_changed=False)
    result=dict(old)
    keep=[(xy,label) for xy,label in zip(old['proposal']['points_xy'],old['proposal']['labels']) if label==1]
    assert len(keep)==3
    points=[xy for xy,_ in keep];labels=[1]*3
    try:
        data=json_object(raw);xy=np.asarray(data['point_2d'],dtype=float)
        if xy.shape!=(2,) or not np.isfinite(xy).all() or (xy<0).any() or (xy>1000).any():raise ValueError('No valid lower-boundary coordinate')
        mapped=xy/1000*[box[2]-box[0],box[3]-box[1]]+box[:2]
        shape=np.load(manifest['masks'])['masks'].shape
        if min(mapped)<0 or mapped[0]>=shape[2] or mapped[1]>=shape[1]:raise ValueError('Point outside frame')
        if min(np.linalg.norm(np.asarray(points)-mapped,axis=1))<3:raise ValueError('Negative too close to positive')
        points.append(mapped.tolist());labels.append(0);record.update(status='well_formed_unverified',source_xy=mapped.tolist())
        result.update(status='ready_for_sam',proposal=dict(status='ok',points_xy=points,labels=labels,evidence='Three retained positives plus one Qwen negative outside the palm/finger lower boundary; not anatomically verified ground truth'))
    except (ValueError,KeyError,TypeError) as e:record.update(status='localization_failed',error=str(e));result['status']='localization_failed'
    (out/'finger_point.json').write_text(json.dumps(record,indent=2))
    (out/'correction.json').write_text(json.dumps(result,indent=2))
    shutil.copy2(base/'manifest.json',out/'manifest.json')
    im=Image.open(base/call['crop']).convert('RGB');draw=ImageDraw.Draw(im)
    for i,(xy,label) in enumerate(zip(points,labels)):
        x,y=np.asarray(xy)-box[:2];color='lime' if label else 'red'
        draw.ellipse((x-2,y-2,x+2,y+2),fill=color);draw.text((x+3,y),('P' if label else 'N')+str(i),fill=color)
    im.resize((im.width*3,im.height*3)).save(out/'point_preview.png')
    print(record,flush=True)


if __name__=='__main__':main()
